"""
Perfect Foresight MILP Teacher for VPP Optimization.

This module implements the offline optimization component that generates
optimal dispatch trajectories given perfect price foresight. These trajectories
serve as training labels for the neural network student model.

Mathematical Formulation:
    max  Σ(P_discharge,t · π_sell - P_charge,t · π_buy)
    s.t. SOC_{t+1} = SOC_t + (P_charge · η_ch - P_discharge / η_dis) · Δt / E_cap
         SOC_min ≤ SOC_t ≤ SOC_max
         0 ≤ P_charge,t ≤ P_max
         0 ≤ P_discharge,t ≤ P_max
"""

import time
from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


@dataclass
class BatteryConstraints:
    """Battery physical constraints for optimization.

    Attributes:
        energy_capacity: E_cap in kWh - total energy storage capacity
        max_power: P_max in kW - maximum charge/discharge power
        charge_efficiency: η_ch (0-1) - charging efficiency
        discharge_efficiency: η_dis (0-1) - discharging efficiency
        soc_min: Minimum allowed state of charge (default 0.1)
        soc_max: Maximum allowed state of charge (default 0.9)
    """
    energy_capacity: float
    max_power: float
    charge_efficiency: float
    discharge_efficiency: float
    soc_min: float = 0.1
    soc_max: float = 0.9

    def __post_init__(self):
        """Validate constraint values."""
        if self.energy_capacity <= 0:
            raise ValueError("energy_capacity must be positive")
        if self.max_power <= 0:
            raise ValueError("max_power must be positive")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("soc_min and soc_max must satisfy 0 <= soc_min < soc_max <= 1")


@dataclass
class TrajectoryResult:
    """Result of perfect foresight optimization.

    Attributes:
        power: Net power trajectory P_t (positive=discharge/sell, negative=charge/buy)
        soc: State of charge trajectory SOC_t
        charge_power: Charging power P_charge,t >= 0
        discharge_power: Discharging power P_discharge,t >= 0
        profit: Total profit (objective value)
        status: Solver status ("optimal", "infeasible", "unbounded", etc.)
        solve_time: Time taken to solve in seconds
    """
    power: np.ndarray
    soc: np.ndarray
    charge_power: np.ndarray
    discharge_power: np.ndarray
    profit: float
    status: str
    solve_time: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "power": self.power.tolist(),
            "soc": self.soc.tolist(),
            "charge_power": self.charge_power.tolist(),
            "discharge_power": self.discharge_power.tolist(),
            "profit": self.profit,
            "status": self.status,
            "solve_time": self.solve_time
        }


class PerfectForesightOptimizer:
    """Perfect foresight optimizer using convex optimization (CVXPY).

    Solves the battery arbitrage problem given perfect knowledge of future prices.
    Uses LP relaxation (no binary variables) - price spread naturally prevents
    simultaneous charging and discharging in most cases.

    Example:
        >>> constraints = BatteryConstraints(
        ...     energy_capacity=100.0,  # 100 kWh battery
        ...     max_power=50.0,         # 50 kW max power
        ...     charge_efficiency=0.95,
        ...     discharge_efficiency=0.95
        ... )
        >>> optimizer = PerfectForesightOptimizer(time_step=0.25)  # 15 min
        >>> prices = np.array([0.10, 0.08, 0.15, 0.20])  # $/kWh
        >>> result = optimizer.solve_horizon(prices, load=None, initial_soc=0.5, constraints=constraints)
        >>> print(f"Profit: ${result.profit:.2f}")
    """

    def __init__(self, time_step: float = 0.25):
        """Initialize the optimizer.

        Args:
            time_step: Duration of each time step in hours (default 0.25 = 15 minutes)
        """
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for PerfectForesightOptimizer. Install with: pip install cvxpy")

        if time_step <= 0:
            raise ValueError("time_step must be positive")

        self.time_step = time_step

    def solve_horizon(
        self,
        prices: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        load: Optional[np.ndarray],
        initial_soc: float,
        constraints: BatteryConstraints
    ) -> TrajectoryResult:
        """Solve the perfect foresight optimization problem.

        Args:
            prices: Either a single price array (buy=sell) or tuple (π_buy, π_sell).
                    Units: $/kWh or €/kWh
            load: Load profile (reserved for future use, currently ignored in pure arbitrage)
            initial_soc: Initial state of charge (0-1)
            constraints: Battery physical constraints

        Returns:
            TrajectoryResult with optimal dispatch trajectory

        Raises:
            ValueError: If inputs are invalid
        """
        start_time = time.time()

        # Parse prices
        if isinstance(prices, tuple):
            prices_buy, prices_sell = prices
            prices_buy = np.asarray(prices_buy, dtype=np.float64)
            prices_sell = np.asarray(prices_sell, dtype=np.float64)
            if len(prices_buy) != len(prices_sell):
                raise ValueError("prices_buy and prices_sell must have same length")
        else:
            prices_buy = np.asarray(prices, dtype=np.float64)
            prices_sell = prices_buy

        T = len(prices_buy)  # Number of time steps

        # Validate initial SOC
        if not constraints.soc_min <= initial_soc <= constraints.soc_max:
            raise ValueError(
                f"initial_soc ({initial_soc}) must be within [{constraints.soc_min}, {constraints.soc_max}]"
            )

        # Decision variables
        P_charge = cp.Variable(T, nonneg=True)      # Charging power >= 0
        P_discharge = cp.Variable(T, nonneg=True)   # Discharging power >= 0
        SOC = cp.Variable(T + 1)                    # SOC at each time step (including final)

        # Objective: Maximize profit
        # Revenue from discharging - Cost of charging
        # Note: P_discharge sells energy, P_charge buys energy
        profit = cp.sum(cp.multiply(P_discharge, prices_sell) - cp.multiply(P_charge, prices_buy)) * self.time_step
        objective = cp.Maximize(profit)

        # Constraints
        constraint_list = []

        # Initial SOC constraint
        constraint_list.append(SOC[0] == initial_soc)

        # SOC dynamics for each time step
        # SOC_{t+1} = SOC_t + (P_charge * η_ch - P_discharge / η_dis) * Δt / E_cap
        for t in range(T):
            energy_in = P_charge[t] * constraints.charge_efficiency * self.time_step
            energy_out = P_discharge[t] / constraints.discharge_efficiency * self.time_step
            delta_soc = (energy_in - energy_out) / constraints.energy_capacity
            constraint_list.append(SOC[t + 1] == SOC[t] + delta_soc)

        # SOC limits for all time steps (including final)
        constraint_list.append(SOC >= constraints.soc_min)
        constraint_list.append(SOC <= constraints.soc_max)

        # Power limits
        constraint_list.append(P_charge <= constraints.max_power)
        constraint_list.append(P_discharge <= constraints.max_power)

        # Formulate and solve the problem
        problem = cp.Problem(objective, constraint_list)

        try:
            problem.solve(solver=cp.ECOS)  # ECOS is reliable for LP/QP
        except cp.SolverError:
            # Try alternative solver
            try:
                problem.solve(solver=cp.SCS)
            except cp.SolverError:
                problem.solve()  # Use default solver

        solve_time = time.time() - start_time

        # Extract results
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            power = P_discharge.value - P_charge.value  # Net power (positive = selling)
            soc = SOC.value
            charge_power = P_charge.value
            discharge_power = P_discharge.value
            profit_value = problem.value
            status = "optimal" if problem.status == cp.OPTIMAL else "optimal_inaccurate"
        else:
            # Return zero trajectory for failed optimization
            power = np.zeros(T)
            soc = np.full(T + 1, initial_soc)
            charge_power = np.zeros(T)
            discharge_power = np.zeros(T)
            profit_value = 0.0
            status = problem.status

        return TrajectoryResult(
            power=power,
            soc=soc,
            charge_power=charge_power,
            discharge_power=discharge_power,
            profit=profit_value,
            status=status,
            solve_time=solve_time
        )

    def solve_rolling_horizon(
        self,
        prices: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        load: Optional[np.ndarray],
        initial_soc: float,
        constraints: BatteryConstraints,
        horizon_length: int,
        step_size: int = 1
    ) -> TrajectoryResult:
        """Solve using rolling horizon (receding horizon) approach.

        Useful for very long time series where solving the full problem at once
        is computationally expensive.

        Args:
            prices: Price array(s) for full period
            load: Load profile (reserved for future use)
            initial_soc: Initial state of charge
            constraints: Battery constraints
            horizon_length: Number of steps to look ahead in each solve
            step_size: Number of steps to advance between solves

        Returns:
            TrajectoryResult for the full period
        """
        # Parse prices
        if isinstance(prices, tuple):
            prices_buy, prices_sell = prices
            prices_buy = np.asarray(prices_buy, dtype=np.float64)
            prices_sell = np.asarray(prices_sell, dtype=np.float64)
        else:
            prices_buy = np.asarray(prices, dtype=np.float64)
            prices_sell = prices_buy

        T = len(prices_buy)

        # Initialize result arrays
        full_power = np.zeros(T)
        full_soc = np.zeros(T + 1)
        full_charge = np.zeros(T)
        full_discharge = np.zeros(T)
        full_soc[0] = initial_soc

        total_profit = 0.0
        total_solve_time = 0.0
        current_soc = initial_soc
        status = "optimal"

        t = 0
        while t < T:
            # Determine horizon for this solve
            horizon_end = min(t + horizon_length, T)

            # Extract prices for this horizon
            if isinstance(prices, tuple):
                horizon_prices = (prices_buy[t:horizon_end], prices_sell[t:horizon_end])
            else:
                horizon_prices = prices_buy[t:horizon_end]

            # Solve for this horizon
            result = self.solve_horizon(
                prices=horizon_prices,
                load=load[t:horizon_end] if load is not None else None,
                initial_soc=current_soc,
                constraints=constraints
            )

            total_solve_time += result.solve_time

            if result.status not in ["optimal", "optimal_inaccurate"]:
                status = result.status
                break

            # Apply first `step_size` decisions
            apply_steps = min(step_size, horizon_end - t)
            full_power[t:t + apply_steps] = result.power[:apply_steps]
            full_charge[t:t + apply_steps] = result.charge_power[:apply_steps]
            full_discharge[t:t + apply_steps] = result.discharge_power[:apply_steps]
            full_soc[t + 1:t + apply_steps + 1] = result.soc[1:apply_steps + 1]

            # Calculate profit for applied steps
            if isinstance(prices, tuple):
                step_profit = np.sum(
                    result.discharge_power[:apply_steps] * prices_sell[t:t + apply_steps] -
                    result.charge_power[:apply_steps] * prices_buy[t:t + apply_steps]
                ) * self.time_step
            else:
                step_profit = np.sum(result.power[:apply_steps] * prices_buy[t:t + apply_steps]) * self.time_step
            total_profit += step_profit

            # Update current SOC and advance
            current_soc = result.soc[apply_steps]
            t += apply_steps

        return TrajectoryResult(
            power=full_power,
            soc=full_soc,
            charge_power=full_charge,
            discharge_power=full_discharge,
            profit=total_profit,
            status=status,
            solve_time=total_solve_time
        )
