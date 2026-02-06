"""
MPC-Based Training Data Generation for VPP Optimization.

This module generates training data using a Model Predictive Control approach:
1. True battery state tracked by NonlinearBatterySimulator (nonlinear physics)
2. PerfectForesightOptimizer plans with linear approximation
3. Only first action taken, then true physics stepped forward
4. Model mismatch creates realistic correction behavior for student to learn

Usage:
    python -m vpp.experiments.generate_data --days 365 --output data/training_data.csv
"""

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..optimization.offline import BatteryConstraints, PerfectForesightOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for data generation."""

    n_days: int = 365
    time_step_minutes: int = 15
    lookahead_hours: int = 24
    reoptimize_hours: int = 24  # Re-solve optimization every N hours (daily = 24)
    random_seed: int = 42

    # Battery parameters
    energy_capacity_kwh: float = 100.0
    max_power_kw: float = 50.0
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    soc_min: float = 0.1
    soc_max: float = 0.9
    initial_soc: float = 0.5

    # Price parameters
    base_price: float = 0.10  # $/kWh
    price_volatility: float = 0.15

    # Load parameters
    base_load_kw: float = 30.0
    load_volatility: float = 0.2

    @property
    def steps_per_day(self) -> int:
        return (24 * 60) // self.time_step_minutes

    @property
    def total_steps(self) -> int:
        return self.n_days * self.steps_per_day

    @property
    def lookahead_steps(self) -> int:
        return (self.lookahead_hours * 60) // self.time_step_minutes

    @property
    def reoptimize_steps(self) -> int:
        return (self.reoptimize_hours * 60) // self.time_step_minutes

    @property
    def time_step_hours(self) -> float:
        return self.time_step_minutes / 60.0


class PriceGenerator:
    """Generates realistic electricity price profiles."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

    def generate(self, n_steps: int) -> np.ndarray:
        """Generate price profile for n_steps.

        Price model:
        - Base price with daily pattern (peak morning/evening)
        - Weekly pattern (lower weekend prices)
        - Random walk component for day-to-day variation
        - Occasional price spikes
        """
        prices = np.zeros(n_steps)
        steps_per_day = self.config.steps_per_day

        # Random walk for daily base variation
        daily_drift = np.cumsum(self.rng.normal(0, 0.005, n_steps // steps_per_day + 1))
        daily_drift = np.clip(daily_drift, -0.3, 0.3)

        for t in range(n_steps):
            day = t // steps_per_day
            step_in_day = t % steps_per_day
            hour = (step_in_day * self.config.time_step_minutes) / 60.0

            # Daily pattern: peaks at 8am and 6pm
            morning_peak = np.exp(-((hour - 8) ** 2) / 8)
            evening_peak = np.exp(-((hour - 18) ** 2) / 8)
            daily_pattern = 1.0 + 0.5 * (morning_peak + 1.2 * evening_peak)

            # Night valley
            if 0 <= hour < 6:
                daily_pattern *= 0.6

            # Weekly pattern (lower on weekends)
            day_of_week = day % 7
            if day_of_week >= 5:  # Weekend
                weekly_factor = 0.85
            else:
                weekly_factor = 1.0

            # Combine factors
            base = self.config.base_price * daily_pattern * weekly_factor

            # Add daily drift
            base *= 1 + daily_drift[day]

            # Add noise
            noise = self.rng.normal(0, self.config.price_volatility * self.config.base_price)
            prices[t] = max(0.02, base + noise)

            # Occasional spikes (0.5% chance)
            if self.rng.random() < 0.005:
                prices[t] *= self.rng.uniform(1.5, 3.0)

        return prices


class LoadGenerator:
    """Generates realistic load profiles."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed + 1)

    def generate(self, n_steps: int) -> np.ndarray:
        """Generate load profile for n_steps.

        Load model:
        - Base load with daily pattern
        - Higher during business hours
        - Lower at night and weekends
        """
        loads = np.zeros(n_steps)
        steps_per_day = self.config.steps_per_day

        for t in range(n_steps):
            day = t // steps_per_day
            step_in_day = t % steps_per_day
            hour = (step_in_day * self.config.time_step_minutes) / 60.0

            # Daily pattern
            if 6 <= hour < 9:  # Morning ramp
                daily_pattern = 0.7 + 0.3 * (hour - 6) / 3
            elif 9 <= hour < 17:  # Business hours
                daily_pattern = 1.0 + 0.1 * np.sin(np.pi * (hour - 9) / 8)
            elif 17 <= hour < 22:  # Evening
                daily_pattern = 1.1 - 0.3 * (hour - 17) / 5
            else:  # Night
                daily_pattern = 0.5

            # Weekly pattern
            day_of_week = day % 7
            if day_of_week >= 5:  # Weekend
                weekly_factor = 0.7
            else:
                weekly_factor = 1.0

            # Combine
            base = self.config.base_load_kw * daily_pattern * weekly_factor

            # Add noise
            noise = self.rng.normal(0, self.config.load_volatility * self.config.base_load_kw)
            loads[t] = max(5.0, base + noise)

        return loads


class NonlinearBatterySimulator:
    """Nonlinear battery simulator for data generation.

    Provides realistic model mismatch vs the linear optimizer through:
    - Energy-based SOC dynamics (correct conservation)
    - C-rate dependent efficiency (η drops at high power)
    - Soft power limits near SOC boundaries
    - Self-discharge

    The linear optimizer assumes constant efficiency and hard SOC limits,
    so this model creates genuine planning/execution mismatch.
    """

    def __init__(
        self,
        energy_capacity: float,
        max_power: float,
        base_charge_efficiency: float,
        base_discharge_efficiency: float,
        soc_min: float,
        soc_max: float,
        initial_soc: float,
        self_discharge_rate: float = 0.0001,  # per hour (0.01%/hr)
    ):
        self.energy_capacity = energy_capacity  # kWh
        self.max_power = max_power  # kW
        self.base_charge_eff = base_charge_efficiency
        self.base_discharge_eff = base_discharge_efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc = initial_soc
        self.self_discharge_rate = self_discharge_rate

    def _effective_efficiency(self, power: float, is_charging: bool) -> float:
        """C-rate dependent efficiency: drops at high power.

        Models internal resistance losses that scale with I^2.
        At 100% power, efficiency drops by ~5% from base.
        At 50% power, drop is ~1.25%.
        """
        c_rate = abs(power) / self.max_power  # 0 to 1
        efficiency_drop = 0.05 * c_rate ** 2  # Quadratic loss
        base = self.base_charge_eff if is_charging else self.base_discharge_eff
        return max(0.7, base - efficiency_drop)

    def _soft_power_limit(self, power: float) -> float:
        """Apply soft power limits near SOC boundaries.

        Linearly reduces available power when SOC is within 5% of limits.
        This prevents the hard boundary behavior the linear optimizer can't predict.
        """
        margin = 0.05  # 5% SOC margin for soft limiting

        if power > 0:  # Charging
            headroom = self.soc_max - self.soc
            if headroom < margin:
                scale = max(0.0, headroom / margin)
                power *= scale
        else:  # Discharging
            headroom = self.soc - self.soc_min
            if headroom < margin:
                scale = max(0.0, headroom / margin)
                power *= scale

        return power

    def update(self, power_setpoint: float, dt_hours: float) -> float:
        """Step the battery forward in time.

        Args:
            power_setpoint: Commanded power in kW (positive=charge, negative=discharge)
            dt_hours: Time step in hours

        Returns:
            Actual power executed (after limits and efficiency)
        """
        # Clip to max power
        actual_power = np.clip(power_setpoint, -self.max_power, self.max_power)

        # Apply soft SOC limits
        actual_power = self._soft_power_limit(actual_power)

        # Calculate energy with nonlinear efficiency
        if actual_power > 0:  # Charging
            eff = self._effective_efficiency(actual_power, is_charging=True)
            energy_stored = actual_power * eff * dt_hours  # kWh into battery
        elif actual_power < 0:  # Discharging
            eff = self._effective_efficiency(actual_power, is_charging=False)
            energy_stored = actual_power / eff * dt_hours  # kWh out of battery (negative)
        else:
            energy_stored = 0.0

        # Update SOC
        delta_soc = energy_stored / self.energy_capacity
        new_soc = self.soc + delta_soc

        # Apply self-discharge
        new_soc -= self.self_discharge_rate * dt_hours

        # Hard clamp (safety)
        new_soc = np.clip(new_soc, self.soc_min, self.soc_max)

        # If we hit a limit, back-calculate what power was actually used
        actual_delta_soc = new_soc - self.soc
        if abs(actual_delta_soc) < abs(delta_soc) and abs(delta_soc) > 1e-10:
            # SOC was clipped - actual power was less than requested
            actual_energy = actual_delta_soc * self.energy_capacity
            if actual_power > 0:
                actual_power = actual_energy / (eff * dt_hours) if dt_hours > 0 else 0.0
            elif actual_power < 0:
                actual_power = actual_energy * eff / dt_hours if dt_hours > 0 else 0.0

        self.soc = new_soc
        return actual_power

    def reset(self, soc: float) -> None:
        """Reset battery to given SOC."""
        self.soc = soc


class DataGenerator:
    """MPC-based training data generator.

    Uses Model Predictive Control approach:
    - NonlinearBatterySimulator tracks true battery state (nonlinear physics)
    - PerfectForesightOptimizer plans with linear approximation
    - Plan applied to true physics, re-solved periodically
    - Model mismatch creates realistic correction behavior
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

        # Initialize price and load generators
        self.price_gen = PriceGenerator(config)
        self.load_gen = LoadGenerator(config)

        # Initialize optimizer (linear model for planning)
        self.optimizer = PerfectForesightOptimizer(time_step=config.time_step_hours)

        # Optimizer constraints (linear approximation)
        self.opt_constraints = BatteryConstraints(
            energy_capacity=config.energy_capacity_kwh,
            max_power=config.max_power_kw,
            charge_efficiency=config.charge_efficiency,
            discharge_efficiency=config.discharge_efficiency,
            soc_min=config.soc_min,
            soc_max=config.soc_max,
        )

        # Initialize true battery model (nonlinear physics)
        self.battery = self._create_battery_model()

    def _create_battery_model(self) -> NonlinearBatterySimulator:
        """Create the true nonlinear physics battery model."""
        return NonlinearBatterySimulator(
            energy_capacity=self.config.energy_capacity_kwh,
            max_power=self.config.max_power_kw,
            base_charge_efficiency=self.config.charge_efficiency,
            base_discharge_efficiency=self.config.discharge_efficiency,
            soc_min=self.config.soc_min,
            soc_max=self.config.soc_max,
            initial_soc=self.config.initial_soc,
        )

    def _extract_features(
        self,
        t: int,
        prices: np.ndarray,
        loads: np.ndarray,
        soc: float,
    ) -> dict:
        """Extract features for timestep t.

        Features (7 dimensions):
        - hour_sin, hour_cos: Cyclical time encoding
        - price_current: Current market price
        - price_avg_4h: 4-hour lookahead average
        - net_load: Current net load
        - soc: Current state of charge
        - dow_sin: Day of week encoding
        """
        steps_per_day = self.config.steps_per_day
        step_in_day = t % steps_per_day
        hour = (step_in_day * self.config.time_step_minutes) / 60.0
        day = t // steps_per_day
        day_of_week = day % 7

        # Cyclical time encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)

        # Price features
        price_current = prices[t]

        # 4-hour lookahead average
        lookahead_4h = (4 * 60) // self.config.time_step_minutes
        end_idx = min(t + lookahead_4h, len(prices))
        price_avg_4h = np.mean(prices[t:end_idx])

        # Load
        net_load = loads[t]

        return {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "price_current": price_current,
            "price_avg_4h": price_avg_4h,
            "net_load": net_load,
            "soc": soc,
            "dow_sin": dow_sin,
        }

    def generate(self, progress_interval: int = 1000) -> pd.DataFrame:
        """Generate training dataset using batch MPC approach.

        Solves optimization every `reoptimize_hours` (default: daily), applies
        the full plan to the true physics model, then re-solves at the next
        interval using the actual (not predicted) battery state.

        This creates model mismatch between linear optimizer and nonlinear
        physics, generating realistic correction behavior in the training data.

        Args:
            progress_interval: Log progress every N steps

        Returns:
            DataFrame with features and targets
        """
        total_steps = self.config.total_steps
        lookahead = self.config.lookahead_steps
        reopt_interval = self.config.reoptimize_steps
        dt_seconds = self.config.time_step_minutes * 60

        logger.info(f"Generating {total_steps:,} steps ({self.config.n_days} days)")
        logger.info(f"Lookahead: {self.config.lookahead_hours}h ({lookahead} steps)")
        logger.info(f"Re-optimize every: {self.config.reoptimize_hours}h ({reopt_interval} steps)")

        # Generate full price and load profiles (need lookahead buffer)
        logger.info("Generating price profile...")
        prices = self.price_gen.generate(total_steps + lookahead)

        logger.info("Generating load profile...")
        loads = self.load_gen.generate(total_steps + lookahead)

        # Storage for results
        features_list = []
        targets = np.zeros(total_steps)
        actual_socs = np.zeros(total_steps + 1)
        actual_powers = np.zeros(total_steps)

        # Initialize SOC tracking
        actual_socs[0] = self.config.initial_soc

        # Reset battery model
        self.battery = self._create_battery_model()

        start_time = time.time()
        n_optimizations = 0

        logger.info("Starting batch MPC loop...")

        # Current optimization plan (will be refreshed periodically)
        current_plan = None
        plan_start_idx = 0

        for t in range(total_steps):
            # 1. Get TRUE state from nonlinear battery model
            true_soc = self.battery.soc

            # 2. Check if we need to re-optimize
            need_reopt = (current_plan is None) or ((t - plan_start_idx) >= reopt_interval)

            if need_reopt:
                # Solve optimization for next lookahead period
                horizon_end = min(t + lookahead, total_steps + lookahead)
                horizon_prices = prices[t:horizon_end]

                # Pad if at end of dataset
                if len(horizon_prices) < lookahead:
                    horizon_prices = np.pad(
                        horizon_prices,
                        (0, lookahead - len(horizon_prices)),
                        mode="edge",
                    )

                # Clamp SOC for optimizer (in case physics pushed slightly out of bounds)
                safe_soc = np.clip(
                    true_soc,
                    self.opt_constraints.soc_min,
                    self.opt_constraints.soc_max,
                )

                result = self.optimizer.solve_horizon(
                    prices=horizon_prices,
                    load=None,
                    initial_soc=safe_soc,
                    constraints=self.opt_constraints,
                )
                n_optimizations += 1

                if result.status in ["optimal", "optimal_inaccurate"]:
                    # Store plan: P_charge - P_discharge (positive = charging)
                    current_plan = result.charge_power - result.discharge_power
                    plan_start_idx = t
                else:
                    # Fallback: idle plan
                    current_plan = np.zeros(lookahead)
                    plan_start_idx = t
                    logger.warning(f"Optimization failed at t={t}: {result.status}")

            # 3. Extract features (using TRUE state)
            features = self._extract_features(t, prices, loads, true_soc)
            features_list.append(features)

            # 4. Get action from current plan
            plan_idx = t - plan_start_idx
            if plan_idx < len(current_plan):
                action = current_plan[plan_idx]
            else:
                action = 0.0  # Fallback if plan exhausted

            targets[t] = action

            # 5. Step TRUE nonlinear physics model
            actual_power = self.battery.update(
                power_setpoint=action,
                dt_hours=self.config.time_step_hours,
            )
            actual_socs[t + 1] = self.battery.soc
            actual_powers[t] = actual_power

            # Progress logging
            if (t + 1) % progress_interval == 0 or t == total_steps - 1:
                elapsed = time.time() - start_time
                rate = (t + 1) / elapsed
                eta = (total_steps - t - 1) / rate

                logger.info(
                    f"Step {t + 1:,}/{total_steps:,} ({100 * (t + 1) / total_steps:.1f}%) | "
                    f"Rate: {rate:.0f} steps/s | ETA: {eta:.1f}s | "
                    f"Opts: {n_optimizations} | SOC: {true_soc:.3f}"
                )

        total_time = time.time() - start_time
        logger.info(f"Generation complete in {total_time:.1f}s ({n_optimizations} optimizations)")

        # Build DataFrame
        df = pd.DataFrame(features_list)
        df["power_setpoint"] = targets

        # Add metadata columns for analysis
        df["actual_soc"] = actual_socs[:-1]
        df["actual_power"] = actual_powers

        return df

    def save(self, df: pd.DataFrame, path: str) -> None:
        """Save dataset to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save main training columns only
        train_cols = [
            "hour_sin",
            "hour_cos",
            "price_current",
            "price_avg_4h",
            "net_load",
            "soc",
            "dow_sin",
            "power_setpoint",
        ]
        df[train_cols].to_csv(path, index=False)
        logger.info(f"Saved training data to {path}")

        # Save full data with metadata
        full_path = path.with_suffix(".full.csv")
        df.to_csv(full_path, index=False)
        logger.info(f"Saved full data (with metadata) to {full_path}")


def generate_training_data(
    n_days: int = 365,
    output_path: str = "data/training_data.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate training data.

    Args:
        n_days: Number of days to generate
        output_path: Path to save CSV
        seed: Random seed

    Returns:
        Generated DataFrame
    """
    config = GenerationConfig(n_days=n_days, random_seed=seed)
    generator = DataGenerator(config)
    df = generator.generate()
    generator.save(df, output_path)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of generated data."""
    print("\n" + "=" * 60)
    print("DATA GENERATION SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\nFeature Statistics:")
    print("-" * 60)
    for col in df.columns:
        print(
            f"  {col:20s}: min={df[col].min():8.4f}, max={df[col].max():8.4f}, "
            f"mean={df[col].mean():8.4f}, std={df[col].std():8.4f}"
        )

    print("\nTarget (power_setpoint) Distribution:")
    print("-" * 60)
    charging = (df["power_setpoint"] > 0.1).sum()
    discharging = (df["power_setpoint"] < -0.1).sum()
    idle = len(df) - charging - discharging
    print(f"  Charging:    {charging:6,} ({100 * charging / len(df):5.1f}%)")
    print(f"  Discharging: {discharging:6,} ({100 * discharging / len(df):5.1f}%)")
    print(f"  Idle:        {idle:6,} ({100 * idle / len(df):5.1f}%)")

    print("\nSOC Statistics:")
    print("-" * 60)
    soc = df["soc"] if "soc" in df.columns else df.get("actual_soc")
    if soc is not None:
        print(f"  Range: [{soc.min():.3f}, {soc.max():.3f}]")
        violations = ((soc < 0.1) | (soc > 0.9)).sum()
        print(f"  Bound violations: {violations} ({100 * violations / len(df):.2f}%)")

    print("\nData Quality Checks:")
    print("-" * 60)
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate MPC training data")
    parser.add_argument("--days", type=int, default=365, help="Number of days")
    parser.add_argument(
        "--output", type=str, default="data/training_data.csv", help="Output path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_training_data(n_days=args.days, output_path=args.output, seed=args.seed)
    print_summary(df)


if __name__ == "__main__":
    main()
