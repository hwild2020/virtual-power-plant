"""
Tests for Perfect Foresight MILP Teacher.

Tests cover:
1. Basic functionality and constraints
2. 24-hour arbitrage scenario
3. Edge cases and validation
"""

import pytest
import numpy as np

from vpp.optimization.offline import (
    BatteryConstraints,
    TrajectoryResult,
    PerfectForesightOptimizer,
)


class TestBatteryConstraints:
    """Tests for BatteryConstraints dataclass."""

    def test_valid_constraints(self):
        """Test creating valid constraints."""
        constraints = BatteryConstraints(
            energy_capacity=100.0,
            max_power=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9
        )
        assert constraints.energy_capacity == 100.0
        assert constraints.max_power == 50.0

    def test_default_soc_limits(self):
        """Test default SOC limits."""
        constraints = BatteryConstraints(
            energy_capacity=100.0,
            max_power=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95
        )
        assert constraints.soc_min == 0.1
        assert constraints.soc_max == 0.9

    def test_invalid_energy_capacity(self):
        """Test that negative energy capacity raises error."""
        with pytest.raises(ValueError, match="energy_capacity must be positive"):
            BatteryConstraints(
                energy_capacity=-100.0,
                max_power=50.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95
            )

    def test_invalid_efficiency(self):
        """Test that invalid efficiency raises error."""
        with pytest.raises(ValueError, match="charge_efficiency"):
            BatteryConstraints(
                energy_capacity=100.0,
                max_power=50.0,
                charge_efficiency=1.5,  # Invalid: > 1
                discharge_efficiency=0.95
            )

    def test_invalid_soc_range(self):
        """Test that invalid SOC range raises error."""
        with pytest.raises(ValueError, match="soc_min and soc_max"):
            BatteryConstraints(
                energy_capacity=100.0,
                max_power=50.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                soc_min=0.9,  # Invalid: min > max
                soc_max=0.1
            )


class TestPerfectForesightOptimizer:
    """Tests for PerfectForesightOptimizer."""

    @pytest.fixture
    def constraints(self):
        """Standard battery constraints for testing."""
        return BatteryConstraints(
            energy_capacity=100.0,  # 100 kWh
            max_power=50.0,         # 50 kW
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9
        )

    @pytest.fixture
    def optimizer(self):
        """Optimizer with 1-hour time step."""
        return PerfectForesightOptimizer(time_step=1.0)

    def test_initialization(self):
        """Test optimizer initialization."""
        opt = PerfectForesightOptimizer(time_step=0.25)
        assert opt.time_step == 0.25

    def test_invalid_time_step(self):
        """Test that invalid time step raises error."""
        with pytest.raises(ValueError, match="time_step must be positive"):
            PerfectForesightOptimizer(time_step=-1.0)

    def test_constant_price_no_arbitrage(self, optimizer, constraints):
        """With constant prices, no round-trip arbitrage is profitable."""
        prices = np.array([0.10] * 24)  # Constant price
        result = optimizer.solve_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        assert result.status == "optimal"
        # With constant prices, there's no incentive to charge (would cost money with efficiency loss)
        # The optimizer may discharge existing SOC to capture value, but won't charge
        assert np.allclose(result.charge_power, 0, atol=1e-4), "Should not charge at constant prices"
        # No simultaneous charge/discharge (would be inefficient)
        simultaneous = np.minimum(result.charge_power, result.discharge_power)
        assert np.allclose(simultaneous, 0, atol=1e-4), "No simultaneous charge/discharge"

    def test_simple_arbitrage(self, optimizer, constraints):
        """Test simple buy-low-sell-high arbitrage."""
        # Low price at hour 0, high price at hour 1
        prices = np.array([0.05, 0.20])  # $/kWh

        result = optimizer.solve_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        assert result.status == "optimal"
        # Should charge at low price (hour 0) and discharge at high price (hour 1)
        assert result.charge_power[0] > 0, "Should charge at low price"
        assert result.discharge_power[1] > 0, "Should discharge at high price"
        assert result.profit > 0, "Should make profit from arbitrage"

    def test_soc_limits_respected(self, optimizer, constraints):
        """Test that SOC limits are respected."""
        prices = np.array([0.05] * 10 + [0.20] * 10)  # Low then high

        result = optimizer.solve_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        assert result.status == "optimal"
        # SOC should stay within limits
        assert np.all(result.soc >= constraints.soc_min - 1e-6)
        assert np.all(result.soc <= constraints.soc_max + 1e-6)

    def test_power_limits_respected(self, optimizer, constraints):
        """Test that power limits are respected."""
        prices = np.array([0.01, 0.50])  # Extreme price difference

        result = optimizer.solve_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        assert result.status == "optimal"
        assert np.all(result.charge_power <= constraints.max_power + 1e-6)
        assert np.all(result.discharge_power <= constraints.max_power + 1e-6)

    def test_initial_soc_validation(self, optimizer, constraints):
        """Test that invalid initial SOC raises error."""
        prices = np.array([0.10] * 24)

        with pytest.raises(ValueError, match="initial_soc"):
            optimizer.solve_horizon(
                prices=prices,
                load=None,
                initial_soc=0.05,  # Below soc_min
                constraints=constraints
            )

    def test_buy_sell_price_spread(self, optimizer, constraints):
        """Test with separate buy and sell prices."""
        prices_buy = np.array([0.12, 0.10, 0.15, 0.20])
        prices_sell = np.array([0.10, 0.08, 0.13, 0.18])  # Sell < Buy (spread)

        result = optimizer.solve_horizon(
            prices=(prices_buy, prices_sell),
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        assert result.status == "optimal"
        assert len(result.power) == 4

    def test_result_to_dict(self, optimizer, constraints):
        """Test TrajectoryResult serialization."""
        prices = np.array([0.10, 0.20])
        result = optimizer.solve_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        result_dict = result.to_dict()
        assert "power" in result_dict
        assert "soc" in result_dict
        assert "profit" in result_dict
        assert isinstance(result_dict["power"], list)


class TestArbitrageScenario24h:
    """Test realistic 24-hour arbitrage scenario."""

    @pytest.fixture
    def constraints(self):
        """Realistic battery constraints."""
        return BatteryConstraints(
            energy_capacity=100.0,   # 100 kWh battery
            max_power=25.0,          # 25 kW (C/4 rate)
            charge_efficiency=0.92,
            discharge_efficiency=0.92,
            soc_min=0.1,
            soc_max=0.9
        )

    @pytest.fixture
    def optimizer(self):
        """Optimizer with 1-hour resolution."""
        return PerfectForesightOptimizer(time_step=1.0)

    @pytest.fixture
    def typical_prices(self):
        """Typical day-ahead prices with morning and evening peaks."""
        # $/kWh - typical pattern with low overnight, peaks at 8am and 6pm
        return np.array([
            0.04, 0.03, 0.03, 0.03,  # 00:00-03:00 (low overnight)
            0.04, 0.05, 0.08, 0.12,  # 04:00-07:00 (morning ramp)
            0.15, 0.14, 0.12, 0.10,  # 08:00-11:00 (morning peak)
            0.09, 0.08, 0.09, 0.10,  # 12:00-15:00 (midday)
            0.12, 0.18, 0.22, 0.20,  # 16:00-19:00 (evening peak)
            0.15, 0.10, 0.06, 0.05   # 20:00-23:00 (evening decline)
        ])

    def test_24h_arbitrage_profitable(self, optimizer, constraints, typical_prices):
        """Test that 24h arbitrage is profitable with typical prices."""
        result = optimizer.solve_horizon(
            prices=typical_prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        assert result.status == "optimal"
        assert result.profit > 0, "Should be profitable with price variation"

        # Print summary for verification
        print(f"\n24h Arbitrage Results:")
        print(f"  Status: {result.status}")
        print(f"  Profit: ${result.profit:.2f}")
        print(f"  Solve time: {result.solve_time:.4f}s")
        print(f"  SOC range: [{result.soc.min():.2f}, {result.soc.max():.2f}]")
        print(f"  Max charge: {result.charge_power.max():.1f} kW")
        print(f"  Max discharge: {result.discharge_power.max():.1f} kW")

    def test_24h_charges_at_low_prices(self, optimizer, constraints, typical_prices):
        """Verify optimizer charges during low-price periods."""
        result = optimizer.solve_horizon(
            prices=typical_prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        # Hours 0-3 have lowest prices - should see charging
        low_price_charging = np.sum(result.charge_power[0:4])
        assert low_price_charging > 0, "Should charge during low-price overnight hours"

    def test_24h_discharges_at_high_prices(self, optimizer, constraints, typical_prices):
        """Verify optimizer discharges during high-price periods."""
        result = optimizer.solve_horizon(
            prices=typical_prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        # Hours 17-19 have highest prices - should see discharging
        high_price_discharging = np.sum(result.discharge_power[17:20])
        assert high_price_discharging > 0, "Should discharge during evening peak"

    def test_24h_energy_conservation(self, optimizer, constraints, typical_prices):
        """Verify energy conservation in SOC trajectory."""
        result = optimizer.solve_horizon(
            prices=typical_prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        # Verify SOC dynamics are consistent
        for t in range(len(typical_prices)):
            energy_in = result.charge_power[t] * constraints.charge_efficiency * optimizer.time_step
            energy_out = result.discharge_power[t] / constraints.discharge_efficiency * optimizer.time_step
            expected_delta = (energy_in - energy_out) / constraints.energy_capacity
            actual_delta = result.soc[t + 1] - result.soc[t]
            assert np.isclose(expected_delta, actual_delta, atol=1e-5), \
                f"SOC dynamics violated at t={t}"

    def test_24h_net_power_consistency(self, optimizer, constraints, typical_prices):
        """Verify net power equals discharge - charge."""
        result = optimizer.solve_horizon(
            prices=typical_prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        expected_power = result.discharge_power - result.charge_power
        assert np.allclose(result.power, expected_power, atol=1e-6)


class TestRollingHorizon:
    """Tests for rolling horizon optimization."""

    @pytest.fixture
    def constraints(self):
        return BatteryConstraints(
            energy_capacity=100.0,
            max_power=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95
        )

    @pytest.fixture
    def optimizer(self):
        return PerfectForesightOptimizer(time_step=1.0)

    def test_rolling_horizon_basic(self, optimizer, constraints):
        """Test basic rolling horizon functionality."""
        prices = np.array([0.05] * 12 + [0.20] * 12)  # 24 hours

        result = optimizer.solve_rolling_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints,
            horizon_length=6,
            step_size=3
        )

        assert result.status == "optimal"
        assert len(result.power) == 24
        assert len(result.soc) == 25

    def test_rolling_vs_full_horizon(self, optimizer, constraints):
        """Compare rolling horizon to full horizon solution."""
        np.random.seed(42)
        prices = 0.10 + 0.05 * np.sin(np.linspace(0, 2*np.pi, 24))

        # Full horizon
        full_result = optimizer.solve_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints
        )

        # Rolling horizon
        rolling_result = optimizer.solve_rolling_horizon(
            prices=prices,
            load=None,
            initial_soc=0.5,
            constraints=constraints,
            horizon_length=24,  # Same as full
            step_size=24        # Single step
        )

        # With same horizon, should get same result
        assert np.allclose(full_result.power, rolling_result.power, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
