# Data Quality Fix: Battery Physics Model Replacement

**Date:** 2026-02-06
**Affected files:** `src/vpp/experiments/generate_data.py`, `src/vpp/models/battery.py`

---

## Problem

The MPC data generation loop was producing unusable training data. Two bugs in `AdvancedElectrochemicalModel` caused a near-total disconnect between what the optimizer commanded and what the battery executed.

### Bug 1: Voltage-Based Power Limiting (Severe)

In `AdvancedElectrochemicalModel.get_max_charge_power()`:

```python
voltage_headroom = max_voltage - state.voltage  # 420 - 400 = 20V
basic_limit = min(
    max_current * nominal_voltage / 1000,       # 125A * 400V / 1000 = 50 kW ✓
    voltage_headroom * max_current / 1000        # 20V * 125A / 1000 = 2.5 kW ← BOTTLENECK
) * charge_efficiency                            # = 2.3 kW (vs 50 kW rated)
```

The `voltage_headroom × max_current` formula capped charge power at **4.6% of rated power**. The battery silently rejected 95% of all charge commands.

### Bug 2: Uncalibrated Concentration Dynamics (Severe)

The electrochemical model derives SOC from `bulk_concentration`, which uses particle-level diffusion physics:
- Diffusion time constant: **167 seconds** (equilibrates in ~3 minutes)
- A 2.3 kW charge for 15 min stored 0.575 kWh (0.53% SOC change expected)
- Actual SOC change: **21%** (40x overshoot)

The particle radius (5μm) and diffusion coefficient (1e-14 m²/s) are material-level parameters that were never calibrated for the 100 kWh battery-system scale.

### Combined Effect

| Metric | Expected | Actual |
|--------|----------|--------|
| Max charge power | 50 kW | 2.3 kW |
| Max discharge power | 50 kW | 10.9 kW |
| SOC change per step | ~10% at full power | 21% at 2.3 kW |
| SOC distribution | Spread across [0.1, 0.9] | Stuck at 0.9 (50% of time) |
| actual_power range | [-50, 50] kW | [-0.001, 11.5] kW |

### Additional Fix

The `super().get_max_charge_power()` call in `AdvancedElectrochemicalModel` returned `None` because the parent class (`BatteryModel`) has abstract methods that return nothing. This caused a `TypeError: '<' not supported between instances of 'float' and 'NoneType'`. Fixed by inlining the base constraint logic directly into the Advanced model's methods.

---

## Solution

Replaced `AdvancedElectrochemicalModel` with a purpose-built `NonlinearBatterySimulator` in the data generation pipeline.

### NonlinearBatterySimulator Design

Located in: `src/vpp/experiments/generate_data.py`

**Energy-based SOC dynamics** (correct conservation):
```
SOC_{t+1} = SOC_t + (P * η_eff * Δt) / E_cap
```

**C-rate dependent efficiency** (nonlinear model mismatch):
```
η_eff = η_base - 0.05 × (P / P_max)²
```
- At idle: η = 0.92 (matches linear optimizer assumption)
- At 50% power: η = 0.9075 (1.4% drop)
- At 100% power: η = 0.87 (5.4% drop — significant mismatch)

**Soft power limits** near SOC boundaries:
- Within 5% of SOC_min/SOC_max, power scales linearly to zero
- Linear optimizer assumes hard limits → creates planning/execution gap

**Self-discharge**: 0.01%/hr continuous SOC loss

### Why This Creates Useful Model Mismatch

The linear optimizer (PerfectForesightOptimizer) assumes:
- Constant efficiency η=0.92
- Hard SOC boundaries
- No self-discharge

The NonlinearBatterySimulator deviates in controlled, physically-motivated ways:
- Efficiency varies with power level
- SOC boundaries are soft
- Self-discharge exists

This means the optimizer's plan gradually diverges from reality, and re-optimization (daily) corrects for accumulated mismatch — exactly the pattern we want the student to learn.

---

## Results Comparison

| Metric | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| SOC mean | 0.837 | **0.419** |
| SOC std | 0.064 | **0.265** |
| actual_power range | [-0.001, 11.5] kW | **[-50, 50] kW** |
| Power fidelity (within 1kW) | ~5% | **93%** |
| Price → Target correlation | -0.522 | **-0.618** |
| Mismatch at low SOC | N/A | 3.43 kW |
| Mismatch at mid SOC | N/A | 0.0 kW |

---

## Files Changed

1. **`src/vpp/experiments/generate_data.py`**
   - Removed `AdvancedElectrochemicalModel` dependency
   - Added `NonlinearBatterySimulator` class
   - Updated `DataGenerator._create_battery_model()` and MPC loop

2. **`src/vpp/models/battery.py`**
   - Fixed `AdvancedElectrochemicalModel.get_max_charge_power()` — replaced `super()` call with inlined logic
   - Fixed `AdvancedElectrochemicalModel.get_max_discharge_power()` — same fix
   - Note: These fixes prevent the `TypeError` but the voltage/concentration issues remain (not addressed since we no longer use this model for data generation)
