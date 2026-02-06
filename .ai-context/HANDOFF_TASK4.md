# Handoff: Task 4 — Training Pipeline

**Date:** 2026-02-06
**Status:** Ready to implement
**File:** `src/vpp/learning/train.py`

---

## Completed Work (Tasks 1-3)

### Task 1: Perfect Foresight MILP Teacher
- **File:** `src/vpp/optimization/offline.py`
- **Class:** `PerfectForesightOptimizer` — CVXPY-based LP solver
- **Method:** `solve_horizon(prices, load, initial_soc, constraints) → TrajectoryResult`
- **Tests:** `tests/test_offline.py` — 21/21 passing
- Maximizes arbitrage profit subject to SOC dynamics, power/SOC limits
- Also provides `solve_rolling_horizon()` for batched solving

### Task 2: Data Generation (MPC Loop)
- **File:** `src/vpp/experiments/generate_data.py`
- **Output:** `data/training_data.csv` (35,040 rows × 8 columns)
- **Method:** Daily re-optimization with `NonlinearBatterySimulator` as true plant
- **Bug fix:** Replaced `AdvancedElectrochemicalModel` (had voltage power-limiting bug and uncalibrated concentration dynamics). See `.ai-context/DATA_QUALITY_FIX.md` for details.
- Key data stats:
  - SOC distribution: mean=0.419, std=0.265 (good spread)
  - Price→Target correlation: -0.618 (arbitrage logic confirmed)
  - Power execution fidelity: 93% of steps within 1 kW of setpoint
  - No NaN/Inf values, zero SOC bound violations

### Task 3: Student Model
- **File:** `src/vpp/learning/models.py`
- **Class:** `VPPStudentModel` — PyTorch MLP
- **Architecture:** Input(7) → Linear(64)+ReLU → Linear(64)+ReLU → Linear(32)+ReLU → Linear(1) → Tanh × P_max
- **Parameters:** 6,785
- **Output bounds:** [-50, 50] kW enforced by Tanh scaling
- **Class:** `VPPDataset` — loads CSV, returns `(features, target)` tensors
- **Tests:** `tests/test_student_model.py` — 18/18 passing

---

## Task 4 Requirements (from IMPLEMENTATION_SPECS.md Section 4)

### Spec

```
1. Data Loading: Read CSV, split Train/Val (80/20).
   - Time-series split (Train on days 0-292, Val on days 293-365)
2. Normalization: StandardScaler for inputs. Targets scaled by P_max.
3. Loss Function: MSELoss
4. Optimizer: Adam (lr=1e-3)
5. Loop: 50-100 Epochs. Early stopping if Val Loss plateaus.
6. Artifacts: Save student_model.pth and scaler.pkl
```

### Agreed Design Decisions

**Preprocessing pipeline (Option B — raw kW targets):**
- **Features:** `StandardScaler` (mean=0, std=1) — required because feature scales differ (price ~0.1, net_load ~22)
- **Target:** Raw kW values, NO normalization — model's Tanh × P_max already outputs [-50, 50] kW
- **Loss:** MSE computed in kW space (model output and target are both in kW)

**Training details:**
- Time-series split at row 28,032 (80% of 35,040) — no shuffling, preserves temporal order
- Early stopping: patience=10 epochs on val loss
- LR scheduler: `ReduceLROnPlateau` (decay LR before stopping)
- Restore best model weights when early stopping triggers
- Max epochs: 100

**Artifacts to save:**
- `artifacts/student_model.pth` — trained model state dict
- `artifacts/scaler.pkl` — fitted StandardScaler (needed at inference)

### Proposed Structure

```python
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config)
    def train(self) -> dict           # Returns training history {epoch, train_loss, val_loss}
    def evaluate(self) -> dict        # Returns val metrics {mse, mae, r2}
    def save_artifacts(self, path)    # Saves model + scaler

def train_student(data_path, output_dir, config) -> dict  # Convenience entry point
```

---

## File Map

```
src/vpp/
├── optimization/
│   └── offline.py          # Task 1: PerfectForesightOptimizer
├── experiments/
│   └── generate_data.py    # Task 2: DataGenerator, NonlinearBatterySimulator
├── learning/
│   ├── __init__.py
│   ├── models.py           # Task 3: VPPStudentModel, VPPDataset
│   └── train.py            # Task 4: TO IMPLEMENT
├── models/
│   └── battery.py          # Bug-fixed but NOT used for data gen
data/
├── training_data.csv       # 35,040 × 8 (features + target only)
└── training_data.full.csv  # 35,040 × 10 (includes actual_soc, actual_power metadata)
tests/
├── test_offline.py         # 21 tests
└── test_student_model.py   # 18 tests
.ai-context/
├── IMPLEMENTATION_SPECS.md # Full spec
├── PROGRESS.md             # Task tracking
├── DATA_QUALITY_FIX.md     # Battery model bug documentation
└── HANDOFF_TASK4.md        # This file
```

---

## Training Data Columns

```csv
hour_sin,hour_cos,price_current,price_avg_4h,net_load,soc,dow_sin,power_setpoint
```

| Column | Type | Range | Role |
|--------|------|-------|------|
| hour_sin | float | [-1, 1] | Feature: cyclical hour encoding |
| hour_cos | float | [-1, 1] | Feature: cyclical hour encoding |
| price_current | float | [0.02, 0.43] | Feature: current electricity price $/kWh |
| price_avg_4h | float | [0.04, 0.17] | Feature: 4-hour lookahead average price |
| net_load | float | [5, 53] | Feature: current net load kW |
| soc | float | [0.1, 0.9] | Feature: battery state of charge |
| dow_sin | float | [-0.97, 0.97] | Feature: cyclical day-of-week encoding |
| power_setpoint | float | [-50, 50] | **Target**: optimal power (positive=charge, negative=discharge) |

---

## Remaining Tasks After Task 4

- **Task 5:** XAI Analysis (`src/vpp/learning/xai.py`) — SHAP explanations
- **Task 6:** Comparative Framework (`src/vpp/analysis/compare.py`) — R², SOC violations, optimality gap
