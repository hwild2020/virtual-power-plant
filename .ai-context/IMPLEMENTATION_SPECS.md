# Implementation Specifications: VPP Optimization Sprint

**Workflow:** MILP Teacher -> NN Student -> XAI Analysis
**Dependencies:** `cvxpy` (optimization), `torch` (learning), `shap` (explanation), `pandas/numpy` (data).

---

## 1. Perfect Foresight Teacher (`offline.py`)

**File Path:** `src/vpp/optimization/offline.py`

### Mathematical Formulation
**Objective:** Maximize Profit over Horizon $T$.
$$
\max \sum_{t=0}^{T} (P_{discharge, t} \cdot \pi_{t, sell} - P_{charge, t} \cdot \pi_{t, buy})
$$
Where:
- $\pi_{t, sell}, \pi_{t, buy}$ are market prices (adjusted for grid fees/spread).
- $P_{discharge, t}, P_{charge, t} \ge 0$.

**Constraints:**
1.  **SOC Dynamics:**
    $$SOC_{t+1} \cdot E_{cap} = SOC_{t} \cdot E_{cap} + (P_{charge, t} \cdot \eta_{ch} - P_{discharge, t} / \eta_{dis}) \cdot \Delta t$$
2.  **SOC Limits:**
    $$SOC_{min} \le SOC_t \le SOC_{max}$$
    (e.g., $0.1 \le SOC \le 0.9$)
3.  **Power Limits:**
    $$0 \le P_{charge, t} \le P_{max}$$
    $$0 \le P_{discharge, t} \le P_{max}$$
4.  **Simultaneous Operation (Relaxed):**
    in Convex Optimization, we rely on price spread to prevent simultaneous charge/discharge. If prices are negative, explicit binary variables (MIP) might be needed, but for standard prices, standard LP/QP works.

### Implementation Details
- **Class:** `PerfectForesightOptimizer`
- **Method:** `solve_horizon(prices, load, initial_soc, constraints)`
- **Library:** `cvxpy`.
- **Return:** Full trajectory of $\{P_t, SOC_t\}$.

---

## 2. Data Generation Experiment (`generate_data.py`)

**File Path:** `src/vpp/experiments/generate_data.py`

### Process
1.  **Initialize:** `Simulator`, `AdvancedElectrochemicalModel`.
2.  **Horizon:** 365 days @ 15 min resolution = ~35,040 steps.
    - *Rationale:* Ensures seasonal coverage and ~10:1 sample-to-parameter ratio.
3.  **Data Generation Loop (Rolling Horizon / MPC):**
    -   **Method:** At each time step $t$:
        1.  Get current true state $S_t$ (SOC, Temp) from `AdvancedElectrochemicalModel`.
        2.  Teacher solves optimization for horizon $[t, t+24h]$ using `SimpleEquivalentCircuitModel` (Linear Proxy).
        3.  Extract first action $P^*_t$.
        4.  Apply $P^*_t$ to `AdvancedElectrochemicalModel` to get $S_{t+1}$.
    -   *Rationale:* This "Hybrid" approach allows the linear teacher to control the complex non-linear plant, correcting for model mismatch (e.g., efficiency differences) at every step [Literature].
4.  **Feature Extraction ($X_t$):**
    - `timestamp_hour_sin`, `timestamp_hour_cos` (Cyclical time)
    - `market_price_current`
    - `market_price_avg_4h` (Lookahead context)
    - `net_load_current`
    - `soc_current`
5.  **Target ($y_t$):**
    - `optimized_power_net` ($P_{charge} - P_{discharge}$) from Teacher.
6.  **Output:** `data/training_data.csv`

---

## 3. Student Model (`models.py`)

**File Path:** `src/vpp/learning/models.py`

### Architecture
**Justification:** Literature [1, 2] suggests MLPs are sufficient for imitating fixed logic, provided adequate capacity.
- **Input Layer:** Dim = 7 (Time features + Price + Price_Future + Load + SOC).
- **Hidden Layers:**
    - Linear(Input -> 64) + ReLU
    - Linear(64 -> 64) + ReLU
    - Linear(64 -> 32) + ReLU
- **Output Layer:** Linear(32 -> 1) (Simple Scalar Power).
    - *Activation:* `Tanh` scaled by $P_{max}$ (to enforce physical bounds directly) OR linear with clamping.

---

## 4. Training Pipeline (`train.py`)

**File Path:** `src/vpp/learning/train.py`

### Procedure
1.  **Data Loading:** Read CSV, split Train/Val (80/20).
    - *Important:* Time-series split (Train on days 0-24, Test on days 25-30) to test generalization.
2.  **Normalization:** `StandardScaler` for inputs. Targets can be scaled by $P_{max}$.
3.  **Loss Function:** `MSELoss` (Mean Squared Error).
4.  **Optimizer:** `Adam` (lr=1e-3).
5.  **Loop:** 50-100 Epochs. Early stopping if Val Loss plateaus.
6.  **Artifacts:** Save `student_model.pth` and `scaler.pkl`.

---

## 5. XAI Analysis (`xai.py`)

**File Path:** `src/vpp/learning/xai.py` *(Note: Moved to learning pkg)*

### Configuration
1.  **Method:** `shap.KernelExplainer` (model-agnostic, robust) or `shap.DeepExplainer` (PyTorch specific, faster).
2.  **Background Dataset:** KMeans summary of training data (k=50) to serve as baseline.
3.  **Outputs:**
    - `shap_values`: Array of [N_samples, N_features].
    - `summary_plot`: Dot plot ranked by feature importance.
    - `dependence_plot`: Price vs SHAP value (To verify arbitrage logic).

---

## 6. Comparative Framework (`compare.py`)

**File Path:** `src/vpp/analysis/compare.py`

### Metrics
1.  **Regression:** $R^2$ Score on Test Set.
2.  **Physical Integrity:**
    - `SOC_Violation_Rate`: $\%$ of steps where resulting SOC $< 0$ or $> 1$.
3.  **Economic Efficiency:**
    - `Teacher_Profit`: $\sum P^*_{t} \cdot \pi_t$
    - `Student_Profit`: $\sum \hat{P}_{t} \cdot \pi_t$ (Simulated forward with physics model to catch SOC constraints).
    - `Optimality_Gap`: $1 - \frac{\text{Student\_Profit}}{\text{Teacher\_Profit}}$

### Visualization
- **Dual Plot:**
    - Top: Price Profile ($/kWh).
    - Bottom: Overlay of Teacher Dispatch (Blue) vs Student Dispatch (Red).
    - Highlights: Annotate regions where Student deviates.
