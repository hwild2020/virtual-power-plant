"""
Quick test: Does current model at least fit the TRAINING data well?

If model can't even fit training data → architecture issue
If model fits training  but fails test → learning/generalization issue
"""

import numpy as np
import pandas as pd
import torch
from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS
from scipy.stats import spearmanr

print("="*70)
print("TRAINING DATA FIT CHECK")
print("="*70)

# Load model
model = VPPStudentModel(input_dim=7, max_power=50.0)
state_dict = torch.load("artifacts/student_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Load training data (use first 1000 samples)
data = pd.read_csv("data/training_data.csv").head(5000)

X = data[FEATURE_COLUMNS].values
y_true = data['power_setpoint'].values

# Predict
X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

# Metrics
mse = np.mean((y_pred - y_true) ** 2)
mae = np.mean(np.abs(y_pred - y_true))
r2 = 1 - (np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

print(f"\nFit on TRAINING data (5000 samples):")
print(f"  MSE: {mse:.2f}")
print(f"  MAE: {mae:.2f} kW")
print(f"  R²:  {r2:.4f}")

# Check price relationship
price = data['price_current'].values
corr_price_true, _ = spearmanr(price, y_true)
corr_price_pred, _ = spearmanr(price, y_pred)

print(f"\nPrice-power correlations:")
print(f"  price vs y_true: {corr_price_true:+.4f} (ground truth)")
print(f"  price vs y_pred: {corr_price_pred:+.4f} (model learned)")

# Check predicted power distribution
charging_pred = (y_pred > 5).sum()
discharging_pred = (y_pred < -5).sum()
idle_pred = len(y_pred) - charging_pred - discharging_pred

charging_true = (y_true > 5).sum()
discharging_true = (y_true < -5).sum()
idle_true = len(y_true) - charging_true - discharging_true

print(f"\nAction distribution comparison:")
print(f"  {'':15s} {'True':>10s} {'Predicted':>10s}")
print(f"  {'Charging':<15s} {charging_true:>10} {charging_pred:>10}")
print(f"  {'Discharging':<15s} {discharging_true:>10} {discharging_pred:>10}")
print(f"  {'Idle':<15s} {idle_true:>10} {idle_pred:>10}")

print(f"\nPrediction statistics:")
print(f"  Mean:   {y_pred.mean():+.2f} kW (true: {y_true.mean():+.2f})")
print(f"  Std:    {y_pred.std():.2f} kW (true: {y_true.std():.2f})")
print(f"  Range:  [{y_pred.min():+.2f}, {y_pred.max():+.2f}] (true: [{y_true.min():+.2f}, {y_true.max():+.2f}])")

# Diagnosis
print(f"\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if r2 < 0.7:
    print(f"\n✗ POOR FIT: Model R² = {r2:.3f} on training data")
    print(f"  → Model CANNOT learn this data with current architecture/training")
elif abs(corr_price_pred) < 0.3:
    print(f"\n✗ WEAK PRICE LEARNING: price-pred correlation = {corr_price_pred:+.3f}")
    print(f"  → Model learned SOMETHING but not price arbitrage")
elif y_pred.std() < y_true.std() * 0.5:
    print(f"\n✗ UNDERFITTING: Predicted std = {y_pred.std():.2f} << true std = {y_true.std():.2f}")
    print(f"  → Model outputs are too conservative/flat")
else:
    print(f"\n? Model fits training data reasonably")
    print(f"  But fails behavioral tests → Overfitting to wrong patterns")
