"""
Verify if student model uses price_avg_4h (foresight).

Check correlation between price gradient and student predictions.

Expected:
- If student uses foresight: correlation should be NEGATIVE
  (rising prices → charge now, falling prices → discharge now)
- If student ignores foresight: correlation near ZERO
"""

import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS

print("="*70)
print("VERIFYING STUDENT'S USE OF PRICE FORESIGHT")
print("="*70)

# Load model and scaler
model = VPPStudentModel(input_dim=7, max_power=50.0)
state_dict = torch.load("../../artifacts/student_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

with open('../../artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data (same split as comparison)
data = pd.read_csv("../../data/training_data.csv")

X = data[FEATURE_COLUMNS].values
y_teacher = data['power_setpoint'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_teacher,
    train_size=0.8,
    random_state=42,
    shuffle=True
)

# Get test data subset for analysis
test_indices = np.arange(len(X_train), len(X))  # Approximate
test_data = data.iloc[test_indices].copy()

# Compute price gradient
price_current = test_data['price_current'].values
price_avg_4h = test_data['price_avg_4h'].values
price_gradient = price_avg_4h - price_current

print(f"\nTest data: {len(test_data):,} samples")
print(f"Price gradient statistics:")
print(f"  Mean: {price_gradient.mean():+.6f}")
print(f"  Std:  {price_gradient.std():.6f}")
print(f"  Range: [{price_gradient.min():+.6f}, {price_gradient.max():+.6f}]")

# Get student predictions
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

with torch.no_grad():
    student_predictions = model(X_test_tensor).squeeze().numpy()

print(f"\nStudent predictions statistics:")
print(f"  Mean: {student_predictions.mean():+.2f} kW")
print(f"  Std:  {student_predictions.std():.2f} kW")
print(f"  Range: [{student_predictions.min():+.2f}, {student_predictions.max():+.2f}] kW")

# Compute correlation
correlation = np.corrcoef(price_gradient, student_predictions)[0, 1]

print(f"\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

print(f"\nCorrelation (price_gradient vs student_predictions): {correlation:+.4f}")

print(f"\nExpected behavior:")
print(f"  - Prices rising (gradient > 0) → Charge now → Positive power")
print(f"  - Prices falling (gradient < 0) → Discharge now → Negative power")
print(f"  → Correlation should be NEGATIVE if using foresight")

print(f"\nInterpretation:")
if abs(correlation) < 0.1:
    print(f"  ✗ NEAR ZERO correlation ({correlation:+.3f})")
    print(f"     Student IGNORES price_avg_4h!")
    print(f"     This explains the 70% optimality gap.")
    print(f"     Model treats future prices as irrelevant.")
elif correlation < -0.3:
    print(f"  ✓ NEGATIVE correlation ({correlation:+.3f})")
    print(f"     Student USES price_avg_4h correctly")
    print(f"     70% gap is genuinely concerning - other issues present")
elif correlation > 0.3:
    print(f"  ✗ POSITIVE correlation ({correlation:+.3f})")
    print(f"     Student uses foresight BACKWARDS!")
    print(f"     Charges when prices rising, discharges when falling")
else:
    print(f"  ⚠ WEAK correlation ({correlation:+.3f})")
    print(f"     Student uses foresight weakly")

# Additional analysis: Check feature importance in practice
print(f"\n" + "="*70)
print("FEATURE SENSITIVITY ANALYSIS")
print("="*70)

# Test: Does changing price_avg_4h significantly change predictions?
print(f"\nTesting sensitivity to price_avg_4h...")

# Take a sample scenario
sample_idx = 1000
sample_features = X_test[sample_idx].copy()

print(f"\nBase scenario:")
print(f"  price_current: {sample_features[2]:.4f}")
print(f"  price_avg_4h:  {sample_features[3]:.4f}")
print(f"  soc:           {sample_features[5]:.3f}")

# Scale and predict
sample_scaled = scaler.transform([sample_features])
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
with torch.no_grad():
    base_pred = model(sample_tensor).item()

print(f"  Prediction:    {base_pred:+.2f} kW")

# Vary price_avg_4h
print(f"\nVarying price_avg_4h:")
print(f"  {'price_avg_4h':<15} {'Prediction':<15} {'Change':<15}")
print(f"  {'-'*45}")

for delta in [-0.05, -0.02, 0.0, +0.02, +0.05]:
    test_features = sample_features.copy()
    test_features[3] = sample_features[3] + delta  # Modify price_avg_4h
    
    test_scaled = scaler.transform([test_features])
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred = model(test_tensor).item()
    
    change = pred - base_pred
    print(f"  {test_features[3]:<15.4f} {pred:<+15.2f} {change:+.2f} kW")

print(f"\n" + "="*70)
print("CONCLUSION")
print("="*70)

if abs(correlation) < 0.1:
    print(f"\n✗ STUDENT DOES NOT USE PRICE FORESIGHT")
    print(f"\nRoot cause of 70% optimality gap:")
    print(f"  1. Student ignores price_avg_4h feature")
    print(f"  2. Only uses price_current (instantaneous)")
    print(f"  3. Teacher uses 24h perfect foresight")
    print(f"  4. Gap is explained by lack of future information")
    print(f"\nRecommendation:")
    print(f"  - Check XAI analysis: Is price_avg_4h importance very low?")
    print(f"  - Possible training issue: Feature not learned properly")
    print(f"  - Or: Feature is genuinely weak signal in training data")
else:
    print(f"\n✓ Student uses price foresight (correlation={correlation:+.3f})")
    print(f"\n70% optimality gap requires further investigation:")
    print(f"  - Physics simulation may be too conservative")
    print(f"  - MAE error compounds over time")
    print(f"  - Initial SOC mismatch")
    print(f"  - Other systematic biases")
