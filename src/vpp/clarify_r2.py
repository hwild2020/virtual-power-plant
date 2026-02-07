"""
Clarification: R² on Actual Train/Val Splits

User's critical question: Was R² = -0.36 on training data or test data?

This script will:
1. Recreate exact train/val split used during training
2. Calculate R² on FULL training set (~28K samples)
3. Calculate R² on FULL validation set (~7K samples)  
4. Reconcile the contradiction
"""

import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS

print("="*70)
print("R² CLARIFICATION: ACTUAL TRAIN/VAL SPLITS")
print("="*70)

# Load model
model = VPPStudentModel(input_dim=7, max_power=50.0)
state_dict = torch.load("artifacts/student_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Load scaler
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load full dataset
print("\nLoading full dataset...")
data = pd.read_csv("data/training_data.csv")
print(f"  Total samples: {len(data):,}")

X_raw = data[FEATURE_COLUMNS].values
y_raw = data['power_setpoint'].values

# Recreate EXACT train/val split used during training
# From train.py line 262-325: train_split=0.8, random_seed=42
print("\nRecreating train/val split (80/20, seed=42)...")
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y_raw, 
    train_size=0.8, 
    random_state=42,
    shuffle=True
)

print(f"  Training set: {len(X_train_raw):,} samples")
print(f"  Validation set: {len(X_val_raw):,} samples")

# Scale features (using same scaler from training)
X_train_scaled = scaler.transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)

#==============================================================================
# COMPUTE R² ON TRAINING SET
#==============================================================================

print("\n" + "="*70)
print("TRAINING SET PERFORMANCE (Data model was trained on)")
print("="*70)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
with torch.no_grad():
    y_train_pred = model(X_train_tensor).squeeze().numpy()

# Metrics
train_mse = np.mean((y_train_pred - y_train) ** 2)
train_mae = np.mean(np.abs(y_train_pred - y_train))
train_ss_res = np.sum((y_train - y_train_pred) ** 2)
train_ss_tot = np.sum((y_train - y_train.mean()) ** 2)
train_r2 = 1 - (train_ss_res / train_ss_tot)

print(f"\nMetrics on FULL training set ({len(y_train):,} samples):")
print(f"  MSE: {train_mse:.2f}")
print(f"  MAE: {train_mae:.2f} kW")
print(f"  R²:  {train_r2:.4f}")

if train_r2 < 0:
    print(f"\n  ✗ CATASTROPHIC: R² < 0 means model is WORSE than mean baseline!")
    print(f"     Model did NOT learn the training data at all.")
elif train_r2 < 0.5:
    print(f"\n  ✗ POOR: R² = {train_r2:.3f} means weak fit to training data")
elif train_r2 < 0.8:
    print(f"\n  ⚠ MEDIOCRE: R² = {train_r2:.3f} means partial fit")
else:
    print(f"\n  ✓ GOOD: R² = {train_r2:.3f} means strong fit")

#==============================================================================
# COMPUTE R² ON VALIDATION SET
#==============================================================================

print("\n" + "="*70)
print("VALIDATION SET PERFORMANCE (Data model was NOT trained on)")
print("="*70)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
with torch.no_grad():
    y_val_pred = model(X_val_tensor).squeeze().numpy()

# Metrics
val_mse = np.mean((y_val_pred - y_val) ** 2)
val_mae = np.mean(np.abs(y_val_pred - y_val))
val_ss_res = np.sum((y_val - y_val_pred) ** 2)
val_ss_tot = np.sum((y_val - y_val.mean()) ** 2)
val_r2 = 1 - (val_ss_res / val_ss_tot)

print(f"\nMetrics on FULL validation set ({len(y_val):,} samples):")
print(f"  MSE: {val_mse:.2f}")
print(f"  MAE: {val_mae:.2f} kW")
print(f"  R²:  {val_r2:.4f}")

if val_r2 < 0:
    print(f"\n  ✗ R² < 0 on validation set")
elif val_r2 < train_r2 - 0.2:
    print(f"\n  ⚠ Significant overfitting (train R²={train_r2:.3f}, val R²={val_r2:.3f})")
else:
    print(f"\n  Validation performance consistent with training")

#==============================================================================
# VERIFY FIRST 10 SAMPLES
#==============================================================================

print("\n" + "="*70)
print("FIRST 10 SAMPLES CHECK (from raw CSV)")
print("="*70)

# These are the first 10 from the CSV (before train/val split)
X_first10_scaled = scaler.transform(X_raw[:10])
y_first10_true = y_raw[:10]

X_first10_tensor = torch.tensor(X_first10_scaled, dtype=torch.float32)
with torch.no_grad():
    y_first10_pred = model(X_first10_tensor).squeeze().numpy()

corr_first10 = np.corrcoef(y_first10_true, y_first10_pred)[0, 1]

print(f"\nFirst 10 samples:")
print(f"  Correlation: {corr_first10:+.4f}")
print(f"  Mean absolute error: {np.mean(np.abs(y_first10_pred - y_first10_true)):.2f} kW")

# Are these 10 samples in train or val set?
# Check if first sample is in training set
is_in_train = False
for i in range(len(X_train_raw)):
    if np.allclose(X_train_raw[i], X_raw[0]):
        is_in_train = True
        break

print(f"  First 10 samples are in: {'TRAINING set' if is_in_train else 'VALIDATION set'}")

#==============================================================================
# RECONCILE PREVIOUS MEASUREMENTS
#==============================================================================

print("\n" + "="*70)
print("RECONCILING PREVIOUS MEASUREMENTS")
print("="*70)

# My previous check_training_fit.py used first 5000 samples
X_first5k_scaled = scaler.transform(X_raw[:5000])
y_first5k_true = y_raw[:5000]

X_first5k_tensor = torch.tensor(X_first5k_scaled, dtype=torch.float32)
with torch.no_grad():
    y_first5k_pred = model(X_first5k_tensor).squeeze().numpy()

first5k_ss_res = np.sum((y_first5k_true - y_first5k_pred) ** 2)
first5k_ss_tot = np.sum((y_first5k_true - y_first5k_true.mean()) ** 2)
first5k_r2 = 1 - (first5k_ss_res / first5k_ss_tot)

print(f"\nPrevious measurement (first 5000 samples from CSV):")
print(f"  R²: {first5k_r2:.4f}")
print(f"  This was reported as R² = -0.36 (approximately)")

# What fraction of first 5000 are in training vs validation?
in_train_count = 0
for i in range(5000):
    for j in range(len(X_train_raw)):
        if np.allclose(X_raw[i], X_train_raw[j]):
            in_train_count += 1
            break

print(f"\n  Of first 5000 samples:")
print(f"    In training set: ~{in_train_count} ({100*in_train_count/5000:.1f}%)")
print(f"    In validation set: ~{5000-in_train_count} ({100*(5000-in_train_count)/5000:.1f}%)")

#==============================================================================
# FINAL VERDICT
#==============================================================================

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

print(f"\nCLARIFICATION:")
print(f"  Training set R² (ALL {len(y_train):,} samples): {train_r2:+.4f}")
print(f"  Validation set R² (ALL {len(y_val):,} samples): {val_r2:+.4f}")
print(f"  First 10 samples correlation: {corr_first10:+.4f}")
print(f"  First 5000 samples R²: {first5k_r2:+.4f} (previous measurement)")

if train_r2 < 0:
    print(f"\n✗ CATASTROPHIC:")
    print(f"   Model CANNOT fit its own training data (R² = {train_r2:.3f})")
    print(f"   This is NOT a hyperparameter problem.")
    print(f"   This is a fundamental training/architecture issue.")
    print(f"\n   Possible causes:")
    print(f"     1. Weight initialization completely wrong")
    print(f"     2. Learning rate way too high (gradients exploding)")
    print(f"     3. Bug in model forward pass")
    print(f"     4. Optimizer not working")
elif train_r2 > 0.7:
    print(f"\n✓ Training worked:")
    print(f"   Model fits training data well (R² = {train_r2:.3f})")
    if val_r2 < 0.5:
        print(f"   But validation is poor (R² = {val_r2:.3f}) → Overfitting")
else:
    print(f"\n⚠ Partial learning:")
    print(f"   Model learned SOMETHING (R² = {train_r2:.3f})")
    print(f"   But not enough → Underconverged / underfitting")

print(f"\n" + "="*70)
