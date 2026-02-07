"""
Critical Diagnostics: Why did training completely fail?

R² = -0.36 on training data is catastrophic.
Running 4 checks to identify root cause before attempting fix.
"""

import numpy as np
import pandas as pd
import torch
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS

print("="*70)
print("CRITICAL DIAGNOSTICS: TRAINING FAILURE ANALYSIS")
print("="*70)

#==============================================================================
# CHECK 1: VERIFY LOSS FUNCTION AND TARGETS
#==============================================================================

print("\n" + "="*70)
print("CHECK 1: LOSS FUNCTION AND TARGETS VERIFICATION")
print("="*70)

# Load model
model = VPPStudentModel(input_dim=7, max_power=50.0)
state_dict = torch.load("artifacts/student_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Load scaler
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load raw training data
data = pd.read_csv("data/training_data.csv")
X_raw = data[FEATURE_COLUMNS].values[:10]
y_raw = data['power_setpoint'].values[:10]

# Scale features (as done during training)
X_scaled = scaler.transform(X_raw)

# Model predictions
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()

print("\nFirst 10 samples comparison:")
print(f"{'Idx':<5} {'Price':<10} {'SOC':<8} {'y_true':<12} {'y_pred':<12} {'Error':<10} {'Sq.Error':<10}")
print("-"*70)

for i in range(10):
    price = X_raw[i, FEATURE_COLUMNS.index('price_current')]
    soc = X_raw[i, FEATURE_COLUMNS.index('soc')]
    error = y_raw[i] - y_pred[i]
    sq_error = error ** 2
    print(f"{i:<5} {price:<10.4f} {soc:<8.3f} {y_raw[i]:<+12.2f} {y_pred[i]:<+12.2f} {error:<+10.2f} {sq_error:<10.2f}")

print(f"\nTarget signs check:")
print(f"  y_true range: [{y_raw.min():+.2f}, {y_raw.max():+.2f}]")
print(f"  y_pred range: [{y_pred.min():+.2f}, {y_pred.max():+.2f}]")

# Check if predictions are systematically inverted
correlation = np.corrcoef(y_raw, y_pred)[0, 1]
print(f"  Correlation (y_true vs y_pred): {correlation:+.4f}")

if correlation < 0:
    print(f"  ⚠ NEGATIVE CORRELATION - Model outputs are inverted!")
elif abs(correlation) < 0.1:
    print(f"  ⚠ NEAR ZERO CORRELATION - Model outputs are random!")
else:
    print(f"  ✓ Positive correlation exists")

#==============================================================================
# CHECK 2: TRAINING HISTORY
#==============================================================================

print("\n" + "="*70)
print("CHECK 2: TRAINING HISTORY ANALYSIS")
print("="*70)

# Load training history
with open('artifacts/training_history.json', 'r') as f:
    history = json.load(f)

epochs = history['epoch']
train_loss = history['train_loss']
val_loss = history['val_loss']
lr_history = history['lr']

print(f"\nTraining summary:")
print(f"  Total epochs: {len(epochs)}")
print(f"  Initial train loss: {train_loss[0]:.2f}")
print(f"  Final train loss:   {train_loss[-1]:.2f}")
print(f"  Best val loss:      {min(val_loss):.2f}")
print(f"  Final val loss:     {val_loss[-1]:.2f}")

# Check if loss decreased
loss_improvement = train_loss[0] - train_loss[-1]
print(f"\nLoss improvement: {loss_improvement:.2f}")

if loss_improvement < 10:
    print(f"  ⚠ MINIMAL IMPROVEMENT - Training barely converged!")
elif loss_improvement < 0:
    print(f"  ✗ LOSS INCREASED - Training diverged!")
else:
    print(f"  Training loss did decrease")

# Still, even final loss is very high
print(f"\nFinal loss magnitude check:")
print(f"  Final train loss: {train_loss[-1]:.2f}")
print(f"  Expected for good fit: < 50")
if train_loss[-1] > 150:
    print(f"  ✗ LOSS TOO HIGH - Model did not converge properly!")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
axes[0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training History')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

axes[1].plot(epochs, lr_history, linewidth=2, color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Schedule')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('output/xai/training_history.png', dpi=200)
print(f"\n  Saved: output/xai/training_history.png")

#==============================================================================
# CHECK 3: DATA PREPROCESSING
#==============================================================================

print("\n" + "="*70)
print("CHECK 3: DATA PREPROCESSING VERIFICATION")
print("="*70)

# Load full dataset
data = pd.read_csv("data/training_data.csv")
X_raw_full = data[FEATURE_COLUMNS].values
y_raw_full = data['power_setpoint'].values

# Apply scaler
X_scaled_full = scaler.transform(X_raw_full)

print(f"\nRaw data:")
print(f"  Features (X):")
print(f"    Shape: {X_raw_full.shape}")
print(f"    price_current range: [{X_raw_full[:, 2].min():.4f}, {X_raw_full[:, 2].max():.4f}]")
print(f"    soc range: [{X_raw_full[:, 5].min():.4f}, {X_raw_full[:, 5].max():.4f}]")
print(f"  Targets (y):")
print(f"    Shape: {y_raw_full.shape}")
print(f"    Range: [{y_raw_full.min():+.2f}, {y_raw_full.max():+.2f}]")
print(f"    Mean: {y_raw_full.mean():+.2f}")
print(f"    Std: {y_raw_full.std():.2f}")

print(f"\nScaled features (X_scaled):")
print(f"    Shape: {X_scaled_full.shape}")
print(f"    Mean: {X_scaled_full.mean():.4f} (should be ~0)")
print(f"    Std: {X_scaled_full.std():.4f} (should be ~1)")

# Check if scaler is correct
print(f"\nScaler parameters:")
print(f"  Fitted on {len(scaler.mean_)} features")
for i, name in enumerate(FEATURE_COLUMNS):
    print(f"    {name:20s}: mean={scaler.mean_[i]:+8.4f}, std={scaler.scale_[i]:8.4f}")

# CRITICAL: Verify y (targets) are NOT scaled
print(f"\nTarget scaling check:")
print(f"  Targets should NOT be scaled/transformed")
print(f"  Raw y range: [{y_raw_full.min():+.2f}, {y_raw_full.max():+.2f}]")
print(f"  If range is [-1, +1] or [0, 1]: ✗ TARGETS WERE INCORRECTLY SCALED!")
if abs(y_raw_full.max()) <= 1.1 and abs(y_raw_full.min()) <= 1.1:
    print(f"  ⚠ WARNING: Target range suggests possible scaling!")
else:
    print(f"  ✓ Targets appear to be in kW units (not scaled)")

#==============================================================================
# CHECK 4: LINEAR BASELINE SANITY CHECK
#==============================================================================

print("\n" + "="*70)
print("CHECK 4: LINEAR BASELINE SANITY CHECK")
print("="*70)

print(f"\nFitting simple linear regression on [price_current, soc]...")

# Use just price and SOC
X_simple = data[['price_current', 'soc']].values[:5000]
y_simple = data['power_setpoint'].values[:5000]

# Fit linear model (NO SCALING - to diagnose if scaling is the issue)
lr = LinearRegression()
lr.fit(X_simple, y_simple)

r2 = lr.score(X_simple, y_simple)
print(f"\nLinear regression results:")
print(f"  R²: {r2:.4f}")
print(f"  Coefficients:")
print(f"    price_current: {lr.coef_[0]:+.4f} (should be NEGATIVE)")
print(f"    soc:           {lr.coef_[1]:+.4f}")
print(f"  Intercept: {lr.intercept_:+.4f}")

if lr.coef_[0] > 0:
    print(f"\n  ✗ PRICE COEFFICIENT IS POSITIVE!")
    print(f"     This is WRONG - should be negative for arbitrage")
    print(f"     → DATA ITSELF may be corrupted!")
else:
    print(f"\n  ✓ Price coefficient is negative (correct)")

if r2 < 0.3:
    print(f"\n  ✗ LINEAR MODEL ALSO FAILS (R²={r2:.3f})")
    print(f"     → Problem is in the DATA, not the neural network!")
elif r2 > 0.5:
    print(f"\n  ✓ Linear model works (R²={r2:.3f})")
    print(f"     → Problem is in NEURAL NETWORK training!")
else:
    print(f"\n  ? Linear model is mediocre (R²={r2:.3f})")

# Now try with ALL features and scaling (as neural net does)
print(f"\nFitting linear regression with ALL features + scaling...")
X_all_scaled = scaler.transform(data[FEATURE_COLUMNS].values[:5000])
y_all = data['power_setpoint'].values[:5000]

lr_scaled = LinearRegression()
lr_scaled.fit(X_all_scaled, y_all)

r2_scaled = lr_scaled.score(X_all_scaled, y_all)
print(f"  R² (scaled): {r2_scaled:.4f}")

if r2_scaled > 0.5 and r2 < 0.3:
    print(f"  → Scaling HELPS linear model")
elif r2_scaled < r2:
    print(f"  ⚠ Scaling HURTS performance ({r2_scaled:.3f} < {r2:.3f})")
    print(f"     → SCALER may be problematic!")

#==============================================================================
# DIAGNOSTIC SUMMARY
#==============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

issues_found = []

# Issue 1: Model output correlation
if correlation < 0:
    issues_found.append("Model outputs NEGATIVELY correlated with targets")
elif abs(correlation) < 0.1:
    issues_found.append("Model outputs UNCORRELATED with targets")

# Issue 2: Training convergence
if train_loss[-1] > 150:
    issues_found.append(f"Training did not converge (final loss {train_loss[-1]:.1f})")

# Issue 3: Linear baseline
if r2 < 0.3:
    issues_found.append(f"Linear model ALSO fails (R²={r2:.3f}) - DATA issue")
elif r2 > 0.5 and r2_scaled > 0.5:
    issues_found.append(f"Linear models work but neural net fails - TRAINING issue")

# Issue 4: Price coefficient
if lr.coef_[0] > 0:
    issues_found.append("Linear regression has POSITIVE price coefficient - DATA corrupted")

print(f"\nIssues identified ({len(issues_found)}):")
if issues_found:
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("  None - unclear what went wrong")

print("\n" + "="*70)
print("ROOT CAUSE HYPOTHESIS")
print("="*70)

if lr.coef_[0] > 0:
    print("\n✗ CRITICAL: Linear model price coefficient is POSITIVE")
    print("  → Training DATA itself is corrupted or wrong")
    print("  → Need to regenerate data or check data generation code")
elif r2 < 0.3 and r2_scaled < 0.3:
    print("\n✗ CRITICAL: Even simple linear models fail")
    print("  → DATA quality issue")
    print("  → Check data generation pipeline")
elif train_loss[-1] > 150:
    print("\n⚠ Neural network training failed to converge")
    print("  Possible causes:")
    print("    1. Learning rate too high/low")
    print("    2. Poor weight initialization")
    print("    3. Architecture mismatch for this problem")
    print("    4. Optimizer stuck in local minimum")
    print("\n  Next steps: Try different hyperparameters or simpler architecture")
else:
    print("\n? Unclear - need more investigation")

print("\n" + "="*70)
