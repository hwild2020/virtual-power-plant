"""
Debug script to investigate XAI contradictions.

Contradiction:
- Q1: price_current rank #6 (low importance)
- Q2: price_current correlation ρ = -0.98 (very high)

This script investigates:
1. Feature correlations (hour_sin vs price_current)
2. Teacher optimization logic (does it use net_load?)
3. SHAP value correctness
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("="*70)
print("DEBUGGING XAI CONTRADICTIONS")
print("="*70)

data = pd.read_csv("data/training_data.csv")
print(f"\nDataset: {len(data)} samples\n")

# Extract features
features = ['hour_sin', 'hour_cos', 'price_current', 'price_avg_4h', 'net_load', 'soc', 'dow_sin']
X = data[features].values
y = data['power_setpoint'].values

print("="*70)
print("INVESTIGATION 1: FEATURE CORRELATIONS")
print("="*70)

# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)
feature_corrs = pd.DataFrame(corr_matrix, index=features, columns=features)

print("\nCorrelation Matrix:")
print(feature_corrs.round(3))

# Focus on price_current correlations
print("\n" + "="*70)
print("PRICE_CURRENT CORRELATIONS WITH OTHER FEATURES:")
print("="*70)
price_idx = features.index('price_current')
for i, feat in enumerate(features):
    if feat != 'price_current':
        corr = corr_matrix[price_idx, i]
        print(f"  {feat:20s}: {corr:+.4f}")

# Check hour_sin vs price
hour_sin_idx = features.index('hour_sin')
price_hour_corr = corr_matrix[hour_sin_idx, price_idx]
print(f"\n>>> hour_sin vs price_current: {price_hour_corr:+.4f}")
if abs(price_hour_corr) > 0.5:
    print("    ⚠ STRONG CORRELATION - hour_sin may be proxy for price!")

print("\n" + "="*70)
print("INVESTIGATION 2: NET_LOAD vs TEACHER OUTPUT")
print("="*70)

# Correlation between net_load and teacher's power setpoint
net_load_idx = features.index('net_load')
net_load_values = X[:, net_load_idx]
corr_netload_power = np.corrcoef(net_load_values, y)[0, 1]
spearman_netload_power, _ = spearmanr(net_load_values, y)

print(f"\nnet_load vs power_setpoint:")
print(f"  Pearson r:  {corr_netload_power:+.4f}")
print(f"  Spearman ρ: {spearman_netload_power:+.4f}")

if abs(corr_netload_power) > 0.3:
    print("  ⚠ SIGNIFICANT CORRELATION - Teacher appears to use net_load!")
    print("  This contradicts pure arbitrage assumption.")
else:
    print("  ✓ Weak correlation - consistent with pure arbitrage.")

print("\n" + "="*70)
print("INVESTIGATION 3: PRICE vs POWER (TEACHER BEHAVIOR)")
print("="*70)

price_values = X[:, price_idx]
corr_price_power = np.corrcoef(price_values, y)[0, 1]
spearman_price_power, _ = spearmanr(price_values, y)

print(f"\nprice_current vs power_setpoint:")
print(f"  Pearson r:  {corr_price_power:+.4f}")
print(f"  Spearman ρ: {spearman_price_power:+.4f}")

if abs(corr_price_power) > 0.5:
    print("  ✓ STRONG CORRELATION - Teacher uses price for arbitrage")
else:
    print("  ⚠ WEAK CORRELATION - Unexpected for arbitrage optimization")

# Visualize price vs power
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Price vs Power
axes[0].scatter(price_values, y, alpha=0.1, s=1)
axes[0].set_xlabel('Price Current ($/kWh)')
axes[0].set_ylabel('Power Setpoint (kW)')
axes[0].set_title(f'Price vs Power\nPearson r={corr_price_power:.3f}')
axes[0].axhline(0, color='red', linestyle='--', alpha=0.3)
axes[0].grid(True, alpha=0.3)

# Hour_sin vs Price
axes[1].scatter(X[:, hour_sin_idx], price_values, alpha=0.1, s=1)
axes[1].set_xlabel('Hour Sin')
axes[1].set_ylabel('Price Current ($/kWh)')
axes[1].set_title(f'Hour vs Price\nPearson r={price_hour_corr:.3f}')
axes[1].grid(True, alpha=0.3)

# Net_load vs Power
axes[2].scatter(net_load_values, y, alpha=0.1, s=1)
axes[2].set_xlabel('Net Load (kW)')
axes[2].set_ylabel('Power Setpoint (kW)')
axes[2].set_title(f'Net Load vs Power\nPearson r={corr_netload_power:.3f}')
axes[2].axhline(0, color='red', linestyle='--', alpha=0.3)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/xai/debug_feature_analysis.png', dpi=200)
print(f"\n  Saved: output/xai/debug_feature_analysis.png")

print("\n" + "="*70)
print("INVESTIGATION 4: SHAP VALUE MAGNITUDE CHECK")
print("="*70)

# Load SHAP analysis results if they exist
try:
    import torch
    from vpp.learning.models import VPPStudentModel
    from vpp.learning.xai import VPPExplainer
    
    print("\nRe-computing SHAP values on small sample for verification...")
    explainer = VPPExplainer(
        model_path="artifacts/student_model.pth",
        data_path="data/training_data.csv",
        n_background=50,
        n_explain=1000,  # Smaller sample for quick check
        output_dir="output/xai_debug",
    )
    
    shap_values = explainer.compute_shap_values()
    
    print(f"\nSHAP Values Shape: {shap_values.shape}")
    print(f"Base Value: {explainer.base_value:.2f} kW")
    
    # Check mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print("\nMean |SHAP| by feature:")
    for i, feat in enumerate(features):
        print(f"  {feat:20s}: {mean_abs_shap[i]:.4f}")
    
    # Verify price_current SHAP correlation
    price_shap = shap_values[:, price_idx]
    price_vals = explainer.X_explain[:, price_idx]
    
    corr_price_shap_pearson = np.corrcoef(price_vals, price_shap)[0, 1]
    corr_price_shap_spearman, _ = spearmanr(price_vals, price_shap)
    
    print(f"\nprice_current value vs its SHAP:")
    print(f"  Pearson r:  {corr_price_shap_pearson:+.4f}")
    print(f"  Spearman ρ: {corr_price_shap_spearman:+.4f}")
    
    # Check additivity manually
    model = explainer.model
    X_tensor = torch.tensor(explainer.X_explain[:100], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().numpy()
    
    shap_sums = explainer.base_value + shap_values[:100].sum(axis=1)
    additivity_errors = np.abs(predictions - shap_sums)
    
    print(f"\nAdditivity Check (first 100 samples):")
    print(f"  Mean absolute error: {additivity_errors.mean():.4f} kW")
    print(f"  Max absolute error:  {additivity_errors.max():.4f} kW")
    print(f"  Relative error:      {(additivity_errors.mean() / 50) * 100:.2f}%")
    
    if additivity_errors.mean() > 1.0:
        print("  ⚠ ADDITIVITY VIOLATION - SHAP values may be incorrect!")
    else:
        print("  ✓ Additivity holds within acceptable tolerance")
    
except Exception as e:
    print(f"\nCould not recompute SHAP: {e}")

print("\n" + "="*70)
print("SUMMARY & DIAGNOSIS")
print("="*70)

print("\nKey Questions:")
print("  1. Is hour_sin correlated with price?")
print(f"     → Correlation: {price_hour_corr:+.4f}")
if abs(price_hour_corr) > 0.5:
    print("     → YES - This explains high hour_sin importance")
    print("     → hour_sin captures indirect price effects")
else:
    print("     → NO - Time and price are independent")

print("\n  2. Why is net_load important?")
print(f"     → net_load vs power correlation: {corr_netload_power:+.4f}")
if abs(corr_netload_power) > 0.3:
    print("     → Teacher DOES use net_load (not pure arbitrage)")
    print("     → May be load-following or grid services component")
else:
    print("     → net_load has minimal impact (pure arbitrage)")

print("\n  3. Are SHAP values correct?")
print("     → Run additivity check above")
print("     → If additivity fails, GradientExplainer may still be flawed")

print("\n" + "="*70)
