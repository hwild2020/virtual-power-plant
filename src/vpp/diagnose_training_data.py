"""
Training Data Diagnosis

Investigate why student model fails to learn price arbitrage.
Check:
1. Does training data contain strong price-power relationships?
2. Is data dominated by idle/constraint scenarios vs arbitrage?
3. Are price signals too weak?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

print("="*70)
print("TRAINING DATA DIAGNOSIS")
print("="*70)

# Load training data
data = pd.read_csv("data/training_data.csv")
print(f"\nDataset: {len(data):,} samples")

# Extract key columns
price = data['price_current'].values
power = data['power_setpoint'].values
soc = data['soc'].values
price_avg = data['price_avg_4h'].values

#==============================================================================
# ANALYSIS 1: Overall Price-Power Relationship
#==============================================================================

print("\n" + "="*70)
print("ANALYSIS 1: PRICE-POWER CORRELATION")
print("="*70)

corr_pearson, _ = pearsonr(price, power)
corr_spearman, _ = spearmanr(price, power)

print(f"\nOverall correlation (price_current vs power_setpoint):")
print(f"  Pearson r:  {corr_pearson:+.4f}")
print(f"  Spearman ρ: {corr_spearman:+.4f}")

# Expected: Strong negative correlation (high price → discharge = negative power)
if corr_pearson < -0.5:
    print(f"  ✓ Strong arbitrage signal in data")
else:
    print(f"  ⚠ WEAK arbitrage signal (expected < -0.5)")

# Binned analysis
print(f"\nPrice binned analysis:")
price_bins = pd.qcut(price, q=5, labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'])
for bin_name in ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high']:
    mask = price_bins == bin_name
    mean_price = price[mask].mean()
    mean_power = power[mask].mean()
    print(f"  {bin_name:10s}: price={mean_price:.4f}, power={mean_power:+.2f} kW")

#==============================================================================
# ANALYSIS 2: Action Distribution
#==============================================================================

print("\n" + "="*70)
print("ANALYSIS 2: ACTION DISTRIBUTION")
print("="*70)

charging = (power > 5).sum()
discharging = (power < -5).sum()
idle = len(data) - charging - discharging

print(f"\nAction distribution:")
print(f"  Charging (>5kW):     {charging:6,} ({100*charging/len(data):5.1f}%)")
print(f"  Discharging (<-5kW): {discharging:6,} ({100*discharging/len(data):5.1f}%)")
print(f"  Idle (|power|<5kW):  {idle:6,} ({100*idle/len(data):5.1f}%)")

if idle > 0.5 * len(data):
    print(f"\n  ⚠ WARNING: Data is {100*idle/len(data):.1f}% idle!")
    print(f"     Student may learn to default to idle behavior.")

#==============================================================================
# ANALYSIS 3: Arbitrage Scenarios in Data
#==============================================================================

print("\n" + "="*70)
print("ANALYSIS 3: ARBITRAGE SCENARIOS IN DATA")
print("="*70)

# Define clear arbitrage scenarios
low_price_threshold = np.percentile(price, 20)
high_price_threshold = np.percentile(price, 80)

# Low price scenarios (should charge)
low_price_mask = price < low_price_threshold
low_price_samples = low_price_mask.sum()
low_price_charging = ((price < low_price_threshold) & (power > 20)).sum()

print(f"\nLow price scenarios (price < {low_price_threshold:.4f}):")
print(f"  Total samples: {low_price_samples:,}")
print(f"  Actually charging (>20kW): {low_price_charging:,} ({100*low_price_charging/low_price_samples:.1f}%)")

if low_price_charging / low_price_samples < 0.3:
    print(f"  ⚠ Only {100*low_price_charging/low_price_samples:.1f}% charge during low prices!")

# High price scenarios (should discharge)
high_price_mask = price > high_price_threshold
high_price_samples = high_price_mask.sum()
high_price_discharging = ((price > high_price_threshold) & (power < -20)).sum()

print(f"\nHigh price scenarios (price > {high_price_threshold:.4f}):")
print(f"  Total samples: {high_price_samples:,}")
print(f"  Actually discharging (<-20kW): {high_price_discharging:,} ({100*high_price_discharging/high_price_samples:.1f}%)")

if high_price_discharging / high_price_samples < 0.3:
    print(f"  ⚠ Only {100*high_price_discharging/high_price_samples:.1f}% discharge during high prices!")

#==============================================================================
# ANALYSIS 4: SOC Distribution
#==============================================================================

print("\n" + "="*70)
print("ANALYSIS 4: SOC DISTRIBUTION")
print("="*70)

print(f"\nSOC statistics:")
print(f"  Mean: {soc.mean():.3f}")
print(f"  Std:  {soc.std():.3f}")
print(f"  Min:  {soc.min():.3f}")
print(f"  Max:  {soc.max():.3f}")

# Check if SOC hits boundaries often
boundary_low = (soc <= 0.15).sum()
boundary_high = (soc >= 0.85).sum()
mid_range = len(data) - boundary_low - boundary_high

print(f"\nSOC distribution:")
print(f"  Boundary low (≤0.15):  {boundary_low:6,} ({100*boundary_low/len(data):5.1f}%)")
print(f"  Mid-range (0.15-0.85): {mid_range:6,} ({100*mid_range/len(data):5.1f}%)")
print(f"  Boundary high (≥0.85): {boundary_high:6,} ({100*boundary_high/len(data):5.1f}%)")

if (boundary_low + boundary_high) > 0.3 * len(data):
    print(f"\n  ⚠ {100*(boundary_low+boundary_high)/len(data):.1f}% of data at SOC boundaries!")
    print(f"     Constraints may dominate over price signals.")

#==============================================================================
# ANALYSIS 5: Feature Importance (Simple Linear Model)
#==============================================================================

print("\n" + "="*70)
print("ANALYSIS 5: LINEAR REGRESSION BASELINE")
print("="*70)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Prepare features
X_features = data[['hour_sin', 'hour_cos', 'price_current', 'price_avg_4h', 'net_load', 'soc', 'dow_sin']].values
y_target = power

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Fit linear model
lr = LinearRegression()
lr.fit(X_scaled, y_target)

# Get coefficients
feature_names = ['hour_sin', 'hour_cos', 'price_current', 'price_avg_4h', 'net_load', 'soc', 'dow_sin']
coeffs = lr.coef_

print(f"\nLinear regression coefficients (standardized):")
coeff_list = [(name, coeff) for name, coeff in zip(feature_names, coeffs)]
coeff_list.sort(key=lambda x: abs(x[1]), reverse=True)

for name, coeff in coeff_list:
    print(f"  {name:20s}: {coeff:+8.2f}")

print(f"\nLinear model R²: {lr.score(X_scaled, y_target):.4f}")

# Check if price coefficients are significant
price_current_coeff = coeffs[feature_names.index('price_current')]
price_avg_coeff = coeffs[feature_names.index('price_avg_4h')]

print(f"\nPrice coefficients:")
print(f"  price_current: {price_current_coeff:+.2f}")
print(f"  price_avg_4h:  {price_avg_coeff:+.2f}")

if abs(price_current_coeff) < 20:
    print(f"  ⚠ price_current coefficient is WEAK ({price_current_coeff:+.2f})")
    print(f"     Even linear model struggles with price signals!")

#==============================================================================
# DIAGNOSTIC SUMMARY
#==============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

issues = []

if abs(corr_pearson) < 0.5:
    issues.append(f"Weak price-power correlation ({corr_pearson:+.3f})")

if idle / len(data) > 0.5:
    issues.append(f"Data dominated by idle ({100*idle/len(data):.1f}%)")

if low_price_charging / low_price_samples < 0.3:
    issues.append(f"Weak charging during low prices ({100*low_price_charging/low_price_samples:.1f}%)")

if high_price_discharging / high_price_samples < 0.3:
    issues.append(f"Weak discharging during high prices ({100*high_price_discharging/high_price_samples:.1f}%)")

if (boundary_low + boundary_high) / len(data) > 0.3:
    issues.append(f"Too much time at SOC boundaries ({100*(boundary_low+boundary_high)/len(data):.1f}%)")

print(f"\nIdentified issues ({len(issues)}):")
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("  None - data appears healthy")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if len(issues) >= 3:
    print("\n✗ CRITICAL: Multiple data quality issues found!")
    print("  Root cause: TRAINING DATA has weak arbitrage signals")
    print("\nRecommended fix:")
    print("  1. Regenerate data with stronger price signals")
    print("  2. OR: Modify training to emphasize arbitrage scenarios")
    print("  3. OR: Change data generation to create more extreme prices")
elif len(issues) >= 1:
    print("\n⚠ WARNING: Some data quality issues found")
    print("  May need data adjustments or training modifications")
else:
    print("\n? UNCLEAR: Data appears OK but student still fails")
    print("  Issue likely in model architecture or training process")

# Save plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price vs Power scatter
axes[0, 0].scatter(price, power, alpha=0.1, s=1)
axes[0, 0].set_xlabel('Price ($/kWh)')
axes[0, 0].set_ylabel('Power (kW)')
axes[0, 0].set_title(f'Price vs Power (ρ={corr_pearson:.3f})')
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)

# Power distribution
axes[0, 1].hist(power, bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Power (kW)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Power Distribution')
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].grid(True, alpha=0.3)

# SOC distribution
axes[1, 0].hist(soc, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('SOC')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('SOC Distribution')
axes[1, 0].axvline(0.1, color='red', linestyle='--', label='Limits')
axes[1, 0].axvline(0.9, color='red', linestyle='--')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Feature importance (linear model)
axes[1, 1].barh([name for name, _ in coeff_list], [abs(coeff) for _, coeff in coeff_list])
axes[1, 1].set_xlabel('Absolute Coefficient (Standardized)')
axes[1, 1].set_title('Linear Model Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('output/xai/training_data_diagnosis.png', dpi=200)
print(f"\nVisualization saved: output/xai/training_data_diagnosis.png")
