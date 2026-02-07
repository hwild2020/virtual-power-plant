"""
Student Model Behavioral Verification

Tests student model on 100 handcrafted scenarios with obvious optimal actions.
Goal: Verify student learned reasonable battery behavior, not just memorized noise.

Sign convention (from data generation line 499):
- Positive power = CHARGING (buying from grid)
- Negative power = DISCHARGING (selling to grid)
"""

import numpy as np
import pandas as pd
import torch
from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS

print("="*70)
print("STUDENT MODEL BEHAVIORAL VERIFICATION")
print("="*70)

# Load trained model
print("\nLoading model...")
model = VPPStudentModel(input_dim=7, max_power=50.0)
state_dict = torch.load("artifacts/student_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

def predict(features_dict):
    """Predict power for feature dict."""
    # Convert to array in correct order
    x = np.array([[
        features_dict['hour_sin'],
        features_dict['hour_cos'],
        features_dict['price_current'],
        features_dict['price_avg_4h'],
        features_dict['net_load'],
        features_dict['soc'],
        features_dict['dow_sin'],
    ]])
    
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        pred = model(x_tensor).item()
    
    return pred

# Create test scenarios
scenarios = []

print("\nCreating test scenarios...")

# Default neutral values
DEFAULT = {
    'hour_sin': 0.0,
    'hour_cos': 1.0,
    'dow_sin': 0.0,
}

#==============================================================================
# SCENARIO SET 1: PRICE ARBITRAGE (40 samples)
#==============================================================================

print("  Set 1: Price Arbitrage (40 scenarios)")

# 1A: High price + High SOC → Should DISCHARGE (sell expensive)
for i in range(10):
    price = np.random.uniform(0.15, 0.20)
    soc = np.random.uniform(0.70, 0.85)
    scenarios.append({
        **DEFAULT,
        'price_current': price,
        'price_avg_4h': price * 0.95,  # Slightly lower future
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'discharge',
        'expected_sign': 'negative',
        'expected_magnitude': '>20kW',
        'scenario_type': 'high_price_high_soc',
        'rationale': 'Sell expensive power when battery is full',
    })

# 1B: Low price + Low SOC → Should CHARGE (buy cheap)
for i in range(10):
    price = np.random.uniform(0.05, 0.08)
    soc = np.random.uniform(0.15, 0.30)
    scenarios.append({
        **DEFAULT,
        'price_current': price,
        'price_avg_4h': price * 1.05,  # Slightly higher future
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'charge',
        'expected_sign': 'positive',
        'expected_magnitude': '>20kW',
        'scenario_type': 'low_price_low_soc',
        'rationale': 'Buy cheap power when battery is empty',
    })

# 1C: Mid price + Mid SOC → Should be IDLE or small action
for i in range(10):
    price = np.random.uniform(0.10, 0.12)
    soc = np.random.uniform(0.40, 0.60)
    scenarios.append({
        **DEFAULT,
        'price_current': price,
        'price_avg_4h': price,
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'idle',
        'expected_sign': 'any',
        'expected_magnitude': '<15kW',
        'scenario_type': 'mid_price_mid_soc',
        'rationale': 'No strong arbitrage signal',
    })

# 1D: Low price + High SOC → Conflicting signal (should idle or small charge)
for i in range(10):
    price = np.random.uniform(0.05, 0.08)
    soc = np.random.uniform(0.70, 0.85)
    scenarios.append({
        **DEFAULT,
        'price_current': price,
        'price_avg_4h': price * 1.1,
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'idle_or_small',
        'expected_sign': 'any',
        'expected_magnitude': '<20kW',
        'scenario_type': 'low_price_high_soc',
        'rationale': 'Cheap power but battery nearly full',
    })

#==============================================================================
# SCENARIO SET 2: SOC CONSTRAINTS (20 samples)
#==============================================================================

print("  Set 2: SOC Constraints (20 scenarios)")

# 2A: SOC at upper limit → CANNOT charge more
for i in range(10):
    price = np.random.uniform(0.05, 0.15)  # Any price
    soc = np.random.uniform(0.85, 0.89)  # Near max (0.9)
    scenarios.append({
        **DEFAULT,
        'price_current': price,
        'price_avg_4h': price,
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'no_charge',
        'expected_sign': 'negative_or_zero',
        'expected_magnitude': 'any',
        'scenario_type': 'high_soc_limit',
        'rationale': 'SOC near maximum, cannot charge',
    })

# 2B: SOC at lower limit → CANNOT discharge more  
for i in range(10):
    price = np.random.uniform(0.05, 0.15)  # Any price
    soc = np.random.uniform(0.11, 0.15)  # Near min (0.1)
    scenarios.append({
        **DEFAULT,
        'price_current': price,
        'price_avg_4h': price,
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'no_discharge',
        'expected_sign': 'positive_or_zero',
        'expected_magnitude': 'any',
        'scenario_type': 'low_soc_limit',
        'rationale': 'SOC near minimum, cannot discharge',
    })

#==============================================================================
# SCENARIO SET 3: PRICE GRADIENT (20 samples)
#==============================================================================

print("  Set 3: Price Gradient (20 scenarios)")

# 3A: Current price low, future price high → CHARGE now (buy cheap, sell later)
for i in range(10):
    price_now = np.random.uniform(0.06, 0.08)
    price_future = np.random.uniform(0.14, 0.18)
    soc = np.random.uniform(0.30, 0.60)
    scenarios.append({
        **DEFAULT,
        'price_current': price_now,
        'price_avg_4h': price_future,
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'charge',
        'expected_sign': 'positive',
        'expected_magnitude': '>10kW',
        'scenario_type': 'price_rising',
        'rationale': 'Price rising, charge now to sell later',
    })

# 3B: Current price high, future price low → DISCHARGE now (sell expensive now)
for i in range(10):
    price_now = np.random.uniform(0.14, 0.18)
    price_future = np.random.uniform(0.06, 0.08)
    soc = np.random.uniform(0.40, 0.70)
    scenarios.append({
        **DEFAULT,
        'price_current': price_now,
        'price_avg_4h': price_future,
        'net_load': np.random.uniform(20, 40),
        'soc': soc,
        'expected_action': 'discharge',
        'expected_sign': 'negative',
        'expected_magnitude': '>10kW',
        'scenario_type': 'price_falling',
        'rationale': 'Price falling, discharge now before drop',
    })

#==============================================================================
# SCENARIO SET 4: EXTREME CASES (20 samples)
#==============================================================================

print("  Set 4: Extreme Cases (20 scenarios)")

# 4A: Maximum arbitrage opportunity
for i in range(5):
    scenarios.append({
        **DEFAULT,
        'price_current': 0.05,  # Very low
        'price_avg_4h': 0.18,  # Very high future
        'net_load': 25.0,
        'soc': 0.20,  # Low SOC, room to charge
        'expected_action': 'charge_max',
        'expected_sign': 'positive',
        'expected_magnitude': '>35kW',
       'scenario_type': 'max_arbitrage_buy',
        'rationale': 'Extreme buy opportunity',
    })

# 4B: Maximum discharge opportunity
for i in range(5):
    scenarios.append({
        **DEFAULT,
        'price_current': 0.18,  # Very high
        'price_avg_4h': 0.05,  # Very low future
        'net_load': 25.0,
        'soc': 0.75,  # High SOC, can discharge
        'expected_action': 'discharge_max',
        'expected_sign': 'negative',
        'expected_magnitude': '>35kW',
        'scenario_type': 'max_arbitrage_sell',
        'rationale': 'Extreme sell opportunity',
    })

# 4C: Impossible scenarios (conflicting constraints)
for i in range(10):
    # High price, low SOC = can't capitalize (no energy to sell)
    scenarios.append({
        **DEFAULT,
        'price_current': np.random.uniform(0.15, 0.18),
        'price_avg_4h': np.random.uniform(0.15, 0.18),
        'net_load': 30.0,
        'soc': np.random.uniform(0.12, 0.18),
        'expected_action': 'idle',
        'expected_sign': 'any',
        'expected_magnitude': '<10kW',
        'scenario_type': 'impossible_discharge',
        'rationale': 'Want to sell but no energy',
    })

print(f"\nTotal scenarios: {len(scenarios)}")

#==============================================================================
# RUN VERIFICATION
#==============================================================================

print("\n" + "="*70)
print("RUNNING TESTS")
print("="*70)

results = []
passed = 0
failed = 0

for i, scenario in enumerate(scenarios):
    prediction = predict(scenario)
    
    # Check expectations
    scenario_passed = False
    failure_reason = ""
    
    expected_action = scenario['expected_action']
    expected_sign = scenario['expected_sign']
    expected_mag = scenario['expected_magnitude']
    
    # Check sign
    sign_ok = True
    if expected_sign == 'positive' and prediction <= 0:
        sign_ok = False
        failure_reason = f"Expected positive (charge), got {prediction:.2f}"
    elif expected_sign == 'negative' and prediction >= 0:
        sign_ok = False
        failure_reason = f"Expected negative (discharge), got {prediction:.2f}"
    elif expected_sign == 'positive_or_zero' and prediction < -0.5:
        sign_ok = False
        failure_reason = f"Expected ≥0 (no discharge), got {prediction:.2f}"
    elif expected_sign == 'negative_or_zero' and prediction > 0.5:
        sign_ok = False
        failure_reason = f"Expected ≤0 (no charge), got {prediction:.2f}"
    
    # Check magnitude
    mag_ok = True
    abs_pred = abs(prediction)
    if '>' in expected_mag:
        threshold = float(expected_mag.split('>')[1].replace('kW', ''))
        if abs_pred < threshold:
            mag_ok = False
            failure_reason = f"Expected |power| > {threshold}, got {abs_pred:.2f}"
    elif '<' in expected_mag:
        threshold = float(expected_mag.split('<')[1].replace('kW', ''))
        if abs_pred > threshold:
            mag_ok = False
            failure_reason = f"Expected |power| < {threshold}, got {abs_pred:.2f}"
    
    scenario_passed = sign_ok and mag_ok
    
    if scenario_passed:
        passed += 1
    else:
        failed += 1
    
    results.append({
        'scenario_id': i,
        'scenario_type': scenario['scenario_type'],
        'price_current': scenario['price_current'],
        'price_avg_4h': scenario['price_avg_4h'],
        'soc': scenario['soc'],
        'prediction': prediction,
        'expected': expected_action,
        'passed': scenario_passed,
        'failure_reason': failure_reason if not scenario_passed else '',
        'rationale': scenario['rationale'],
    })

# Print summary
print(f"\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

total = len(scenarios)
pass_rate = 100 * passed / total

print(f"\nTotal scenarios: {total}")
print(f"Passed: {passed} ({pass_rate:.1f}%)")
print(f"Failed: {failed} ({100 - pass_rate:.1f}%)")

# Breakdown by scenario type
print(f"\n" + "-"*70)
print("PASS RATE BY SCENARIO TYPE")
print("-"*70)

df = pd.DataFrame(results)
type_summary = df.groupby('scenario_type').agg({
    'passed': ['sum', 'count']
}).reset_index()
type_summary.columns = ['scenario_type', 'passed', 'total']
type_summary['pass_rate'] = 100 * type_summary['passed'] / type_summary['total']
type_summary = type_summary.sort_values('pass_rate')

for _, row in type_summary.iterrows():
    print(f"{row['scenario_type']:25s}: {row['passed']:2.0f}/{row['total']:2.0f} ({row['pass_rate']:5.1f}%)")

# Show failures
print(f"\n" + "="*70)
print(f"FAILURE ANALYSIS ({failed} failures)")
print("="*70)

failures = df[~df['passed']].head(20)  # Show first 20 failures

if len(failures) > 0:
    print(f"\n{'Type':<25} {'Price':<8} {'SOC':<6} {'Pred':<8} {'Reason':<40}")
    print("-"*70)
    for _, row in failures.iterrows():
        print(f"{row['scenario_type']:<25} "
              f"{row['price_current']:<8.4f} "
              f"{row['soc']:<6.3f} "
              f"{row['prediction']:<+8.2f} "
              f"{row['failure_reason']:<40}")
else:
    print("\n✓ NO FAILURES - All scenarios passed!")

# Final verdict
print(f"\n" + "="*70)
print("VERDICT")
print("="*70)

THRESHOLD = 80.0  # 80% pass rate

print(f"\nThreshold: {THRESHOLD}% pass rate")
print(f"Observed: {pass_rate:.1f}% pass rate")

if pass_rate >= THRESHOLD:
    print(f"\n✓ PASS - Student model exhibits reasonable battery behavior!")
    print(f"  Teacher-student pipeline appears healthy.")
    print(f"  Safe to proceed with SHAP debugging.")
else:
    print(f"\n✗ FAIL - Student model behavior is problematic!")
    print(f"  Pass rate {pass_rate:.1f}% below threshold {THRESHOLD}%")
    print(f"  Training may be fundamentally broken.")
    print(f"  DO NOT proceed with SHAP - fix training first!")

# Save results
df.to_csv('output/xai/student_behavior_verification.csv', index=False)
print(f"\nDetailed results saved: output/xai/student_behavior_verification.csv")
