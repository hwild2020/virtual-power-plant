"""
PIPELINE INTEGRITY VERIFICATION

Step 1: Verify Teacher Data Integrity
- Load training_data.csv
- For 100 random samples, re-run PerfectForesightOptimizer
- Compare optimizer output vs power_setpoint column
- Threshold: If mismatch > 5kW, data generation is BROKEN

This script validates that training data was correctly generated.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Import teacher optimizer
from vpp.optimization.offline import PerfectForesightOptimizer, BatteryConstraints

print("="*70)
print("STEP 1: TEACHER DATA INTEGRITY VERIFICATION")
print("="*70)

# Load training data
data_path = "data/training_data.csv"
print(f"\nLoading training data: {data_path}")
data = pd.read_csv(data_path)
print(f"  Total samples: {len(data)}")

# Sample 100 random rows
np.random.seed(42)
sample_indices = np.random.choice(len(data), size=100, replace=False)
sample_data = data.iloc[sample_indices].reset_index(drop=True)

print(f"\nSampled 100 random rows for verification")

# Initialize teacher optimizer
optimizer = PerfectForesightOptimizer()

# Battery constraints (from training specs)
constraints = BatteryConstraints(
    energy_capacity=100.0,  # kWh
    max_power=50.0,         # kW
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    soc_min=0.1,
    soc_max=0.9,
)

print(f"\nBattery Constraints:")
print(f"  energy_capacity: {constraints.energy_capacity} kWh")
print(f"  max_power: {constraints.max_power} kW")
print(f"  charge_efficiency: {constraints.charge_efficiency}")
print(f"  discharge_efficiency: {constraints.discharge_efficiency}")
print(f"  soc_min: {constraints.soc_min}")
print(f"  soc_max: {constraints.soc_max}")

# Verification arrays
errors = []
original_powers = []
recomputed_powers = []

print(f"\n" + "="*70)
print("RE-RUNNING OPTIMIZER ON SAMPLED DATA")
print("="*70)

# Group by episode (assuming data has episode structure)
# For now, we'll verify single-step decisions which is what the student learns

print("\nVerifying single-step optimization decisions...")
print("(Note: Teacher uses multi-period optimization, so single-step")
print(" comparison may show differences due to foresight)")

for i, row in sample_data.iterrows():
    # Extract features
    price_current = row['price_current']
    net_load = row['net_load']
    soc = row['soc']
    power_setpoint_recorded = row['power_setpoint']
    
    # For single-step verification, we need price horizon
    # In training, teacher had 4-hour lookahead
    # For this test, use price_avg_4h as proxy for future prices
    price_avg_4h = row['price_avg_4h']
    
    # Create 4-hour price horizon (simplified: assume constant at avg)
    horizon_length = 4
    prices = np.full(horizon_length, price_avg_4h)
    load = np.full(horizon_length, net_load)
    
    # Run optimizer
    try:
        result = optimizer.solve_horizon(
            prices=prices,
            load=load,
            initial_soc=soc,
            constraints=constraints,
        )
        
        # First timestep is the decision
        power_recomputed = result.power[0]
        
        # Compare
        error = abs(power_recomputed - power_setpoint_recorded)
        errors.append(error)
        original_powers.append(power_setpoint_recorded)
        recomputed_powers.append(power_recomputed)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/100 samples...")
            
    except Exception as e:
        print(f"  ERROR at sample {i}: {e}")
        errors.append(np.nan)
        original_powers.append(power_setpoint_recorded)
        recomputed_powers.append(np.nan)

# Compute statistics
errors = np.array(errors)
valid_mask = ~np.isnan(errors)
valid_errors = errors[valid_mask]

print(f"\n" + "="*70)
print("VERIFICATION RESULTS")
print("="*70)

print(f"\nValid samples: {valid_mask.sum()}/100")

if len(valid_errors) > 0:
    print(f"\nError Statistics (|recomputed - recorded|):")
    print(f"  Mean error:   {valid_errors.mean():.2f} kW")
    print(f"  Median error: {np.median(valid_errors):.2f} kW")
    print(f"  StdDev:       {valid_errors.std():.2f} kW")
    print(f"  Max error:    {valid_errors.max():.2f} kW")
    print(f"  Min error:    {valid_errors.min():.2f} kW")
    
    # Distribution
    print(f"\nError Distribution:")
    print(f"  < 1 kW:    {(valid_errors < 1).sum()} samples ({(valid_errors < 1).sum()/len(valid_errors)*100:.1f}%)")
    print(f"  < 5 kW:    {(valid_errors < 5).sum()} samples ({(valid_errors < 5).sum()/len(valid_errors)*100:.1f}%)")
    print(f"  < 10 kW:   {(valid_errors < 10).sum()} samples ({(valid_errors < 10).sum()/len(valid_errors)*100:.1f}%)")
    print(f"  >= 10 kW:  {(valid_errors >= 10).sum()} samples ({(valid_errors >= 10).sum()/len(valid_errors)*100:.1f}%)")
    
    # Threshold check
    mean_error = valid_errors.mean()
    THRESHOLD = 5.0  # kW
    
    print(f"\n" + "="*70)
    print("INTEGRITY CHECK")
    print("="*70)
    print(f"\nThreshold: Mean error must be < {THRESHOLD} kW")
    print(f"Observed:  Mean error = {mean_error:.2f} kW")
    
    if mean_error < THRESHOLD:
        print(f"\n✓ PASS - Teacher data integrity verified!")
        print(f"  Training data appears to be correctly generated.")
    else:
        print(f"\n✗ FAIL - Teacher data integrity compromised!")
        print(f"  Mean error {mean_error:.2f} kW exceeds threshold {THRESHOLD} kW")
        print(f"\n  CRITICAL: Data generation may be BROKEN")
        print(f"  Need to investigate data generation pipeline.")
    
    # Sample comparison
    print(f"\n" + "="*70)
    print("SAMPLE COMPARISONS (First 10)")
    print("="*70)
    print(f"\n{'Idx':<6} {'SOC':<8} {'Price':<10} {'Recorded':<12} {'Recomputed':<12} {'Error':<10}")
    print("-"*70)
    
    for i in range(min(10, len(sample_data))):
        if valid_mask[i]:
            idx = sample_indices[i]
            soc = sample_data.iloc[i]['soc']
            price = sample_data.iloc[i]['price_current']
            rec = original_powers[i]
            recomp = recomputed_powers[i]
            err = errors[i]
            print(f"{idx:<6} {soc:<8.3f} {price:<10.4f} {rec:<12.2f} {recomp:<12.2f} {err:<10.2f}")
    
    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter: recorded vs recomputed
        axes[0].scatter(original_powers, recomputed_powers, alpha=0.5)
        axes[0].plot([-50, 50], [-50, 50], 'r--', label='Perfect match')
        axes[0].set_xlabel('Recorded Power (kW)')
        axes[0].set_ylabel('Recomputed Power (kW)')
        axes[0].set_title('Teacher Data Integrity Check')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error histogram
        axes[1].hist(valid_errors, bins=30, edgecolor='black')
        axes[1].axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD} kW)')
        axes[1].axvline(mean_error, color='green', linestyle='-', label=f'Mean ({mean_error:.2f} kW)')
        axes[1].set_xlabel('Absolute Error (kW)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = 'output/xai/teacher_integrity_check.png'
        plt.savefig(out_path, dpi=200)
        print(f"\n  Saved visualization: {out_path}")
        
    except Exception as e:
        print(f"\n  Could not generate visualization: {e}")

else:
    print("\n✗ CRITICAL ERROR - No valid samples!")
    print("  All optimizer runs failed. Check constraints and data format.")

print(f"\n" + "="*70)
print("NEXT STEP")
print("="*70)

if len(valid_errors) > 0 and valid_errors.mean() < THRESHOLD:
    print("\nTeacher data integrity: ✓ VERIFIED")
    print("Proceed to Step 2: Verify student learned correctly")
    print("  Run: python src/vpp/verify_student.py")
else:
    print("\nTeacher data integrity: ✗ FAILED")
    print("DO NOT PROCEED - Fix data generation first!")
    print("  Check: src/vpp/data/generate_training_data.py")
