"""
Script to apply counterfactual explanations to real test set scenarios.
Answers specific "Why didn't you do X?" questions to understand the model's logic.
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

# Adjust path if script is run from project root
import sys
sys.path.insert(0, 'src')

from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS
from vpp.learning.counterfactuals import CounterfactualExplainer, visualize_cf

def load_environment():
    """Load model, scaler, and training data."""
    model = VPPStudentModel(input_dim=7, max_power=50.0)
    state_dict = torch.load("../artifacts/student_model.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Model remains in evaluation mode through the explainer
    with open('../artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    data = pd.read_csv("../data/training_data.csv")
    
    return model, scaler, data

def main():
    print("="*60)
    print("COUNTERFACTUAL EXPLANATIONS: VIRTUAL POWER PLANT")
    print("="*60)
    
    model, scaler, data = load_environment()
    explainer = CounterfactualExplainer(model, scaler, FEATURE_COLUMNS)
    
    output_dir = Path("output/xai/counterfactuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Features we allow the explainer to modify.
    # We do NOT allow modifying time (hour_sin/cos, dow_sin) or external load (net_load)
    # as the grid operator cannot change those.
    # We can modify market parameters (prices) or physical state boundary (soc).
    mutable_features = ['price_current', 'price_avg_4h', 'soc']
    
    # Get features as numpy array
    X = data[FEATURE_COLUMNS].values
    
    # Scale all features to get original model predictions easily
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).squeeze().numpy()
    
    # =========================================================================
    # SCENARIO 1: "Why didn't you charge here?" 
    # Find a timestep where price is low, SOC is low, but model DISCHARGED or IDLED (pred < 5)
    # We want to know: What would it take for the model to charge?
    # =========================================================================
    prices = data['price_current'].values
    socs = data['soc'].values
    
    # Look for: Price < 20th percentile, SOC < 0.3, prediction < 5 kW
    price_low_thres = np.percentile(prices, 20)
    candidates_1 = np.where((prices < price_low_thres) & (socs < 0.3) & (preds < 5.0))[0]
    
    if len(candidates_1) > 0:
        idx1 = candidates_1[0]
        orig_features1 = X[idx1]
        orig_pred1 = preds[idx1]
        
        print("\n--- SCENARIO 1: Why didn't you charge? ---")
        print(f"Index: {idx1}")
        print(f"Current State: Price=${orig_features1[2]:.4f}/kWh, Future_Price=${orig_features1[3]:.4f}/kWh, SOC={orig_features1[5]:.2f}")
        print(f"Original Prediction: {orig_pred1:.2f} kW (not charging much)")
        
        # Target: Force a solid charge (+30 kW)
        target1 = 30.0
        
        # Run explainer
        cf_features1, cf_pred1, loss1 = explainer.generate(
            original_features=orig_features1,
            target_power=target1,
            mutable_features=mutable_features,
            max_iters=1000,
            lr=0.05,
            l1_lambda=0.5  # High L1 to isolate the most important feature
        )
        
        print(f"Achieved Counterfactual Prediction: {cf_pred1:.2f} kW")
        visualize_cf(
            orig_features1, cf_features1, orig_pred1, cf_pred1, 
            FEATURE_COLUMNS, str(output_dir / "cf_scenario1_charge.png")
        )
        
        # Print differences
        for i, name in enumerate(FEATURE_COLUMNS):
            if abs(cf_features1[i] - orig_features1[i]) > 0.001:
                print(f"  > Changed {name}: {orig_features1[i]:.4f} -> {cf_features1[i]:.4f}")
    
    # =========================================================================
    # SCENARIO 2: "Why didn't you discharge here?" 
    # Find a timestep where price is very high, SOC is high, but model IDLED or CHARGED (pred > -5)
    # =========================================================================
    price_high_thres = np.percentile(prices, 80)
    candidates_2 = np.where((prices > price_high_thres) & (socs > 0.7) & (preds > -5.0))[0]
    
    if len(candidates_2) > 0:
        idx2 = candidates_2[0]  # Take first
        orig_features2 = X[idx2]
        orig_pred2 = preds[idx2]
        
        print("\n--- SCENARIO 2: Why didn't you discharge? ---")
        print(f"Index: {idx2}")
        print(f"Current State: Price=${orig_features2[2]:.4f}/kWh, Future_Price=${orig_features2[3]:.4f}/kWh, SOC={orig_features2[5]:.2f}")
        print(f"Original Prediction: {orig_pred2:.2f} kW (not discharging much)")
        
        # Target: Force a solid discharge (-30 kW)
        target2 = -30.0
        
        cf_features2, cf_pred2, loss2 = explainer.generate(
            original_features=orig_features2,
            target_power=target2,
            mutable_features=mutable_features,
            max_iters=1000,
            lr=0.05,
            l1_lambda=0.5
        )
        
        print(f"Achieved Counterfactual Prediction: {cf_pred2:.2f} kW")
        visualize_cf(
            orig_features2, cf_features2, orig_pred2, cf_pred2, 
            FEATURE_COLUMNS, str(output_dir / "cf_scenario2_discharge.png")
        )
        
        for i, name in enumerate(FEATURE_COLUMNS):
            if abs(cf_features2[i] - orig_features2[i]) > 0.001:
                print(f"  > Changed {name}: {orig_features2[i]:.4f} -> {cf_features2[i]:.4f}")

    # =========================================================================
    # SCENARIO 3: Test Sensitivity to Foresight (Price_avg_4h)
    # Since we discovered previously the student ignores price_avg_4h mostly,
    # let's prove it by restricting mutable_features to ONLY price_avg_4h.
    # Can it flip a decision by ONLY changing the future price?
    # =========================================================================
    print("\n--- SCENARIO 3: Can future price alone flip a decision? ---")
    if len(candidates_1) > 0:
        # Try to make scenario 1 charge by ONLY changing future pricing
        cf_f3, cf_p3, _ = explainer.generate(
            original_features=orig_features1,
            target_power=30.0,
            mutable_features=['price_avg_4h'], # ONLY modifying foresight
            max_iters=1000,
            lr=0.05,
            l1_lambda=0.01  # Lower lambda, let it change as much as it wants
        )
        print(f"Original Prediction: {orig_pred1:.2f} kW")
        print(f"Prediction after maxing out future price perturbation: {cf_p3:.2f} kW")
        fprice_idx = FEATURE_COLUMNS.index('price_avg_4h')
        print(f"Perturbed Future Price: {orig_features1[fprice_idx]:.4f} -> {cf_f3[fprice_idx]:.4f}")
        
        if abs(cf_p3 - orig_pred1) < 5.0:
            print("  Conclusion: The model is highly insensitive to future prices. It refuses to change decision.")
        else:
            print("  Conclusion: The model IS sensitive to future prices when pushed to extremes.")

if __name__ == "__main__":
    main()
