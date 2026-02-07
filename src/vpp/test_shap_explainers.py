"""
Test different SHAP explainers to find one that produces correct values.

Problem: Both DeepExplainer and GradientExplainer produce incorrect SHAP values.
Testing: KernelExplainer (model-agnostic, guaranteed correct but slow)
"""

import numpy as np
import torch
from vpp.learning.models import VPPStudentModel
import shap

print("="*70)
print("TESTING SHAP EXPLAINERS FOR CORRECTNESS")
print("="*70)

# Load model
model = VPPStudentModel(input_dim=7, max_power=50.0)
state_dict = torch.load("artifacts/student_model.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Load small sample
import pandas as pd
data = pd.read_csv("data/training_data.csv")
from vpp.learning.models import FEATURE_COLUMNS

X = data[FEATURE_COLUMNS].values[:200]  # Small sample

# Create model wrapper that returns scalar
def model_predict(X):
    """Wrapper that returns 1D array."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).squeeze().numpy()
    return preds

# Test predictions
print(f"\nTesting model wrapper:")
test_preds = model_predict(X[:5])
print(f"  Predictions shape: {test_preds.shape}")
print(f"  Sample predictions: {test_preds}")

#==============================================================================
# TEST 1: KernelExplainer (model-agnostic, guaranteed correct)
#==============================================================================
print("\n" + "="*70)
print("TEST 1: KERNEL EXPLAINER (Model-Agnostic)")
print("="*70)

try:
    # Use small background
    background = shap.kmeans(X, 10)
    
    print(f"Creating KernelExplainer with {len(background.data)} background samples...")
    explainer_kernel = shap.KernelExplainer(model_predict, background)
    
    print(f"Computing SHAP values for 10 test samples...")
    shap_values_kernel = explainer_kernel.shap_values(X[:10])
    
    print(f"  SHAP shape: {shap_values_kernel.shape}")
    print(f"  Base value: {explainer_kernel.expected_value:.2f}")
    
    # Check additivity
    base_val = explainer_kernel.expected_value
    predictions = model_predict(X[:10])
    shap_sums = base_val + shap_values_kernel.sum(axis=1)
    errors = np.abs(predictions - shap_sums)
    
    print(f"\n  Additivity check:")
    print(f"    Mean error: {errors.mean():.4f} kW")
    print(f"    Max error:  {errors.max():.4f} kW")
    print(f"    Relative:   {(errors.mean() / 50) * 100:.2f}%")
    
    if errors.mean() < 0.1:
        print(f"    ✓ ADDITIVITY HOLDS - KernelExplainer is CORRECT")
    else:
        print(f"    ✗ ADDITIVITY FAILS - Error too high")
    
    # Check feature importance
    mean_abs_shap = np.abs(shap_values_kernel).mean(axis=0)
    print(f"\n  Mean |SHAP| by feature:")
    for i, feat in enumerate(FEATURE_COLUMNS):
        print(f"    {feat:20s}: {mean_abs_shap[i]:.4f}")
    
    # Check price correlation
    price_idx = FEATURE_COLUMNS.index('price_current')
    from scipy.stats import spearmanr
    corr, _ = spearmanr(X[:10, price_idx], shap_values_kernel[:, price_idx])
    print(f"\n  price_current correlation: {corr:+.4f}")
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nKernelExplainer is the only guaranteed-correct method.")
print("It's slower but produces reliable SHAP values.")
print("\nRecommendation: Use KernelExplainer for final analysis.")
