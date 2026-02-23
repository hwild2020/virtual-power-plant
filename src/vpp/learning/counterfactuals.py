"""
Counterfactual Explanations for the Virtual Power Plant Student Model.

Provides insights into "what-if" scenarios by finding the minimum feature 
perturbation needed to change the model's prediction to a target value.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Need to make sure the imports work correctly when called as a module or script
from .models import FEATURE_COLUMNS


class CounterfactualExplainer:
    """
    Finds counterfactual explanations for a differential PyTorch model using gradient descent.
    """
    def __init__(
        self, 
        model: nn.Module, 
        scaler: StandardScaler, 
        feature_names: List[str] = FEATURE_COLUMNS,
    ):
        """
        Initialize the explainer.
        
        Args:
            model: PyTorch model to explain
            scaler: StandardScaler used to scale the input features
            feature_names: Names of the features, parallel to the model's input
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Ensure model is in eval mode and doesn't require gradients for its parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _create_feature_mask(self, mutable_features: List[str]) -> torch.Tensor:
        """
        Create a binary mask: 1 for features we can change, 0 for frozen features.
        
        Args:
            mutable_features: List of feature names that the optimizer is allowed to alter
            
        Returns:
            Tensor mask of shape (num_features,)
        """
        mask = torch.zeros(len(self.feature_names), dtype=torch.float32)
        for i, name in enumerate(self.feature_names):
            if name in mutable_features:
                mask[i] = 1.0
        return mask

    def generate(
        self, 
        original_features: np.ndarray, 
        target_power: float, 
        mutable_features: List[str],
        max_iters: int = 500,
        lr: float = 0.05,
        target_tolerance: float = 1.0,
        l1_lambda: float = 0.1,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Find a counterfactual feature vector that achieves the target power.
        
        Args:
            original_features: 1D array of the original Unscaled feature values.
            target_power: The desired power output (kW). Positive=Charge, Negative=Discharge.
            mutable_features: Features allowed to change (e.g., ['price_current', 'soc']).
            max_iters: Maximum optimization steps.
            lr: Learning rate for the perturbation optimization.
            target_tolerance: Acceptable distance from the target power (kW).
            l1_lambda: Regularization strength for the perturbation (higher = encourages sparsity).
            
        Returns:
            Tuple of:
            - Counterfactual unscaled feature vector
            - Final prediction achieved
            - Final loss
        """
        # 1. Prepare mask
        # Example mutable: ['price_current', 'price_avg_4h', 'soc']
        mask = self._create_feature_mask(mutable_features)
        
        # 2. Scale the original feature vector
        # Model operates in scaled space
        orig_scaled = self.scaler.transform(original_features.reshape(1, -1))[0]
        orig_tensor = torch.tensor(orig_scaled, dtype=torch.float32)
        
        # 3. Initialize perturbation parameter (starts at 0)
        # We optimize the perturbation (delta) rather than the absolute value
        delta = nn.Parameter(torch.zeros_like(orig_tensor))
        
        optimizer = optim.Adam([delta], lr=lr)
        criterion = nn.MSELoss()
        
        target_tensor = torch.tensor([target_power], dtype=torch.float32)
        
        best_delta = None
        best_loss = float('inf')
        
        # [NEW] Pre-calculate physical bounds for SOC in scaled space
        # Assuming SOC is bound between 0.0 and 1.0
        # Find the index of SOC in the feature columns
        soc_idx = self.feature_names.index('soc') if 'soc' in self.feature_names else -1
        
        if soc_idx != -1:
            # Create dummy arrays to find the scaled values of SOC=0 and SOC=1
            dummy_min = np.zeros((1, len(self.feature_names)))
            dummy_max = np.zeros((1, len(self.feature_names)))
            dummy_max[0, soc_idx] = 1.0
            
            scaled_soc_min = self.scaler.transform(dummy_min)[0, soc_idx]
            scaled_soc_max = self.scaler.transform(dummy_max)[0, soc_idx]

        # 4. Optimization Loop
        for i in range(max_iters):
            optimizer.zero_grad()
            
            # Apply mask to delta
            cf_tensor_scaled = orig_tensor + mask * delta
            
            # Forward pass
            prediction = self.model(cf_tensor_scaled.unsqueeze(0)).squeeze()
            
            # [FIXED] Component 1: One-Sided Hinge Loss for Thresholds
            # If target > 0 (Charge), we want prediction >= target
            # If target < 0 (Discharge), we want prediction <= target
            if target_power > 0:
                pred_loss = torch.relu(target_tensor - prediction)**2
            else:
                pred_loss = torch.relu(prediction - target_tensor)**2
                
            # Component 2: Penalty for changing features
            reg_loss = l1_lambda * torch.sum(torch.abs(mask * delta))
            
            total_loss = pred_loss + reg_loss
            
            # Keep track of the best solution
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta = delta.clone().detach()
                
            total_loss.backward()
            optimizer.step()
            
            # [FIXED] Projected Gradient Descent (PGD) - Mandatory Physical Constraints
            with torch.no_grad():
                # Reconstruct the current scaled tensor
                current_cf_scaled = orig_tensor + mask * delta
                
                # Clamp SOC strictly within its physical bounds (in scaled space)
                if soc_idx != -1:
                    current_cf_scaled[soc_idx] = torch.clamp(
                        current_cf_scaled[soc_idx], 
                        min=float(scaled_soc_min), 
                        max=float(scaled_soc_max)
                    )
                
                # Re-update delta to reflect the clamped reality
                delta.data = current_cf_scaled - orig_tensor
        
        # 5. Extract results
        final_cf_scaled = orig_tensor + mask * best_delta
        
        with torch.no_grad():
            final_pred = self.model(final_cf_scaled.unsqueeze(0)).squeeze().item()
            
        # Inverse transform back to unscaled space
        final_cf_unscaled = self.scaler.inverse_transform(final_cf_scaled.unsqueeze(0).numpy())[0]
        
        return final_cf_unscaled, final_pred, best_loss

def visualize_cf(
    orig_features: np.ndarray, 
    cf_features: np.ndarray, 
    orig_pred: float, 
    cf_pred: float, 
    feature_names: List[str],
    save_path: str = None
):
    """
    Visualizes the difference between the original and counterfactual features.
    Only shows features that changed significantly.
    """
    diffs = cf_features - orig_features
    
    # Filter for features that actually changed (> 1% relative change or significant absolute)
    # We use a small epsilon to avoid division by zero
    eps = 1e-6
    rel_diffs = np.abs(diffs) / (np.abs(orig_features) + eps)
    
    # Identify changed features (e.g. changed more than 0.1% or absolute diff > 0.001)
    changed_indices = np.where((rel_diffs > 0.001) | (np.abs(diffs) > 0.001))[0]
    
    if len(changed_indices) == 0:
        print("Note: The optimizer could not find a counterfactual that changed the features.")
        return
        
    names = [feature_names[i] for i in changed_indices]
    o_vals = orig_features[changed_indices]
    c_vals = cf_features[changed_indices]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2]})
    
    # Subplot 1: Predictions
    ax1.bar(['Original', 'Counterfactual'], [orig_pred, cf_pred], color=['blue', 'red'], alpha=0.7)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Model Prediction')
    
    # Subplot 2: Changed Features
    rects1 = ax2.bar(x - width/2, o_vals, width, label='Original', color='blue', alpha=0.7)
    rects2 = ax2.bar(x + width/2, c_vals, width, label='Counterfactual', color='red', alpha=0.7)
    
    ax2.set_ylabel('Feature Value')
    ax2.set_title('Feature Changes Required')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.legend()
    
    # Attach labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1, ax2)
    autolabel(rects2, ax2)
    
    fig.suptitle(f"Counterfactual Explanation: Shift from {orig_pred:.1f}kW to {cf_pred:.1f}kW", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved Counterfactual visualization to {save_path}")
    else:
        plt.show()
    plt.close()
