"""
XAI Analysis for VPP Student Model using SHAP.

This module provides explainability analysis to answer:
1. Which features drive student model decisions?
2. Does the student learn correct price arbitrage logic?
3. Where/when does the student deviate from the teacher?

Uses SHAP (SHapley Additive exPlanations) with GradientExplainer for PyTorch models.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from vpp.learning.models import VPPStudentModel, FEATURE_COLUMNS, TARGET_COLUMN


class VPPExplainer:
    """SHAP-based explainability analysis for VPP student model.
    
    Implements GradientExplainer for reliable PyTorch model analysis with stratified
    sampling strategy to ensure representative coverage of price regimes and
    SOC ranges.
    
    Args:
        model_path: Path to trained student model (.pth file)
        data_path: Path to training data CSV
        n_background: Number of background samples for SHAP (default: 50)
        n_explain: Number of samples to explain (default: 5000)
        output_dir: Directory for saving plots and reports
        device: PyTorch device ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        n_background: int = 50,
        n_explain: int = 5000,
        output_dir: str = "output/xai",
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.n_background = n_background
        self.n_explain = n_explain
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        print(f"Loading model from {self.model_path}...")
        self.model = self._load_model()
        
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Extract features and targets
        self.X = self.data[FEATURE_COLUMNS].values
        self.y = self.data[TARGET_COLUMN].values
        
        print(f"Dataset: {len(self.X)} samples, {self.X.shape[1]} features")
        
        # Create stratified sample for explanation
        self.X_explain, self.y_explain, self.explain_indices = self._create_stratified_sample()
        
        # Create background dataset using KMeans
        self.X_background = self._create_background_dataset()
        
        # Initialize SHAP explainer
        print("Initializing SHAP GradientExplainer...")
        self.explainer = self._create_explainer()
        
        # Compute SHAP values
        print(f"Computing SHAP values for {len(self.X_explain)} samples...")
        self.shap_values = None
        self.base_value = None
        
    def _load_model(self) -> VPPStudentModel:
        """Load trained student model."""
        model = VPPStudentModel(input_dim=len(FEATURE_COLUMNS), max_power=50.0)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model
    
    def _create_stratified_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create stratified sample covering price quantiles and SOC ranges.
        
        Strategy:
        - Stratify by price quantiles (low/medium/high)
        - Ensure coverage of full SOC range
        - Oversample SOC boundaries and price extremes (per user request)
        
        Returns:
            X_sample, y_sample, sample_indices
        """
        df = self.data.copy()
        df['price_quantile'] = pd.qcut(df['price_current'], q=3, labels=['low', 'med', 'high'])
        df['soc_range'] = pd.cut(
            df['soc'],
            bins=[0, 0.2, 0.8, 1.0],
            labels=['boundary_low', 'normal', 'boundary_high']
        )
        
        # Compute base sample sizes
        base_per_group = self.n_explain // 9  # 3 price × 3 soc = 9 groups
        
        # Oversample focus areas (SOC boundaries + price extremes)
        focus_multiplier = 1.5
        
        samples = []
        for price_q in ['low', 'med', 'high']:
            for soc_r in ['boundary_low', 'normal', 'boundary_high']:
                mask = (df['price_quantile'] == price_q) & (df['soc_range'] == soc_r)
                group_df = df[mask]
                
                # Determine sample size (oversample focus areas)
                if soc_r in ['boundary_low', 'boundary_high'] or price_q in ['low', 'high']:
                    n_samples = int(base_per_group * focus_multiplier)
                else:
                    n_samples = base_per_group
                
                # Sample from group
                if len(group_df) > 0:
                    n_samples = min(n_samples, len(group_df))
                    group_sample = group_df.sample(n=n_samples, random_state=42)
                    samples.append(group_sample)
        
        # Combine all samples
        sampled_df = pd.concat(samples, ignore_index=False)
        
        # If we have too many, downsample uniformly
        if len(sampled_df) > self.n_explain:
            sampled_df = sampled_df.sample(n=self.n_explain, random_state=42)
        
        indices = sampled_df.index.values
        X_sample = sampled_df[FEATURE_COLUMNS].values
        y_sample = sampled_df[TARGET_COLUMN].values
        
        print(f"Stratified sample: {len(X_sample)} samples")
        print(f"  Price ranges: {sampled_df.groupby('price_quantile').size().to_dict()}")
        print(f"  SOC ranges: {sampled_df.groupby('soc_range').size().to_dict()}")
        
        return X_sample, y_sample, indices
    
    def _create_background_dataset(self) -> np.ndarray:
        """Create background dataset using KMeans clustering.
        
        Returns k-means centroids as representative background samples.
        """
        print(f"Creating background dataset (k={self.n_background})...")
        kmeans = KMeans(n_clusters=self.n_background, random_state=42, n_init=10)
        kmeans.fit(self.X)
        return kmeans.cluster_centers_
    
    def _create_explainer(self) -> shap.GradientExplainer:
        """Create SHAP GradientExplainer for PyTorch model.
        
        Note: Using GradientExplainer instead of DeepExplainer because the model
        uses tanh activation which DeepExplainer doesn't handle correctly,
        leading to a 37% additivity error. GradientExplainer uses gradients
        to compute attributions, which correctly handles all differentiable operations.
        """
        # Convert background to tensor
        background_tensor = torch.tensor(
            self.X_background, dtype=torch.float32, device=self.device
        )
        
        # Create explainer (GradientExplainer handles tanh correctly)
        explainer = shap.GradientExplainer(self.model, background_tensor)
        return explainer
    
    def compute_shap_values(self) -> np.ndarray:
        """Compute SHAP values for explanation samples.
        
        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        if self.shap_values is not None:
            return self.shap_values
        
        # Convert to tensor
        X_tensor = torch.tensor(self.X_explain, dtype=torch.float32, device=self.device)
        
        # Compute SHAP values
        shap_values_raw = self.explainer.shap_values(X_tensor)
        
        # GradientExplainer returns shape (n_samples, n_features, n_outputs)
        # For single output model, squeeze the last dimension
        if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            shap_values_raw = shap_values_raw.squeeze(-1)  # (n_samples, n_features)
        elif isinstance(shap_values_raw, list):
            # If list of arrays, extract first and squeeze
            shap_values_raw = shap_values_raw[0]
            if shap_values_raw.ndim == 3:
                shap_values_raw = shap_values_raw.squeeze(-1)
        
        # Convert to numpy
        self.shap_values = shap_values_raw
        
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, (list, np.ndarray)):
                self.base_value = self.base_value[0]
        else:
            # Fallback: compute mean prediction on background
            with torch.no_grad():
                background_tensor = torch.tensor(
                    self.X_background, dtype=torch.float32, device=self.device
                )
                self.base_value = self.model(background_tensor).mean().item()
        
        print(f"SHAP values computed: shape={self.shap_values.shape}")
        print(f"Base value (expected): {self.base_value:.2f} kW")
        
        return self.shap_values
    
    def plot_summary(self, save: bool = True) -> None:
        """Generate SHAP summary plot (global feature importance).
        
        Creates a beeswarm plot showing feature importance ranked by
        mean absolute SHAP value.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        print("\nGenerating SHAP summary plot...")
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            features=self.X_explain,
            feature_names=FEATURE_COLUMNS,
            show=False,
        )
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "shap_summary.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        save: bool = True,
    ) -> None:
        """Generate SHAP dependence plot.
        
        Args:
            feature: Primary feature name
            interaction_feature: Optional interaction feature for coloring
            save: Whether to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if feature not in FEATURE_COLUMNS:
            raise ValueError(f"Feature '{feature}' not in {FEATURE_COLUMNS}")
        
        feature_idx = FEATURE_COLUMNS.index(feature)
        
        print(f"\nGenerating dependence plot for '{feature}'...")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            features=self.X_explain,
            feature_names=FEATURE_COLUMNS,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"shap_dependence_{feature}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def plot_force(
        self,
        sample_idx: int,
        scenario_name: str = "sample",
        save: bool = True,
    ) -> None:
        """Generate SHAP force plot for individual prediction.
        
        Args:
            sample_idx: Index in X_explain to visualize
            scenario_name: Descriptive name for saving
            save: Whether to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        print(f"\nGenerating force plot for {scenario_name} (idx={sample_idx})...")
        
        # Create force plot
        force_plot = shap.force_plot(
            self.base_value,
            self.shap_values[sample_idx, :],
            features=self.X_explain[sample_idx, :],
            feature_names=FEATURE_COLUMNS,
            matplotlib=False,  # Use interactive HTML
        )
        
        if save:
            save_path = self.output_dir / f"shap_force_{scenario_name}.html"
            shap.save_html(str(save_path), force_plot)
            print(f"  Saved: {save_path}")
    
    def analyze_arbitrage_logic(self) -> Dict[str, float]:
        """Analyze price arbitrage logic learning.
        
        Validates that the model learns correct arbitrage behavior:
        - High price → negative SHAP (push toward discharge)
        - Low price → positive SHAP (push toward charge)
        
        Returns:
            Dictionary with correlation metrics and statistics
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        print("\n" + "="*60)
        print("PRICE ARBITRAGE LOGIC ANALYSIS")
        print("="*60)
        
        # Get price_current feature index
        price_idx = FEATURE_COLUMNS.index('price_current')
        price_values = self.X_explain[:, price_idx]
        price_shap = self.shap_values[:, price_idx]
        
        # Compute Spearman correlation (monotonic relationship)
        corr_spearman, p_value = spearmanr(price_values, price_shap)
        
        # Compute Pearson correlation (linear relationship)
        corr_pearson = np.corrcoef(price_values, price_shap)[0, 1]
        
        # Analyze by price quantiles
        price_quantiles = pd.qcut(price_values, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        quantile_stats = {}
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            mask = price_quantiles == q
            quantile_stats[q] = {
                'mean_price': price_values[mask].mean(),
                'mean_shap': price_shap[mask].mean(),
                'n_samples': mask.sum(),
            }
        
        # Print results
        print(f"\nCorrelation Metrics:")
        print(f"  Spearman ρ: {corr_spearman:.4f} (p={p_value:.4e})")
        print(f"  Pearson r:  {corr_pearson:.4f}")
        print(f"  Target: ρ < -0.5 (negative correlation)")
        
        if corr_spearman < -0.5:
            print(f"  ✓ PASS: Model learns price arbitrage logic!")
        else:
            print(f"  ✗ FAIL: Weak arbitrage correlation (target: ρ < -0.5)")
        
        print(f"\nPrice Quantile Analysis:")
        print(f"  {'Quantile':<10} {'Mean Price':>12} {'Mean SHAP':>12} {'N':>8}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
        for q, stats in quantile_stats.items():
            print(f"  {q:<10} {stats['mean_price']:>12.4f} {stats['mean_shap']:>12.2f} {stats['n_samples']:>8}")
        
        return {
            'spearman_correlation': corr_spearman,
            'spearman_p_value': p_value,
            'pearson_correlation': corr_pearson,
            'quantile_stats': quantile_stats,
        }
    
    def analyze_deviations(self) -> Dict:
        """Analyze student-teacher deviations.
        
        Identifies where and when the student model deviates from teacher,
        focusing on:
        - SOC boundaries (0.1-0.2 and 0.8-0.9)
        - Price extremes (top/bottom 10%)
        - Transition periods (rapid price changes)
        
        Returns:
            Dictionary with deviation statistics and insights
        """
        print("\n" + "="*60)
        print("STUDENT-TEACHER DEVIATION ANALYSIS")
        print("="*60)
        
        # Get student predictions
        with torch.no_grad():
            X_tensor = torch.tensor(self.X_explain, dtype=torch.float32, device=self.device)
            y_student = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Compute errors
        errors = np.abs(y_student - self.y_explain)
        
        # Extract features
        price_idx = FEATURE_COLUMNS.index('price_current')
        soc_idx = FEATURE_COLUMNS.index('soc')
        
        prices = self.X_explain[:, price_idx]
        socs = self.X_explain[:, soc_idx]
        
        # Define focus regions
        soc_boundary_low = (socs >= 0.1) & (socs <= 0.2)
        soc_boundary_high = (socs >= 0.8) & (socs <= 0.9)
        soc_normal = (socs > 0.2) & (socs < 0.8)
        
        price_low = prices <= np.percentile(prices, 10)
        price_high = prices >= np.percentile(prices, 90)
        price_normal = ~(price_low | price_high)
        
        # Compute rapid price changes
        full_prices = self.data['price_current'].values
        price_changes = np.abs(np.diff(full_prices))
        rapid_transitions = price_changes >= np.percentile(price_changes, 90)
        rapid_transition_mask = np.zeros(len(self.X_explain), dtype=bool)
        for i, idx in enumerate(self.explain_indices):
            if idx > 0 and idx < len(rapid_transitions):
                rapid_transition_mask[i] = rapid_transitions[idx - 1]
        
        # Compute statistics
        stats = {
            'overall': {
                'mean_error': errors.mean(),
                'max_error': errors.max(),
                'std_error': errors.std(),
            },
            'soc_regions': {
                'boundary_low': errors[soc_boundary_low].mean() if soc_boundary_low.sum() > 0 else 0,
                'normal': errors[soc_normal].mean() if soc_normal.sum() > 0 else 0,
                'boundary_high': errors[soc_boundary_high].mean() if soc_boundary_high.sum() > 0 else 0,
            },
            'price_regions': {
                'low': errors[price_low].mean() if price_low.sum() > 0 else 0,
                'normal': errors[price_normal].mean() if price_normal.sum() > 0 else 0,
                'high': errors[price_high].mean() if price_high.sum() > 0 else 0,
            },
            'transitions': {
                'rapid_changes': errors[rapid_transition_mask].mean() if rapid_transition_mask.sum() > 0 else 0,
                'normal': errors[~rapid_transition_mask].mean() if (~rapid_transition_mask).sum() > 0 else 0,
            },
        }
        
        # Print results
        print(f"\nOverall Performance:")
        print(f"  Mean Absolute Error: {stats['overall']['mean_error']:.2f} kW")
        print(f"  Max Absolute Error:  {stats['overall']['max_error']:.2f} kW")
        print(f"  Std Deviation:       {stats['overall']['std_error']:.2f} kW")
        
        print(f"\nError by SOC Region:")
        print(f"  Boundary Low (0.1-0.2):  {stats['soc_regions']['boundary_low']:.2f} kW")
        print(f"  Normal (0.2-0.8):        {stats['soc_regions']['normal']:.2f} kW")
        print(f"  Boundary High (0.8-0.9): {stats['soc_regions']['boundary_high']:.2f} kW")
        
        print(f"\nError by Price Region:")
        print(f"  Low (<10th percentile):   {stats['price_regions']['low']:.2f} kW")
        print(f"  Normal:                   {stats['price_regions']['normal']:.2f} kW")
        print(f"  High (>90th percentile):  {stats['price_regions']['high']:.2f} kW")
        
        print(f"\nError by Price Transition:")
        print(f"  Rapid changes (>90th %ile): {stats['transitions']['rapid_changes']:.2f} kW")
        print(f"  Normal:                      {stats['transitions']['normal']:.2f} kW")
        
        # Create deviation heatmap
        self._plot_deviation_heatmap(prices, socs, errors)
        
        return stats
    
    def _plot_deviation_heatmap(
        self,
        prices: np.ndarray,
        socs: np.ndarray,
        errors: np.ndarray,
    ) -> None:
        """Plot 2D heatmap of deviations across price-SOC space."""
        print(f"\nGenerating deviation heatmap...")
        
        # Create bins
        price_bins = np.linspace(prices.min(), prices.max(), 20)
        soc_bins = np.linspace(0.1, 0.9, 20)
        
        # Compute 2D histogram of errors
        H, xedges, yedges = np.histogram2d(
            prices, socs, bins=[price_bins, soc_bins], weights=errors
        )
        counts, _, _ = np.histogram2d(prices, socs, bins=[price_bins, soc_bins])
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            H_mean = H / counts
            H_mean[~np.isfinite(H_mean)] = 0
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            H_mean.T,
            origin='lower',
            aspect='auto',
            extent=[price_bins[0], price_bins[-1], soc_bins[0], soc_bins[-1]],
            cmap='YlOrRd',
        )
        
        ax.set_xlabel('Market Price ($/kWh)', fontsize=12)
        ax.set_ylabel('State of Charge (SOC)', fontsize=12)
        ax.set_title('Student-Teacher Deviation Heatmap\n(Mean Absolute Error)', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Absolute Error (kW)', fontsize=11)
        
        plt.tight_layout()
        save_path = self.output_dir / "deviation_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def generate_report(
        self,
        arbitrage_results: Dict,
        deviation_results: Dict,
    ) -> None:
        """Generate comprehensive XAI report answering key questions.
        
        Args:
            arbitrage_results: Results from analyze_arbitrage_logic()
            deviation_results: Results from analyze_deviations()
        """
        print("\n" + "="*60)
        print("GENERATING XAI REPORT")
        print("="*60)
        
        # Compute global feature importance
        if self.shap_values is None:
            self.compute_shap_values()
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = [
            (FEATURE_COLUMNS[i], mean_abs_shap[i])
            for i in range(len(FEATURE_COLUMNS))
        ]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Write report
        report_path = self.output_dir / "xai_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VPP STUDENT MODEL - XAI ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Data: {self.data_path}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples Analyzed: {len(self.X_explain):,} (stratified from {len(self.X):,})\n")
            f.write("\n")
            
            # Question 1: Feature Importance
            f.write("="*60 + "\n")
            f.write("Q1: WHICH FEATURES DRIVE STUDENT DECISIONS MOST?\n")
            f.write("="*60 + "\n\n")
            
            f.write("Global Feature Importance (Mean |SHAP|):\n")
            f.write(f"  {'Rank':<6} {'Feature':<20} {'Importance':>15}\n")
            f.write(f"  {'-'*6} {'-'*20} {'-'*15}\n")
            for rank, (feature, importance) in enumerate(feature_importance, 1):
                f.write(f"  {rank:<6} {feature:<20} {importance:>15.4f}\n")
            
            f.write("\nKey Insights:\n")
            top_3 = [f[0] for f in feature_importance[:3]]
            f.write(f"  - Top 3 features: {', '.join(top_3)}\n")
            
            if 'price_current' in top_3 or 'price_avg_4h' in top_3:
                f.write(f"  - ✓ Price features are among top drivers (expected for arbitrage)\n")
            if 'soc' in top_3:
                f.write(f"  - ✓ SOC is a key driver (expected for constraint awareness)\n")
            
            f.write("\n")
            
            # Question 2: Price Arbitrage Logic
            f.write("="*60 + "\n")
            f.write("Q2: DOES STUDENT LEARN CORRECT PRICE ARBITRAGE LOGIC?\n")
            f.write("="*60 + "\n\n")
            
            spearman_corr = arbitrage_results['spearman_correlation']
            f.write(f"Correlation Analysis (price_current vs SHAP):\n")
            f.write(f"  Spearman ρ: {spearman_corr:.4f}\n")
            f.write(f"  Target: ρ < -0.5 (negative correlation)\n")
            f.write(f"  Result: {'✓ PASS' if spearman_corr < -0.5 else '✗ FAIL'}\n\n")
            
            f.write("Expected Behavior:\n")
            f.write("  - High price → Negative SHAP → Push toward discharge\n")
            f.write("  - Low price → Positive SHAP → Push toward charge\n\n")
            
            f.write("Observed Behavior (by Price Quantile):\n")
            for q, stats in arbitrage_results['quantile_stats'].items():
                direction = "discharge" if stats['mean_shap'] < 0 else "charge"
                f.write(f"  {q}: Price={stats['mean_price']:.4f}, SHAP={stats['mean_shap']:+.2f} → {direction}\n")
            
            f.write("\nConclusion:\n")
            if spearman_corr < -0.5:
                f.write("  ✓ Student successfully learns price arbitrage logic.\n")
                f.write("    Model exhibits correct negative correlation between price and SHAP values.\n")
            else:
                f.write("  ⚠ Student shows weak arbitrage learning.\n")
                f.write("    Correlation is weaker than expected (ρ < -0.5).\n")
            
            f.write("\n")
            
            # Question 3: Deviations
            f.write("="*60 + "\n")
            f.write("Q3: WHERE/WHEN DOES STUDENT DEVIATE FROM TEACHER?\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall Performance:\n")
            f.write(f"  Mean Absolute Error: {deviation_results['overall']['mean_error']:.2f} kW\n")
            f.write(f"  Max Absolute Error:  {deviation_results['overall']['max_error']:.2f} kW\n\n")
            
            f.write("Deviation Hotspots:\n\n")
            
            f.write("  1. SOC Boundaries:\n")
            soc_stats = deviation_results['soc_regions']
            f.write(f"     - Low boundary (0.1-0.2):  {soc_stats['boundary_low']:.2f} kW\n")
            f.write(f"     - Normal range (0.2-0.8):  {soc_stats['normal']:.2f} kW\n")
            f.write(f"     - High boundary (0.8-0.9): {soc_stats['boundary_high']:.2f} kW\n")
            
            if soc_stats['boundary_low'] > soc_stats['normal'] * 1.2:
                f.write(f"     → Higher errors at low SOC boundary\n")
            if soc_stats['boundary_high'] > soc_stats['normal'] * 1.2:
                f.write(f"     → Higher errors at high SOC boundary\n")
            
            f.write("\n  2. Price Extremes:\n")
            price_stats = deviation_results['price_regions']
            f.write(f"     - Low prices (<10%ile):  {price_stats['low']:.2f} kW\n")
            f.write(f"     - Normal prices:         {price_stats['normal']:.2f} kW\n")
            f.write(f"     - High prices (>90%ile): {price_stats['high']:.2f} kW\n")
            
            if price_stats['low'] > price_stats['normal'] * 1.2:
                f.write(f"     → Higher errors at low prices\n")
            if price_stats['high'] > price_stats['normal'] * 1.2:
                f.write(f"     → Higher errors at high prices\n")
            
            f.write("\n  3. Price Transitions:\n")
            trans_stats = deviation_results['transitions']
            f.write(f"     - Rapid changes (>90%ile): {trans_stats['rapid_changes']:.2f} kW\n")
            f.write(f"     - Normal changes:          {trans_stats['normal']:.2f} kW\n")
            
            if trans_stats['rapid_changes'] > trans_stats['normal'] * 1.2:
                f.write(f"     → Higher errors during rapid price transitions\n")
            
            f.write("\nConclusion:\n")
            f.write("  Student deviates most from teacher in:\n")
            
            # Identify top deviation scenarios
            deviations = [
                ("SOC low boundary", soc_stats['boundary_low']),
                ("SOC high boundary", soc_stats['boundary_high']),
                ("Price extremes (low)", price_stats['low']),
                ("Price extremes (high)", price_stats['high']),
                ("Rapid price changes", trans_stats['rapid_changes']),
            ]
            deviations.sort(key=lambda x: x[1], reverse=True)
            
            for i, (scenario, error) in enumerate(deviations[:3], 1):
                f.write(f"    {i}. {scenario}: {error:.2f} kW MAE\n")
            
            f.write("\n")
            f.write("="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")
        
        print(f"  Saved: {report_path}")
    
    def run_full_analysis(self) -> None:
        """Run complete XAI analysis pipeline.
        
        Generates all plots and report answering the 3 key questions.
        """
        print("\n" + "="*60)
        print("STARTING FULL XAI ANALYSIS")
        print("="*60)
        
        # Compute SHAP values
        self.compute_shap_values()
        
        # Generate visualizations
        self.plot_summary(save=True)
        self.plot_dependence('price_current', save=True)
        self.plot_dependence('soc', save=True)
        self.plot_dependence('price_avg_4h', save=True)
        
        # Generate force plots for key scenarios
        # Find high price + high SOC scenario
        price_idx = FEATURE_COLUMNS.index('price_current')
        soc_idx = FEATURE_COLUMNS.index('soc')
        
        high_price_high_soc = np.argmax(
            self.X_explain[:, price_idx] + self.X_explain[:, soc_idx]
        )
        self.plot_force(high_price_high_soc, scenario_name="high_price_high_soc", save=True)
        
        # Find low price + low SOC scenario
        low_price_low_soc = np.argmin(
            self.X_explain[:, price_idx] + self.X_explain[:, soc_idx]
        )
        self.plot_force(low_price_low_soc, scenario_name="low_price_low_soc", save=True)
        
        # Analyze arbitrage logic
        arbitrage_results = self.analyze_arbitrage_logic()
        
        # Analyze deviations
        deviation_results = self.analyze_deviations()
        
        # Generate report
        self.generate_report(arbitrage_results, deviation_results)
        
        print("\n" + "="*60)
        print("XAI ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    """CLI entry point for XAI analysis."""
    parser = argparse.ArgumentParser(
        description="SHAP-based explainability analysis for VPP student model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/student_model.pth",
        help="Path to trained student model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training_data.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/xai",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=50,
        help="Number of background samples for SHAP",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=5000,
        help="Number of samples to explain",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (cpu or cuda)",
    )
    
    args = parser.parse_args()
    
    # Create explainer and run analysis
    explainer = VPPExplainer(
        model_path=args.model,
        data_path=args.data,
        n_background=args.n_background,
        n_explain=args.n_explain,
        output_dir=args.output,
        device=args.device,
    )
    
    explainer.run_full_analysis()


if __name__ == "__main__":
    main()
