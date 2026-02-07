"""
Comparative Framework for Teacher-Student Model Evaluation.

This module implements comprehensive comparison between the MILP teacher model
and the Neural Network student model across multiple dimensions:
1. Regression accuracy (R² score)
2. Physical constraint adherence (SOC violations)
3. Economic efficiency (profit comparison, optimality gap)

Usage:
    python -m vpp.analysis.compare --test-data data/training_data.csv --output output/comparison
"""

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..learning.models import VPPStudentModel, FEATURE_COLUMNS
from ..experiments.generate_data import NonlinearBatterySimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResults:
    """Results from teacher-student comparison."""
    
    # Regression metrics
    r2_score: float
    mae: float
    rmse: float
    
    # Physical integrity
    soc_violation_rate: float
    soc_violations_count: int
    
    # Economic efficiency
    teacher_profit: float
    student_profit: float
    optimality_gap: float
    
    # Trajectories for visualization
    teacher_dispatch: np.ndarray
    student_dispatch: np.ndarray
    student_dispatch_actual: np.ndarray  # After physics simulation
    soc_trajectory: np.ndarray
    prices: np.ndarray
    timestamps: np.ndarray
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'regression_metrics': {
                'r2_score': float(self.r2_score),
                'mae': float(self.mae),
                'rmse': float(self.rmse),
            },
            'physical_integrity': {
                'soc_violation_rate': float(self.soc_violation_rate),
                'soc_violations_count': int(self.soc_violations_count),
            },
            'economic_efficiency': {
                'teacher_profit': float(self.teacher_profit),
                'student_profit': float(self.student_profit),
                'optimality_gap': float(self.optimality_gap),
            },
        }


class ComparativeAnalyzer:
    """Comparative analysis between teacher and student models."""
    
    def __init__(
        self,
        student_model_path: str,
        scaler_path: Optional[str] = None,
        battery_config: Optional[dict] = None,
        time_step_hours: float = 0.25
    ):
        """
        Initialize comparative analyzer.
        
        Args:
            student_model_path: Path to trained student model (.pth)
            scaler_path: Path to fitted scaler (.pkl). If None, looks for scaler.pkl next to model.
            battery_config: Battery parameters (if None, uses defaults from data generation)
            time_step_hours: Time step duration in hours
        """
        # Load student model
        self.student = VPPStudentModel(input_dim=7, max_power=50.0)
        state_dict = torch.load(student_model_path, map_location='cpu')
        self.student.load_state_dict(state_dict)
        self.student.eval()
        
        # Load scaler
        if scaler_path is None:
            # Default: look for scaler.pkl in same directory as model
            model_dir = Path(student_model_path).parent
            scaler_path = model_dir / 'scaler.pkl'
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Battery configuration (match data generation defaults)
        if battery_config is None:
            battery_config = {
                'energy_capacity': 100.0,  # kWh
                'max_power': 50.0,  # kW
                'base_charge_efficiency': 0.92,
                'base_discharge_efficiency': 0.92,
                'soc_min': 0.1,
                'soc_max': 0.9,
            }
        
        self.battery_config = battery_config
        self.time_step_hours = time_step_hours
        
        logger.info(f"Loaded student model from {student_model_path}")
        logger.info(f"Battery config: {battery_config}")
    
    def simulate_student_dispatch(
        self,
        features: np.ndarray,
        initial_soc: float,
        prices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate student dispatch with physics model.
        
        This accounts for:
        - SOC limits (hard bounds at 0.1 and 0.9)
        - Soft power limits near boundaries
        - C-rate dependent efficiency
        - Self-discharge
        
        Args:
            features: Feature array (n_samples, n_features)
            initial_soc: Starting SOC
            prices: Price series for profit calculation
            
        Returns:
            Tuple of:
            - student_dispatch_raw: Student's raw predictions (kW)
            - student_dispatch_actual: Actually executed power (kW)
            - soc_trajectory: SOC over time (n_samples + 1,)
        """
        n_samples = len(features)
        
        # Initialize battery simulator
        battery = NonlinearBatterySimulator(
            energy_capacity=self.battery_config['energy_capacity'],
            max_power=self.battery_config['max_power'],
            base_charge_efficiency=self.battery_config['base_charge_efficiency'],
            base_discharge_efficiency=self.battery_config['base_discharge_efficiency'],
            soc_min=self.battery_config['soc_min'],
            soc_max=self.battery_config['soc_max'],
            initial_soc=initial_soc,
        )
        
        # Storage
        student_dispatch_raw = np.zeros(n_samples)
        student_dispatch_actual = np.zeros(n_samples)
        soc_trajectory = np.zeros(n_samples + 1)
        soc_trajectory[0] = initial_soc
        
        # Predict and simulate
        # Scale features 
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            student_dispatch_raw = self.student(features_tensor).squeeze().numpy()
        
        # Simulate forward with physics
        for t in range(n_samples):
            # Apply student's command to physics model
            actual_power = battery.update(
                power_setpoint=student_dispatch_raw[t],
                dt_hours=self.time_step_hours
            )
            
            student_dispatch_actual[t] = actual_power
            soc_trajectory[t + 1] = battery.soc
        
        return student_dispatch_raw, student_dispatch_actual, soc_trajectory
    
    def compute_soc_violations(self, soc_trajectory: np.ndarray) -> Tuple[float, int]:
        """
        Compute SOC violation rate.
        
        Args:
            soc_trajectory: SOC values over time
            
        Returns:
            Tuple of (violation_rate, violation_count)
        """
        violations = (soc_trajectory < 0.0) | (soc_trajectory > 1.0)
        count = violations.sum()
        rate = count / len(soc_trajectory)
        
        return rate, count
    
    def compute_profit(
        self,
        dispatch: np.ndarray,
        prices: np.ndarray,
    ) -> float:
        """
        Compute profit from dispatch strategy.
        
        Profit = Σ(P_t · π_t · Δt)
        
        Positive power (charging) = buying from grid = cost (negative profit)
        Negative power (discharging) = selling to grid = revenue (positive profit)
        
        Args:
            dispatch: Power dispatch (kW), positive=charge, negative=discharge
            prices: Electricity prices ($/kWh)
            
        Returns:
            Total profit ($)
        """
        # Revenue from selling - cost of buying
        # dispatch is positive for charging (cost), negative for discharging (revenue)
        profit = -np.sum(dispatch * prices * self.time_step_hours)
        
        return profit
    
    def compare(
        self,
        test_data_path: str,
        train_split: float = 0.8,
        random_seed: int = 42,
    ) -> ComparisonResults:
        """
        Run comprehensive comparison on test data.
        
        Args:
            test_data_path: Path to training_data.csv
            train_split: Fraction used for training (remainder is test set)
            random_seed: Random seed for reproducibility
            
        Returns:
            ComparisonResults with all metrics
        """
        logger.info(f"Loading data from {test_data_path}")
        data = pd.read_csv(test_data_path)
        
        # Split data (use same split as training)
        X = data[FEATURE_COLUMNS].values
        y_teacher = data['power_setpoint'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_teacher,
            train_size=train_split,
            random_state=random_seed,
            shuffle=True
        )
        
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Get prices and initial SOC from test set
        test_data = data.iloc[len(X_train):]  # Approximate test indices
        prices_test = test_data['price_current'].values
        initial_soc = test_data['soc'].iloc[0]
        
        logger.info(f"Initial SOC: {initial_soc:.3f}")
        
        # === 1. REGRESSION METRICS ===
        logger.info("Computing regression metrics...")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_student_pred = self.student(X_test_tensor).squeeze().numpy()
        
        r2 = r2_score(y_test, y_student_pred)
        mae = mean_absolute_error(y_test, y_student_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_student_pred))
        
        logger.info(f"  R² = {r2:.4f}")
        logger.info(f"  MAE = {mae:.2f} kW")
        logger.info(f"  RMSE = {rmse:.2f} kW")
        
        # === 2. SIMULATE STUDENT WITH PHYSICS ===
        logger.info("Simulating student dispatch with physics model...")
        
        student_raw, student_actual, soc_traj = self.simulate_student_dispatch(
            features=X_test,
            initial_soc=initial_soc,
            prices=prices_test
        )
        
        # === 3. SOC VIOLATIONS ===
        logger.info("Checking SOC violations...")
        
        viol_rate, viol_count = self.compute_soc_violations(soc_traj)
        
        logger.info(f"  Violation rate: {viol_rate*100:.2f}%")
        logger.info(f"  Violations: {viol_count}/{len(soc_traj)}")
        
        # === 4. PROFIT COMPARISON ===
        logger.info("Computing profits...")
        
        teacher_profit = self.compute_profit(y_test, prices_test)
        student_profit = self.compute_profit(student_actual, prices_test)
        
        optimality_gap = 1 - (student_profit / teacher_profit) if teacher_profit != 0 else 0.0
        
        logger.info(f"  Teacher profit: ${teacher_profit:.2f}")
        logger.info(f"  Student profit: ${student_profit:.2f}")
        logger.info(f"  Optimality gap: {optimality_gap*100:.2f}%")
        
        # === 5. CREATE RESULTS ===
        results = ComparisonResults(
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            soc_violation_rate=viol_rate,
            soc_violations_count=viol_count,
            teacher_profit=teacher_profit,
            student_profit=student_profit,
            optimality_gap=optimality_gap,
            teacher_dispatch=y_test,
            student_dispatch=y_student_pred,
            student_dispatch_actual=student_actual,
            soc_trajectory=soc_traj,
            prices=prices_test,
            timestamps=np.arange(len(y_test)),
        )
        
        return results


def visualize_comparison(
    results: ComparisonResults,
    output_path: str,
    show_window: int = 500,  # Show first N timesteps for clarity
):
    """
    Create dual plot visualization comparing teacher and student.
    
    Args:
        results: ComparisonResults object
        output_path: Path to save figure
        show_window: Number of timesteps to display
    """
    # Limit to window for visibility
    n = min(show_window, len(results.timestamps))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # === TOP PANEL: PRICE PROFILE ===
    ax1.plot(results.timestamps[:n], results.prices[:n], 
             color='green', linewidth=1.5, label='Electricity Price')
    ax1.set_ylabel('Price ($/kWh)', fontsize=12)
    ax1.set_title('Teacher vs Student Dispatch Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # === BOTTOM PANEL: DISPATCH OVERLAY ===
    ax2.plot(results.timestamps[:n], results.teacher_dispatch[:n],
             color='blue', linewidth=2, alpha=0.7, label='Teacher (Optimal)')
    ax2.plot(results.timestamps[:n], results.student_dispatch_actual[:n],
             color='red', linewidth=2, alpha=0.7, label='Student (Actual)')
    
    # Highlight deviation regions (where |error| > 20 kW)
    errors = np.abs(results.teacher_dispatch - results.student_dispatch_actual)
    large_errors = errors > 20
    
    if large_errors.any():
        deviation_indices = np.where(large_errors[:n])[0]
        if len(deviation_indices) > 0:
            ax2.scatter(results.timestamps[deviation_indices],
                       results.student_dispatch_actual[deviation_indices],
                       color='orange', s=30, alpha=0.6,
                       label=f'Large Deviations (>20kW)', zorder=5)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Power (kW)', fontsize=12)
    ax2.set_ylim(-60, 60)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add metrics text box
    metrics_text = (
        f"R² = {results.r2_score:.4f}\n"
        f"MAE = {results.mae:.2f} kW\n"
        f"Optimality Gap = {results.optimality_gap*100:.2f}%\n"
        f"SOC Violations = {results.soc_violation_rate*100:.2f}%"
    )
    ax2.text(0.02, 0.98, metrics_text,
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def generate_report(results: ComparisonResults, output_path: str):
    """Generate text report summarizing comparison."""
    
    report = []
    report.append("=" * 70)
    report.append("TEACHER-STUDENT COMPARATIVE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("="* 70)
    report.append("1. REGRESSION METRICS")
    report.append("=" * 70)
    report.append(f"  R² Score:       {results.r2_score:.4f}")
    report.append(f"  MAE:            {results.mae:.2f} kW")
    report.append(f"  RMSE:           {results.rmse:.2f} kW")
    report.append("")
    
    if results.r2_score > 0.8:
        report.append("  ✓ EXCELLENT: Student closely matches teacher predictions")
    elif results.r2_score > 0.6:
        report.append("  ✓ GOOD: Student captures most of teacher's behavior")
    else:
        report.append("  ⚠ POOR: Student struggles to replicate teacher")
    report.append("")
    
    report.append("=" * 70)
    report.append("2. PHYSICAL INTEGRITY")
    report.append("=" * 70)
    report.append(f"  SOC Violation Rate: {results.soc_violation_rate*100:.2f}%")
    report.append(f"  Violations Count:   {results.soc_violations_count}")
    report.append("")
    
    if results.soc_violation_rate < 0.01:
        report.append("  ✓ EXCELLENT: Student respects SOC constraints")
    elif results.soc_violation_rate < 0.05:
        report.append("  ✓ ACCEPTABLE: Minor SOC violations")
    else:
        report.append("  ✗ POOR: Significant SOC constraint violations")
    report.append("")
    
    report.append("=" * 70)
    report.append("3. ECONOMIC EFFICIENCY")
    report.append("=" * 70)
    report.append(f"  Teacher Profit:     ${results.teacher_profit:.2f}")
    report.append(f"  Student Profit:     ${results.student_profit:.2f}")
    report.append(f"  Optimality Gap:     {results.optimality_gap*100:.2f}%")
    report.append("")
    
    if results.optimality_gap < 0.10:
        report.append("  ✓ EXCELLENT: Student achieves near-optimal profit")
    elif results.optimality_gap < 0.25:
        report.append("  ✓ GOOD: Student is reasonably efficient")
    else:
        report.append("  ⚠ SUBOPTIMAL: Significant profit loss vs teacher")
    report.append("")
    
    report.append("=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    report.append(f"The student model achieves R² = {results.r2_score:.4f} on test data,")
    report.append(f"with an MAE of {results.mae:.2f} kW. Economic performance shows")
    report.append(f"{results.optimality_gap*100:.1f}% optimality gap, meaning the student")
    report.append(f"captures {(1-results.optimality_gap)*100:.1f}% of teacher's profit potential.")
    report.append(f"SOC violations are {results.soc_violation_rate*100:.2f}%, indicating")
    report.append("physical constraints are well-learned." if results.soc_violation_rate < 0.01 
                  else "some constraint violations occur.")
    report.append("")
    report.append("=" * 70)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Saved report to {output_path}")


def compare_models(
    test_data_path: str = 'data/training_data.csv',
    student_model_path: str = 'artifacts/student_model.pth',
    output_dir: str = 'output/comparison',
    train_split: float = 0.8,
    random_seed: int = 42,
) -> ComparisonResults:
    """
    Run full comparative analysis.
    
    Args:
        test_data_path: Path to training data CSV
        student_model_path: Path to trained student model
        output_dir: Directory to save outputs
        train_split: Training fraction (remainder is test)
        random_seed: Random seed
        
    Returns:
        ComparisonResults object
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ComparativeAnalyzer(
        student_model_path=student_model_path,
        time_step_hours=0.25  # 15-minute intervals
    )
    
    # Run comparison
    results = analyzer.compare(
        test_data_path=test_data_path,
        train_split=train_split,
        random_seed=random_seed
    )
    
    # Save metrics
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate visualization
    viz_path = output_path / 'teacher_vs_student.png'
    visualize_comparison(results, str(viz_path))
    
    # Generate report
    report_path = output_path / 'report.txt'
    generate_report(results, str(report_path))
    
    logger.info("Comparative analysis complete!")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare teacher and student models')
    parser.add_argument('--test-data', type=str, default='data/training_data.csv',
                       help='Path to test data')
    parser.add_argument('--model', type=str, default='artifacts/student_model.pth',
                       help='Path to student model')
    parser.add_argument('--output', type=str, default='output/comparison',
                       help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training split fraction')
    
    args = parser.parse_args()
    
    results = compare_models(
        test_data_path=args.test_data,
        student_model_path=args.model,
        output_dir=args.output,
        train_split=args.train_split
    )
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"R²:              {results.r2_score:.4f}")
    print(f"MAE:             {results.mae:.2f} kW")
    print(f"Optimality Gap:  {results.optimality_gap*100:.2f}%")
    print(f"SOC Violations:  {results.soc_violation_rate*100:.2f}%")
    print("="*70)
