"""
Tests for VPP Student Model.

Tests cover:
1. Model architecture and forward pass
2. Output bounds enforcement
3. Dataset loading
4. Parameter counting
"""

import pytest
import numpy as np
import torch

from vpp.learning.models import VPPStudentModel, VPPDataset, FEATURE_COLUMNS


class TestVPPStudentModel:
    """Tests for the student MLP model."""

    @pytest.fixture
    def model(self):
        """Default model matching spec."""
        return VPPStudentModel(input_dim=7, hidden_dims=[64, 64, 32], max_power=50.0)

    def test_forward_pass_shape(self, model):
        """Test that forward pass produces correct output shape."""
        x = torch.randn(32, 7)
        y = model(x)
        assert y.shape == (32, 1)

    def test_single_sample(self, model):
        """Test forward pass with single sample."""
        x = torch.randn(1, 7)
        y = model(x)
        assert y.shape == (1, 1)

    def test_output_bounds(self, model):
        """Test that output is always within [-P_max, P_max]."""
        # Test with many random inputs including extreme values
        x = torch.randn(10000, 7) * 100  # Large inputs
        y = model(x)
        assert y.min() >= -50.0
        assert y.max() <= 50.0

    def test_output_bounds_extreme(self, model):
        """Test bounds with very extreme inputs."""
        x = torch.full((100, 7), 1e6)
        y = model(x)
        assert y.min() >= -50.0
        assert y.max() <= 50.0

        x = torch.full((100, 7), -1e6)
        y = model(x)
        assert y.min() >= -50.0
        assert y.max() <= 50.0

    def test_custom_max_power(self):
        """Test that max_power parameter is respected."""
        model = VPPStudentModel(max_power=100.0)
        x = torch.randn(1000, 7) * 100
        y = model(x)
        assert y.min() >= -100.0
        assert y.max() <= 100.0

    def test_parameter_count(self, model):
        """Test parameter count matches expected architecture."""
        n_params = model.count_parameters()
        # Input(7) -> 64: 7*64 + 64 = 512
        # 64 -> 64: 64*64 + 64 = 4160
        # 64 -> 32: 64*32 + 32 = 2080
        # 32 -> 1: 32*1 + 1 = 33
        expected = 512 + 4160 + 2080 + 33
        assert n_params == expected, f"Expected {expected}, got {n_params}"

    def test_summary(self, model):
        """Test model summary output."""
        summary = model.summary()
        assert "VPPStudentModel" in summary
        assert "50.0 kW" in summary
        assert "Tanh" in summary

    def test_predict_numpy(self, model):
        """Test numpy convenience method."""
        x = np.random.randn(16, 7).astype(np.float32)
        y = model.predict(x)
        assert isinstance(y, np.ndarray)
        assert y.shape == (16,)
        assert np.all(y >= -50.0)
        assert np.all(y <= 50.0)

    def test_custom_architecture(self):
        """Test model with non-default architecture."""
        model = VPPStudentModel(input_dim=10, hidden_dims=[128, 64], max_power=25.0)
        x = torch.randn(8, 10)
        y = model(x)
        assert y.shape == (8, 1)
        assert y.min() >= -25.0
        assert y.max() <= 25.0

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(8, 7, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestVPPDataset:
    """Tests for the dataset loader."""

    @pytest.fixture
    def dataset(self):
        """Load the actual training data."""
        return VPPDataset("data/training_data.csv", max_power=50.0)

    def test_load_data(self, dataset):
        """Test that data loads correctly."""
        assert len(dataset) == 35040

    def test_item_shapes(self, dataset):
        """Test that individual items have correct shapes."""
        features, target = dataset[0]
        assert features.shape == (7,)
        assert target.shape == (1,)

    def test_feature_types(self, dataset):
        """Test that features are float tensors."""
        features, target = dataset[0]
        assert features.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_input_dim(self, dataset):
        """Test input_dim property."""
        assert dataset.input_dim == 7

    def test_feature_stats(self, dataset):
        """Test feature statistics are reasonable."""
        stats = dataset.get_feature_stats()
        # hour_sin and hour_cos should be in [-1, 1]
        assert stats["min"][0] >= -1.01  # hour_sin
        assert stats["max"][0] <= 1.01
        # soc should be in [0, 1]
        assert stats["min"][5] >= 0.0  # soc
        assert stats["max"][5] <= 1.0

    def test_target_stats(self, dataset):
        """Test target statistics match known data."""
        stats = dataset.get_target_stats()
        assert stats["min"] >= -55.0  # Should be near -50
        assert stats["max"] <= 55.0   # Should be near +50

    def test_normalized_target(self):
        """Test target normalization to [-1, 1]."""
        dataset = VPPDataset(
            "data/training_data.csv",
            max_power=50.0,
            normalize_target=True,
        )
        stats = dataset.get_target_stats()
        assert stats["min"] >= -1.1
        assert stats["max"] <= 1.1

    def test_model_dataset_compatibility(self, dataset):
        """Test that model and dataset work together."""
        model = VPPStudentModel(input_dim=dataset.input_dim, max_power=50.0)
        features, target = dataset[0]
        output = model(features.unsqueeze(0))
        assert output.shape == target.unsqueeze(0).shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
