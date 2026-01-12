"""Tests for CircularScale bijection."""

import jax.numpy as jnp
import pytest

from flowjax.bijections import CircularScale


class TestCircularScale:
    """Tests for CircularScale bijection."""

    def test_basic_forward_transform(self):
        """Test forward transform scales circular dims to [0, 2pi]."""
        is_circular = jnp.array([True, False, True, False])
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.5, 0.5, 0.5, 0.5])
        y = scale.transform(x)

        # Circular dims (0, 2) should be scaled: 0.5 * 2pi = pi
        assert jnp.allclose(y[0], jnp.pi)
        assert jnp.allclose(y[2], jnp.pi)

        # Linear dims (1, 3) should NOT be scaled: 0.5 stays 0.5
        assert jnp.allclose(y[1], 0.5)
        assert jnp.allclose(y[3], 0.5)

    def test_linear_dims_not_scaled(self):
        """CRITICAL: Ensure linear dimensions are identity-mapped (not scaled by 2pi).

        If the mask logic in CircularScale is inverted, linear dims get scaled,
        destroying density estimation. This test catches that bug.
        """
        is_circular = jnp.array([True, False, True, False])
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.5, 0.5, 0.5, 0.5])
        y = scale.transform(x)

        # Linear dims should be unchanged
        assert jnp.isclose(y[1], 0.5)
        assert jnp.isclose(y[3], 0.5)

        # Make sure they're NOT scaled by 2pi
        assert not jnp.isclose(y[1], 0.5 * 2 * jnp.pi)
        assert not jnp.isclose(y[3], 0.5 * 2 * jnp.pi)

    def test_forward_inverse_consistency(self):
        """Test that inverse undoes forward transform."""
        is_circular = jnp.array([True, False, True, False])
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.25, 0.75, 0.9, 0.1])
        y = scale.transform(x)
        x_rec = scale.inverse(y)

        assert jnp.allclose(x, x_rec, atol=1e-6)

    def test_transform_and_log_det_forward(self):
        """Test forward transform with log determinant."""
        is_circular = jnp.array([True, True, False])  # 2 circular, 1 linear
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.5, 0.5, 0.5])
        y, log_det = scale.transform_and_log_det(x)

        # Log det should be n_circular * log(2pi)
        expected_log_det = 2 * jnp.log(2 * jnp.pi)
        assert jnp.allclose(log_det, expected_log_det, atol=1e-6)

    def test_transform_and_log_det_inverse(self):
        """Test inverse transform with log determinant."""
        is_circular = jnp.array([True, True, False])  # 2 circular, 1 linear
        scale = CircularScale(is_circular, forward_to_2pi=True)

        y = jnp.array([jnp.pi, jnp.pi, 0.5])
        x, log_det = scale.inverse_and_log_det(y)

        # Log det should be -n_circular * log(2pi)
        expected_log_det = -2 * jnp.log(2 * jnp.pi)
        assert jnp.allclose(log_det, expected_log_det, atol=1e-6)

    def test_log_det_consistency(self):
        """Test that forward and inverse log dets are negatives of each other."""
        is_circular = jnp.array([True, False, True, True, False])
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.3, 0.7, 0.5, 0.2, 0.8])
        y, log_det_forward = scale.transform_and_log_det(x)
        x_rec, log_det_inverse = scale.inverse_and_log_det(y)

        assert jnp.allclose(log_det_forward, -log_det_inverse, atol=1e-6)
        assert jnp.allclose(x, x_rec, atol=1e-6)

    def test_forward_to_2pi_false(self):
        """Test with forward_to_2pi=False (reverse direction)."""
        is_circular = jnp.array([True, False])
        scale = CircularScale(is_circular, forward_to_2pi=False)

        # Forward now maps [0, 2pi] -> [0, 1]
        x = jnp.array([jnp.pi, 0.5])  # pi should become 0.5
        y = scale.transform(x)

        assert jnp.allclose(y[0], 0.5)  # pi / (2pi) = 0.5
        assert jnp.allclose(y[1], 0.5)  # Linear unchanged

    def test_shape_property(self):
        """Test that shape property is correctly set."""
        is_circular = jnp.array([True, False, True])
        scale = CircularScale(is_circular, forward_to_2pi=True)

        assert scale.shape == (3,)

    def test_all_circular(self):
        """Test with all dimensions circular."""
        is_circular = jnp.ones(4, dtype=bool)
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.25, 0.5, 0.75, 1.0])
        y = scale.transform(x)

        expected = x * 2 * jnp.pi
        assert jnp.allclose(y, expected)

    def test_all_linear(self):
        """Test with all dimensions linear (should be identity)."""
        is_circular = jnp.zeros(4, dtype=bool)
        scale = CircularScale(is_circular, forward_to_2pi=True)

        x = jnp.array([0.25, 0.5, 0.75, 1.0])
        y, log_det = scale.transform_and_log_det(x)

        # Should be identity transform
        assert jnp.allclose(y, x)
        assert jnp.allclose(log_det, 0.0)

    def test_boundary_values(self):
        """Test transform at boundary values."""
        is_circular = jnp.array([True, False])
        scale = CircularScale(is_circular, forward_to_2pi=True)

        # At boundaries
        x = jnp.array([0.0, 0.0])
        y = scale.transform(x)
        assert jnp.allclose(y[0], 0.0)
        assert jnp.allclose(y[1], 0.0)

        x = jnp.array([1.0, 1.0])
        y = scale.transform(x)
        assert jnp.allclose(y[0], 2 * jnp.pi)
        assert jnp.allclose(y[1], 1.0)
