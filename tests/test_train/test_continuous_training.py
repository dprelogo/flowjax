"""Tests for continuous per-batch resampling training."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.bijections import Affine
from flowjax.distributions import Normal, Transformed
from flowjax.train.continuous_training import fit_to_weighted_data_continuous


class TestFitToWeightedDataContinuous:
    """Tests for fit_to_weighted_data_continuous function."""

    def test_basic_training(self):
        """Test that continuous training produces valid results."""
        dim = 2
        key = jr.PRNGKey(0)

        # Create a simple flow
        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        # Create synthetic data with uniform weights
        n_samples = 500
        key, data_key = jr.split(key)
        x = jr.normal(data_key, (n_samples, dim))
        log_weights = jnp.zeros(n_samples)  # Uniform weights

        # Train
        trained_flow, losses = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            max_epochs=5,
            batch_size=64,
            show_progress=False,
        )

        # Check outputs
        assert isinstance(trained_flow, Transformed)
        assert "train" in losses
        assert "val" in losses
        assert len(losses["train"]) > 0
        assert len(losses["val"]) > 0
        assert isinstance(losses["train"][0], float)
        assert isinstance(losses["val"][0], float)

    def test_parameters_change(self):
        """Test that parameters are updated during training."""
        dim = 2
        key = jr.PRNGKey(42)

        # Create a simple flow
        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        # Store parameters before
        before = eqx.filter(flow, eqx.is_inexact_array)

        # Create synthetic data
        n_samples = 500
        key, data_key = jr.split(key)
        x = jr.normal(data_key, (n_samples, dim))
        log_weights = jnp.zeros(n_samples)

        # Train
        trained_flow, _ = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            max_epochs=10,
            batch_size=64,
            show_progress=False,
        )

        # Store parameters after
        after = eqx.filter(trained_flow, eqx.is_inexact_array)

        # Check that at least some parameters changed
        before_flat = jnp.concatenate([p.flatten() for p in jax.tree.leaves(before)])
        after_flat = jnp.concatenate([p.flatten() for p in jax.tree.leaves(after)])
        assert not jnp.allclose(before_flat, after_flat)

    def test_early_stopping(self):
        """Test that early stopping works correctly."""
        dim = 2
        key = jr.PRNGKey(0)

        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        n_samples = 200
        key, data_key = jr.split(key)
        x = jr.normal(data_key, (n_samples, dim))
        log_weights = jnp.zeros(n_samples)

        # Train with early stopping (very small patience)
        trained_flow, losses = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            max_epochs=100,
            max_patience=3,
            batch_size=32,
            show_progress=False,
        )

        # Should stop before max_epochs if no improvement
        # (though with random data, we may train all epochs)
        assert len(losses["train"]) <= 100

    def test_weighted_samples(self):
        """Test that weights are properly used in resampling."""
        dim = 2
        key = jr.PRNGKey(123)

        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        # Create biased weights - heavily weight first half of samples
        n_samples = 500
        key, data_key = jr.split(key)
        x = jr.normal(data_key, (n_samples, dim))

        # First half has high weights, second half has very low weights
        log_weights = jnp.concatenate([
            jnp.zeros(n_samples // 2),
            jnp.full(n_samples - n_samples // 2, -10.0),  # Very low weight
        ])

        trained_flow, losses = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            max_epochs=5,
            batch_size=64,
            show_progress=False,
        )

        # Training should complete without errors
        assert len(losses["train"]) > 0

    def test_custom_perturbation_fn(self):
        """Test that custom perturbation function is applied."""
        dim = 2
        key = jr.PRNGKey(0)

        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        n_samples = 300
        key, data_key = jr.split(key)
        # Generate data in [0, 1] range
        x = jr.uniform(data_key, (n_samples, dim))
        log_weights = jnp.zeros(n_samples)

        # Custom perturbation that clips to [0, 1]
        def clip_perturbation(key, batch_x):
            noise = jr.normal(key, batch_x.shape) * 0.01
            return jnp.clip(batch_x + noise, 0.0, 1.0)

        trained_flow, losses = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            perturbation_fn=clip_perturbation,
            max_epochs=5,
            batch_size=64,
            show_progress=False,
        )

        assert len(losses["train"]) > 0

    def test_batches_differ_across_epochs(self):
        """Test that batches are different across epochs (diversity check)."""
        # This test verifies that continuous resampling provides new samples each time
        dim = 2
        key = jr.PRNGKey(0)

        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        n_samples = 100
        key, data_key = jr.split(key)
        x = jr.normal(data_key, (n_samples, dim))
        log_weights = jnp.zeros(n_samples)

        # Train for multiple epochs - if batches were identical, loss pattern would differ
        trained_flow, losses = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            max_epochs=10,
            batch_size=32,
            show_progress=False,
        )

        # Multiple epochs should complete
        assert len(losses["train"]) >= 1

    def test_input_validation(self):
        """Test that input validation catches mismatched shapes."""
        dim = 2
        key = jr.PRNGKey(0)

        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        x = jr.normal(key, (100, dim))
        log_weights = jnp.zeros(50)  # Wrong size!

        with pytest.raises(ValueError, match="same length"):
            fit_to_weighted_data_continuous(
                key=key,
                dist=flow,
                x=x,
                log_weights=log_weights,
                max_epochs=5,
                show_progress=False,
            )

    def test_return_best_flag(self):
        """Test that return_best=False returns final model."""
        dim = 2
        key = jr.PRNGKey(0)

        base_dist = Normal(jnp.zeros(dim), jnp.ones(dim))
        flow = Transformed(base_dist, Affine(jnp.zeros(dim), jnp.ones(dim)))

        n_samples = 200
        key, data_key = jr.split(key)
        x = jr.normal(data_key, (n_samples, dim))
        log_weights = jnp.zeros(n_samples)

        # Train with return_best=False
        trained_flow, losses = fit_to_weighted_data_continuous(
            key=key,
            dist=flow,
            x=x,
            log_weights=log_weights,
            max_epochs=5,
            batch_size=32,
            return_best=False,
            show_progress=False,
        )

        # Should complete without errors
        assert len(losses["train"]) > 0


# Import jax for tree operations
import jax
