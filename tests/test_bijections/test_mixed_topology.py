"""Tests for mixed topology bijections."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import grad

from flowjax.bijections import (
    MixedMaskedAutoregressive,
    MixedTransformer,
    RationalQuadraticSpline,
    CircularRationalQuadraticSpline,
)
from flowjax.distributions import MixedUniform
from flowjax.flows import mixed_masked_autoregressive_flow


class TestMixedTransformer:
    """Tests for MixedTransformer."""

    def test_basic_transform_simple(self):
        """Test basic transform with simple setup using MixedMaskedAutoregressive factory."""
        # Test the MixedTransformer indirectly through MixedMaskedAutoregressive
        # which properly creates the vectorized transformers
        key = jr.key(42)

        # R^1 × T^1 case
        is_circular = jnp.array([False, True])

        def linear_factory(params):
            return RationalQuadraticSpline(knots=4, interval=(-2.0, 2.0))

        def circular_factory(params):
            return CircularRationalQuadraticSpline(num_bins=4)

        # Create a MixedMaskedAutoregressive bijection (which creates MixedTransformer internally)
        bijection = MixedMaskedAutoregressive(
            key=key,
            is_circular=is_circular,
            dim=2,
            linear_transformer_factory=linear_factory,
            circular_transformer_factory=circular_factory,
            nn_width=16,
        )

        # Test that the bijection works (which tests MixedTransformer)
        x = jnp.array([0.5, jnp.pi])
        y, log_det = bijection.transform_and_log_det(x)

        # Basic checks that transformation produces finite results
        assert jnp.isfinite(y).all()
        assert jnp.isfinite(log_det)

    def test_only_linear_dimensions(self):
        """Test with only linear dimensions using MixedMaskedAutoregressive."""
        key = jr.key(42)

        # Only linear dimensions
        is_circular = jnp.array([False, False])

        def linear_factory(params):
            return RationalQuadraticSpline(knots=4, interval=(-2.0, 2.0))

        def circular_factory(params):
            return CircularRationalQuadraticSpline(num_bins=4)

        bijection = MixedMaskedAutoregressive(
            key=key,
            is_circular=is_circular,
            dim=2,
            linear_transformer_factory=linear_factory,
            circular_transformer_factory=circular_factory,
            nn_width=16,
        )

        x = jnp.array([0.5, -1.2])
        y, log_det = bijection.transform_and_log_det(x)

        assert jnp.isfinite(y).all()
        assert jnp.isfinite(log_det)

    def test_only_circular_dimensions(self):
        """Test with only circular dimensions using MixedMaskedAutoregressive."""
        key = jr.key(42)

        # Only circular dimensions
        is_circular = jnp.array([True, True])

        def linear_factory(params):
            return RationalQuadraticSpline(knots=4, interval=(-2.0, 2.0))

        def circular_factory(params):
            return CircularRationalQuadraticSpline(num_bins=4)

        bijection = MixedMaskedAutoregressive(
            key=key,
            is_circular=is_circular,
            dim=2,
            linear_transformer_factory=linear_factory,
            circular_transformer_factory=circular_factory,
            nn_width=16,
        )

        x = jnp.array([jnp.pi, 0.5])
        y, log_det = bijection.transform_and_log_det(x)

        assert jnp.isfinite(y).all()
        assert jnp.isfinite(log_det)


class TestMixedMaskedAutoregressive:
    """Tests for MixedMaskedAutoregressive."""

    def test_basic_functionality(self):
        """Test basic masked autoregressive flow functionality."""
        key = jr.key(42)

        # R^2 × T^1 case
        is_circular = jnp.array([False, False, True])

        def linear_factory(params):
            return RationalQuadraticSpline(knots=4, interval=(-2.0, 2.0))

        def circular_factory(params):
            return CircularRationalQuadraticSpline(num_bins=4)

        bijection = MixedMaskedAutoregressive(
            key=key,
            is_circular=is_circular,
            dim=3,
            linear_transformer_factory=linear_factory,
            circular_transformer_factory=circular_factory,
            nn_width=16,
            nn_depth=1,
        )

        # Test basic invertibility
        x = jnp.array([1.0, -0.5, jnp.pi])
        y, log_det = bijection.transform_and_log_det(x)
        x_rec, neg_log_det = bijection.inverse_and_log_det(y)

        # Should be close (autoregressive approximation)
        assert jnp.allclose(x, x_rec, atol=1e-3)

    def test_embedding_dimensions(self):
        """Test that embedding works correctly."""
        key = jr.key(42)

        # Test with R^1 × T^1
        is_circular = jnp.array([False, True])

        def linear_factory(params):
            return RationalQuadraticSpline(knots=4, interval=(-2.0, 2.0))

        def circular_factory(params):
            return CircularRationalQuadraticSpline(num_bins=4)

        bijection = MixedMaskedAutoregressive(
            key=key,
            is_circular=is_circular,
            dim=2,
            linear_transformer_factory=linear_factory,
            circular_transformer_factory=circular_factory,
            nn_width=16,
        )

        # Test embedding function
        x = jnp.array([1.0, jnp.pi])
        embedded = bijection._embed_mixed(x)

        # Should be [r, cos(θ), sin(θ)]
        expected = jnp.array([1.0, jnp.cos(jnp.pi), jnp.sin(jnp.pi)])
        assert jnp.allclose(embedded, expected)


class TestMixedUniform:
    """Tests for MixedUniform distribution."""

    def test_basic_creation(self):
        """Test basic mixed uniform creation."""
        is_circular = jnp.array([False, True, False])
        dist = MixedUniform(is_circular, linear_bounds=(-2.0, 2.0))

        assert dist.shape == (3,)

        expected_min = jnp.array([-2.0, 0.0, -2.0])
        expected_max = jnp.array([2.0, 2*jnp.pi, 2.0])

        assert jnp.allclose(dist.minval, expected_min)
        assert jnp.allclose(dist.maxval, expected_max)

    def test_sampling(self):
        """Test that sampling works and produces values in correct ranges."""
        key = jr.key(123)
        is_circular = jnp.array([False, True])
        dist = MixedUniform(is_circular, linear_bounds=(-1.0, 1.0))

        samples = dist.sample(key, sample_shape=(100,))

        # Check shapes
        assert samples.shape == (100, 2)

        # Check ranges
        linear_samples = samples[:, 0]
        circular_samples = samples[:, 1]

        assert jnp.all(linear_samples >= -1.0)
        assert jnp.all(linear_samples <= 1.0)
        assert jnp.all(circular_samples >= 0.0)
        assert jnp.all(circular_samples <= 2*jnp.pi)


class TestMixedFlow:
    """Tests for mixed_masked_autoregressive_flow."""

    def test_basic_flow_creation(self):
        """Test that flow can be created and used."""
        key = jr.key(0)

        # Create R^2 × T^1 flow
        is_circular = jnp.array([False, False, True])
        base_dist = MixedUniform(is_circular, linear_bounds=(-2.0, 2.0))

        flow = mixed_masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            is_circular=is_circular,
            flow_layers=2,
            nn_width=16,
        )

        # Test basic functionality
        sample_key = jr.key(1)
        x = flow.sample(sample_key)
        log_p = flow.log_prob(x)

        assert x.shape == (3,)
        assert jnp.isfinite(log_p)

    def test_flow_invertibility(self):
        """Test basic flow invertibility."""
        key = jr.key(42)

        # Simple R^1 × T^1 case
        is_circular = jnp.array([False, True])
        base_dist = MixedUniform(is_circular, linear_bounds=(-1.0, 1.0))

        flow = mixed_masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            is_circular=is_circular,
            flow_layers=1,
            nn_width=8,
        )

        # Test sample -> log_prob consistency
        sample_key = jr.key(123)
        x, log_p_sample = flow.sample_and_log_prob(sample_key)
        log_p_direct = flow.log_prob(x)

        assert jnp.allclose(log_p_sample, log_p_direct, atol=1e-5)

    @pytest.mark.parametrize("is_circular", [
        jnp.array([True, False]),
        jnp.array([False, True]),
        jnp.array([True, True]),
        jnp.array([False, False]),
    ])
    def test_different_topologies(self, is_circular):
        """Test flow works with different topology combinations."""
        key = jr.key(0)

        base_dist = MixedUniform(is_circular, linear_bounds=(-1.0, 1.0))

        flow = mixed_masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            is_circular=is_circular,
            flow_layers=1,
            nn_width=8,
        )

        # Basic smoke test
        sample_key = jr.key(1)
        x = flow.sample(sample_key)
        log_p = flow.log_prob(x)

        assert x.shape == (len(is_circular),)
        assert jnp.isfinite(log_p)