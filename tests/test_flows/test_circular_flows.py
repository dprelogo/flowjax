"""Tests for circular/toroidal flow architectures."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.distributions import TorusUniform
from flowjax.flows import circular_coupling_flow, circular_masked_autoregressive_flow


class TestCircularCouplingFlow:
    """Tests for circular_coupling_flow."""

    def test_basic_flow(self):
        """Test basic flow creation and operations."""
        key = jr.key(0)
        dim = 4
        base_dist = TorusUniform(dim)

        flow = circular_coupling_flow(
            key,
            base_dist=base_dist,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Test sampling
        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (5,))
        assert samples.shape == (5, dim)

        # Test log_prob
        x = jnp.ones(dim) * jnp.pi
        log_p = flow.log_prob(x)
        assert jnp.isfinite(log_p)

    def test_invertibility(self):
        """Test that the flow is invertible."""
        key = jr.key(0)
        dim = 4
        base_dist = TorusUniform(dim)

        flow = circular_coupling_flow(
            key,
            base_dist=base_dist,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        x = jnp.ones(dim) * jnp.pi
        y, log_det = flow.bijection.transform_and_log_det(x)
        x_rec, neg_log_det = flow.bijection.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_conditional(self):
        """Test conditional circular coupling flow."""
        key = jr.key(0)
        dim = 4
        cond_dim = 2
        base_dist = TorusUniform(dim)

        flow = circular_coupling_flow(
            key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        x = jnp.ones(dim) * jnp.pi
        condition = jnp.array([0.5, 0.5])

        log_p = flow.log_prob(x, condition)
        assert jnp.isfinite(log_p)

        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (3,), condition)
        assert samples.shape == (3, dim)


class TestCircularMaskedAutoregressiveFlow:
    """Tests for circular_masked_autoregressive_flow."""

    def test_basic_flow(self):
        """Test basic flow creation and operations."""
        key = jr.key(0)
        dim = 3
        base_dist = TorusUniform(dim)

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Test sampling
        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (5,))
        assert samples.shape == (5, dim)

        # Test log_prob
        x = jnp.ones(dim) * jnp.pi
        log_p = flow.log_prob(x)
        assert jnp.isfinite(log_p)

    def test_invertibility(self):
        """Test that the flow is invertible."""
        key = jr.key(0)
        dim = 3
        base_dist = TorusUniform(dim)

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        x = jnp.ones(dim) * jnp.pi
        y, log_det = flow.bijection.transform_and_log_det(x)
        x_rec, neg_log_det = flow.bijection.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_conditional(self):
        """Test conditional circular MAF."""
        key = jr.key(0)
        dim = 3
        cond_dim = 2
        base_dist = TorusUniform(dim)

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            cond_dim=cond_dim,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        x = jnp.ones(dim) * jnp.pi
        condition = jnp.array([0.5, 0.5])

        log_p = flow.log_prob(x, condition)
        assert jnp.isfinite(log_p)

        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (3,), condition)
        assert samples.shape == (3, dim)


class TestTorusUniform:
    """Tests for TorusUniform distribution."""

    def test_basic(self):
        """Test basic TorusUniform properties."""
        dim = 4
        dist = TorusUniform(dim)

        assert dist.shape == (dim,)
        assert dist.cond_shape is None

        # Sample
        key = jr.key(0)
        samples = dist.sample(key, (10,))
        assert samples.shape == (10, dim)

        # Samples should be in [0, 2π]
        two_pi = 2 * jnp.pi
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= two_pi)

    def test_log_prob(self):
        """Test log probability is uniform."""
        dim = 3
        dist = TorusUniform(dim)
        two_pi = 2 * jnp.pi

        # Log prob should be -dim * log(2π) everywhere on the torus
        x1 = jnp.ones(dim) * jnp.pi
        x2 = jnp.ones(dim) * jnp.pi / 2

        log_p1 = dist.log_prob(x1)
        log_p2 = dist.log_prob(x2)

        expected = -dim * jnp.log(two_pi)
        assert jnp.allclose(log_p1, expected)
        assert jnp.allclose(log_p2, expected)
