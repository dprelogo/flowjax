"""Tests for circular/toroidal flow architectures."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from flowjax.distributions import TorusUniform, MixedUniformBase, Uniform
from flowjax.flows import (
    circular_coupling_flow,
    circular_masked_autoregressive_flow,
    mixed_masked_autoregressive_flow,
)


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


class TestUnitHypercubeMode:
    """Tests for unit_hypercube mode in flows.

    These tests verify that flows with unit_hypercube=True produce samples
    in [0, 1]^D instead of the default [0, 2π]^D domain.
    """

    def test_circular_maf_unit_hypercube_samples_in_01(self):
        """Test that circular MAF with unit_hypercube produces samples in [0, 1].

        Note: When unit_hypercube=True, must use Uniform[0, 1] base distribution,
        not TorusUniform (which is in [0, 2pi]).
        """
        key = jr.key(0)
        dim = 4
        # Use Uniform[0, 1] base distribution for unit_hypercube mode
        base_dist = Uniform(jnp.zeros(dim), jnp.ones(dim))

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            unit_hypercube=True,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Sample from flow
        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (100,))

        assert samples.shape == (100, dim)
        # ALL samples should be in [0, 1]
        assert jnp.all(samples >= 0.0), f"Min value: {jnp.min(samples)}"
        assert jnp.all(samples <= 1.0), f"Max value: {jnp.max(samples)}"

    def test_circular_maf_unit_hypercube_log_prob_finite(self):
        """Test that log_prob is finite for unit_hypercube mode."""
        key = jr.key(0)
        dim = 3
        # Use Uniform[0, 1] for unit_hypercube mode
        base_dist = Uniform(jnp.zeros(dim), jnp.ones(dim))

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            unit_hypercube=True,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Points in [0, 1]
        x = jnp.array([0.25, 0.5, 0.75])
        log_p = flow.log_prob(x)
        assert jnp.isfinite(log_p)

    def test_circular_maf_unit_hypercube_invertibility(self):
        """Test that unit_hypercube flow is invertible."""
        key = jr.key(0)
        dim = 3
        # Use Uniform[0, 1] for unit_hypercube mode
        base_dist = Uniform(jnp.zeros(dim), jnp.ones(dim))

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            unit_hypercube=True,
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        x = jnp.array([0.25, 0.5, 0.75])
        y, log_det = flow.bijection.transform_and_log_det(x)
        x_rec, neg_log_det = flow.bijection.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_mixed_maf_unit_hypercube_samples_in_01(self):
        """CRITICAL: Test mixed MAF with unit_hypercube produces samples in [0, 1].

        This is the most important test - verifies that both circular AND linear
        dimensions stay in [0, 1] when using unit_hypercube=True.
        """
        key = jr.key(0)
        is_circular = jnp.array([True, True, False, False])  # 2 circular, 2 linear
        base_dist = MixedUniformBase(is_circular)

        flow = mixed_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            is_circular=is_circular,
            unit_hypercube=True,
            linear_bounds=(0.0, 1.0),  # CRITICAL: must match [0,1] domain
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Sample from flow
        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (100,))

        assert samples.shape == (100, 4)

        # ALL dimensions should be in [0, 1]
        assert jnp.all(samples >= 0.0), f"Min values: {jnp.min(samples, axis=0)}"
        assert jnp.all(samples <= 1.0), f"Max values: {jnp.max(samples, axis=0)}"

        # Log prob should be finite
        log_probs = flow.log_prob(samples)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_mixed_maf_unit_hypercube_linear_dims_not_scaled(self):
        """CRITICAL: Verify linear dimensions are NOT scaled by 2π.

        If the mask logic is inverted, linear dims would be scaled to [0, 2π],
        which would destroy density estimation for bounded parameters.
        """
        key = jr.key(0)
        is_circular = jnp.array([True, False, True, False])  # Alternating
        base_dist = MixedUniformBase(is_circular)

        flow = mixed_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            is_circular=is_circular,
            unit_hypercube=True,
            linear_bounds=(0.0, 1.0),
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Sample from flow
        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (1000,))

        # Linear dimensions (1, 3) should definitely be in [0, 1]
        linear_samples = samples[:, [1, 3]]
        assert jnp.all(linear_samples >= 0.0)
        assert jnp.all(linear_samples <= 1.0)

        # If they were scaled by 2π, max would be ~6.28
        # So this should pass easily
        assert jnp.max(linear_samples) < 2.0, (
            f"Linear dims appear to be scaled! Max: {jnp.max(linear_samples)}"
        )

    def test_mixed_maf_unit_hypercube_invertibility(self):
        """Test that mixed MAF with unit_hypercube is invertible."""
        key = jr.key(0)
        is_circular = jnp.array([True, False, True])
        base_dist = MixedUniformBase(is_circular)

        flow = mixed_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            is_circular=is_circular,
            unit_hypercube=True,
            linear_bounds=(0.0, 1.0),
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        x = jnp.array([0.3, 0.5, 0.7])
        y, log_det = flow.bijection.transform_and_log_det(x)
        x_rec, neg_log_det = flow.bijection.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_domain_consistency_throughout_flow(self):
        """Test that domain is consistent throughout flow with unit_hypercube.

        This verifies that:
        1. Base distribution produces [0, 1] samples
        2. After flow transform, samples are still in [0, 1]
        3. log_prob evaluates correctly for [0, 1] inputs
        """
        key = jr.key(0)
        is_circular = jnp.array([True, True, False, False])
        base_dist = MixedUniformBase(is_circular)

        flow = mixed_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            is_circular=is_circular,
            unit_hypercube=True,
            linear_bounds=(0.0, 1.0),
            flow_layers=3,
            nn_width=16,
            nn_depth=1,
        )

        # Sample from base distribution
        sample_key = jr.key(1)
        base_samples = base_dist.sample(sample_key, (100,))
        assert jnp.all(base_samples >= 0.0) and jnp.all(base_samples <= 1.0)

        # Transform through flow
        flow_samples = flow.sample(sample_key, (100,))
        assert jnp.all(flow_samples >= 0.0), f"Min: {jnp.min(flow_samples)}"
        assert jnp.all(flow_samples <= 1.0), f"Max: {jnp.max(flow_samples)}"

        # Log prob should be finite for valid [0, 1] inputs
        test_points = jnp.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.5, 0.5],
            [0.9, 0.8, 0.7, 0.6],
        ])
        log_probs = flow.log_prob(test_points)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_without_unit_hypercube_samples_in_2pi(self):
        """Verify that without unit_hypercube, samples are in [0, 2π] for circular."""
        key = jr.key(0)
        dim = 3
        base_dist = TorusUniform(dim)

        flow = circular_masked_autoregressive_flow(
            key,
            base_dist=base_dist,
            unit_hypercube=False,  # Default behavior
            flow_layers=2,
            nn_width=16,
            nn_depth=1,
        )

        # Sample from flow
        sample_key = jr.key(1)
        samples = flow.sample(sample_key, (100,))

        # Samples should be in [0, 2π], NOT [0, 1]
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 2 * jnp.pi)
        # At least some samples should be > 1 (in the 2π domain)
        assert jnp.max(samples) > 1.0
