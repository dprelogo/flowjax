"""Tests for circular/toroidal bijections."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import grad

from flowjax.bijections import (
    CircularCoupling,
    CircularMaskedAutoregressive,
    CircularRationalQuadraticSpline,
    ConvexCombination,
    Affine,
)


class TestCircularRationalQuadraticSpline:
    """Tests for CircularRationalQuadraticSpline."""

    def test_basic_transform(self):
        """Test basic forward/inverse transform."""
        spline = CircularRationalQuadraticSpline(num_bins=8)
        x = jnp.array(jnp.pi)

        y, log_det = spline.transform_and_log_det(x)
        x_rec, neg_log_det = spline.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_periodicity(self):
        """Test that f(θ + 2π) = f(θ) + 2π (circle diffeomorphism property)."""
        spline = CircularRationalQuadraticSpline(num_bins=8)
        x = jnp.array(1.5)
        two_pi = 2 * jnp.pi

        y1, _ = spline.transform_and_log_det(x)
        y2, _ = spline.transform_and_log_det(x + two_pi)

        assert jnp.allclose(y2, y1 + two_pi, atol=1e-5)

    def test_derivative_continuity(self):
        """Test derivative continuity at boundaries (d_0 = d_K)."""
        spline = CircularRationalQuadraticSpline(num_bins=8)

        def f(x):
            y, _ = spline.transform_and_log_det(x)
            return y

        # Derivative at 0 and 2π should be equal
        eps = 1e-4
        grad_at_0 = grad(f)(jnp.array(eps))
        grad_at_2pi = grad(f)(jnp.array(2 * jnp.pi - eps))

        assert jnp.allclose(grad_at_0, grad_at_2pi, atol=1e-3)

    def test_monotonicity(self):
        """Test that the spline is monotonically increasing."""
        spline = CircularRationalQuadraticSpline(num_bins=8)
        xs = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 20)

        ys = []
        for x in xs:
            y, log_det = spline.transform_and_log_det(x)
            ys.append(y)
            # Positive log det implies positive derivative
            assert log_det > -10  # Not degenerate

        ys = jnp.array(ys)
        # Should be increasing
        assert jnp.all(jnp.diff(ys) > 0)

    def test_winding_numbers(self):
        """Test correct handling of winding numbers."""
        spline = CircularRationalQuadraticSpline(num_bins=8)
        two_pi = 2 * jnp.pi

        # Test multiple winding numbers
        for k in [-2, -1, 0, 1, 2]:
            x = jnp.array(jnp.pi + k * two_pi)
            y, log_det = spline.transform_and_log_det(x)
            x_rec, _ = spline.inverse_and_log_det(y)

            assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_gradient_flow_at_boundary(self):
        """Test that gradients flow correctly through the boundary x=0/2π.

        This ensures there are no numerical issues or discontinuities
        in the gradient computation at the periodic boundaries.
        """
        spline = CircularRationalQuadraticSpline(num_bins=8)
        two_pi = 2 * jnp.pi

        def loss_fn(x):
            """Simple loss function to test gradient flow."""
            y, log_det = spline.transform_and_log_det(x)
            return y + log_det

        # Test gradients at various points including near boundaries
        test_points = [
            jnp.array(0.01),      # Near 0
            jnp.array(jnp.pi),    # Middle
            jnp.array(two_pi - 0.01),  # Near 2π
            jnp.array(two_pi + 0.01),  # Just past 2π (wraps to near 0)
            jnp.array(-0.01),     # Just before 0 (wraps to near 2π)
        ]

        for x in test_points:
            g = grad(loss_fn)(x)
            # Gradient should be finite and not NaN
            assert jnp.isfinite(g), f"Non-finite gradient at x={x}: {g}"
            # Gradient should not be zero (spline should be non-trivial)
            assert jnp.abs(g) > 1e-10, f"Zero gradient at x={x}"


class TestConvexCombination:
    """Tests for ConvexCombination bijection."""

    def test_basic_transform(self):
        """Test basic forward/inverse transform."""
        bij1 = Affine(loc=jnp.array(0.5), scale=jnp.array(1.2))
        bij2 = Affine(loc=jnp.array(-0.3), scale=jnp.array(0.8))
        convex = ConvexCombination(bijections=(bij1, bij2))

        x = jnp.array(1.0)
        y, log_det = convex.transform_and_log_det(x)
        x_rec, neg_log_det = convex.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_weights_sum_to_one(self):
        """Test that weights are valid probabilities."""
        bij1 = Affine(loc=jnp.array(0.0), scale=jnp.array(1.0))
        bij2 = Affine(loc=jnp.array(0.0), scale=jnp.array(1.0))
        convex = ConvexCombination(bijections=(bij1, bij2))

        weights = convex.weights
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0)

    def test_single_bijection_identity(self):
        """Single bijection should behave like the original."""
        bij = Affine(loc=jnp.array(2.0), scale=jnp.array(3.0))
        convex = ConvexCombination(bijections=(bij,))

        x = jnp.array(1.5)
        y_convex, log_det_convex = convex.transform_and_log_det(x)
        y_orig, log_det_orig = bij.transform_and_log_det(x)

        assert jnp.allclose(y_convex, y_orig)
        assert jnp.allclose(log_det_convex, log_det_orig)

    def test_shape_one_tuple_invertibility(self):
        """Test that ConvexCombination is fully invertible with shape=(1,) bijections."""
        # Create bijections with shape (1,) instead of ()
        bij1 = Affine(loc=jnp.array([0.5]), scale=jnp.array([1.2]))
        bij2 = Affine(loc=jnp.array([-0.3]), scale=jnp.array([0.8]))

        assert bij1.shape == (1,)
        assert bij2.shape == (1,)

        convex = ConvexCombination(bijections=(bij1, bij2))
        assert convex.shape == (1,)

        x = jnp.array([1.0])
        y, log_det = convex.transform_and_log_det(x)

        # Forward transform should work
        assert y.shape == (1,)
        assert jnp.isfinite(y).all()
        assert jnp.isfinite(log_det)

        # Inverse should now work for shape=(1,) using the adapter pattern
        x_rec, neg_log_det = convex.inverse_and_log_det(y)

        assert x_rec.shape == (1,)
        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)


class TestCircularCoupling:
    """Tests for CircularCoupling bijection."""

    def test_basic_transform(self):
        """Test basic forward/inverse transform."""
        key = jr.key(0)
        coupling = CircularCoupling(
            key=key,
            transformer=CircularRationalQuadraticSpline(num_bins=4),
            untransformed_dim=2,
            dim=4,
            nn_width=8,
            nn_depth=1,
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y, log_det = coupling.transform_and_log_det(x)
        x_rec, neg_log_det = coupling.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_untransformed_dims_unchanged(self):
        """Test that untransformed dimensions are unchanged."""
        key = jr.key(0)
        coupling = CircularCoupling(
            key=key,
            transformer=CircularRationalQuadraticSpline(num_bins=4),
            untransformed_dim=2,
            dim=4,
            nn_width=8,
            nn_depth=1,
        )

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y, _ = coupling.transform_and_log_det(x)

        # First 2 dimensions should be unchanged
        assert jnp.allclose(x[:2], y[:2])

    def test_conditional(self):
        """Test conditional circular coupling."""
        key = jr.key(0)
        coupling = CircularCoupling(
            key=key,
            transformer=CircularRationalQuadraticSpline(num_bins=4),
            untransformed_dim=1,
            dim=2,
            cond_dim=3,
            nn_width=8,
            nn_depth=1,
        )

        x = jnp.array([1.0, 2.0])
        condition = jnp.array([0.5, 0.5, 0.5])

        y, log_det = coupling.transform_and_log_det(x, condition)
        x_rec, neg_log_det = coupling.inverse_and_log_det(y, condition)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)


class TestCircularMaskedAutoregressive:
    """Tests for CircularMaskedAutoregressive bijection."""

    def test_basic_transform(self):
        """Test basic forward/inverse transform."""
        key = jr.key(0)
        maf = CircularMaskedAutoregressive(
            key=key,
            transformer=CircularRationalQuadraticSpline(num_bins=4),
            dim=3,
            nn_width=8,
            nn_depth=1,
        )

        x = jnp.array([1.0, 2.0, 3.0])
        y, log_det = maf.transform_and_log_det(x)
        x_rec, neg_log_det = maf.inverse_and_log_det(y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_conditional(self):
        """Test conditional circular MAF."""
        key = jr.key(0)
        maf = CircularMaskedAutoregressive(
            key=key,
            transformer=CircularRationalQuadraticSpline(num_bins=4),
            dim=2,
            cond_dim=3,
            nn_width=8,
            nn_depth=1,
        )

        x = jnp.array([1.0, 2.0])
        condition = jnp.array([0.5, 0.5, 0.5])

        y, log_det = maf.transform_and_log_det(x, condition)
        x_rec, neg_log_det = maf.inverse_and_log_det(y, condition)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(log_det, -neg_log_det, atol=1e-5)

    def test_autoregressive_structure(self):
        """Test that the transformation is autoregressive."""
        key = jr.key(0)
        maf = CircularMaskedAutoregressive(
            key=key,
            transformer=CircularRationalQuadraticSpline(num_bins=4),
            dim=3,
            nn_width=8,
            nn_depth=1,
        )

        x1 = jnp.array([1.0, 2.0, 3.0])
        x2 = jnp.array([1.0, 2.0, 5.0])  # Only last dim different

        y1, _ = maf.transform_and_log_det(x1)
        y2, _ = maf.transform_and_log_det(x2)

        # First two outputs should be the same (autoregressive)
        assert jnp.allclose(y1[:2], y2[:2])
        # Last output should differ
        assert not jnp.allclose(y1[2], y2[2])
