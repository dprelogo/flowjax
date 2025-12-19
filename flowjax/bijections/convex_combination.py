"""Convex combination of bijections for increased expressivity.

Based on https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres),
which suggests alternating between functional composition and convex combinations
for flows on circular domains.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.utils import NumericalInverse
from flowjax.root_finding import bisection_search, root_finder_to_inverter
from flowjax.utils import merge_cond_shapes


class ConvexCombination(AbstractBijection):
    """Convex combination of bijections: f(x) = sum_i rho_i * f_i(x).

    Creates a new bijection as the weighted sum of multiple bijections, where
    the weights are non-negative and sum to 1 (parameterized via softmax).

    For 1D monotonic bijections (like splines with positive derivatives), the
    convex combination preserves monotonicity. This is useful for building
    expressive flows on circular domains.

    The derivative is: df/dx = sum_i rho_i * df_i/dx
    The log determinant uses logsumexp for numerical stability.

    Note: The inverse requires numerical methods since convex combinations of
    invertible functions don't have closed-form inverses in general.

    Args:
        bijections: Tuple of bijections to combine. All must have the same
            shape and cond_shape.

    Example:
        >>> from flowjax.bijections import CircularRationalQuadraticSpline
        >>> splines = tuple(CircularRationalQuadraticSpline(num_bins=8) for _ in range(3))
        >>> convex = ConvexCombination(splines)
    """

    bijections: tuple[AbstractBijection, ...]
    weight_logits: Array
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def __init__(self, bijections: tuple[AbstractBijection, ...]):
        if not bijections:
            raise ValueError("At least one bijection required.")

        # Validate all bijections have compatible shapes
        shapes = [b.shape for b in bijections]
        cond_shapes = [b.cond_shape for b in bijections]

        if len(set(shapes)) > 1:
            raise ValueError(
                f"All bijections must have the same shape. Got: {shapes}"
            )

        self.bijections = bijections
        self.shape = bijections[0].shape
        self.cond_shape = merge_cond_shapes(cond_shapes)

        # Initialize weights uniformly (logits = 0 -> equal weights after softmax)
        self.weight_logits = jnp.zeros(len(bijections))

    @property
    def weights(self) -> Array:
        """The normalized weights (sum to 1)."""
        return jax.nn.softmax(self.weight_logits)

    def transform_and_log_det(self, x, condition=None):
        """Transform x via convex combination and compute log determinant."""
        weights = self.weights

        # Compute transforms and log derivatives for all bijections
        ys = []
        log_dets = []
        for bij in self.bijections:
            y_i, log_det_i = bij.transform_and_log_det(x, condition)
            ys.append(y_i)
            log_dets.append(log_det_i)

        ys = jnp.stack(ys)
        log_dets = jnp.stack(log_dets)

        # Convex combination of outputs: y = sum_i rho_i * y_i
        if x.ndim > 0:
            y = jnp.sum(weights[:, None] * ys, axis=0)
        else:
            y = jnp.dot(weights, ys)

        # Log determinant: log(sum_i rho_i * exp(log_det_i))
        # Use logsumexp for numerical stability
        log_det = jax.scipy.special.logsumexp(log_dets + jnp.log(weights))

        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        """Inverse transform using numerical root finding."""
        # Use bisection search for numerical inverse
        # For circular domains, search in [0, 2pi] with some margin

        if self.shape == ():
            # Case 1: Pure scalar - use standard helper
            lower = y - 4 * jnp.pi  # Wide search range
            upper = y + 4 * jnp.pi

            inverter = root_finder_to_inverter(
                partial(bisection_search, lower=lower, upper=upper, atol=1e-6)
            )
            numerical_inv = NumericalInverse(self, inverter)
            return numerical_inv.inverse_and_log_det(y, condition)

        elif self.shape == (1,):
            # Case 2: Shape (1,) - Adapter Pattern
            # bisection_search requires scalar bounds, so we create a custom inverter
            # that handles the reshape logic internally.

            def inverter_1d(bijection, y_vec, condition):
                # 1. Adapt vector y to scalar for bisection bounds
                y_scalar = y_vec.reshape(())
                lower = y_scalar - 4 * jnp.pi
                upper = y_scalar + 4 * jnp.pi

                # 2. Define objective that adapts scalar x -> vector x for bijection
                def objective_fn(x_scalar):
                    # Reshape scalar -> vector to satisfy bijection.shape constraint
                    x_vec = x_scalar.reshape(bijection.shape)
                    y_pred, _ = bijection.transform_and_log_det(x_vec, condition)
                    # Reshape vector -> scalar for bisection
                    return y_pred.reshape(()) - y_scalar

                # 3. Run scalar search
                root_scalar, _ = bisection_search(
                    objective_fn, lower, upper, atol=1e-6
                )

                # 4. Return vector result
                return root_scalar.reshape(bijection.shape)

            numerical_inv = NumericalInverse(self, inverter_1d)
            return numerical_inv.inverse_and_log_det(y, condition)

        else:
            # Case 3: General vector - not yet implemented
            # For proper N-dimensional support, would need vmap over dimensions
            raise NotImplementedError(
                "ConvexCombination inverse currently only supports shape=() or "
                f"shape=(1,). Got shape={self.shape}."
            )
