"""Circular rational quadratic spline bijection for flows on tori.

This implements a circle diffeomorphism using rational quadratic splines,
based on https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres).
"""

from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from paramax import AbstractUnwrappable, Parameterize
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.rational_quadratic_spline import (
    _rqs_forward_unbounded,
    _rqs_inverse_unbounded,
)


def _real_to_increasing_on_interval_circular(
    arr: Float[Array, " num_bins"],
    interval: tuple[float, float],
    softmax_adjust: float = 1e-2,
) -> Float[Array, " num_knots"]:
    """Transform unconstrained vector to monotonic positions for circular splines.

    Unlike the linear version, this includes both endpoints (0 and 2π) in the output
    since circular splines need explicit boundary knots.

    Args:
        arr: Parameter vector of length num_bins.
        interval: The interval (typically (0, 2π)).
        softmax_adjust: Controls minimum bin width via softmax rescaling.

    Returns:
        Array of length num_bins + 1 with positions [interval[0], ..., interval[1]].
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")

    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)

    # Cumulative sum gives positions, scaled to interval
    scale = interval[1] - interval[0]
    positions = interval[0] + scale * jnp.cumsum(widths)

    # Prepend the start of interval
    positions = jnp.concatenate([jnp.array([interval[0]]), positions])

    return positions


def _params_to_circular_derivatives(arr, min_derivative):
    """Convert unconstrained params to positive derivatives with d_0 = d_K."""
    derivs = jax.nn.softplus(arr) + min_derivative
    # Append first derivative to end for periodicity: d_K = d_0
    return jnp.concatenate([derivs, derivs[:1]])


class CircularRationalQuadraticSpline(AbstractBijection):
    """Circular rational quadratic spline implementing a circle diffeomorphism.

    This bijection maps R → R with the property f(θ + 2π) = f(θ) + 2π, making it
    suitable for normalizing flows on circular domains. It satisfies the boundary
    conditions required for circle diffeomorphisms:

    1. f(0) = 0 (by construction)
    2. f(2π) = 2π (by construction)
    3. f'(θ) > 0 for all θ (monotonicity via positive derivatives)
    4. f'(0) = f'(2π) (derivative continuity at wrap-around, via shared d_0 = d_K)

    The implementation uses winding number decomposition to handle inputs outside
    [0, 2π] correctly, ensuring continuous gradients for autodiff.

    Refs:
        - https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres)
        - https://arxiv.org/abs/1906.04032 (Neural Spline Flows)

    Args:
        num_bins: Number of spline bins. More bins = more flexibility. Defaults to 8.
        min_derivative: Minimum derivative value for numerical stability. Defaults to 1e-3.
        softmax_adjust: Controls minimum bin width/height via softmax rescaling.
            0 = no adjustment, higher values promote more uniform widths. Defaults to 1e-2.
    """

    num_bins: int
    min_derivative: float
    softmax_adjust: float

    # Learnable parameters (unconstrained)
    width_params: Array | AbstractUnwrappable[Array]
    height_params: Array | AbstractUnwrappable[Array]
    derivative_params: Array | AbstractUnwrappable[Array]

    shape: ClassVar[tuple[int, ...]] = ()
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        *,
        num_bins: int = 8,
        min_derivative: float = 1e-3,
        softmax_adjust: float = 1e-2,
    ):
        self.num_bins = num_bins
        self.min_derivative = min_derivative
        self.softmax_adjust = softmax_adjust

        interval = (0.0, 2 * jnp.pi)

        # Use vectorized functions for proper vmap support (same pattern as RationalQuadraticSpline)
        to_interval = jnp.vectorize(
            partial(
                _real_to_increasing_on_interval_circular,
                interval=interval,
                softmax_adjust=softmax_adjust,
            ),
            signature="(a)->(b)",
        )

        # x positions via parameterization
        self.width_params = Parameterize(to_interval, jnp.zeros(num_bins))

        # y positions via parameterization
        self.height_params = Parameterize(to_interval, jnp.zeros(num_bins))

        # Derivatives with d_0 = d_K (periodic boundary condition)
        # Use vectorized function for proper vmap support
        to_derivatives = jnp.vectorize(
            partial(_params_to_circular_derivatives, min_derivative=min_derivative),
            signature="(a)->(b)",
        )

        self.derivative_params = Parameterize(
            to_derivatives,
            jnp.full(num_bins, inv_softplus(1 - min_derivative)),
        )

    def transform_and_log_det(self, x, condition=None):
        """Transform x and compute log determinant.

        Uses winding number decomposition: x = 2πk + r where r ∈ [0, 2π).
        This ensures f(x + 2π) = f(x) + 2π and maintains continuous gradients.
        """
        x_pos = self.width_params
        y_pos = self.height_params
        derivatives = self.derivative_params

        # Winding number decomposition
        two_pi = 2 * jnp.pi
        winding = jnp.floor(x / two_pi)
        remainder = x - winding * two_pi

        # Handle edge case: remainder exactly at 2π (map to 0 for computation)
        # This can happen due to floating point issues
        at_boundary = remainder >= two_pi
        remainder_safe = jnp.where(at_boundary, 0.0, remainder)

        # Apply spline transform to remainder
        y_remainder, log_det = _rqs_forward_unbounded(
            remainder_safe, x_pos, y_pos, derivatives
        )

        # Handle boundary: if at 2π, output should be 2π
        y_remainder = jnp.where(at_boundary, two_pi, y_remainder)

        # Recombine with winding number
        y = winding * two_pi + y_remainder

        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        """Inverse transform and compute log determinant of forward transform."""
        x_pos = self.width_params
        y_pos = self.height_params
        derivatives = self.derivative_params

        # Winding number decomposition
        two_pi = 2 * jnp.pi
        winding = jnp.floor(y / two_pi)
        remainder = y - winding * two_pi

        # Handle edge case at boundary
        at_boundary = remainder >= two_pi
        remainder_safe = jnp.where(at_boundary, 0.0, remainder)

        # Apply inverse spline to remainder
        x_remainder, log_det_fwd = _rqs_inverse_unbounded(
            remainder_safe, x_pos, y_pos, derivatives
        )

        # Handle boundary
        x_remainder = jnp.where(at_boundary, two_pi, x_remainder)

        # Recombine with winding number
        x = winding * two_pi + x_remainder

        return x, -log_det_fwd
