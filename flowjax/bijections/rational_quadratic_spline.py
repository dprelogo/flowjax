"""Rational quadratic spline bijections (https://arxiv.org/abs/1906.04032)."""

from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from paramax import AbstractUnwrappable, Parameterize
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection


# =============================================================================
# Core RQ Spline Computation Functions (Shared by linear and circular variants)
# =============================================================================


def _rqs_forward_unbounded(
    x: Float[Array, ""],
    x_pos: Float[Array, " n_knots"],
    y_pos: Float[Array, " n_knots"],
    derivatives: Float[Array, " n_knots"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Core RQ spline forward pass for a single scalar.

    Computes y = f(x) and log|df/dx| using rational quadratic spline interpolation.
    This function assumes x is within the valid interval [x_pos[0], x_pos[-1]].

    Args:
        x: Input scalar (assumed in bounds).
        x_pos: Knot x positions (monotonically increasing).
        y_pos: Knot y positions (monotonically increasing).
        derivatives: Derivative values at each knot (all positive).

    Returns:
        Tuple of (y, log_derivative).
    """
    # Find bin index k such that x_pos[k] <= x < x_pos[k+1]
    k = jnp.searchsorted(x_pos, x) - 1
    k = jnp.clip(k, 0, len(x_pos) - 2)  # Safety clip for boundary

    # Normalized position within bin [0, 1]
    xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])

    # Bin slope
    sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])

    # Derivatives at bin boundaries
    dk, dk1 = derivatives[k], derivatives[k + 1]
    yk, yk1 = y_pos[k], y_pos[k + 1]

    # Eq. 4 from Neural Spline Flows paper
    num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
    den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
    y = yk + num / den

    # Eq. 5 - derivative dy/dx
    num_deriv = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    den_deriv = den**2
    log_deriv = jnp.log(num_deriv / den_deriv)

    return y, log_deriv


def _rqs_inverse_unbounded(
    y: Float[Array, ""],
    x_pos: Float[Array, " n_knots"],
    y_pos: Float[Array, " n_knots"],
    derivatives: Float[Array, " n_knots"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Core RQ spline inverse pass for a single scalar.

    Computes x = f^{-1}(y) and log|df/dx| (forward derivative) using the
    quadratic formula for RQ spline inversion.
    This function assumes y is within the valid interval [y_pos[0], y_pos[-1]].

    Args:
        y: Input scalar (assumed in bounds).
        x_pos: Knot x positions (monotonically increasing).
        y_pos: Knot y positions (monotonically increasing).
        derivatives: Derivative values at each knot (all positive).

    Returns:
        Tuple of (x, log_derivative) where log_derivative is log|df/dx|.
    """
    # Find bin index k such that y_pos[k] <= y < y_pos[k+1]
    k = jnp.searchsorted(y_pos, y) - 1
    k = jnp.clip(k, 0, len(y_pos) - 2)  # Safety clip for boundary

    xk, xk1 = x_pos[k], x_pos[k + 1]
    yk, yk1 = y_pos[k], y_pos[k + 1]

    # Bin slope
    sk = (yk1 - yk) / (xk1 - xk)

    # Quadratic formula coefficients
    dk, dk1 = derivatives[k], derivatives[k + 1]
    y_delta_s_term = (y - yk) * (dk1 + dk - 2 * sk)
    a = (yk1 - yk) * (sk - dk) + y_delta_s_term
    b = (yk1 - yk) * dk - y_delta_s_term
    c = -sk * (y - yk)

    # Solve quadratic for normalized position xi
    sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
    xi = (2 * c) / (-b - sqrt_term)

    # Convert back to x
    x = xi * (xk1 - xk) + xk

    # Compute forward derivative at x for log det
    _, log_deriv = _rqs_forward_unbounded(x, x_pos, y_pos, derivatives)

    return x, log_deriv


def _rqs_derivative_unbounded(
    x: Float[Array, ""],
    x_pos: Float[Array, " n_knots"],
    y_pos: Float[Array, " n_knots"],
    derivatives: Float[Array, " n_knots"],
) -> Float[Array, ""]:
    """Compute the derivative dy/dx at point x.

    Args:
        x: Input scalar (assumed in bounds).
        x_pos: Knot x positions (monotonically increasing).
        y_pos: Knot y positions (monotonically increasing).
        derivatives: Derivative values at each knot (all positive).

    Returns:
        The derivative dy/dx at x.
    """
    k = jnp.searchsorted(x_pos, x) - 1
    k = jnp.clip(k, 0, len(x_pos) - 2)

    xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
    sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
    dk, dk1 = derivatives[k], derivatives[k + 1]

    # Eq. 5 from Neural Spline Flows paper
    num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2

    return num / den


# =============================================================================
# Parameter Transformation Functions
# =============================================================================


def _real_to_increasing_on_interval(
    arr: Float[Array, " dim"],
    interval: tuple[int | float, int | float],
    softmax_adjust: float = 1e-2,
    *,
    pad_with_ends: bool = True,
):
    """Transform unconstrained vector to monotonically increasing positions on [-B, B].

    Args:
        arr: Parameter vector.
        interval: Interval to transform output. Defaults to 1.
        softmax_adjust : Rescales softmax output using
            ``(widths + softmax_adjust/widths.size) / (1 + softmax_adjust)``. e.g.
            0=no adjustment, 1=average softmax output with evenly spaced widths, >1
            promotes more evenly spaced widths.
        pad_with_ends: Whether to pad the with -interval and interval. Defaults to True.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")

    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    scale = interval[1] - interval[0]
    pos = interval[0] + scale * jnp.cumsum(widths)

    if pad_with_ends:
        pos = jnp.pad(pos, pad_width=1, constant_values=interval)

    return pos


class RationalQuadraticSpline(AbstractBijection):
    """Scalar RationalQuadraticSpline transformation (https://arxiv.org/abs/1906.04032).

    Args:
        knots: Number of knots.
        interval: Interval to transform, if a scalar value, uses [-interval, interval],
            if a tuple, uses [interval[0], interval[1]]
        min_derivative: Minimum dervivative. Defaults to 1e-3.
        softmax_adjust: Controls minimum bin width and height by rescaling softmax
            output, e.g. 0=no adjustment, 1=average softmax output with evenly spaced
            widths, >1 promotes more evenly spaced widths. See
            ``real_to_increasing_on_interval``. Defaults to 1e-2.
        boundary_derivatives: If set, fixes the boundary derivatives to this value
            instead of learning them. For identity tails (outside the interval),
            this should be 1.0 to ensure C1 continuity. If None, all derivatives
            are learned. Defaults to None for backward compatibility.
    """

    knots: int
    interval: tuple[int | float, int | float]
    softmax_adjust: float | int
    min_derivative: float
    boundary_derivatives: float | None
    x_pos: Array | AbstractUnwrappable[Array]
    y_pos: Array | AbstractUnwrappable[Array]
    derivatives: Array | AbstractUnwrappable[Array]
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        *,
        knots: int,
        interval: float | int | tuple[int | float, int | float],
        min_derivative: float = 1e-3,
        softmax_adjust: float | int = 1e-2,
        boundary_derivatives: float | None = None,
    ):
        self.knots = knots
        interval = interval if isinstance(interval, tuple) else (-interval, interval)
        self.interval = interval
        self.softmax_adjust = softmax_adjust
        self.min_derivative = min_derivative
        self.boundary_derivatives = boundary_derivatives

        to_interval = jnp.vectorize(
            partial(
                _real_to_increasing_on_interval,
                interval=interval,
                softmax_adjust=softmax_adjust,
            ),
            signature="(a)->(b)",
        )

        self.x_pos = Parameterize(to_interval, jnp.zeros(knots))
        self.y_pos = Parameterize(to_interval, jnp.zeros(knots))

        if boundary_derivatives is not None:
            # Fix boundary derivatives, only learn internal derivatives
            self.derivatives = Parameterize(
                lambda arr: jax.nn.softplus(arr) + min_derivative,
                jnp.full(knots, inv_softplus(1 - min_derivative)),
            )
        else:
            # Original behavior: learn all derivatives including boundaries
            self.derivatives = Parameterize(
                lambda arr: jax.nn.softplus(arr) + min_derivative,
                jnp.full(knots + 2, inv_softplus(1 - min_derivative)),
            )

    def _get_derivatives(self) -> Array:
        """Get the full derivatives array, padding with boundary values if needed."""
        if self.boundary_derivatives is not None:
            # Pad learned internal derivatives with fixed boundary values
            pad = jnp.array([self.boundary_derivatives])
            return jnp.concatenate([pad, self.derivatives, pad])
        return self.derivatives

    def transform_and_log_det(self, x, condition=None):
        # Following notation from the paper
        x_pos, y_pos = self.x_pos, self.y_pos
        derivatives = self._get_derivatives()
        in_bounds = jnp.logical_and(x >= self.interval[0], x <= self.interval[1])
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1  # k is bin number
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        y = yk + num / den  # eq. 4

        # avoid numerical precision issues transforming from in -> out of bounds
        y = jnp.clip(y, self.interval[0], self.interval[1])
        y = jnp.where(in_bounds, y, x)

        return y, jnp.log(self.derivative(x)).sum()

    def inverse_and_log_det(self, y, condition=None):
        # Following notation from the paper
        x_pos, y_pos = self.x_pos, self.y_pos
        derivatives = self._get_derivatives()
        in_bounds = jnp.logical_and(y >= self.interval[0], y <= self.interval[1])
        y_robust = jnp.where(in_bounds, y, 0)  # To avoid nans
        k = jnp.searchsorted(y_pos, y_robust) - 1
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y_robust - yk) * (
            derivatives[k + 1] + derivatives[k] - 2 * sk
        )
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y_robust - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk

        # avoid numerical precision issues transforming from in -> out of bounds
        x = jnp.clip(x, self.interval[0], self.interval[1])
        x = jnp.where(in_bounds, x, y)

        return x, -jnp.log(self.derivative(x)).sum()

    def derivative(self, x) -> Array:
        """The derivative dy/dx of the forward transformation."""
        # Following notation from the paper (eq. 5)
        x_pos, y_pos = self.x_pos, self.y_pos
        derivatives = self._get_derivatives()
        in_bounds = jnp.logical_and(x >= self.interval[0], x <= self.interval[1])
        x_robust = jnp.where(in_bounds, x, 0)  # To avoid nans
        k = jnp.searchsorted(x_pos, x_robust) - 1
        xi = (x_robust - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1 = derivatives[k], derivatives[k + 1]
        num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        derivative = num / den
        return jnp.where(in_bounds, derivative, 1.0)
