"""Circular scaling bijection for unit hypercube support.

This module provides a bijection that scales circular dimensions between
[0, 1] and [0, 2π] while leaving linear dimensions unchanged.
"""

from typing import ClassVar

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from flowjax.bijections.bijection import AbstractBijection


class CircularScale(AbstractBijection):
    """Scale circular dimensions between [0, 1] and [0, 2π].

    This bijection applies scaling only to dimensions marked as circular,
    leaving linear dimensions unchanged. It enables normalizing flows to
    work natively with unit hypercube data [0, 1]^D while using the natural
    [0, 2π] domain internally for circular embeddings.

    The forward direction (when ``forward_to_2pi=True``) maps:
    - Circular dimensions: [0, 1] → [0, 2π]
    - Linear dimensions: unchanged

    Args:
        is_circular: Boolean array of shape (dim,) indicating which dimensions
            are circular (True) vs linear (False).
        forward_to_2pi: If True (default), forward maps [0, 1] → [0, 2π].
            If False, forward maps [0, 2π] → [0, 1].

    Example:
        >>> import jax.numpy as jnp
        >>> is_circular = jnp.array([True, False, True])  # dims 0, 2 are circular
        >>> scale = CircularScale(is_circular, forward_to_2pi=True)
        >>> x = jnp.array([0.5, 0.5, 0.5])
        >>> y, log_det = scale.transform_and_log_det(x)
        >>> # y ≈ [π, 0.5, π]  (circular dims scaled, linear unchanged)
    """

    is_circular: Array
    forward_to_2pi: bool
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    # Pre-computed constants for efficiency
    _n_circular: int
    _log_2pi: float

    def __init__(
        self,
        is_circular: ArrayLike,
        forward_to_2pi: bool = True,
    ):
        self.is_circular = jnp.asarray(is_circular, dtype=bool)
        self.forward_to_2pi = forward_to_2pi
        self.shape = (len(self.is_circular),)

        # Pre-compute for log-det calculation
        self._n_circular = int(jnp.sum(self.is_circular))
        self._log_2pi = float(jnp.log(2 * jnp.pi))

    def transform(self, x: Array, condition: Array | None = None) -> Array:
        """Apply forward transformation."""
        if self.forward_to_2pi:
            # [0, 1] → [0, 2π] for circular dims
            scale = jnp.where(self.is_circular, 2 * jnp.pi, 1.0)
        else:
            # [0, 2π] → [0, 1] for circular dims
            scale = jnp.where(self.is_circular, 1.0 / (2 * jnp.pi), 1.0)
        return x * scale

    def transform_and_log_det(
        self, x: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        """Apply forward transformation and compute log determinant.

        The Jacobian is diagonal with entries either 2π (circular) or 1 (linear).
        log|det(J)| = n_circular * log(2π) for forward_to_2pi=True.
        """
        y = self.transform(x, condition)
        if self.forward_to_2pi:
            log_det = self._n_circular * self._log_2pi
        else:
            log_det = -self._n_circular * self._log_2pi
        return y, jnp.array(log_det)

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        """Apply inverse transformation."""
        if self.forward_to_2pi:
            # Inverse: [0, 2π] → [0, 1] for circular dims
            scale = jnp.where(self.is_circular, 1.0 / (2 * jnp.pi), 1.0)
        else:
            # Inverse: [0, 1] → [0, 2π] for circular dims
            scale = jnp.where(self.is_circular, 2 * jnp.pi, 1.0)
        return y * scale

    def inverse_and_log_det(
        self, y: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        """Apply inverse transformation and compute log determinant."""
        x = self.inverse(y, condition)
        if self.forward_to_2pi:
            log_det = -self._n_circular * self._log_2pi
        else:
            log_det = self._n_circular * self._log_2pi
        return x, jnp.array(log_det)
