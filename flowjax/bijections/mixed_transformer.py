"""Mixed topology transformer for handling both linear and circular transformations."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from flowjax.bijections.bijection import AbstractBijection


class MixedTransformer(AbstractBijection):
    """Composite transformer that handles heterogeneous topology types separately.

    This bijection applies different transformers to different dimensions based on
    their topology type (linear vs circular). It uses scatter/gather logic to
    route inputs to the correct transformer type.

    Args:
        is_circular: Boolean array indicating which dimensions are circular.
        linear_transformer: Vectorized transformer for linear dimensions.
        circular_transformer: Vectorized transformer for circular dimensions.

    Note:
        This addresses the "heterogeneous transformer problem" where you cannot
        vmap over different PyTree structures (RationalQuadraticSpline vs
        CircularRationalQuadraticSpline) in JAX.
    """

    is_circular: Array
    linear_transformer: AbstractBijection | None
    circular_transformer: AbstractBijection | None
    linear_indices: Array
    circular_indices: Array
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def __init__(
        self,
        *,
        is_circular: Array,
        linear_transformer: AbstractBijection | None = None,
        circular_transformer: AbstractBijection | None = None,
        linear_indices: Array | None = None,
        circular_indices: Array | None = None,
    ):
        self.is_circular = jnp.asarray(is_circular, dtype=bool)

        # Use pre-computed indices if provided, otherwise compute them
        if linear_indices is not None:
            self.linear_indices = linear_indices
        else:
            self.linear_indices = jnp.where(~self.is_circular)[0]

        if circular_indices is not None:
            self.circular_indices = circular_indices
        else:
            self.circular_indices = jnp.where(self.is_circular)[0]

        # Set shape and conditioning shape
        self.shape = (len(self.is_circular),)
        self.cond_shape = None

        # Store transformers (can be None if no dimensions of that type)
        self.linear_transformer = linear_transformer
        self.circular_transformer = circular_transformer

        # Validate that we have transformers for existing dimensions
        if len(self.linear_indices) > 0 and self.linear_transformer is None:
            raise ValueError("linear_transformer required when linear dimensions exist")
        if len(self.circular_indices) > 0 and self.circular_transformer is None:
            raise ValueError("circular_transformer required when circular dimensions exist")

    def transform_and_log_det(self, x: Array, condition: Array | None = None) -> tuple[Array, Array]:
        """Apply mixed transformations and compute log determinant.

        Uses diagonal vmap to apply transformer i to input i, where the transformers
        have batched parameters from eqx.filter_vmap.
        """
        total_log_det = jnp.array(0.0)
        y = jnp.zeros_like(x)

        # Transform linear dimensions using diagonal vmap
        if len(self.linear_indices) > 0:
            # Ensure we have at least 1D array for vmap
            x_linear = jnp.atleast_1d(x[self.linear_indices])

            # Diagonal application: transformer[i] applies to x[i]
            # The linear_transformer has batched parameters from eqx.filter_vmap
            # We must use eqx.filter_vmap since the transformer has non-array leaves
            def _apply_transform(bij, xi):
                return bij.transform_and_log_det(xi, condition)

            y_linear, ld_linear = eqx.filter_vmap(_apply_transform)(
                self.linear_transformer, x_linear
            )
            y = y.at[self.linear_indices].set(y_linear)
            total_log_det = total_log_det + ld_linear.sum()

        # Transform circular dimensions using diagonal vmap
        if len(self.circular_indices) > 0:
            # Ensure we have at least 1D array for vmap
            x_circular = jnp.atleast_1d(x[self.circular_indices])

            def _apply_transform(bij, xi):
                return bij.transform_and_log_det(xi, condition)

            y_circular, ld_circular = eqx.filter_vmap(_apply_transform)(
                self.circular_transformer, x_circular
            )
            y = y.at[self.circular_indices].set(y_circular)
            total_log_det = total_log_det + ld_circular.sum()

        return y, total_log_det

    def inverse_and_log_det(self, y: Array, condition: Array | None = None) -> tuple[Array, Array]:
        """Apply inverse mixed transformations and compute log determinant.

        Uses diagonal vmap to apply inverse transformer i to input i.
        """
        total_log_det = jnp.array(0.0)
        x = jnp.zeros_like(y)

        # Inverse transform linear dimensions using diagonal vmap
        if len(self.linear_indices) > 0:
            # Ensure we have at least 1D array for vmap
            y_linear = jnp.atleast_1d(y[self.linear_indices])

            def _apply_inverse(bij, yi):
                return bij.inverse_and_log_det(yi, condition)

            # Use eqx.filter_vmap since the transformer has non-array leaves
            x_linear, ld_linear = eqx.filter_vmap(_apply_inverse)(
                self.linear_transformer, y_linear
            )
            x = x.at[self.linear_indices].set(x_linear)
            total_log_det = total_log_det + ld_linear.sum()

        # Inverse transform circular dimensions using diagonal vmap
        if len(self.circular_indices) > 0:
            # Ensure we have at least 1D array for vmap
            y_circular = jnp.atleast_1d(y[self.circular_indices])

            def _apply_inverse(bij, yi):
                return bij.inverse_and_log_det(yi, condition)

            # Use eqx.filter_vmap since the transformer has non-array leaves
            x_circular, ld_circular = eqx.filter_vmap(_apply_inverse)(
                self.circular_transformer, y_circular
            )
            x = x.at[self.circular_indices].set(x_circular)
            total_log_det = total_log_det + ld_circular.sum()

        return x, total_log_det