"""Coupling layer with periodic embeddings for flows on tori.

Based on https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres).
"""

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import paramax
from jaxtyping import PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.utils import Array, get_ravelled_pytree_constructor


class CircularCoupling(AbstractBijection):
    """Coupling layer with periodic embeddings for circular domains.

    Like the standard :class:`~flowjax.bijections.Coupling` layer, but embeds
    conditioning angles as (cos θ, sin θ) pairs before passing to the conditioner
    network. This ensures the conditioner respects the periodicity of the circular
    domain.

    For a D-dimensional torus, the first `untransformed_dim` angles are used to
    condition the transformation of the remaining angles.

    Refs:
        - https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres)
        - https://arxiv.org/abs/1605.08803 (RealNVP)

    Args:
        key: PRNG key for initialization.
        transformer: Scalar bijection (shape=()) to parameterize, e.g.,
            :class:`~flowjax.bijections.CircularRationalQuadraticSpline`.
            Parameters wrapped with ``NonTrainable`` are excluded.
        untransformed_dim: Number of conditioning dimensions (not transformed).
        dim: Total dimension of the input.
        cond_dim: Optional external conditioning dimension. Defaults to None.
        nn_width: Conditioner network hidden layer width.
        nn_depth: Conditioner network depth (number of hidden layers).
        nn_activation: Activation function. Defaults to relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    untransformed_dim: int
    dim: int
    transformer_constructor: Callable
    conditioner: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        untransformed_dim: int,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ):
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        self.transformer_constructor = constructor
        self.untransformed_dim = untransformed_dim
        self.dim = dim
        self.shape = (dim,)
        self.cond_shape = (cond_dim,) if cond_dim is not None else None

        # Input size: 2 * untransformed_dim (cos, sin pairs) + cond_dim
        # The (cos, sin) embedding doubles the effective input dimension
        embedded_input_size = 2 * untransformed_dim
        if cond_dim is not None:
            # External conditions are NOT embedded (may not be circular)
            embedded_input_size += cond_dim

        conditioner_output_size = num_params * (dim - untransformed_dim)

        self.conditioner = eqx.nn.MLP(
            in_size=embedded_input_size,
            out_size=conditioner_output_size,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

    def _embed_angles(self, angles: Array) -> Array:
        """Embed angles as (cos, sin) pairs for periodicity.

        This ensures the conditioner network respects the circular topology:
        angles θ and θ + 2π map to the same embedding.

        Args:
            angles: Array of angles.

        Returns:
            Concatenated [cos(angles), sin(angles)].
        """
        return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])

    def transform_and_log_det(self, x, condition=None):
        """Transform x, keeping first untransformed_dim fixed."""
        x_cond = x[: self.untransformed_dim]
        x_trans = x[self.untransformed_dim :]

        # Embed conditioning angles for periodicity
        embedded = self._embed_angles(x_cond)
        if condition is not None:
            embedded = jnp.concatenate([embedded, condition])

        # Get transformer parameters from conditioner network
        transformer_params = self.conditioner(embedded)
        transformer = self._flat_params_to_transformer(transformer_params)

        # Transform the non-conditioning part
        y_trans, log_det = transformer.transform_and_log_det(x_trans)
        y = jnp.concatenate([x_cond, y_trans])

        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        """Inverse transform."""
        x_cond = y[: self.untransformed_dim]
        y_trans = y[self.untransformed_dim :]

        # Embed conditioning angles (same as forward)
        embedded = self._embed_angles(x_cond)
        if condition is not None:
            embedded = jnp.concatenate([embedded, condition])

        # Get transformer parameters
        transformer_params = self.conditioner(embedded)
        transformer = self._flat_params_to_transformer(transformer_params)

        # Inverse transform
        x_trans, log_det = transformer.inverse_and_log_det(y_trans)
        x = jnp.concatenate([x_cond, x_trans])

        return x, log_det

    def _flat_params_to_transformer(self, params: Array):
        """Reshape flat params to per-dimension params, then vmap transformer."""
        dim = self.dim - self.untransformed_dim
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Vmap(transformer, in_axes=eqx.if_array(0))
