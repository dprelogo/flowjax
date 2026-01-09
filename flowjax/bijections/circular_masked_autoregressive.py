"""Masked autoregressive bijection with periodic embeddings for flows on tori.

Based on https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres).
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import paramax
from jaxtyping import Array, Int, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.bijections.masked_autoregressive import masked_autoregressive_mlp
from flowjax.masks import rank_based_mask
from flowjax.utils import get_ravelled_pytree_constructor


class CircularMaskedAutoregressive(AbstractBijection):
    """Masked autoregressive bijection with periodic embeddings for circular domains.

    Like the standard :class:`~flowjax.bijections.MaskedAutoregressive`, but embeds
    input angles as (cos θ, sin θ) pairs before the autoregressive network. This
    ensures the conditioner respects the periodicity of the circular domain.

    The key difference from standard MAF is the mask grouping: inputs (cos θᵢ, sin θᵢ)
    are grouped together with the same rank as θᵢ, ensuring they are masked
    consistently relative to output θᵢ.

    Refs:
        - https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres)
        - https://arxiv.org/abs/1705.07057 (Masked Autoregressive Flow)

    Args:
        key: PRNG key for initialization.
        transformer: Scalar bijection (shape=()) to parameterize.
            Parameters wrapped with ``NonTrainable`` are excluded.
        dim: Dimension of the input (number of angles).
        cond_dim: Optional external conditioning dimension. Defaults to None.
        nn_width: Neural network hidden layer width.
        nn_depth: Neural network depth (number of hidden layers).
        nn_activation: Activation function. Defaults to relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    transformer_constructor: Callable
    masked_autoregressive_mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        # Input ranks for (cos, sin) embedding:
        # Inputs are [cos θ_0, sin θ_0, cos θ_1, sin θ_1, ...]
        # Both cos θ_i and sin θ_i get rank i (grouped together)
        in_ranks = jnp.repeat(jnp.arange(dim), 2)  # [0, 0, 1, 1, 2, 2, ...]

        if cond_dim is None:
            self.cond_shape = None
            # If dim=1, hidden ranks all zero -> all weights masked in final layer
            hidden_ranks = jnp.arange(nn_width) % max(1, dim - 1)
        else:
            self.cond_shape = (cond_dim,)
            # Conditioning variables get rank -1 (connect to all outputs)
            cond_ranks = -jnp.ones(cond_dim, dtype=jnp.int32)
            in_ranks = jnp.concatenate([in_ranks, cond_ranks])
            # If dim=1, hidden ranks all -1 -> outputs only depend on condition
            hidden_ranks = (jnp.arange(nn_width) % dim) - 1

        # Output ranks: each output θ_i gets repeated for its num_params
        out_ranks = jnp.repeat(jnp.arange(dim), num_params)

        self.masked_autoregressive_mlp = _circular_masked_autoregressive_mlp(
            in_ranks,
            hidden_ranks,
            out_ranks,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

    def _embed_angles(self, angles: Array) -> Array:
        """Embed angles as interleaved (cos, sin) pairs.

        Args:
            angles: Array of shape (dim,).

        Returns:
            Array of shape (2*dim,) with [cos θ_0, sin θ_0, cos θ_1, sin θ_1, ...].
        """
        cos_vals = jnp.cos(angles)
        sin_vals = jnp.sin(angles)
        # Interleave: [cos_0, sin_0, cos_1, sin_1, ...]
        return jnp.ravel(jnp.stack([cos_vals, sin_vals], axis=-1))

    def transform_and_log_det(self, x, condition=None):
        """Transform x using masked autoregressive network."""
        # Embed input angles
        embedded = self._embed_angles(x)
        nn_input = embedded if condition is None else jnp.concatenate([embedded, condition])

        # Get transformer parameters from autoregressive network
        transformer_params = self.masked_autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)

        return transformer.transform_and_log_det(x)

    def inverse_and_log_det(self, y, condition=None):
        """Inverse transform via sequential scan."""
        init = (y, 0)
        fn = partial(self._inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))

        # Compute log det from forward pass at recovered x
        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det

    def _inv_scan_fn(self, init, _, condition):
        """One 'step' in computing the inverse."""
        y, rank = init

        # Embed current (partially inverted) y
        embedded = self._embed_angles(y)
        nn_input = embedded if condition is None else jnp.concatenate([embedded, condition])

        # Get transformer params
        transformer_params = self.masked_autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)

        # Inverse transform at current rank
        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])

        return (x, rank + 1), None

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to dim × params_per_dim, then vmap."""
        dim = self.shape[-1]
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Vmap(transformer, in_axes=eqx.if_array(0))


def _circular_masked_autoregressive_mlp(
    in_ranks: Int[Array, " in_size"],
    hidden_ranks: Int[Array, " hidden_size"],
    out_ranks: Int[Array, " out_size"],
    **kwargs,
) -> eqx.nn.MLP:
    """Returns an equinox MLP with autoregressive masks for circular inputs.

    This is similar to the standard masked_autoregressive_mlp, but designed for
    inputs where (cos θᵢ, sin θᵢ) pairs share the same rank.

    The weight matrices are wrapped using :class:`~paramax.wrappers.Parameterize`,
    which will apply the masking when :class:`~paramax.wrappers.unwrap` is called.

    Args:
        in_ranks: The ranks of the inputs (typically [0, 0, 1, 1, ...] for cos/sin pairs).
        hidden_ranks: The ranks of the hidden dimensions.
        out_ranks: The ranks of the output dimensions.
        **kwargs: Keyword arguments passed to equinox.nn.MLP.
    """
    in_ranks, hidden_ranks, out_ranks = (
        jnp.asarray(a, jnp.int32) for a in (in_ranks, hidden_ranks, out_ranks)
    )

    mlp = eqx.nn.MLP(
        in_size=len(in_ranks),
        out_size=len(out_ranks),
        width_size=len(hidden_ranks),
        **kwargs,
    )
    ranks = [in_ranks, *[hidden_ranks] * mlp.depth, out_ranks]

    masked_layers = []
    for i, linear in enumerate(mlp.layers):
        # For last layer, use strict inequality (output can't depend on same rank input)
        # For other layers, use <=
        mask = rank_based_mask(ranks[i], ranks[i + 1], eq=i != len(mlp.layers) - 1)
        masked_linear = eqx.tree_at(
            lambda linear: linear.weight,
            linear,
            paramax.Parameterize(jnp.where, mask, linear.weight, 0),
        )
        masked_layers.append(masked_linear)

    return eqx.tree_at(lambda mlp: mlp.layers, mlp, replace=tuple(masked_layers))
