"""Mixed topology masked autoregressive bijection for R^N × T^M flows."""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import paramax
from jaxtyping import Array, Int, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.mixed_transformer import MixedTransformer
from flowjax.bijections.masked_autoregressive import masked_autoregressive_mlp
from flowjax.utils import get_ravelled_pytree_constructor


class MixedMaskedAutoregressive(AbstractBijection):
    """Masked autoregressive bijection for mixed R^N × T^M topologies.

    This bijection handles flows on mixed topology spaces combining Euclidean (R^N)
    and circular/toroidal (T^M) dimensions. It uses different transformers for each
    topology type while maintaining autoregressive structure.

    Key features:
    - Mixed embedding: identity for linear dims, (cos θ, sin θ) for circular dims
    - Topology-aware parameter routing using MixedTransformer
    - Proper rank assignment for cos/sin pairs
    - Supports external conditioning

    Based on "Normalizing Flows on Tori and Spheres" (Rezende et al., 2020):
    "More generally, autoregressive flows can be applied in the same way on any
    manifold that can be written as a Cartesian product of circles and intervals"

    Args:
        key: PRNG key for initialization.
        is_circular: Boolean array indicating which dimensions are circular.
        dim: Total number of dimensions (linear + circular).
        linear_transformer_factory: Factory function for linear (RQS) transformers.
        circular_transformer_factory: Factory function for circular (CRQS) transformers.
        linear_bounds: Interval bounds for linear transformers.
        cond_dim: Optional external conditioning dimension.
        nn_width: Neural network hidden layer width.
        nn_depth: Neural network depth.
        nn_activation: Activation function.
        **transformer_kwargs: Additional kwargs for transformer factories.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    is_circular: Array
    linear_transformer_factory: Callable
    circular_transformer_factory: Callable
    linear_bounds: tuple[float, float]
    linear_param_count: int
    circular_param_count: int
    linear_embed_indices: Array
    circular_embed_indices: Array
    _linear_param_starts: list[int]
    _linear_param_sizes: list[int]
    _circular_param_starts: list[int]
    _circular_param_sizes: list[int]
    _is_circular_list: list[bool]  # Python list for interleaved embedding
    masked_autoregressive_mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        is_circular: Array,
        dim: int,
        linear_transformer_factory: Callable,
        circular_transformer_factory: Callable,
        linear_bounds: tuple[float, float] = (-5.0, 5.0),
        cond_dim: int | None = None,
        nn_width: int = 50,
        nn_depth: int = 1,
        nn_activation: Callable = jnn.relu,
        **transformer_kwargs,
    ):
        self.is_circular = jnp.asarray(is_circular, dtype=bool)
        self.linear_bounds = linear_bounds
        self.linear_transformer_factory = linear_transformer_factory
        self.circular_transformer_factory = circular_transformer_factory

        if len(self.is_circular) != dim:
            raise ValueError(f"is_circular length {len(self.is_circular)} != dim {dim}")

        # Get parameter counts by creating sample transformers
        self.linear_param_count = self._get_linear_param_count(**transformer_kwargs)
        self.circular_param_count = self._get_circular_param_count(**transformer_kwargs)

        # Compute dimensions and sizes
        n_linear = (~self.is_circular).sum()
        n_circular = self.is_circular.sum()

        # Mixed embedding size: linear dims contribute 1, circular dims contribute 2
        embedded_input_size = n_linear + 2 * n_circular

        # Total output size: sum of all parameter requirements
        output_size = n_linear * self.linear_param_count + n_circular * self.circular_param_count

        # Compute ranks for autoregressive masking
        in_ranks = self._compute_input_ranks(self.is_circular)
        out_ranks = self._compute_output_ranks(self.is_circular)

        # Add conditioning if specified
        if cond_dim is not None:
            self.cond_shape = (cond_dim,)
            # Conditioning variables get rank -1 (connect to all outputs)
            cond_ranks = -jnp.ones(cond_dim, dtype=jnp.int32)
            in_ranks = jnp.concatenate([in_ranks, cond_ranks])
            embedded_input_size += cond_dim
            # Hidden ranks accounting for conditioning
            hidden_ranks = (jnp.arange(nn_width) % dim) - 1
        else:
            self.cond_shape = None
            # Standard hidden ranks without conditioning
            hidden_ranks = jnp.arange(nn_width) % max(1, dim - 1)

        # Pre-compute embedding indices to avoid JAX tracing issues
        self.linear_embed_indices = jnp.array([i for i, is_circ in enumerate(self.is_circular) if not is_circ])
        self.circular_embed_indices = jnp.array([i for i, is_circ in enumerate(self.is_circular) if is_circ])

        # Pre-compute interleaved embedding info for _embed_mixed
        # This stores (dim_idx, is_circular) for each input dimension in order
        # to enable interleaved embedding without iterating over is_circular at trace time
        self._is_circular_list = [bool(is_circ) for is_circ in self.is_circular]

        # Pre-compute parameter extraction information as Python lists
        # Keep these as Python types to avoid JAX tracing issues
        self._linear_param_starts = []
        self._linear_param_sizes = []
        self._circular_param_starts = []
        self._circular_param_sizes = []

        param_idx = 0
        for i, is_circ in enumerate(self.is_circular):
            if is_circ:
                self._circular_param_starts.append(int(param_idx))
                self._circular_param_sizes.append(int(self.circular_param_count))
                param_idx += self.circular_param_count
            else:
                self._linear_param_starts.append(int(param_idx))
                self._linear_param_sizes.append(int(self.linear_param_count))
                param_idx += self.linear_param_count

        # Create masked autoregressive MLP
        self.masked_autoregressive_mlp = masked_autoregressive_mlp(
            in_ranks=in_ranks,
            hidden_ranks=hidden_ranks,
            out_ranks=out_ranks,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

        self.shape = (dim,)

    def _get_linear_param_count(self, **kwargs):
        """Get parameter count for linear transformer."""
        # Create a sample linear transformer to count parameters
        from flowjax.bijections import RationalQuadraticSpline
        sample_transformer = RationalQuadraticSpline(
            knots=kwargs.get('knots', 8),
            interval=self.linear_bounds,
            boundary_derivatives=kwargs.get('boundary_derivatives', 1.0),
        )
        _, param_count = get_ravelled_pytree_constructor(
            sample_transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        return param_count

    def _get_circular_param_count(self, **kwargs):
        """Get parameter count for circular transformer."""
        # Create a sample circular transformer to count parameters
        from flowjax.bijections import CircularRationalQuadraticSpline
        sample_transformer = CircularRationalQuadraticSpline(
            num_bins=kwargs.get('num_bins', 8),
        )
        _, param_count = get_ravelled_pytree_constructor(
            sample_transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        return param_count

    def _compute_input_ranks(self, is_circular: Array) -> Array:
        """Compute input ranks for mixed embedding with interleaved order.

        Embeddings are ordered to match input dimension order:
        [embed(x_0), embed(x_1), ...] where circular dims contribute (cos, sin).

        Example: [circular, linear] -> [cos(θ0), sin(θ0), r1]
        Ranks:   [0,       0,       1]

        This interleaved approach ensures monotonic ranks for stable MLP training
        while preserving the user's causal dimension ordering.
        """
        ranks = []
        for i, is_circ in enumerate(is_circular):
            if is_circ:
                # Circular dim i contributes 2 values (cos, sin), both rank i
                ranks.extend([i, i])
            else:
                # Linear dim i contributes 1 value, rank i
                ranks.append(i)
        return jnp.array(ranks, dtype=jnp.int32)

    def _compute_output_ranks(self, is_circular: Array) -> Array:
        """Compute output ranks: repeat dimension index for each parameter."""
        out_ranks = []
        for i, is_circ in enumerate(is_circular):
            if is_circ:
                out_ranks.extend([i] * self.circular_param_count)
            else:
                out_ranks.extend([i] * self.linear_param_count)
        return jnp.array(out_ranks, dtype=jnp.int32)

    def _embed_mixed(self, x: Array) -> Array:
        """Embed inputs with interleaved order: [embed(x_0), embed(x_1), ...].

        This preserves the user's dimension ordering for correct causal structure
        in autoregressive flows. Each dimension is embedded in order:
        - Linear: identity (1 value)
        - Circular: (cos, sin) (2 values)

        Example: x = [θ, r] with is_circular = [True, False]
        Output:  [cos(θ), sin(θ), r]
        """
        components = []
        # Use precomputed Python list to avoid tracing issues
        for i, is_circ in enumerate(self._is_circular_list):
            if is_circ:
                components.append(jnp.cos(x[i]))
                components.append(jnp.sin(x[i]))
            else:
                components.append(x[i])

        # Ensure all components are 1D arrays before concatenating
        flat_components = [jnp.atleast_1d(c) for c in components]
        return jnp.concatenate(flat_components)

    def _flat_params_to_mixed_transformer(self, flat_params: Array) -> MixedTransformer:
        """Convert flat parameter vector to MixedTransformer.

        Uses pre-computed parameter extraction indices to avoid JAX tracing issues.
        The returned transformers have batched parameters from eqx.filter_vmap,
        and MixedTransformer applies them diagonally using jax.vmap.
        """
        # Create transformers using factory patterns
        linear_transformer = None
        circular_transformer = None

        # Extract linear parameters using pre-computed indices
        if len(self._linear_param_starts) > 0:
            linear_params_list = []
            for i in range(len(self._linear_param_starts)):
                start = self._linear_param_starts[i]
                size = self._linear_param_sizes[i]
                # Use dynamic_slice for JAX tracing compatibility
                params = jax.lax.dynamic_slice(flat_params, (start,), (size,))
                linear_params_list.append(params)

            linear_params = jnp.stack(linear_params_list)
            # Create batched-parameter transformer using eqx.filter_vmap
            # MixedTransformer will apply these diagonally using jax.vmap
            linear_transformer = eqx.filter_vmap(self.linear_transformer_factory)(linear_params)

        # Extract circular parameters using pre-computed indices
        if len(self._circular_param_starts) > 0:
            circular_params_list = []
            for i in range(len(self._circular_param_starts)):
                start = self._circular_param_starts[i]
                size = self._circular_param_sizes[i]
                # Use dynamic_slice for JAX tracing compatibility
                params = jax.lax.dynamic_slice(flat_params, (start,), (size,))
                circular_params_list.append(params)

            circular_params = jnp.stack(circular_params_list)
            # Create batched-parameter transformer using eqx.filter_vmap
            # MixedTransformer will apply these diagonally using jax.vmap
            circular_transformer = eqx.filter_vmap(self.circular_transformer_factory)(circular_params)

        # Create MixedTransformer with pre-computed indices to avoid tracing issues
        return MixedTransformer(
            is_circular=self.is_circular,
            linear_transformer=linear_transformer,
            circular_transformer=circular_transformer,
            linear_indices=self.linear_embed_indices,
            circular_indices=self.circular_embed_indices,
        )

    def transform_and_log_det(self, x: Array, condition: Array | None = None) -> tuple[Array, Array]:
        """Apply mixed topology masked autoregressive transformation."""
        # Embed inputs based on topology
        embedded = self._embed_mixed(x)

        # Concatenate with conditioning if provided
        nn_input = embedded if condition is None else jnp.concatenate([embedded, condition])

        # Get transformer parameters from masked MLP
        transformer_params = self.masked_autoregressive_mlp(nn_input)

        # Create mixed transformer and apply
        transformer = self._flat_params_to_mixed_transformer(transformer_params)
        return transformer.transform_and_log_det(x)

    def inverse_and_log_det(self, y: Array, condition: Array | None = None) -> tuple[Array, Array]:
        """Inverse transformation via sequential scan."""
        init = (y, 0)
        fn = partial(self._inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))

        # Compute log det from forward pass at recovered x
        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det

    def _inv_scan_fn(self, init, _, condition):
        """One step in computing the inverse."""
        y, rank = init

        # Embed current (partially inverted) y
        embedded = self._embed_mixed(y)
        nn_input = embedded if condition is None else jnp.concatenate([embedded, condition])

        # Get transformer parameters
        transformer_params = self.masked_autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_mixed_transformer(transformer_params)

        # Inverse transform at current rank only
        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])

        return (x, rank + 1), None