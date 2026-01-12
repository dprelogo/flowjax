"""Premade versions of common flow architetctures from ``flowjax.flows``.

All these functions return a :class:`~flowjax.distributions.Transformed` distribution.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Linear
from jax.nn import softplus
from jax.nn.initializers import glorot_uniform
from jaxtyping import Array, PRNGKeyArray
import paramax
from paramax import Parameterize, WeightNormalization
from paramax.utils import inv_softplus

from flowjax.bijections import (
    AbstractBijection,
    AdditiveCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    CircularCoupling,
    CircularMaskedAutoregressive,
    CircularRationalQuadraticSpline,
    CircularScale,
    Coupling,
    Flip,
    Invert,
    LeakyTanh,
    MaskedAutoregressive,
    MixedMaskedAutoregressive,
    NumericalInverse,
    Permute,
    Planar,
    RationalQuadraticSpline,
    Sandwich,
    Scan,
    TriangularAffine,
    Vmap,
)
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.root_finding import (
    bisect_check_expand_search,
    root_finder_to_inverter,
)
from flowjax.utils import get_ravelled_pytree_constructor


def _affine_with_min_scale(min_scale: float = 1e-2) -> Affine:
    scale = Parameterize(lambda x: softplus(x) + min_scale, inv_softplus(1 - min_scale))
    return eqx.tree_at(where=lambda aff: aff.scale, pytree=Affine(), replace=scale)


def coupling_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Create a coupling flow (https://arxiv.org/abs/1605.08803).

    Args:
        key: Jax random key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        transformer: Bijection to be parameterised by conditioner. Defaults to
            affine.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        flow_layers: Number of coupling layers. Defaults to 8.
        nn_width: Conditioner hidden layer size. Defaults to 50.
        nn_depth: Conditioner depth. Defaults to 1.
        nn_activation: Conditioner activation function. Defaults to jnn.relu.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            `inverse` methods, leading to faster `log_prob`, False will prioritise
            faster `transform` methods, leading to faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = _affine_with_min_scale()

    dim = base_dist.shape[-1]

    def make_layer(key):  # coupling layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = Coupling(
            key=bij_key,
            transformer=transformer,
            untransformed_dim=dim // 2,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Masked autoregressive flow.

    Parameterises a transformer bijection with an autoregressive neural network.
    Refs: https://arxiv.org/abs/1606.04934; https://arxiv.org/abs/1705.07057v4.

    Args:
        key: Random seed.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        transformer: Bijection parameterised by autoregressive network. Defaults to
            affine.
        cond_dim: Dimension of the conditioning variable. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        nn_width: Number of hidden layers in neural network. Defaults to 50.
        nn_depth: Depth of neural network. Defaults to 1.
        nn_activation: _description_. Defaults to jnn.relu.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            inverse, leading to faster `log_prob`, False will prioritise faster forward,
            leading to faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = _affine_with_min_scale()

    dim = base_dist.shape[-1]

    def make_layer(key):  # masked autoregressive layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = MaskedAutoregressive(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def block_neural_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    nn_depth: int = 1,
    nn_block_dim: int = 8,
    flow_layers: int = 1,
    invert: bool = True,
    activation: AbstractBijection | Callable | None = None,
    inverter: Callable[[AbstractBijection, Array, Array | None], Array] | None = None,
) -> Transformed:
    """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

    Each flow layer contains a
    :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`
    bijection. The bijection does not have an analytic inverse, so must be inverted
    using numerical methods (by default a bisection search). Note that this means
    that only one of ``log_prob`` or ``sample{_and_log_prob}`` can be efficient,
    controlled by the ``invert`` argument. Note, ensuring reasonably scaled base and
    target distributions will be beneficial for the efficiency of the numerical inverse.

    Args:
        key: Jax key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: Dimension of conditional variables. Defaults to None.
        nn_depth: Number of hidden layers within the networks. Defaults to 1.
        nn_block_dim: Block size. Hidden layer width is dim*nn_block_dim. Defaults to 8.
        flow_layers: Number of BNAF layers. Defaults to 1.
        invert: Use `True` for efficient ``log_prob`` (e.g. when fitting by maximum
            likelihood), and `False` for efficient ``sample`` and
            ``sample_and_log_prob`` methods (e.g. for fitting variationally).
        activation: Activation function used within block neural autoregressive
            networks. Note this should be bijective and in some use cases should map
            real -> real. For more information, see
            :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`.
            Defaults to :class:`~flowjax.bijections.tanh.LeakyTanh`.
        inverter: Callable that implements the required numerical method to
            invert the ``BlockAutoregressiveNetwork`` bijection. Passed to
            :py:class:`~flowjax.bijections.NumericalInverse`. Defaults to
            using ``elementwise_autoregressive_bisection``.
    """
    dim = base_dist.shape[-1]

    if inverter is None:
        inverter = root_finder_to_inverter(
            partial(
                bisect_check_expand_search,
                midpoint=jnp.zeros(dim),
                width=5,
            )
        )

    def make_layer(key):  # bnaf layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = NumericalInverse(
            BlockAutoregressiveNetwork(
                bij_key,
                dim=base_dist.shape[-1],
                cond_dim=cond_dim,
                depth=nn_depth,
                block_dim=nn_block_dim,
                activation=activation,
            ),
            inverter=inverter,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def planar_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    invert: bool = True,
    negative_slope: float | None = None,
    **mlp_kwargs,
) -> Transformed:
    """Planar flow as introduced in https://arxiv.org/pdf/1505.05770.pdf.

    This alternates between :class:`~flowjax.bijections.planar.Planar` layers and
    permutations. Note the definition here is inverted compared to the original paper.

    Args:
        key: Jax key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        invert: Whether to invert the bijection. Broadly, True will prioritise a faster
            `inverse` methods, leading to faster `log_prob`, False will prioritise
            faster `transform` methods, leading to faster `sample`. Defaults to True.
        negative_slope: A positive float. If provided, then a leaky relu activation
            (with the corresponding negative slope) is used instead of tanh. This also
            provides the advantage that the bijection can be inverted analytically.
        **mlp_kwargs: Keyword arguments (excluding in_size and out_size) passed to
            the MLP (equinox.nn.MLP). Ignored when cond_dim is None.
    """

    def make_layer(key):  # Planar layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = Planar(
            bij_key,
            dim=base_dist.shape[-1],
            cond_dim=cond_dim,
            negative_slope=negative_slope,
            **mlp_kwargs,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def triangular_spline_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    knots: int = 8,
    tanh_max_val: float | int = 3.0,
    invert: bool = True,
    init: Callable | None = None,
) -> Transformed:
    """Triangular spline flow.

    Each layer consists of a triangular affine transformation with weight normalisation,
    and an elementwise rational quadratic spline. Tanh is used to constrain to the input
    to [-1, 1] before spline transformations.

    Args:
        key: Jax random key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: The number of conditioning features. Defaults to None.
        flow_layers: Number of flow layers. Defaults to 8.
        knots: Number of knots in the splines. Defaults to 8.
        tanh_max_val: Maximum absolute value beyond which we use linear "tails" in the
            tanh function. Defaults to 3.0.
        invert: Whether to invert the bijection before transforming the base
            distribution. Defaults to True.
        init: Initialisation method for the lower triangular weights.
            Defaults to glorot_uniform().
    """
    init = init if init is not None else glorot_uniform()
    dim = base_dist.shape[-1]

    def get_splines():
        fn = partial(RationalQuadraticSpline, knots=knots, interval=1)
        spline = eqx.filter_vmap(fn, axis_size=dim)()
        return Vmap(spline, in_axes=eqx.if_array(0))

    def make_layer(key):
        lt_key, perm_key, cond_key = jr.split(key, 3)
        weights = init(lt_key, (dim, dim))
        lt_weights = weights.at[jnp.diag_indices(dim)].set(1)
        tri_aff = TriangularAffine(jnp.zeros(dim), lt_weights)
        tri_aff = eqx.tree_at(
            lambda t: t.triangular, tri_aff, replace_fn=WeightNormalization
        )
        bijections = [
            Sandwich(get_splines(), LeakyTanh(tanh_max_val, (dim,))),
            tri_aff,
        ]

        if cond_dim is not None:
            linear_condition = AdditiveCondition(
                Linear(cond_dim, dim, use_bias=False, key=cond_key),
                (dim,),
                (cond_dim,),
            )
            bijections.append(linear_condition)

        bijection = Chain(bijections)
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def circular_coupling_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    num_bins: int = 8,
    invert: bool = True,
) -> Transformed:
    """Coupling flow for circular/toroidal domains.

    Uses :class:`~flowjax.bijections.CircularCoupling` layers with periodic
    (cos, sin) embeddings to respect the circular topology. The default transformer
    is :class:`~flowjax.bijections.CircularRationalQuadraticSpline`, which implements
    a circle diffeomorphism.

    Refs:
        - https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres)
        - https://arxiv.org/abs/1605.08803 (RealNVP)

    Args:
        key: Jax random key.
        base_dist: Base distribution on the torus, with ``base_dist.ndim==1``.
            Typically ``TorusUniform(dim)`` for uniform distribution on [0, 2π]^D.
        transformer: Scalar bijection (shape=()) to be parameterised by conditioner.
            Defaults to :class:`~flowjax.bijections.CircularRationalQuadraticSpline`.
        cond_dim: Dimension of external conditioning variables. Defaults to None.
        flow_layers: Number of coupling layers. Defaults to 8.
        nn_width: Conditioner hidden layer size. Defaults to 50.
        nn_depth: Conditioner depth. Defaults to 1.
        nn_activation: Conditioner activation function. Defaults to jnn.relu.
        num_bins: Number of spline bins (only used if transformer is None). Defaults to 8.
        invert: Whether to invert the bijection. True prioritises faster `log_prob`,
            False prioritises faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = CircularRationalQuadraticSpline(num_bins=num_bins)

    dim = base_dist.shape[-1]

    def make_layer(key):  # circular coupling layer + flip
        bijection = CircularCoupling(
            key=key,
            transformer=transformer,
            untransformed_dim=dim // 2,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        # Use Flip for circular domains (deterministic, preserves topology)
        return Chain([bijection, Flip((dim,))]).merge_chains()

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)


def circular_masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    unit_hypercube: bool = False,
    transformer: AbstractBijection | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    num_bins: int = 8,
    invert: bool = True,
) -> Transformed:
    """Masked autoregressive flow for circular/toroidal domains.

    Uses :class:`~flowjax.bijections.CircularMaskedAutoregressive` layers with
    periodic (cos, sin) embeddings. Input angles are embedded as interleaved
    (cos θᵢ, sin θᵢ) pairs with grouped ranks to ensure proper autoregressive
    masking that respects circularity.

    Refs:
        - https://arxiv.org/abs/2002.02428 (Normalizing Flows on Tori and Spheres)
        - https://arxiv.org/abs/1705.07057 (Masked Autoregressive Flow)

    Args:
        key: Jax random key.
        base_dist: Base distribution on the torus, with ``base_dist.ndim==1``.
            Typically ``TorusUniform(dim)`` for uniform distribution on [0, 2π]^D.
            When ``unit_hypercube=True``, use ``Uniform(jnp.zeros(dim), jnp.ones(dim))``
            for uniform distribution on [0, 1]^D.
        unit_hypercube: If True, the flow operates on [0, 1]^D domain. All
            dimensions are scaled to [0, 2π] internally for proper circular
            embeddings, then scaled back to [0, 1] at output. Defaults to False.
        transformer: Scalar bijection (shape=()) to be parameterised by the
            autoregressive network. Defaults to
            :class:`~flowjax.bijections.CircularRationalQuadraticSpline`.
        cond_dim: Dimension of external conditioning variables. Defaults to None.
        flow_layers: Number of MAF layers. Defaults to 8.
        nn_width: Hidden layer size in the masked autoregressive network. Defaults to 50.
        nn_depth: Depth of the masked autoregressive network. Defaults to 1.
        nn_activation: Activation function. Defaults to jnn.relu.
        num_bins: Number of spline bins (only used if transformer is None). Defaults to 8.
        invert: Whether to invert the bijection. True prioritises faster `log_prob`,
            False prioritises faster `sample`. Defaults to True.
    """
    if transformer is None:
        transformer = CircularRationalQuadraticSpline(num_bins=num_bins)

    dim = base_dist.shape[-1]

    def make_layer(key):  # circular MAF layer + flip
        bijection = CircularMaskedAutoregressive(
            key=key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        # Use Flip for circular domains (deterministic, preserves topology)
        return Chain([bijection, Flip((dim,))]).merge_chains()

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Scan(layers)

    # Wrap with CircularScale for unit hypercube support
    # For fully circular flows, all dimensions are scaled
    if unit_hypercube:
        is_circular = jnp.ones(dim, dtype=bool)  # All dims are circular
        scale_in = CircularScale(is_circular, forward_to_2pi=True)
        scale_out = CircularScale(is_circular, forward_to_2pi=False)
        bijection = Chain([scale_in, bijection, scale_out]).merge_chains()

    if invert:
        bijection = Invert(bijection)

    return Transformed(base_dist, bijection)


def mixed_masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    is_circular: Array,
    unit_hypercube: bool = False,
    linear_bounds: tuple[float, float] = (-5.0, 5.0),
    linear_boundary_derivatives: float | None = 1.0,
    linear_transformer_kwargs: dict | None = None,
    circular_transformer_kwargs: dict | None = None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Masked autoregressive flow for mixed R^N × T^M topologies.

    Creates flows that can model joint distributions on product spaces combining
    Euclidean (R^N) and toroidal (T^M) dimensions. Uses different transformers
    for each topology type while maintaining autoregressive structure with
    mixed embeddings.

    Based on "Normalizing Flows on Tori and Spheres" (Rezende et al., 2020):
    "More generally, autoregressive flows can be applied in the same way on any
    manifold that can be written as a Cartesian product of circles and intervals."

    Key features:
    - Mixed embedding: identity for linear dims, (cos θ, sin θ) for circular dims
    - Interleaved embedding order preserves dimension ordering for causal structure
    - Topology-aware parameter routing using MixedTransformer
    - Full permutation with topology tracking for maximum expressivity
    - Automatic bounds matching between base distribution and transformers

    Note on dimension ordering:
        The flow respects your input dimension ordering. For correlated distributions
        where one variable depends on another (e.g., r = f(θ)), you should order
        dimensions so that conditioning variables come first. In autoregressive flows,
        dimension i is conditioned on dimensions 0, 1, ..., i-1.

    Args:
        key: Jax random key.
        base_dist: Base distribution with shape matching dimension count.
            **Recommended**: Use ``MixedBase(is_circular)`` for numerical stability
            (StandardNormal for R dims, Uniform for T dims). ``MixedUniform`` may
            cause ``-inf`` log_prob when samples map slightly outside bounds.
            When ``unit_hypercube=True``, use ``MixedUniformBase(is_circular)`` instead.
        is_circular: Boolean array indicating which dimensions are circular.
            Length must match base_dist.shape[-1].
        unit_hypercube: If True, the flow operates on [0, 1]^D domain. Circular
            dimensions are internally scaled to [0, 2π] for proper circular embeddings,
            then scaled back to [0, 1] at output. Use with ``MixedUniformBase`` and
            ``linear_bounds=(0.0, 1.0)`` for best results. Defaults to False.
        linear_bounds: Bounds for linear dimensions (used in RationalQuadraticSpline).
            Defaults to (-5.0, 5.0). **Important**: When using ``unit_hypercube=True``,
            set this to ``(0.0, 1.0)`` to match the [0, 1] data domain.
        linear_boundary_derivatives: Fixed boundary derivative value for linear
            transformers. Use 1.0 for C1 continuity with identity tails (default),
            or None to learn boundary derivatives freely. Defaults to 1.0.
        linear_transformer_kwargs: Additional kwargs for linear transformers
            (RationalQuadraticSpline). Defaults to None.
        circular_transformer_kwargs: Additional kwargs for circular transformers
            (CircularRationalQuadraticSpline). Defaults to None.
        cond_dim: Dimension of external conditioning variables. Defaults to None.
        flow_layers: Number of mixed MAF layers. Defaults to 8.
        nn_width: Hidden layer size in autoregressive networks. Defaults to 50.
        nn_depth: Depth of autoregressive networks. Defaults to 1.
        nn_activation: Activation function. Defaults to jnn.relu.
        invert: Whether to invert the bijection. True prioritises faster log_prob,
            False prioritises faster sample. Defaults to True.

    Returns:
        Transformed distribution implementing mixed topology normalizing flow.

    Example:
        >>> import jax.random as jr
        >>> from flowjax.flows import mixed_masked_autoregressive_flow
        >>> from flowjax.distributions import MixedBase
        >>>
        >>> # Create R^2 × T^1 flow (2 linear, 1 circular dimension)
        >>> key = jr.key(0)
        >>> is_circular = jnp.array([False, False, True])  # [R, R, T]
        >>> base_dist = MixedBase(is_circular)  # Recommended for numerical stability
        >>> flow = mixed_masked_autoregressive_flow(
        ...     key, base_dist=base_dist, is_circular=is_circular
        ... )
    """
    is_circular = jnp.asarray(is_circular, dtype=bool)
    dim = base_dist.shape[-1]

    if len(is_circular) != dim:
        raise ValueError(f"is_circular length {len(is_circular)} != dim {dim}")

    # Default transformer kwargs
    linear_kwargs = linear_transformer_kwargs or {"knots": 8}
    circular_kwargs = circular_transformer_kwargs or {"num_bins": 8}

    # Create sample transformers ONCE to build constructors
    # boundary_derivatives=1.0 (default) ensures C1 continuity with identity tails
    # Use boundary_derivatives=None to learn boundary derivatives freely (useful for
    # bounded distributions like Uniform[0,1] where identity tails aren't needed)
    linear_sample = RationalQuadraticSpline(
        interval=linear_bounds,
        boundary_derivatives=linear_boundary_derivatives,
        **linear_kwargs,
    )
    circular_sample = CircularRationalQuadraticSpline(**circular_kwargs)

    # Build constructors ONCE outside the factory closures (critical for performance)
    # This avoids expensive PyTree traversal on every factory call during tracing
    filter_spec = eqx.is_inexact_array
    is_leaf = lambda leaf: isinstance(leaf, paramax.NonTrainable)

    linear_constructor, _ = get_ravelled_pytree_constructor(
        linear_sample, filter_spec=filter_spec, is_leaf=is_leaf
    )
    circular_constructor, _ = get_ravelled_pytree_constructor(
        circular_sample, filter_spec=filter_spec, is_leaf=is_leaf
    )

    # Factories are now simple wrappers around pre-computed constructors
    def linear_transformer_factory(params: Array) -> AbstractBijection:
        """Factory function that creates RQS transformer from flat parameters."""
        return linear_constructor(params)

    def circular_transformer_factory(params: Array) -> AbstractBijection:
        """Factory function that creates CRQS transformer from flat parameters."""
        return circular_constructor(params)

    # Track topology mask across layers for full permutation strategy
    # Use original user-specified order to respect causal structure
    current_mask = is_circular

    def make_layer(layer_data):
        """Create Mixed MAF layer + permutation with topology tracking."""
        layer_key, current_topology = layer_data
        bij_key, perm_key = jr.split(layer_key)

        # Create Mixed MAF bijection using current topology configuration
        # Include boundary_derivatives in kwargs so MixedMaskedAutoregressive
        # computes correct parameter count in _get_linear_param_count
        transformer_kwargs = {
            **linear_kwargs,
            **circular_kwargs,
            "boundary_derivatives": linear_boundary_derivatives,
        }
        bijection = MixedMaskedAutoregressive(
            key=bij_key,
            is_circular=current_topology,
            dim=dim,
            linear_transformer_factory=linear_transformer_factory,
            circular_transformer_factory=circular_transformer_factory,
            linear_bounds=linear_bounds,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
            **transformer_kwargs
        )

        # Add permutation (except for last layer handled separately)
        if dim > 1:
            perm_indices = jr.permutation(perm_key, jnp.arange(dim))
            perm = Permute(perm_indices)
            # Update topology mask for next layer
            next_topology = current_topology[perm_indices]
            return Chain([bijection, perm]).merge_chains(), next_topology
        else:
            return bijection, current_topology

    # Create layer data: (key, topology_mask) pairs
    keys = jr.split(key, flow_layers)

    # Build layers sequentially to track topology evolution
    layers = []
    topology_masks = [current_mask]

    for i, layer_key in enumerate(keys):
        current_topology = topology_masks[-1]

        # For last layer, don't add random permutation
        if i == flow_layers - 1:
            # Use same transformer_kwargs as make_layer to ensure consistent param counts
            last_layer_kwargs = {
                **linear_kwargs,
                **circular_kwargs,
                "boundary_derivatives": linear_boundary_derivatives,
            }
            bijection = MixedMaskedAutoregressive(
                key=layer_key,
                is_circular=current_topology,
                dim=dim,
                linear_transformer_factory=linear_transformer_factory,
                circular_transformer_factory=circular_transformer_factory,
                linear_bounds=linear_bounds,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
                **last_layer_kwargs
            )
            layers.append(bijection)
        else:
            layer, next_topology = make_layer((layer_key, current_topology))
            layers.append(layer)
            topology_masks.append(next_topology)

    # Chain all layers
    bijection = Chain(layers) if len(layers) > 1 else layers[0]

    # Wrap with CircularScale for unit hypercube support
    # This scales circular dims [0,1] <-> [0,2π] at flow boundaries
    # IMPORTANT: scale_out must use the FINAL topology mask after all permutations
    if unit_hypercube:
        # scale_in uses original is_circular (input dimension ordering)
        scale_in = CircularScale(is_circular, forward_to_2pi=True)
        # scale_out uses final topology mask (output dimension ordering after permutations)
        final_topology = topology_masks[-1]
        scale_out = CircularScale(final_topology, forward_to_2pi=False)
        bijection = Chain([scale_in, bijection, scale_out]).merge_chains()

    if invert:
        bijection = Invert(bijection)

    return Transformed(base_dist, bijection)


def _add_default_permute(bijection: AbstractBijection, dim: int, key: PRNGKeyArray):
    if dim == 1:
        return bijection
    if dim == 2:
        return Chain([bijection, Flip((dim,))]).merge_chains()

    perm = Permute(jr.permutation(key, jnp.arange(dim)))
    return Chain([bijection, perm]).merge_chains()
