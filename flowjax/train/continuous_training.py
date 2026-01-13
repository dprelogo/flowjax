"""Continuous per-batch resampling training for normalizing flows.

This module provides training utilities for fitting normalizing flows to
weighted samples using continuous resampling. At each training step, a fresh
batch is drawn from the weighted samples with optional noise added, providing
more sample diversity than one-time resampling approaches.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import paramax
from jaxtyping import Array, PRNGKeyArray

from flowjax.distributions import AbstractDistribution
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import step

# Type alias for perturbation function
PerturbationFn = Callable[[PRNGKeyArray, Array], Array]


def fit_to_weighted_data_continuous(
    key: PRNGKeyArray,
    dist: AbstractDistribution,
    x: Array,
    log_weights: Array,
    *,
    perturbation_fn: PerturbationFn | None = None,
    noise_scale: float = 1e-3,
    max_epochs: int = 100,
    max_patience: int = 10,
    batch_size: int = 256,
    val_prop: float = 0.1,
    n_val_batches: int = 5,
    learning_rate: float = 1e-3,
    optimizer: optax.GradientTransformation | None = None,
    loss_fn: MaximumLikelihoodLoss | None = None,
    show_progress: bool = True,
    return_best: bool = True,
) -> tuple[AbstractDistribution, dict[str, list[float]]]:
    """Train a distribution using continuous per-batch resampling.

    At each training step, a fresh batch is drawn from x according to
    the importance weights, with optional noise added. This approximates
    sampling from the continuous target distribution defined by the weighted
    samples, providing more diversity than one-time resampling.

    Args:
        key: Random key.
        dist: The distribution to train.
        x: Training samples of shape (n, dim).
        log_weights: Log importance weights of shape (n,).
        perturbation_fn: Optional function (key, batch_x) -> batch_x to apply
            noise and boundary handling. If None, adds Gaussian noise with
            std=noise_scale. Use this to inject custom boundary handling logic
            (e.g., clipping to [0, 1] for bounded data).
        noise_scale: Standard deviation of Gaussian noise added to resampled
            batches. Only used if perturbation_fn is None. Defaults to 1e-3.
        max_epochs: Maximum training epochs. Defaults to 100.
        max_patience: Early stopping patience (epochs without improvement).
            Defaults to 10.
        batch_size: Samples per batch. Defaults to 256.
        val_prop: Fraction of data for validation. Defaults to 0.1.
        n_val_batches: Number of validation batches to average for stable loss
            estimate. Each batch is resampled with perturbation. Defaults to 5.
        learning_rate: Learning rate (if optimizer not provided). Defaults to 1e-3.
        optimizer: Optional custom optimizer. If None, uses Adam with learning_rate.
        loss_fn: Loss function. Defaults to MaximumLikelihoodLoss.
        show_progress: Whether to show progress bar. Defaults to True.
        return_best: If True, return best model based on validation loss.
            Otherwise return final model. Defaults to True.

    Returns:
        Tuple of (trained distribution, loss history dict with 'train' and 'val' keys).

    Raises:
        ValueError: If x and log_weights have different lengths.

    Example:
        >>> import jax.random as jr
        >>> from flowjax.flows import masked_autoregressive_flow
        >>> from flowjax.distributions import Normal
        >>> from flowjax.train import fit_to_weighted_data_continuous
        >>>
        >>> key = jr.PRNGKey(0)
        >>> flow = masked_autoregressive_flow(key, base_dist=Normal(jnp.zeros(2)))
        >>> samples = jr.normal(key, (1000, 2))
        >>> log_weights = -jnp.sum(samples**2, axis=1)  # Example weights
        >>> trained_flow, losses = fit_to_weighted_data_continuous(
        ...     key, flow, samples, log_weights, max_epochs=50
        ... )

        Custom perturbation with boundary handling:

        >>> def clip_perturbation(key, batch_x):
        ...     noise = jr.normal(key, batch_x.shape) * 0.01
        ...     return jnp.clip(batch_x + noise, 0.0, 1.0)
        >>> trained_flow, losses = fit_to_weighted_data_continuous(
        ...     key, flow, samples, log_weights,
        ...     perturbation_fn=clip_perturbation, max_epochs=50
        ... )
    """
    # Input validation
    if x.shape[0] != log_weights.shape[0]:
        raise ValueError(
            f"x and log_weights must have same length: {x.shape[0]} vs {log_weights.shape[0]}"
        )

    if optimizer is None:
        optimizer = optax.adam(learning_rate)
    if loss_fn is None:
        loss_fn = MaximumLikelihoodLoss()

    # Define default perturbation if none provided
    if perturbation_fn is None:

        def default_perturbation(key: PRNGKeyArray, batch_x: Array) -> Array:
            return batch_x + jr.normal(key, batch_x.shape) * noise_scale

        perturbation_fn = default_perturbation

    # Normalize weights to probabilities for resampling
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_normalized)

    n_total = x.shape[0]
    n_val = max(1, int(n_total * val_prop))
    n_train = n_total - n_val

    # Randomly split into train/val sets
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, n_total)

    train_x = x[perm[:n_train]]
    val_x = x[perm[n_train:]]
    train_weights = weights[perm[:n_train]]
    val_weights = weights[perm[n_train:]]

    # Re-normalize weights within their splits
    train_weights = train_weights / train_weights.sum()
    val_weights = val_weights / val_weights.sum()

    # Partition model
    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    opt_state = optimizer.init(params)

    # JIT-compiled batch generator with proper key splitting
    @jax.jit
    def get_batch(key, samples, probs):
        # Split key to avoid correlation between selection and perturbation
        idx_key, perturb_key = jr.split(key)

        # Resample according to weights
        indices = jr.choice(idx_key, len(samples), shape=(batch_size,), p=probs)
        batch_x = samples[indices]

        # Apply perturbation (noise + optional boundary handling)
        batch_x = perturbation_fn(perturb_key, batch_x)

        return batch_x

    # JIT-compiled validation loss with resampling and perturbation
    # Resamples from val_x according to val_weights, applies perturbation,
    # and computes unweighted mean NLL (since resampling already accounts for weights)
    @eqx.filter_jit
    def compute_val_batch_loss(params, static, val_x, val_weights, key):
        """Compute validation loss for a single resampled batch."""
        idx_key, perturb_key = jr.split(key)

        # Resample according to weights
        indices = jr.choice(idx_key, len(val_x), shape=(batch_size,), p=val_weights)
        batch_x = val_x[indices]

        # Apply same perturbation as training
        batch_x = perturbation_fn(perturb_key, batch_x)

        # Compute unweighted mean NLL (weights already accounted for by resampling)
        dist = eqx.combine(params, static)
        log_probs = jax.vmap(dist.log_prob)(batch_x)
        return -jnp.mean(log_probs)

    def compute_val_loss(params, static, val_x, val_weights, key):
        """Compute averaged validation loss over multiple resampled batches."""
        keys = jr.split(key, n_val_batches)
        losses = jnp.array([
            compute_val_batch_loss(params, static, val_x, val_weights, k)
            for k in keys
        ])
        return jnp.mean(losses)

    # Training loop
    steps_per_epoch = max(1, n_train // batch_size)
    losses = {"train": [], "val": []}
    best_params = params
    best_val_loss = float("inf")
    patience_counter = 0

    if show_progress:
        try:
            from tqdm.auto import tqdm

            epoch_iter = tqdm(range(max_epochs), desc="Continuous training")
        except ImportError:
            epoch_iter = range(max_epochs)
    else:
        epoch_iter = range(max_epochs)

    for epoch in epoch_iter:
        epoch_train_losses = []

        for _ in range(steps_per_epoch):
            key, batch_key, step_key = jr.split(key, 3)
            batch_x = get_batch(batch_key, train_x, train_weights)

            # Training step
            params, opt_state, loss = step(
                params,
                static,
                batch_x,
                optimizer=optimizer,
                opt_state=opt_state,
                loss_fn=loss_fn,
                key=step_key,
            )
            epoch_train_losses.append(float(loss))

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        losses["train"].append(avg_train_loss)

        # Validation with resampling and perturbation (averaged over n_val_batches)
        key, val_key = jr.split(key)
        val_loss = float(
            compute_val_loss(params, static, val_x, val_weights, val_key)
        )
        losses["val"].append(val_loss)

        if show_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                if show_progress and hasattr(epoch_iter, "write"):
                    epoch_iter.write(f"Early stopping at epoch {epoch}")
                break

    final_params = best_params if return_best else params
    return eqx.combine(final_params, static), losses
