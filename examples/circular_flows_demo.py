"""Demonstration of circular/toroidal normalizing flows.

This script demonstrates the circular flow implementations on:
1. 1D circle (S^1): A wrapped distribution that is continuous at 0/2π boundary
2. 2D torus (T^2): A correlated distribution that wraps in both dimensions

Inspired by arXiv:2002.02428 "Normalizing Flows on Tori and Spheres"
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from functools import partial

from flowjax.distributions import TorusUniform
from flowjax.flows import circular_coupling_flow, circular_masked_autoregressive_flow
from flowjax.train import fit_to_data

# Set up for reproducibility
plt.style.use('seaborn-v0_8-whitegrid')
TWO_PI = 2 * jnp.pi


# =============================================================================
# Target Distributions (von Mises based, as in the paper)
# =============================================================================

def log_bessel_i0(kappa):
    """Log of modified Bessel function I_0(kappa).

    Uses jax.scipy.special.i0e (exponentially scaled) for numerical stability:
    I_0(x) = i0e(x) * exp(|x|), so log(I_0(x)) = log(i0e(x)) + |x|
    """
    return jnp.log(jax.scipy.special.i0e(kappa)) + jnp.abs(kappa)


def von_mises_log_prob_normalized(theta, mu, kappa):
    """Normalized log probability of von Mises distribution.

    log p(θ; μ, κ) = κ cos(θ - μ) - log(2π) - log(I_0(κ))
    """
    return kappa * jnp.cos(theta - mu) - jnp.log(TWO_PI) - log_bessel_i0(kappa)


def von_mises_log_prob(theta, mu, kappa):
    """Log probability of von Mises distribution (up to normalization)."""
    return kappa * jnp.cos(theta - mu)


def sample_von_mises(key, mu, kappa, shape=()):
    """Sample from von Mises distribution using rejection sampling."""
    # Simple rejection sampling for von Mises
    n_samples = int(np.prod(shape)) if shape else 1
    samples = []
    key_iter = key

    while len(samples) < n_samples:
        key_iter, subkey1, subkey2 = jr.split(key_iter, 3)
        # Propose from uniform
        theta = jr.uniform(subkey1, (n_samples * 10,), minval=0, maxval=TWO_PI)
        # Accept with probability proportional to exp(kappa * cos(theta - mu))
        log_u = jnp.log(jr.uniform(subkey2, (n_samples * 10,)))
        log_accept = kappa * (jnp.cos(theta - mu) - 1)  # Normalized so max is 0
        accepted = theta[log_u < log_accept]
        samples.extend(accepted.tolist())

    samples = jnp.array(samples[:n_samples])
    if shape:
        samples = samples.reshape(shape)
    return samples


# =============================================================================
# Example 1: 1D Circle - Bimodal wrapped distribution
# =============================================================================

def demo_1d_circle():
    """1D demonstration: bimodal distribution on the circle."""
    print("=" * 60)
    print("Example 1: 1D Circle (S^1) - Bimodal Distribution")
    print("=" * 60)

    # Target: mixture of two von Mises distributions
    # One mode near 0/2π boundary to show wrap-around
    mu1, kappa1 = 0.3, 5.0  # Mode near 0
    mu2, kappa2 = 3.5, 8.0  # Mode in the middle

    # Mixture weights (must match the sampling strategy: 50/50)
    log_w1, log_w2 = jnp.log(0.5), jnp.log(0.5)

    def target_log_prob(theta):
        """Log probability of 50/50 mixture of normalized von Mises distributions.

        Uses properly normalized von Mises components so the plotted density
        matches the training data (which is sampled with equal weights).
        """
        # Normalized log probs for each component
        lp1 = von_mises_log_prob_normalized(theta, mu1, kappa1) + log_w1
        lp2 = von_mises_log_prob_normalized(theta, mu2, kappa2) + log_w2
        # Log-sum-exp for mixture
        return jax.scipy.special.logsumexp(jnp.stack([lp1, lp2]), axis=0)

    # Generate training data
    key = jr.key(42)
    key, subkey1, subkey2 = jr.split(key, 3)
    n_samples = 5000

    # Sample from mixture
    samples1 = sample_von_mises(subkey1, mu1, kappa1, (n_samples // 2,))
    samples2 = sample_von_mises(subkey2, mu2, kappa2, (n_samples // 2,))
    train_data = jnp.concatenate([samples1, samples2])[:, None]  # Shape (N, 1)

    # Create and train circular flow
    key, subkey = jr.split(key)
    base_dist = TorusUniform(1)

    flow = circular_masked_autoregressive_flow(
        subkey,
        base_dist=base_dist,
        flow_layers=4,
        nn_width=32,
        nn_depth=2,
        num_bins=16,
    )

    # Train
    print("Training 1D circular flow...")
    key, subkey = jr.split(key)
    flow, losses = fit_to_data(
        subkey,
        flow,
        train_data,
        max_epochs=500,
        max_patience=50,
        batch_size=256,
        learning_rate=2e-4,
        show_progress=True,
    )
    print(f"Final loss: {losses['train'][-1]:.4f}")

    # Generate samples from trained flow
    key, subkey = jr.split(key)
    flow_samples = flow.sample(subkey, (5000,))

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Grid for density evaluation
    theta_grid = jnp.linspace(0, TWO_PI, 200)

    # Target density (already properly normalized)
    target_log_probs = jax.vmap(target_log_prob)(theta_grid)
    target_density = jnp.exp(target_log_probs)

    # Learned density
    learned_log_prob = jax.vmap(lambda x: flow.log_prob(x[None]))(theta_grid)
    learned_density = jnp.exp(learned_log_prob)

    # Plot 1: Target vs Learned density
    ax = axes[0]
    ax.plot(theta_grid, target_density, 'b-', lw=2, label='Target')
    ax.plot(theta_grid, learned_density, 'r--', lw=2, label='Learned')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(TWO_PI, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Density')
    ax.set_title('Target vs Learned Density')
    ax.legend()
    ax.set_xlim(0, TWO_PI)

    # Plot 2: Histograms showing wrap-around
    ax = axes[1]
    # Extend data beyond boundaries to show wrap-around
    extended_train = np.concatenate([
        np.array(train_data.flatten()) - TWO_PI,
        np.array(train_data.flatten()),
        np.array(train_data.flatten()) + TWO_PI
    ])
    extended_flow = np.concatenate([
        np.array(flow_samples.flatten()) - TWO_PI,
        np.array(flow_samples.flatten()),
        np.array(flow_samples.flatten()) + TWO_PI
    ])

    ax.hist(extended_train, bins=80, density=True, alpha=0.5,
            label='Target samples', color='blue', range=(-1, TWO_PI + 1))
    ax.hist(extended_flow, bins=80, density=True, alpha=0.5,
            label='Flow samples', color='red', range=(-1, TWO_PI + 1))
    ax.axvline(0, color='black', linestyle='-', lw=2, label='Boundary')
    ax.axvline(TWO_PI, color='black', linestyle='-', lw=2)
    ax.axvspan(-1, 0, alpha=0.1, color='gray')
    ax.axvspan(TWO_PI, TWO_PI + 1, alpha=0.1, color='gray')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Density')
    ax.set_title('Continuity at Boundary (0 = 2π)')
    ax.legend()
    ax.set_xlim(-1, TWO_PI + 1)

    # Plot 3: Polar plot
    ax = axes[2]
    ax = fig.add_subplot(1, 3, 3, projection='polar')
    ax.hist(np.array(train_data.flatten()), bins=50, density=True,
            alpha=0.5, label='Target', color='blue')
    ax.hist(np.array(flow_samples.flatten()), bins=50, density=True,
            alpha=0.5, label='Flow', color='red')
    ax.set_title('Polar View')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('circular_1d_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: circular_1d_demo.png")

    return flow, train_data


# =============================================================================
# Example 2: 2D Torus - Correlated distribution wrapping in both dimensions
# =============================================================================

def demo_2d_torus():
    """2D demonstration: correlated distribution on the torus T^2."""
    print("\n" + "=" * 60)
    print("Example 2: 2D Torus (T^2) - Correlated Distribution")
    print("=" * 60)

    # Target: Correlated von Mises (as in paper Table 2)
    # p(θ1, θ2) ∝ exp[β * cos(θ1 + θ2 - φ)]
    # This creates a diagonal correlation that wraps around both boundaries
    phi = 1.5
    beta = 4.0

    def target_log_prob(theta):
        """Log probability of correlated von Mises."""
        theta1, theta2 = theta[0], theta[1]
        return beta * jnp.cos(theta1 + theta2 - phi)

    # Generate training data via rejection sampling
    key = jr.key(123)
    n_samples = 8000
    samples = []

    print("Generating training data via rejection sampling...")
    while len(samples) < n_samples:
        key, subkey1, subkey2 = jr.split(key, 3)
        # Propose from uniform on torus
        theta = jr.uniform(subkey1, (n_samples * 5, 2), minval=0, maxval=TWO_PI)
        # Accept with probability proportional to target
        log_u = jnp.log(jr.uniform(subkey2, (n_samples * 5,)))
        log_accept = jax.vmap(target_log_prob)(theta) - beta  # Normalize max to 0
        accepted = theta[log_u < log_accept]
        samples.extend(accepted.tolist())

    train_data = jnp.array(samples[:n_samples])
    print(f"Generated {len(train_data)} samples")

    # Create and train circular flow
    key, subkey = jr.split(key)
    base_dist = TorusUniform(2)

    flow = circular_coupling_flow(
        subkey,
        base_dist=base_dist,
        flow_layers=6,
        nn_width=64,
        nn_depth=2,
        num_bins=12,
    )

    # Train
    print("Training 2D circular flow...")
    key, subkey = jr.split(key)
    flow, losses = fit_to_data(
        subkey,
        flow,
        train_data,
        max_epochs=800,
        max_patience=50,
        batch_size=256,
        learning_rate=2e-4,
        show_progress=True,
    )
    print(f"Final loss: {losses['train'][-1]:.4f}")

    # Generate samples from trained flow
    key, subkey = jr.split(key)
    flow_samples = flow.sample(subkey, (5000,))

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Standard view [0, 2π]²
    # Target samples
    ax = axes[0, 0]
    ax.scatter(train_data[:, 0], train_data[:, 1], alpha=0.3, s=1, c='blue')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Target Samples')
    ax.set_xlim(0, TWO_PI)
    ax.set_ylim(0, TWO_PI)
    ax.set_aspect('equal')

    # Flow samples
    ax = axes[0, 1]
    ax.scatter(flow_samples[:, 0], flow_samples[:, 1], alpha=0.3, s=1, c='red')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Flow Samples')
    ax.set_xlim(0, TWO_PI)
    ax.set_ylim(0, TWO_PI)
    ax.set_aspect('equal')

    # Density comparison (heatmap)
    ax = axes[0, 2]
    grid_size = 50
    theta1_grid = jnp.linspace(0, TWO_PI, grid_size)
    theta2_grid = jnp.linspace(0, TWO_PI, grid_size)
    T1, T2 = jnp.meshgrid(theta1_grid, theta2_grid)
    grid_points = jnp.stack([T1.flatten(), T2.flatten()], axis=-1)

    # Target density
    target_log_probs = jax.vmap(target_log_prob)(grid_points)
    target_density = jnp.exp(target_log_probs - target_log_probs.max())
    target_density = target_density.reshape(grid_size, grid_size)

    im = ax.imshow(target_density, origin='lower', extent=[0, TWO_PI, 0, TWO_PI],
                   aspect='equal', cmap='viridis')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Target Density')
    plt.colorbar(im, ax=ax)

    # Row 2: Extended view showing wrap-around
    # Extended target samples (replicated to show periodicity)
    ax = axes[1, 0]
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            offset = jnp.array([di * TWO_PI, dj * TWO_PI])
            shifted = train_data + offset
            alpha = 0.3 if (di == 0 and dj == 0) else 0.1
            color = 'blue' if (di == 0 and dj == 0) else 'lightblue'
            ax.scatter(shifted[:, 0], shifted[:, 1], alpha=alpha, s=1, c=color)

    # Draw boundary lines
    for i in range(4):
        ax.axhline(i * TWO_PI, color='black', linestyle='--', alpha=0.5)
        ax.axvline(i * TWO_PI, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Target - Extended (showing periodicity)')
    ax.set_xlim(-TWO_PI/2, 2.5 * TWO_PI)
    ax.set_ylim(-TWO_PI/2, 2.5 * TWO_PI)
    ax.set_aspect('equal')

    # Extended flow samples
    ax = axes[1, 1]
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            offset = jnp.array([di * TWO_PI, dj * TWO_PI])
            shifted = flow_samples + offset
            alpha = 0.3 if (di == 0 and dj == 0) else 0.1
            color = 'red' if (di == 0 and dj == 0) else 'lightcoral'
            ax.scatter(shifted[:, 0], shifted[:, 1], alpha=alpha, s=1, c=color)

    for i in range(4):
        ax.axhline(i * TWO_PI, color='black', linestyle='--', alpha=0.5)
        ax.axvline(i * TWO_PI, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Flow - Extended (showing periodicity)')
    ax.set_xlim(-TWO_PI/2, 2.5 * TWO_PI)
    ax.set_ylim(-TWO_PI/2, 2.5 * TWO_PI)
    ax.set_aspect('equal')

    # Learned density heatmap
    ax = axes[1, 2]
    learned_log_probs = jax.vmap(flow.log_prob)(grid_points)
    learned_density = jnp.exp(learned_log_probs - learned_log_probs.max())
    learned_density = learned_density.reshape(grid_size, grid_size)

    im = ax.imshow(learned_density, origin='lower', extent=[0, TWO_PI, 0, TWO_PI],
                   aspect='equal', cmap='viridis')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Learned Density')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('circular_2d_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: circular_2d_demo.png")

    return flow, train_data


# =============================================================================
# Example 3: 2D Torus - Multimodal distribution (as in paper Figure 3)
# =============================================================================

def demo_2d_multimodal():
    """2D demonstration: multimodal distribution on T^2 (3 modes)."""
    print("\n" + "=" * 60)
    print("Example 3: 2D Torus (T^2) - Multimodal Distribution")
    print("=" * 60)

    # Target: Mixture of 3 bivariate von Mises (as in paper)
    # Modes placed to demonstrate wrap-around at boundaries
    # Using similar kappa values so modes have similar visual prominence
    modes = [
        (0.3, 5.8, 5.0),   # Mode near (0, 2π) corner - wraps around!
        (2.0, 3.0, 5.0),   # Mode in center (changed from 6.0 to 5.0)
        (5.5, 1.2, 5.0),   # Mode near (2π, 0) corner (changed from 4.0 to 5.0)
    ]
    n_modes = len(modes)
    log_mixture_weight = jnp.log(1.0 / n_modes)  # Equal weights

    def bivariate_von_mises_log_prob_normalized(theta, mu1, mu2, kappa):
        """Normalized log prob of independent bivariate von Mises.

        For independent von Mises in each dimension:
        log p(θ1, θ2) = κ(cos(θ1-μ1) + cos(θ2-μ2)) - 2*log(2π) - 2*log(I_0(κ))
        """
        theta1, theta2 = theta[0], theta[1]
        log_partition = 2 * (jnp.log(TWO_PI) + log_bessel_i0(kappa))
        return kappa * (jnp.cos(theta1 - mu1) + jnp.cos(theta2 - mu2)) - log_partition

    def target_log_prob(theta):
        """Log probability of equal-weight mixture of normalized bivariate von Mises."""
        log_probs = []
        for mu1, mu2, kappa in modes:
            lp = bivariate_von_mises_log_prob_normalized(theta, mu1, mu2, kappa)
            log_probs.append(lp + log_mixture_weight)
        return jax.scipy.special.logsumexp(jnp.stack(log_probs))

    # Generate training data by sampling equally from each mode
    key = jr.key(456)
    n_samples = 10000
    samples_per_mode = n_samples // n_modes

    print("Generating training data (equal samples per mode)...")
    all_samples = []
    for i, (mu1, mu2, kappa) in enumerate(modes):
        key, subkey = jr.split(key)
        # Sample from bivariate independent von Mises via rejection sampling
        mode_samples = []
        while len(mode_samples) < samples_per_mode:
            key, subkey1, subkey2 = jr.split(key, 3)
            theta = jr.uniform(subkey1, (samples_per_mode * 10, 2), minval=0, maxval=TWO_PI)
            log_u = jnp.log(jr.uniform(subkey2, (samples_per_mode * 10,)))
            # Accept prob for independent von Mises: exp(κ(cos(θ1-μ1) + cos(θ2-μ2) - 2))
            log_accept = kappa * (jnp.cos(theta[:, 0] - mu1) + jnp.cos(theta[:, 1] - mu2) - 2)
            accepted = theta[log_u < log_accept]
            mode_samples.extend(accepted.tolist())
        all_samples.extend(mode_samples[:samples_per_mode])

    train_data = jnp.array(all_samples)

    # Train flow
    key, subkey = jr.split(key)
    base_dist = TorusUniform(2)

    flow = circular_masked_autoregressive_flow(
        subkey,
        base_dist=base_dist,
        flow_layers=8,
        nn_width=64,
        nn_depth=2,
        num_bins=16,
    )

    print("Training...")
    key, subkey = jr.split(key)
    flow, losses = fit_to_data(
        subkey,
        flow,
        train_data,
        max_epochs=1000,
        max_patience=50,
        batch_size=256,
        learning_rate=2e-4,
        show_progress=True,
    )
    print(f"Final loss: {losses['train'][-1]:.4f}")

    # Sample from flow
    key, subkey = jr.split(key)
    flow_samples = flow.sample(subkey, (5000,))

    # Plotting - emphasize wrap-around
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Density heatmaps
    grid_size = 80
    theta1_grid = jnp.linspace(0, TWO_PI, grid_size)
    theta2_grid = jnp.linspace(0, TWO_PI, grid_size)
    T1, T2 = jnp.meshgrid(theta1_grid, theta2_grid)
    grid_points = jnp.stack([T1.flatten(), T2.flatten()], axis=-1)

    target_log_probs = jax.vmap(target_log_prob)(grid_points)
    target_density = jnp.exp(target_log_probs - target_log_probs.max())

    learned_log_probs = jax.vmap(flow.log_prob)(grid_points)
    learned_density = jnp.exp(learned_log_probs - learned_log_probs.max())

    # Target density
    ax = axes[0, 0]
    im = ax.imshow(target_density.reshape(grid_size, grid_size),
                   origin='lower', extent=[0, TWO_PI, 0, TWO_PI],
                   aspect='equal', cmap='hot')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Target Density')
    plt.colorbar(im, ax=ax)

    # Learned density
    ax = axes[0, 1]
    im = ax.imshow(learned_density.reshape(grid_size, grid_size),
                   origin='lower', extent=[0, TWO_PI, 0, TWO_PI],
                   aspect='equal', cmap='hot')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Learned Density')
    plt.colorbar(im, ax=ax)

    # Extended view showing wrap-around (target)
    ax = axes[1, 0]
    # Create extended density by tiling
    extended_target = np.tile(np.array(target_density.reshape(grid_size, grid_size)), (3, 3))
    extent = [-TWO_PI, 2*TWO_PI, -TWO_PI, 2*TWO_PI]
    ax.imshow(extended_target, origin='lower', extent=extent, aspect='equal', cmap='hot')

    # Highlight the fundamental domain
    rect = plt.Rectangle((0, 0), TWO_PI, TWO_PI, fill=False,
                          edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Target - Extended (white box = fundamental domain)')
    ax.set_xlim(-TWO_PI/2, 2.5*TWO_PI)
    ax.set_ylim(-TWO_PI/2, 2.5*TWO_PI)

    # Extended view (learned)
    ax = axes[1, 1]
    extended_learned = np.tile(np.array(learned_density.reshape(grid_size, grid_size)), (3, 3))
    ax.imshow(extended_learned, origin='lower', extent=extent, aspect='equal', cmap='hot')
    rect = plt.Rectangle((0, 0), TWO_PI, TWO_PI, fill=False,
                          edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title('Learned - Extended (white box = fundamental domain)')
    ax.set_xlim(-TWO_PI/2, 2.5*TWO_PI)
    ax.set_ylim(-TWO_PI/2, 2.5*TWO_PI)

    plt.tight_layout()
    plt.savefig('circular_2d_multimodal.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: circular_2d_multimodal.png")

    return flow, train_data


if __name__ == "__main__":
    print("Circular Flows Demonstration")
    print("Based on arXiv:2002.02428 'Normalizing Flows on Tori and Spheres'")
    print()

    # Run demonstrations
    demo_1d_circle()
    demo_2d_torus()
    demo_2d_multimodal()

    print("\nAll demonstrations complete!")
