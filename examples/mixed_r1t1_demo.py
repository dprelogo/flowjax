"""Demonstration of mixed topology normalizing flows on R x T (cylinder).

This script demonstrates the mixed topology flow implementation on:
1. Correlated distribution: Linear-circular correlation that wraps at the boundary
2. Multimodal distribution: Multiple modes spanning both topologies

Inspired by arXiv:2002.02428 "Normalizing Flows on Tori and Spheres"
Extended to mixed R^N x T^M topologies.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import optax
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from flowjax.distributions import MixedBase
from flowjax.flows import mixed_masked_autoregressive_flow
from flowjax.train import fit_to_data

# Set up for reproducibility
jax.config.update("jax_enable_x64", True)
plt.style.use('seaborn-v0_8-whitegrid')
TWO_PI = 2 * jnp.pi


# =============================================================================
# Helper Functions
# =============================================================================

def log_bessel_i0(kappa):
    """Log of modified Bessel function I_0(kappa).

    Uses jax.scipy.special.i0e (exponentially scaled) for numerical stability:
    I_0(x) = i0e(x) * exp(|x|), so log(I_0(x)) = log(i0e(x)) + |x|
    """
    return jnp.log(jax.scipy.special.i0e(kappa)) + jnp.abs(kappa)


def von_mises_log_prob_normalized(theta, mu, kappa):
    """Normalized log probability of von Mises distribution."""
    return kappa * jnp.cos(theta - mu) - jnp.log(TWO_PI) - log_bessel_i0(kappa)


def gaussian_log_prob_normalized(x, mu, sigma):
    """Normalized log probability of Gaussian distribution."""
    return -0.5 * ((x - mu) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)


# =============================================================================
# Example 1: Correlated Distribution on R x T (Cylinder)
# =============================================================================

def demo_correlated_cylinder():
    """2D demonstration: correlated distribution on the cylinder R x T.

    The linear dimension r depends on the circular dimension theta,
    creating a "wave" pattern that wraps around the cylinder.
    """
    print("=" * 60)
    print("Example 1: R x T Cylinder - Correlated Distribution")
    print("=" * 60)

    # Target: p(theta, r) = p(r | theta) * p(theta)
    # theta ~ von Mises(mu=pi, kappa=2)
    # r | theta ~ Normal(mu = A*cos(theta), sigma=0.5)
    # This creates a sinusoidal correlation that wraps at the boundary
    #
    # IMPORTANT: We use [theta, r] ordering so that r can depend on theta
    # in the autoregressive flow (each dim depends on previous dims only).

    theta_mu, theta_kappa = jnp.pi, 2.0
    r_amplitude = 1.5
    r_sigma = 0.4

    def target_log_prob(x):
        """Log probability of correlated T x R distribution."""
        theta, r = x[0], x[1]  # theta first, then r
        # p(theta) - von Mises
        log_p_theta = von_mises_log_prob_normalized(theta, theta_mu, theta_kappa)
        # p(r | theta) - Gaussian with theta-dependent mean
        r_mean = r_amplitude * jnp.cos(theta)
        log_p_r = gaussian_log_prob_normalized(r, r_mean, r_sigma)
        return log_p_theta + log_p_r

    # Generate training data via rejection sampling
    key = jr.key(42)
    n_samples = 8000
    samples = []

    print("Generating training data via rejection sampling...")

    # Estimate max log prob for rejection sampling
    # Using [theta, r] ordering
    test_grid = jnp.stack([
        jnp.tile(jnp.linspace(0, TWO_PI, 50), 50),  # theta
        jnp.linspace(-3, 3, 50).repeat(50),         # r
    ], axis=1)
    test_log_probs = jax.vmap(target_log_prob)(test_grid)
    max_log_prob = jnp.max(test_log_probs) + 0.1

    while len(samples) < n_samples:
        key, subkey1, subkey2 = jr.split(key, 3)
        # Propose: theta ~ Uniform(0, 2pi), r ~ Uniform(-3, 3)
        theta_prop = jr.uniform(subkey1, (n_samples * 5,), minval=0, maxval=TWO_PI)
        r_prop = jr.uniform(subkey1, (n_samples * 5,), minval=-3, maxval=3)
        proposals = jnp.stack([theta_prop, r_prop], axis=1)  # [theta, r] ordering

        log_u = jnp.log(jr.uniform(subkey2, (n_samples * 5,)))
        log_accept = jax.vmap(target_log_prob)(proposals) - max_log_prob
        accepted = proposals[log_u < log_accept]
        samples.extend(accepted.tolist())

    train_data = jnp.array(samples[:n_samples])
    print(f"Generated {len(train_data)} samples")

    # Create and train mixed topology flow
    # Using [theta, r] ordering so r can depend on theta in autoregressive flow
    key, subkey = jr.split(key)
    is_circular = jnp.array([True, False])  # [T, R] - theta first, then r
    base_dist = MixedBase(is_circular)

    flow = mixed_masked_autoregressive_flow(
        subkey,
        base_dist=base_dist,
        is_circular=is_circular,
        linear_bounds=(-5.0, 5.0),
        flow_layers=8,
        nn_width=64,
        nn_depth=2,
    )

    # Train with lower learning rate for stability
    print("Training mixed topology flow...")
    key, subkey = jr.split(key)
    flow, losses = fit_to_data(
        subkey,
        flow,
        train_data,
        max_epochs=1000,
        max_patience=80,
        batch_size=256,
        learning_rate=1e-4,
        show_progress=True,
    )
    print(f"Final loss: {losses['train'][-1]:.4f}")

    # Generate samples from trained flow
    key, subkey = jr.split(key)
    flow_samples = flow.sample(subkey, (5000,))

    # Plotting - data is now [theta, r] ordering
    fig = plt.figure(figsize=(18, 12))

    # Row 1: Standard 2D views
    # Target samples - theta on x-axis (index 0), r on y-axis (index 1)
    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(train_data[:, 0], train_data[:, 1], alpha=0.3, s=2, c='blue')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Target Samples', fontsize=12, fontweight='bold')
    ax.set_xlim(0, TWO_PI)
    ax.set_xticks([0, jnp.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

    # Flow samples
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(flow_samples[:, 0], flow_samples[:, 1], alpha=0.3, s=2, c='red')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Flow Samples', fontsize=12, fontweight='bold')
    ax.set_xlim(0, TWO_PI)
    ax.set_xticks([0, jnp.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

    # Density comparison (heatmap)
    ax = fig.add_subplot(2, 3, 3)
    grid_size = 60
    theta_grid = jnp.linspace(0, TWO_PI, grid_size)
    r_grid = jnp.linspace(-3, 3, grid_size)
    T, R = jnp.meshgrid(theta_grid, r_grid)
    grid_points = jnp.stack([T.flatten(), R.flatten()], axis=-1)  # [theta, r] ordering

    target_log_probs = jax.vmap(target_log_prob)(grid_points)
    target_density = jnp.exp(target_log_probs - target_log_probs.max())
    target_density = target_density.reshape(grid_size, grid_size)

    im = ax.imshow(target_density.T, origin='lower', extent=[0, TWO_PI, -3, 3],
                   aspect='auto', cmap='viridis')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Target Density', fontsize=12, fontweight='bold')
    ax.set_xticks([0, jnp.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.colorbar(im, ax=ax)

    # Row 2: Extended views and 3D
    # Extended target samples (showing periodicity in theta)
    ax = fig.add_subplot(2, 3, 4)
    for di in [-1, 0, 1]:
        offset = jnp.array([di * TWO_PI, 0])  # offset theta (index 0)
        shifted = train_data + offset
        alpha = 0.3 if di == 0 else 0.1
        color = 'blue' if di == 0 else 'lightblue'
        ax.scatter(shifted[:, 0], shifted[:, 1], alpha=alpha, s=2, c=color)

    for i in range(-1, 3):
        ax.axvline(i * TWO_PI, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Target - Extended (showing periodicity)', fontsize=12, fontweight='bold')
    ax.set_xlim(-TWO_PI/2, 2.5 * TWO_PI)

    # Extended flow samples
    ax = fig.add_subplot(2, 3, 5)
    for di in [-1, 0, 1]:
        offset = jnp.array([di * TWO_PI, 0])  # offset theta (index 0)
        shifted = flow_samples + offset
        alpha = 0.3 if di == 0 else 0.1
        color = 'red' if di == 0 else 'lightcoral'
        ax.scatter(shifted[:, 0], shifted[:, 1], alpha=alpha, s=2, c=color)

    for i in range(-1, 3):
        ax.axvline(i * TWO_PI, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Flow - Extended (showing periodicity)', fontsize=12, fontweight='bold')
    ax.set_xlim(-TWO_PI/2, 2.5 * TWO_PI)

    # Learned density heatmap
    ax = fig.add_subplot(2, 3, 6)
    learned_log_probs = jax.vmap(flow.log_prob)(grid_points)
    learned_density = jnp.exp(learned_log_probs - learned_log_probs.max())
    learned_density = learned_density.reshape(grid_size, grid_size)

    im = ax.imshow(learned_density.T, origin='lower', extent=[0, TWO_PI, -3, 3],
                   aspect='auto', cmap='viridis')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Learned Density', fontsize=12, fontweight='bold')
    ax.set_xticks([0, jnp.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('mixed_r1t1_correlated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: mixed_r1t1_correlated.png")

    # Additional: 3D cylinder visualization
    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # Map to cylinder: x = cos(theta), y = sin(theta), z = r
    # Data ordering: [theta, r] -> theta at index 0, r at index 1
    t_theta, t_r = np.array(train_data[:, 0]), np.array(train_data[:, 1])
    idx = np.random.choice(len(t_theta), min(2000, len(t_theta)), replace=False)
    ax.scatter(np.cos(t_theta[idx]), np.sin(t_theta[idx]), t_r[idx],
               alpha=0.4, s=3, c='blue', label='Target')
    ax.set_xlabel(r'$\cos(\theta)$')
    ax.set_ylabel(r'$\sin(\theta)$')
    ax.set_zlabel(r'$r$')
    ax.set_title('Target on Cylinder', fontsize=12, fontweight='bold')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    f_theta, f_r = np.array(flow_samples[:, 0]), np.array(flow_samples[:, 1])
    idx = np.random.choice(len(f_theta), min(2000, len(f_theta)), replace=False)
    ax.scatter(np.cos(f_theta[idx]), np.sin(f_theta[idx]), f_r[idx],
               alpha=0.4, s=3, c='red', label='Flow')
    ax.set_xlabel(r'$\cos(\theta)$')
    ax.set_ylabel(r'$\sin(\theta)$')
    ax.set_zlabel(r'$r$')
    ax.set_title('Flow on Cylinder', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('mixed_r1t1_correlated_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: mixed_r1t1_correlated_3d.png")

    # Save training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses['train'], label='Train', alpha=0.8)
    ax.plot(losses['val'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=11)
    ax.set_title('Training Progress - Correlated Cylinder', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mixed_r1t1_correlated_loss.png', dpi=150)
    plt.close()
    print("Saved: mixed_r1t1_correlated_loss.png")

    return flow, train_data


# =============================================================================
# Example 2: Multimodal Distribution on R x T (Cylinder)
# =============================================================================

def demo_multimodal_cylinder():
    """2D demonstration: multimodal distribution on the cylinder R x T.

    Multiple modes at different positions, with one mode near the
    theta boundary to demonstrate wrap-around handling.
    """
    print("\n" + "=" * 60)
    print("Example 2: R x T Cylinder - Multimodal Distribution")
    print("=" * 60)

    # Target: Mixture of 3 independent Gaussian x von Mises distributions
    # Mode 1: near theta=0 boundary (wraps around!)
    # Mode 2: in the center
    # Mode 3: near theta=2pi boundary
    modes = [
        # (r_mu, r_sigma, theta_mu, theta_kappa)
        (1.5, 0.3, 0.3, 8.0),     # Mode near theta=0, positive r
        (-0.5, 0.4, 3.2, 6.0),    # Mode in center, negative r
        (0.8, 0.35, 5.8, 7.0),    # Mode near theta=2pi, positive r
    ]
    n_modes = len(modes)
    log_mixture_weight = jnp.log(1.0 / n_modes)

    def target_log_prob(x):
        """Log probability of mixture of Gaussian x von Mises."""
        r, theta = x[0], x[1]
        log_probs = []
        for r_mu, r_sigma, theta_mu, theta_kappa in modes:
            log_p_r = gaussian_log_prob_normalized(r, r_mu, r_sigma)
            log_p_theta = von_mises_log_prob_normalized(theta, theta_mu, theta_kappa)
            log_probs.append(log_p_r + log_p_theta + log_mixture_weight)
        return jax.scipy.special.logsumexp(jnp.stack(log_probs))

    # Generate training data by sampling equally from each mode
    key = jr.key(123)
    n_samples = 10000
    samples_per_mode = n_samples // n_modes

    print("Generating training data (equal samples per mode)...")
    all_samples = []

    for i, (r_mu, r_sigma, theta_mu, theta_kappa) in enumerate(modes):
        key, subkey = jr.split(key)
        mode_samples = []

        while len(mode_samples) < samples_per_mode:
            key, subkey1, subkey2, subkey3 = jr.split(key, 4)
            # Sample r from Gaussian (use rejection sampling for bounded proposal)
            r_prop = jr.normal(subkey1, (samples_per_mode * 3,)) * r_sigma + r_mu
            # Sample theta via rejection sampling from von Mises
            theta_prop = jr.uniform(subkey2, (samples_per_mode * 3,), minval=0, maxval=TWO_PI)
            log_u = jnp.log(jr.uniform(subkey3, (samples_per_mode * 3,)))
            log_accept = theta_kappa * (jnp.cos(theta_prop - theta_mu) - 1)

            # Accept theta samples
            theta_accepted = theta_prop[log_u < log_accept]
            r_accepted = r_prop[:len(theta_accepted)]

            for r, theta in zip(r_accepted.tolist(), theta_accepted.tolist()):
                if len(mode_samples) < samples_per_mode:
                    mode_samples.append([r, theta])

        all_samples.extend(mode_samples[:samples_per_mode])

    train_data = jnp.array(all_samples)
    print(f"Generated {len(train_data)} samples")

    # Create and train mixed topology flow
    key, subkey = jr.split(key)
    is_circular = jnp.array([False, True])  # [R, T]
    base_dist = MixedBase(is_circular)

    flow = mixed_masked_autoregressive_flow(
        subkey,
        base_dist=base_dist,
        is_circular=is_circular,
        linear_bounds=(-5.0, 5.0),
        flow_layers=8,
        nn_width=64,
        nn_depth=2,
    )

    # Train
    print("Training mixed topology flow...")
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

    # Generate samples from trained flow
    key, subkey = jr.split(key)
    flow_samples = flow.sample(subkey, (5000,))

    # Plotting - emphasize wrap-around
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Density heatmaps
    grid_size = 80
    r_grid = jnp.linspace(-2, 3, grid_size)
    theta_grid = jnp.linspace(0, TWO_PI, grid_size)
    R, T = jnp.meshgrid(r_grid, theta_grid)
    grid_points = jnp.stack([R.flatten(), T.flatten()], axis=-1)

    target_log_probs = jax.vmap(target_log_prob)(grid_points)
    target_density = jnp.exp(target_log_probs - target_log_probs.max())

    learned_log_probs = jax.vmap(flow.log_prob)(grid_points)
    learned_density = jnp.exp(learned_log_probs - learned_log_probs.max())

    # Target density
    ax = axes[0, 0]
    im = ax.imshow(target_density.reshape(grid_size, grid_size).T,
                   origin='lower', extent=[0, TWO_PI, -2, 3],
                   aspect='auto', cmap='hot')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Target Density', fontsize=12, fontweight='bold')
    ax.set_xticks([0, jnp.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.colorbar(im, ax=ax)

    # Learned density
    ax = axes[0, 1]
    im = ax.imshow(learned_density.reshape(grid_size, grid_size).T,
                   origin='lower', extent=[0, TWO_PI, -2, 3],
                   aspect='auto', cmap='hot')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Learned Density', fontsize=12, fontweight='bold')
    ax.set_xticks([0, jnp.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.colorbar(im, ax=ax)

    # Extended view showing wrap-around (target)
    ax = axes[1, 0]
    # Create extended density by tiling in theta direction only
    target_2d = np.array(target_density.reshape(grid_size, grid_size).T)
    extended_target = np.tile(target_2d, (1, 3))
    extent = [-TWO_PI, 2*TWO_PI, -2, 3]
    ax.imshow(extended_target, origin='lower', extent=extent, aspect='auto', cmap='hot')

    # Highlight the fundamental domain
    rect = plt.Rectangle((0, -2), TWO_PI, 5, fill=False,
                          edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Target - Extended (white box = fundamental domain)', fontsize=12, fontweight='bold')
    ax.set_xlim(-TWO_PI/2, 2.5*TWO_PI)

    # Extended view (learned)
    ax = axes[1, 1]
    learned_2d = np.array(learned_density.reshape(grid_size, grid_size).T)
    extended_learned = np.tile(learned_2d, (1, 3))
    ax.imshow(extended_learned, origin='lower', extent=extent, aspect='auto', cmap='hot')
    rect = plt.Rectangle((0, -2), TWO_PI, 5, fill=False,
                          edgecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Learned - Extended (white box = fundamental domain)', fontsize=12, fontweight='bold')
    ax.set_xlim(-TWO_PI/2, 2.5*TWO_PI)

    plt.tight_layout()
    plt.savefig('mixed_r1t1_multimodal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: mixed_r1t1_multimodal.png")

    # Additional: 3D cylinder visualization
    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    t_theta, t_r = np.array(train_data[:, 1]), np.array(train_data[:, 0])
    idx = np.random.choice(len(t_theta), min(2000, len(t_theta)), replace=False)
    ax.scatter(np.cos(t_theta[idx]), np.sin(t_theta[idx]), t_r[idx],
               alpha=0.4, s=3, c='blue', label='Target')
    ax.set_xlabel(r'$\cos(\theta)$')
    ax.set_ylabel(r'$\sin(\theta)$')
    ax.set_zlabel(r'$r$')
    ax.set_title('Target on Cylinder', fontsize=12, fontweight='bold')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    f_theta, f_r = np.array(flow_samples[:, 1]), np.array(flow_samples[:, 0])
    idx = np.random.choice(len(f_theta), min(2000, len(f_theta)), replace=False)
    ax.scatter(np.cos(f_theta[idx]), np.sin(f_theta[idx]), f_r[idx],
               alpha=0.4, s=3, c='red', label='Flow')
    ax.set_xlabel(r'$\cos(\theta)$')
    ax.set_ylabel(r'$\sin(\theta)$')
    ax.set_zlabel(r'$r$')
    ax.set_title('Flow on Cylinder', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('mixed_r1t1_multimodal_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: mixed_r1t1_multimodal_3d.png")

    # Save training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses['train'], label='Train', alpha=0.8)
    ax.plot(losses['val'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=11)
    ax.set_title('Training Progress - Multimodal Cylinder', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mixed_r1t1_multimodal_loss.png', dpi=150)
    plt.close()
    print("Saved: mixed_r1t1_multimodal_loss.png")

    return flow, train_data


# =============================================================================
# Example 3: Density Comparison with KDE Overlay
# =============================================================================

def demo_density_comparison():
    """2D demonstration with detailed density comparison plots.

    Uses seaborn KDE contours to show target vs learned density overlap.
    """
    print("\n" + "=" * 60)
    print("Example 3: R x T Cylinder - Density Comparison")
    print("=" * 60)

    # Simple but interesting target: Gaussian ring at r=1, uniform-ish in theta
    # but with correlation: higher r -> theta concentrated near pi
    def target_log_prob(x):
        """Log probability with r-theta correlation."""
        r, theta = x[0], x[1]
        # p(r) ~ Normal(1, 0.3)
        log_p_r = gaussian_log_prob_normalized(r, 1.0, 0.3)
        # p(theta | r) ~ von Mises with concentration depending on r
        # Higher r -> more concentrated at pi
        kappa = 1.0 + 2.0 * jnp.clip(r, 0, 2)
        log_p_theta = von_mises_log_prob_normalized(theta, jnp.pi, kappa)
        return log_p_r + log_p_theta

    # Generate training data
    key = jr.key(789)
    n_samples = 8000
    samples = []

    print("Generating training data...")

    # Estimate max for rejection sampling
    test_grid = jnp.stack([
        jnp.linspace(-1, 3, 40).repeat(40),
        jnp.tile(jnp.linspace(0, TWO_PI, 40), 40)
    ], axis=1)
    test_log_probs = jax.vmap(target_log_prob)(test_grid)
    max_log_prob = jnp.max(test_log_probs) + 0.1

    while len(samples) < n_samples:
        key, subkey1, subkey2 = jr.split(key, 3)
        r_prop = jr.uniform(subkey1, (n_samples * 5,), minval=-1, maxval=3)
        theta_prop = jr.uniform(subkey1, (n_samples * 5,), minval=0, maxval=TWO_PI)
        proposals = jnp.stack([r_prop, theta_prop], axis=1)

        log_u = jnp.log(jr.uniform(subkey2, (n_samples * 5,)))
        log_accept = jax.vmap(target_log_prob)(proposals) - max_log_prob
        accepted = proposals[log_u < log_accept]
        samples.extend(accepted.tolist())

    train_data = jnp.array(samples[:n_samples])

    # Train flow
    key, subkey = jr.split(key)
    is_circular = jnp.array([False, True])
    base_dist = MixedBase(is_circular)

    flow = mixed_masked_autoregressive_flow(
        subkey,
        base_dist=base_dist,
        is_circular=is_circular,
        linear_bounds=(-5.0, 5.0),
        flow_layers=8,
        nn_width=64,
        nn_depth=2,
    )

    print("Training...")
    key, subkey = jr.split(key)
    flow, losses = fit_to_data(
        subkey,
        flow,
        train_data,
        max_epochs=800,
        max_patience=80,
        batch_size=256,
        learning_rate=1e-4,
        show_progress=True,
    )
    print(f"Final loss: {losses['train'][-1]:.4f}")

    # Generate flow samples
    key, subkey = jr.split(key)
    flow_samples = flow.sample(subkey, (5000,))

    # Create comprehensive density comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    t_data = np.array(train_data)
    f_data = np.array(flow_samples)

    # Row 1: 2D views
    # Target scatter with KDE
    ax = axes[0, 0]
    ax.scatter(t_data[:, 1], t_data[:, 0], alpha=0.2, s=2, c='blue')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Target Samples', fontsize=12, fontweight='bold')
    ax.set_xlim(0, TWO_PI)
    ax.set_xticks([0, np.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

    # Flow scatter with KDE
    ax = axes[0, 1]
    ax.scatter(f_data[:, 1], f_data[:, 0], alpha=0.2, s=2, c='red')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Flow Samples', fontsize=12, fontweight='bold')
    ax.set_xlim(0, TWO_PI)
    ax.set_xticks([0, np.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

    # KDE overlay comparison
    ax = axes[0, 2]
    sns.kdeplot(x=t_data[:, 1], y=t_data[:, 0], ax=ax,
                fill=True, color='blue', alpha=0.3, levels=8)
    sns.kdeplot(x=f_data[:, 1], y=f_data[:, 0], ax=ax,
                color='red', linewidths=2, levels=8)

    legend_elements = [
        Patch(facecolor='blue', alpha=0.3, label='Target'),
        Line2D([0], [0], color='red', linewidth=2, label='Flow')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel(r'$r$ (linear)', fontsize=11)
    ax.set_title('Density Comparison (KDE)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, TWO_PI)
    ax.set_xticks([0, np.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])

    # Row 2: Marginals and 3D
    # r marginal
    ax = axes[1, 0]
    ax.hist(t_data[:, 0], bins=50, density=True, alpha=0.4, color='blue', label='Target')
    sns.kdeplot(x=f_data[:, 0], ax=ax, color='red', linewidth=2, label='Flow')
    ax.set_xlabel(r'$r$ (linear)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Marginal: Linear Dimension', fontsize=12, fontweight='bold')
    ax.legend()

    # theta marginal
    ax = axes[1, 1]
    ax.hist(t_data[:, 1], bins=50, density=True, alpha=0.4, color='blue', label='Target')
    sns.kdeplot(x=f_data[:, 1], ax=ax, color='red', linewidth=2, label='Flow')
    ax.set_xlabel(r'$\theta$ (circular)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Marginal: Circular Dimension', fontsize=12, fontweight='bold')
    ax.set_xlim(0, TWO_PI)
    ax.set_xticks([0, np.pi, TWO_PI])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.legend()

    # Training curve
    ax = axes[1, 2]
    ax.plot(losses['train'], label='Train', alpha=0.8)
    ax.plot(losses['val'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Negative Log-Likelihood', fontsize=11)
    ax.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mixed_r1t1_density_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: mixed_r1t1_density_comparison.png")

    return flow, train_data


if __name__ == "__main__":
    print("Mixed Topology R x T (Cylinder) Flows Demonstration")
    print("Based on arXiv:2002.02428 'Normalizing Flows on Tori and Spheres'")
    print("Extended to mixed R^N x T^M topologies")
    print()

    # Run demonstrations
    demo_correlated_cylinder()
    demo_multimodal_cylinder()
    demo_density_comparison()

    print("\nAll demonstrations complete!")
    print("\nOutput files:")
    print("  - mixed_r1t1_correlated.png (+ _3d.png, _loss.png)")
    print("  - mixed_r1t1_multimodal.png (+ _3d.png, _loss.png)")
    print("  - mixed_r1t1_density_comparison.png")
