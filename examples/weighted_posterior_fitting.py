#!/usr/bin/env python3
"""
Weighted Posterior Fitting with Mixed Topology Normalizing Flows.

This script demonstrates fitting normalizing flows to importance-weighted samples,
comparing three approaches:
1. Standard MAF (Normal base, all dims treated as linear)
2. Circular MAF (TorusUniform base, all dims treated as circular)
3. Mixed MAF (MixedBase, respects actual topology)

Supports three data modes:
- Synthetic data: Use built-in examples with ground truth
- Real PolyChord data: Load dead points with NS weights
- IS-reweighted data: Load PolyChord chains + aims energies for IS reweighting

IS Mode Features:
- Diagnostic plots: logL correlation and weight distribution
- ESS (Effective Sample Size) warning threshold
- Optional comparison with direct aims PolyChord run

Usage:
    # Synthetic data (self-contained demo)
    python weighted_posterior_fitting.py --synthetic correlated_torus

    # Real PolyChord data (NS weights only)
    python weighted_posterior_fitting.py --chains-root /path/to/chains/root \\
        --n-params 5 --is-circular TTFFF

    # IS-reweighted data (tblite chains + aims energies)
    python weighted_posterior_fitting.py --chains-root /path/to/tblite/chains \\
        --is-csv /path/to/aims_energies.csv --n-params 5 --is-circular TTFFF

    # IS-reweighted with direct aims comparison (ground truth)
    python weighted_posterior_fitting.py --chains-root /path/to/tblite/chains \\
        --is-csv /path/to/aims_energies.csv \\
        --aims-chains-root /path/to/aims/chains \\
        --n-params 5 --is-circular TTFFF

    # List available synthetic examples
    python weighted_posterior_fitting.py --list-synthetic
"""

import argparse
import json
from pathlib import Path
from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optax
import paramax
from jaxtyping import Array, Float, PRNGKeyArray
from scipy.special import logsumexp

# FlowJAX imports
from flowjax.distributions import Normal, MixedBase, Uniform, Transformed, TorusUniform
from flowjax.bijections import (
    RationalQuadraticSpline,
    CircularRationalQuadraticSpline,
    Affine,
    Chain,
    Vmap,
)
from flowjax.flows import (
    masked_autoregressive_flow,
    circular_masked_autoregressive_flow,
    mixed_masked_autoregressive_flow,
)
from flowjax.train import fit_to_data
from flowjax.train.losses import MaximumLikelihoodLoss


# =============================================================================
# Constants
# =============================================================================

# Epsilon for numerical stability with Uniform base distributions
# Data is clipped to [EPS, 1-EPS] to avoid -inf log_prob at boundaries
EPS = 1e-6

# Physical constants for IS reweighting
K_BOLTZMANN_EV = 8.617333262e-5  # eV/K
DEFAULT_TEMPERATURE = 300.0  # K

# Try to import corner for nice plots
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

# Try to import anesthetic for PolyChord loading
try:
    import anesthetic
    HAS_ANESTHETIC = True
except ImportError:
    HAS_ANESTHETIC = False


# =============================================================================
# Data Structures
# =============================================================================

class DataResult(NamedTuple):
    """Result from data loading/generation."""
    samples: np.ndarray          # (N, D) samples
    weights: np.ndarray          # (N,) normalized importance weights
    is_circular: np.ndarray      # (D,) boolean mask
    ground_truth: np.ndarray | None  # (N, D) ground truth samples (synthetic only)
    param_names: list[str] | None    # Parameter names
    # IS diagnostic fields (only present when using --is-csv)
    logL_tblite: np.ndarray | None = None  # Log-likelihoods from tblite
    logL_aims: np.ndarray | None = None    # Log-likelihoods from aims
    log_w_ns: np.ndarray | None = None     # NS log-weights (before IS correction)
    ess: float | None = None               # Effective sample size


# =============================================================================
# Weighted Loss Function
# =============================================================================

class WeightedMaximumLikelihoodLoss:
    """
    Weighted negative log-likelihood loss for density estimation.

    Computes: L = -sum(w_i * log p(x_i)) / sum(w_i)

    Usage with fit_to_data:
        loss_fn = WeightedMaximumLikelihoodLoss()
        flow, losses = fit_to_data(
            key=key,
            dist=flow,
            data=(samples, weights),  # Pass as tuple
            loss_fn=loss_fn,
            ...
        )
    """

    @eqx.filter_jit
    def __call__(
        self,
        params,
        static,
        x: Float[Array, "batch dim"],
        weights: Float[Array, "batch"],
        condition: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute weighted negative log-likelihood."""
        dist = paramax.unwrap(eqx.combine(params, static))
        log_probs = dist.log_prob(x, condition)
        return -jnp.sum(weights * log_probs) / jnp.sum(weights)


# =============================================================================
# PolyChord Dead Points Loader
# =============================================================================

def load_polychord_dead_points(
    filepath: str | Path,
    n_params: int,
    param_names: list[str] | None = None,
) -> DataResult:
    """
    Load PolyChord dead points and compute nested sampling weights.

    Supports both dead.txt and dead-birth.txt formats.

    Args:
        filepath: Path to dead points file
        n_params: Number of physical parameters to extract
        param_names: Optional parameter names

    Returns:
        DataResult with samples, weights, and metadata
    """
    filepath = Path(filepath)
    print(f"Loading dead points from: {filepath}")

    # Load data
    data = np.loadtxt(filepath)
    n_points = len(data)
    print(f"  Loaded {n_points} dead points")

    # Determine format (dead.txt vs dead-birth.txt)
    # dead-birth.txt has: logL, logL_birth, params...
    # dead.txt has: logL, params...
    # We detect by checking if second column looks like logL values

    if data.shape[1] >= 2 + n_params:
        # Assume dead-birth format
        logL = data[:, 0]
        logL_birth = data[:, 1]
        samples = data[:, 2:2 + n_params]
        print("  Detected dead-birth format")

        # Compute NS weights using prior compression
        # log(X_i) ≈ -i/n_live for simple estimate
        # Better: use logL_birth to estimate compression
        # w_i ∝ L_i * ΔX_i where ΔX_i = X_{i-1} - X_i
        # For dead-birth: logL_birth gives the likelihood at birth
        # Prior volume at death: X_i ≈ exp(-i/n_live)

        # Simple approximation: weight by likelihood
        # More accurate would require n_live information
        log_w = logL
    else:
        # Simple dead.txt format
        logL = data[:, 0]
        samples = data[:, 1:1 + n_params]
        print("  Detected dead.txt format")
        log_w = logL

    # Normalize weights
    log_w_norm = log_w - logsumexp(log_w)
    weights = np.exp(log_w_norm)

    # Compute ESS
    ess = 1.0 / np.sum(weights ** 2)
    print(f"  Effective Sample Size (ESS): {ess:.1f}")

    # Generate parameter names if not provided
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]

    # Default is_circular: all False (user should override)
    is_circular = np.zeros(n_params, dtype=bool)

    return DataResult(
        samples=samples,
        weights=weights,
        is_circular=is_circular,
        ground_truth=None,
        param_names=param_names,
    )


def load_anesthetic_chains(
    chains_root: str | Path,
    n_params: int,
    param_names: list[str] | None = None,
) -> DataResult:
    """
    Load PolyChord samples using anesthetic library (if available).

    This provides proper NS weight computation.

    Args:
        chains_root: Path to chains directory root (without _dead.txt suffix)
        n_params: Number of physical parameters
        param_names: Optional parameter names

    Returns:
        DataResult with samples, weights, and metadata
    """
    try:
        import anesthetic
    except ImportError:
        raise ImportError(
            "anesthetic package required for this loader. "
            "Install with: pip install anesthetic"
        )

    print(f"Loading chains from: {chains_root}")
    ns_samples = anesthetic.read_chains(str(chains_root))

    # Get log-likelihood
    logL = ns_samples.logL.values

    # Extract physical parameters
    samples = ns_samples.iloc[:, :n_params].values

    # Get parameter names from anesthetic if not provided
    if param_names is None:
        columns = ns_samples.columns[:n_params]
        param_names = [str(c[0]) if isinstance(c, tuple) else str(c) for c in columns]

    # Get NS posterior weights and normalize
    # Use get_weights() to retrieve posterior weights (Likelihood * Volume)
    # NOT logw() which only returns prior volume weights
    weights = np.asarray(ns_samples.get_weights())
    weights = weights / weights.sum()

    # Compute ESS
    ess = 1.0 / np.sum(weights ** 2)
    print(f"  Loaded {len(samples)} samples, ESS: {ess:.1f}")

    # Default is_circular
    is_circular = np.zeros(n_params, dtype=bool)

    return DataResult(
        samples=samples,
        weights=weights,
        is_circular=is_circular,
        ground_truth=None,
        param_names=param_names,
    )


def load_is_reweighted_data(
    chains_root: str | Path,
    is_csv_path: str | Path,
    n_params: int,
    param_names: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> DataResult:
    """
    Load PolyChord chains with importance sampling reweighting.

    This loads tblite PolyChord chains and reweights them using aims energies
    from a CSV file. The IS weights combine NS weights with the likelihood ratio:

        log_w_total = log_w_NS + (logL_aims - logL_tblite)
        w_final = exp(log_w_total - logsumexp(log_w_total))

    Args:
        chains_root: Path to tblite PolyChord chains root
        is_csv_path: Path to CSV with columns [index, energy_eV]
        n_params: Number of physical parameters
        param_names: Optional parameter names
        temperature: Temperature in Kelvin for energy->logL conversion (default 300K)

    Returns:
        DataResult with IS-reweighted samples and weights
    """
    import pandas as pd

    if not HAS_ANESTHETIC:
        raise ImportError(
            "anesthetic package required for IS loading. "
            "Install with: pip install anesthetic"
        )

    print(f"Loading IS-reweighted data...")
    print(f"  Chains root: {chains_root}")
    print(f"  IS CSV: {is_csv_path}")

    # Load tblite PolyChord chains
    ns_samples = anesthetic.read_chains(str(chains_root))
    n_tblite = len(ns_samples)
    print(f"  Loaded {n_tblite} tblite samples")

    # Get original tblite log-likelihoods
    logL_tblite_full = ns_samples.logL.values.flatten()

    # Get original tblite POSTERIOR weights from nested sampling
    # anesthetic computes these from logw + logL internally
    w_tblite_full = np.asarray(ns_samples.get_weights())

    # Also get log prior volume weights for diagnostics
    log_w_ns_full = ns_samples.logw().values

    # Load aims energies CSV
    aims_df = pd.read_csv(is_csv_path)
    if 'index' not in aims_df.columns or 'energy_eV' not in aims_df.columns:
        raise ValueError(f"IS CSV must have 'index' and 'energy_eV' columns. Found: {aims_df.columns.tolist()}")
    print(f"  Loaded {len(aims_df)} aims energies")

    # Get dead point indices from MultiIndex (same as reference implementation)
    if isinstance(ns_samples.index, pd.MultiIndex):
        dead_point_indices = ns_samples.index.get_level_values(0).to_numpy()
    else:
        dead_point_indices = ns_samples.index.to_numpy()

    # Create energy map from CSV
    aims_energy_map = dict(zip(aims_df['index'], aims_df['energy_eV']))

    # Compute aims log-likelihood for all samples (NaN where not available)
    logL_aims_full = np.full(n_tblite, np.nan)
    n_matched = 0
    for i, dp_idx in enumerate(dead_point_indices):
        if dp_idx in aims_energy_map:
            # logL = -E / (k_B * T)
            logL_aims_full[i] = -aims_energy_map[dp_idx] / (K_BOLTZMANN_EV * temperature)
            n_matched += 1

    print(f"  Matched {n_matched}/{n_tblite} samples ({100*n_matched/n_tblite:.1f}%)")

    # Create mask for samples with valid aims energies
    valid_mask = np.isfinite(logL_aims_full)
    n_valid = np.sum(valid_mask)

    # Compute IS-corrected weights using the formula:
    # log_w_aims_IS = log_w_tblite + logL_aims - logL_tblite
    #
    # This is the standard importance sampling formula:
    #   w_i^(IS) ∝ w_i^(tblite) × exp(logL_aims - logL_tblite)
    #            = w_i^(tblite) × (L_aims / L_tblite)
    #
    # Where:
    # - w_tblite = posterior weights from tblite NS run (already includes L_tblite)
    # - logL_tblite = tblite log-likelihood values
    # - logL_aims = aims log-likelihood values (= -beta * E)
    log_w_tblite = np.log(np.clip(w_tblite_full, 1e-300, None))
    log_w_aims_IS = np.full(n_tblite, -np.inf)

    # Only compute for samples with valid aims energies
    # Using exact same order as reference implementation
    log_w_aims_IS[valid_mask] = (
        log_w_tblite[valid_mask] + logL_aims_full[valid_mask] - logL_tblite_full[valid_mask]
    )

    # Diagnostic output
    logL_diff = logL_aims_full[valid_mask] - logL_tblite_full[valid_mask]
    print(f"  logL diff (aims - tblite): mean={np.mean(logL_diff):.2f}, std={np.std(logL_diff):.2f}")

    # Normalize the IS weights to sum to 1
    log_norm = logsumexp(log_w_aims_IS[valid_mask])
    log_w_aims_IS_norm = log_w_aims_IS - log_norm
    weights_full = np.exp(log_w_aims_IS_norm)

    # Set weights for invalid samples to 0
    weights_full[~valid_mask] = 0.0

    # Ensure normalization (numerical safety)
    weights_full = weights_full / weights_full.sum() if weights_full.sum() > 0 else weights_full

    # Filter to only valid samples for output
    valid_indices = np.where(valid_mask)[0]
    ns_samples_valid = ns_samples.iloc[valid_indices]
    logL_tblite = logL_tblite_full[valid_indices]
    logL_aims = logL_aims_full[valid_indices]
    log_w_ns = log_w_ns_full[valid_indices]
    weights = weights_full[valid_indices]

    # Re-normalize weights after filtering
    weights = weights / weights.sum()

    # Compute ESS
    ess = 1.0 / np.sum(weights ** 2)
    print(f"  Effective Sample Size (ESS): {ess:.1f}")

    # Extract physical parameters
    samples = ns_samples_valid.iloc[:, :n_params].values

    # Get parameter names
    if param_names is None:
        columns = ns_samples_valid.columns[:n_params]
        param_names = [str(c[0]) if isinstance(c, tuple) else str(c) for c in columns]

    # Default is_circular
    is_circular = np.zeros(n_params, dtype=bool)

    print(f"  Final samples: {len(samples)}")

    return DataResult(
        samples=samples,
        weights=weights,
        is_circular=is_circular,
        ground_truth=None,
        param_names=param_names,
        # IS diagnostic fields
        logL_tblite=logL_tblite,
        logL_aims=logL_aims,
        log_w_ns=log_w_ns,
        ess=ess,
    )


def load_aims_direct_samples(
    chains_root: str | Path,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load samples from direct aims PolyChord run for ground truth comparison.

    Args:
        chains_root: Path to aims PolyChord chains root
        n_params: Number of physical parameters

    Returns:
        Tuple of (samples, weights, param_names)
    """
    if not HAS_ANESTHETIC:
        raise ImportError(
            "anesthetic package required for aims loading. "
            "Install with: pip install anesthetic"
        )

    print(f"Loading direct aims samples from {chains_root}...")

    # Load chains
    ns_samples = anesthetic.read_chains(str(chains_root))
    print(f"  Loaded {len(ns_samples)} samples")

    # Get NS posterior weights and normalize
    # Use get_weights() to retrieve posterior weights (Likelihood * Volume)
    # NOT logw() which only returns prior volume weights
    weights = np.asarray(ns_samples.get_weights())
    weights = weights / weights.sum()

    # Extract physical parameters
    samples = ns_samples.iloc[:, :n_params].values

    # Get parameter names
    columns = ns_samples.columns[:n_params]
    param_names = [str(c[0]) if isinstance(c, tuple) else str(c) for c in columns]

    # Compute ESS
    ess = 1.0 / np.sum(weights ** 2)
    print(f"  ESS: {ess:.1f}")

    return samples, weights, param_names


# =============================================================================
# Synthetic Data Generators
# =============================================================================

def generate_correlated_torus(
    key: PRNGKeyArray,
    n_samples: int = 10000,
) -> DataResult:
    """
    Generate samples from a correlated distribution on T^2 × R^3.

    All output is in [0, 1] range (unit hypercube, like PolyChord output).

    This distribution is designed to test boundary handling:
    - Circular dims (θ₁, θ₂): Peaks near 0/1 boundary to test wrapping
    - Linear dims (x₁, x₂, x₃): Beta distribution with wider tail for gradient signal

    If circular dims are wrongly treated as linear: gap at 0/1 boundary
    If linear dims are wrongly treated as circular: samples wrap around incorrectly

    Target distribution (in unit hypercube):
    - θ₁, θ₂: von Mises in [0, 2π] → scaled to [0, 1], peaks near 0 and 1
    - x₁, x₂, x₃: Beta(α=2, β=4) distribution peaked near 0.25 with wider tail

    Returns:
        DataResult with samples, weights, is_circular mask, and ground truth
        All samples are in [0, 1] range.
    """
    from scipy.stats import vonmises, beta

    print("Generating correlated torus synthetic data (unit hypercube [0, 1])...")
    print("  Circular dims: peaks near 0/1 boundary (tests wrapping)")
    print("  Linear dims: Beta(2, 4) with wider tail (tests gradient signal)")

    # Convert JAX key to numpy seed
    key_data = jax.random.key_data(key)
    np.random.seed(int(key_data[0]) % (2**31))

    # === Target parameters ===
    # Circular: von Mises centered near 0 and 2π (will straddle 0/1 in unit space)
    kappa1, kappa2 = 3.0, 3.0  # Moderate concentration
    mu1 = 0.3                   # Near 0 in [0, 2π] → near 0 in [0, 1]
    mu2 = 2 * np.pi - 0.3      # Near 2π in [0, 2π] → near 1 in [0, 1]

    # Linear: Beta(α=2, β=4) gives distribution peaked near 0.25 with wider tail
    # This ensures more gradient signal in the [0.5, 0.9] region
    beta_alpha = 2.0
    beta_beta = 4.0

    # === Define target log-probability in [0, 1] space ===
    def target_log_prob(samples_01):
        """Log probability of target distribution in unit hypercube."""
        theta1_01, theta2_01 = samples_01[:, 0], samples_01[:, 1]
        x1, x2, x3 = samples_01[:, 2], samples_01[:, 3], samples_01[:, 4]

        # Transform [0, 1] → [0, 2π] for von Mises evaluation
        theta1 = theta1_01 * 2 * np.pi
        theta2 = theta2_01 * 2 * np.pi

        # Angular part: von Mises (with Jacobian for transform)
        # log p(θ_01) = log p(θ) + log(2π) where θ = θ_01 * 2π
        log_p_theta1 = vonmises.logpdf(theta1, kappa1, loc=mu1) + np.log(2 * np.pi)
        log_p_theta2 = vonmises.logpdf(theta2, kappa2, loc=mu2) + np.log(2 * np.pi)

        # Linear part: Beta distribution (naturally in [0, 1])
        log_p_x1 = beta.logpdf(x1, beta_alpha, beta_beta)
        log_p_x2 = beta.logpdf(x2, beta_alpha, beta_beta)
        log_p_x3 = beta.logpdf(x3, beta_alpha, beta_beta)

        return log_p_theta1 + log_p_theta2 + log_p_x1 + log_p_x2 + log_p_x3

    # === Define proposal log-probability (uniform on [0, 1]^5) ===
    def proposal_log_prob(samples_01):
        """Log probability of uniform proposal on unit hypercube."""
        # Uniform on [0, 1]^5 has constant density = 1, log = 0
        return np.zeros(len(samples_01))

    # === Sample from proposal (uniform on [0, 1]^5) ===
    samples_q = np.random.uniform(0, 1, (n_samples, 5))

    # === Compute importance weights ===
    log_w = target_log_prob(samples_q) - proposal_log_prob(samples_q)

    # Handle -inf (samples at exactly 0 or 1 for Beta can be problematic)
    log_w = np.where(np.isfinite(log_w), log_w, -1e10)

    log_w_norm = log_w - logsumexp(log_w)
    weights = np.exp(log_w_norm)

    # === Generate ground truth (direct sampling from target in [0, 1]) ===
    # Sample angles from von Mises, then scale to [0, 1]
    theta1_gt = vonmises.rvs(kappa1, loc=mu1, size=n_samples)
    theta1_gt = (theta1_gt % (2 * np.pi)) / (2 * np.pi)  # [0, 1]

    theta2_gt = vonmises.rvs(kappa2, loc=mu2, size=n_samples)
    theta2_gt = (theta2_gt % (2 * np.pi)) / (2 * np.pi)  # [0, 1]

    # Sample linear dims from Beta (already in [0, 1])
    x1_gt = beta.rvs(beta_alpha, beta_beta, size=n_samples)
    x2_gt = beta.rvs(beta_alpha, beta_beta, size=n_samples)
    x3_gt = beta.rvs(beta_alpha, beta_beta, size=n_samples)

    ground_truth = np.stack([theta1_gt, theta2_gt, x1_gt, x2_gt, x3_gt], axis=1)

    # Verify output range
    assert samples_q.min() >= 0 and samples_q.max() <= 1, "Samples must be in [0, 1]"
    assert ground_truth.min() >= 0 and ground_truth.max() <= 1, "Ground truth must be in [0, 1]"

    # Compute ESS
    ess = 1.0 / np.sum(weights ** 2)
    print(f"  Generated {n_samples} samples, ESS: {ess:.1f}")

    is_circular = np.array([True, True, False, False, False])
    param_names = ["θ₁", "θ₂", "x₁", "x₂", "x₃"]

    return DataResult(
        samples=samples_q,
        weights=weights,
        is_circular=is_circular,
        ground_truth=ground_truth,
        param_names=param_names,
    )


def generate_bimodal_wrapped(
    key: PRNGKeyArray,
    n_samples: int = 10000,
) -> DataResult:
    """
    Generate bimodal distribution with wrapping effects.

    All output is in [0, 1] range (unit hypercube, like PolyChord output).

    Two modes near the 0/1 boundary demonstrate the importance of proper
    circular handling (the modes should be close, not far apart).

    Target on T^2 × R^3 (in unit hypercube):
    - Mode 1: θ ≈ 0.05 (near 0), x ≈ 0.2 (near 0 in [0, 1])
    - Mode 2: θ ≈ 0.95 (near 1), x ≈ 0.8 (near 1 in [0, 1])

    Returns:
        DataResult with samples, weights, is_circular mask, and ground truth
        All samples are in [0, 1] range.
    """
    from scipy.stats import vonmises, beta

    print("Generating bimodal wrapped synthetic data (unit hypercube [0, 1])...")
    print("  Two modes near 0/1 boundary (tests circular wrapping)")

    # Convert JAX key to numpy seed
    key_data = jax.random.key_data(key)
    np.random.seed(int(key_data[0]) % (2**31))

    # Mode parameters (in [0, 2π] for von Mises, then scaled to [0, 1])
    mode1_theta = 0.3                 # Near 0 in [0, 2π] → ~0.05 in [0, 1]
    mode2_theta = 2 * np.pi - 0.3    # Near 2π in [0, 2π] → ~0.95 in [0, 1]
    kappa = 5.0                       # Tight concentration

    # Beta parameters for linear dims
    # Mode 1: peaked near 0 → Beta(2, 8)
    # Mode 2: peaked near 1 → Beta(8, 2)
    beta_a1, beta_b1 = 2.0, 8.0  # Mode 1 linear dims (peaked near 0.2)
    beta_a2, beta_b2 = 8.0, 2.0  # Mode 2 linear dims (peaked near 0.8)

    # === Define target (mixture of two modes in [0, 1] space) ===
    def target_log_prob(samples_01):
        theta1_01, theta2_01 = samples_01[:, 0], samples_01[:, 1]
        x1, x2, x3 = samples_01[:, 2], samples_01[:, 3], samples_01[:, 4]

        # Transform [0, 1] → [0, 2π] for von Mises
        theta1 = theta1_01 * 2 * np.pi
        theta2 = theta2_01 * 2 * np.pi

        # Mode 1: angles near 0, linear near 0.2
        log_p1_theta1 = vonmises.logpdf(theta1, kappa, loc=mode1_theta)
        log_p1_theta2 = vonmises.logpdf(theta2, kappa, loc=mode1_theta)
        log_p1_x = (beta.logpdf(x1, beta_a1, beta_b1) +
                    beta.logpdf(x2, beta_a1, beta_b1) +
                    beta.logpdf(x3, beta_a1, beta_b1))
        log_p1 = log_p1_theta1 + log_p1_theta2 + log_p1_x + 2 * np.log(2 * np.pi)

        # Mode 2: angles near 2π, linear near 0.8
        log_p2_theta1 = vonmises.logpdf(theta1, kappa, loc=mode2_theta)
        log_p2_theta2 = vonmises.logpdf(theta2, kappa, loc=mode2_theta)
        log_p2_x = (beta.logpdf(x1, beta_a2, beta_b2) +
                    beta.logpdf(x2, beta_a2, beta_b2) +
                    beta.logpdf(x3, beta_a2, beta_b2))
        log_p2 = log_p2_theta1 + log_p2_theta2 + log_p2_x + 2 * np.log(2 * np.pi)

        # Mixture (equal weights)
        return np.logaddexp(log_p1 + np.log(0.5), log_p2 + np.log(0.5))

    # === Proposal: uniform on [0, 1]^5 ===
    def proposal_log_prob(samples_01):
        return np.zeros(len(samples_01))

    # === Sample from proposal ===
    samples_q = np.random.uniform(0, 1, (n_samples, 5))

    # === Compute importance weights ===
    log_w = target_log_prob(samples_q) - proposal_log_prob(samples_q)
    log_w = np.where(np.isfinite(log_w), log_w, -1e10)
    log_w_norm = log_w - logsumexp(log_w)
    weights = np.exp(log_w_norm)

    # === Generate ground truth (in [0, 1]) ===
    n_mode1 = n_samples // 2
    n_mode2 = n_samples - n_mode1

    # Mode 1: angles near 0, linear near 0.2
    theta1_m1 = vonmises.rvs(kappa, loc=mode1_theta, size=n_mode1)
    theta1_m1 = (theta1_m1 % (2 * np.pi)) / (2 * np.pi)  # [0, 1]
    theta2_m1 = vonmises.rvs(kappa, loc=mode1_theta, size=n_mode1)
    theta2_m1 = (theta2_m1 % (2 * np.pi)) / (2 * np.pi)  # [0, 1]
    x1_m1 = beta.rvs(beta_a1, beta_b1, size=n_mode1)
    x2_m1 = beta.rvs(beta_a1, beta_b1, size=n_mode1)
    x3_m1 = beta.rvs(beta_a1, beta_b1, size=n_mode1)

    # Mode 2: angles near 1, linear near 0.8
    theta1_m2 = vonmises.rvs(kappa, loc=mode2_theta, size=n_mode2)
    theta1_m2 = (theta1_m2 % (2 * np.pi)) / (2 * np.pi)  # [0, 1]
    theta2_m2 = vonmises.rvs(kappa, loc=mode2_theta, size=n_mode2)
    theta2_m2 = (theta2_m2 % (2 * np.pi)) / (2 * np.pi)  # [0, 1]
    x1_m2 = beta.rvs(beta_a2, beta_b2, size=n_mode2)
    x2_m2 = beta.rvs(beta_a2, beta_b2, size=n_mode2)
    x3_m2 = beta.rvs(beta_a2, beta_b2, size=n_mode2)

    ground_truth = np.vstack([
        np.stack([theta1_m1, theta2_m1, x1_m1, x2_m1, x3_m1], axis=1),
        np.stack([theta1_m2, theta2_m2, x1_m2, x2_m2, x3_m2], axis=1),
    ])
    np.random.shuffle(ground_truth)

    # Verify output range
    assert samples_q.min() >= 0 and samples_q.max() <= 1, "Samples must be in [0, 1]"
    assert ground_truth.min() >= 0 and ground_truth.max() <= 1, "Ground truth must be in [0, 1]"

    # Compute ESS
    ess = 1.0 / np.sum(weights ** 2)
    print(f"  Generated {n_samples} samples, ESS: {ess:.1f}")

    is_circular = np.array([True, True, False, False, False])
    param_names = ["θ₁", "θ₂", "x₁", "x₂", "x₃"]

    return DataResult(
        samples=samples_q,
        weights=weights,
        is_circular=is_circular,
        ground_truth=ground_truth,
        param_names=param_names,
    )


# Registry of synthetic data generators
SYNTHETIC_GENERATORS: dict[str, Callable] = {
    "correlated_torus": generate_correlated_torus,
    "bimodal_wrapped": generate_bimodal_wrapped,
}


# =============================================================================
# Data Preprocessing
# =============================================================================

def clip_weights(weights: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Clip weights to prevent gradient explosion from extreme values."""
    w_clip = np.percentile(weights, percentile)
    weights_clipped = np.minimum(weights, w_clip)
    weights_clipped = weights_clipped / weights_clipped.sum()
    return weights_clipped


def clip_to_interior(samples: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip samples to interior of [0, 1] for numerical stability with Uniform base.

    When using Uniform[0, 1] as the base distribution, values exactly at 0 or 1
    (or slightly outside due to floating point) will give -inf log_prob.
    This clips data to [eps, 1-eps] to prevent this issue.

    Args:
        samples: Input samples in [0, 1] range
        eps: Small epsilon for boundary margin (default: EPS global constant)

    Returns:
        Samples clipped to [eps, 1-eps]
    """
    return np.clip(samples, eps, 1.0 - eps)


def save_trained_flow(
    flow,
    flow_type: str,
    is_circular: np.ndarray,
    param_names: list[str] | None,
    output_dir: Path,
) -> Path:
    """Save a trained flow model with metadata for later loading.

    Args:
        flow: Trained flow (Transformed distribution)
        flow_type: Type of flow ("standard", "circular", "mixed")
        is_circular: Boolean array indicating circular dimensions
        param_names: Parameter names for plotting
        output_dir: Directory to save model files

    Returns:
        Path to the saved model file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_dir / f"flow_{flow_type}.eqx"
    eqx.tree_serialise_leaves(model_path, flow)

    # Save metadata needed for loading and sampling
    metadata = {
        "flow_type": flow_type,
        "is_circular": is_circular.tolist(),
        "param_names": param_names,
        "dim": len(is_circular),
    }
    metadata_path = output_dir / f"flow_{flow_type}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {flow_type} flow to: {model_path}")
    return model_path


class WeightedDataLoader:
    """Data loader for importance-weighted samples with two training modes.

    Modes:
        - "weighted": Iterates through shuffled data with importance weights.
          Returns (batch_samples, batch_weights) tuples.
        - "resample": Draws samples according to weights (importance resampling).
          Returns (batch_samples, None) tuples (weights implicit in sampling).

    The loader tracks samples drawn to define epoch boundaries (total = dataset size).

    Args:
        samples: Sample array of shape (n_samples, n_dims)
        weights: Importance weights array of shape (n_samples,)
        batch_size: Number of samples per batch
        mode: Training mode, either "weighted" or "resample"
        add_noise: If True, add small Gaussian noise in resample mode
        is_circular: Boolean array indicating circular dimensions (for topology-aware noise)
        key: JAX random key
    """

    NOISE_STD = 1e-3  # Fixed noise level when enabled

    def __init__(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
        batch_size: int,
        mode: str = "weighted",
        add_noise: bool = False,
        is_circular: np.ndarray | None = None,
        key: PRNGKeyArray | None = None,
    ):
        self.samples = jnp.array(samples)
        # Normalize weights to sum to 1
        weights = np.asarray(weights)
        self.weights = jnp.array(weights / weights.sum())
        self.batch_size = batch_size
        self.mode = mode
        self.add_noise = add_noise
        self.is_circular = (
            is_circular if is_circular is not None
            else np.zeros(samples.shape[1], dtype=bool)
        )
        self.key = key if key is not None else jr.PRNGKey(0)
        self.n_samples = len(samples)
        self._samples_drawn = 0
        self._shuffled_indices = None
        self._batch_start = 0

    def __iter__(self):
        """Reset for new epoch. Shuffles indices for weighted mode."""
        self._samples_drawn = 0
        self._batch_start = 0
        # CRITICAL: Shuffle indices at epoch start for weighted mode
        # PolyChord data is often sorted by likelihood, which would violate IID assumption
        if self.mode == "weighted":
            self.key, subkey = jr.split(self.key)
            self._shuffled_indices = jr.permutation(subkey, self.n_samples)
        return self

    def __next__(self) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Get next batch of samples (and weights for weighted mode)."""
        if self._samples_drawn >= self.n_samples:
            raise StopIteration

        if self.mode == "weighted":
            return self._get_weighted_batch()
        elif self.mode == "resample":
            return self._get_resampled_batch()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_weighted_batch(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get batch with shuffled samples and their weights."""
        batch_end = min(self._batch_start + self.batch_size, self.n_samples)
        batch_indices = self._shuffled_indices[self._batch_start:batch_end]
        batch_samples = self.samples[batch_indices]
        batch_weights = self.weights[batch_indices]
        self._batch_start = batch_end
        self._samples_drawn += len(batch_indices)
        return batch_samples, batch_weights

    def _get_resampled_batch(self) -> tuple[jnp.ndarray, None]:
        """Get batch by sampling according to weights."""
        self.key, subkey = jr.split(self.key)
        batch_indices = jr.choice(
            subkey,
            self.n_samples,
            shape=(self.batch_size,),
            p=self.weights,
        )
        batch_samples = self.samples[batch_indices]

        # Apply topology-aware noise if enabled
        if self.add_noise:
            batch_samples = self._apply_noise(batch_samples)

        self._samples_drawn += self.batch_size
        return batch_samples, None  # No weights needed (implicit in sampling)

    def _apply_noise(self, samples: jnp.ndarray) -> jnp.ndarray:
        """Apply topology-aware noise to samples.

        - Circular dimensions: wrap via modulo (periodic boundary)
        - Linear dimensions: reflect at boundaries
        """
        self.key, subkey = jr.split(self.key)
        noise = jr.normal(subkey, samples.shape) * self.NOISE_STD
        noisy_samples = samples + noise

        # Handle each dimension according to topology
        for i in range(samples.shape[1]):
            if self.is_circular[i]:
                # Circular: wrap via modulo (periodic boundary)
                noisy_samples = noisy_samples.at[:, i].set(
                    noisy_samples[:, i] % 1.0
                )
            else:
                # Linear: reflect at boundaries
                # Reflection: if x < 0, x = -x; if x > 1, x = 2 - x
                col = noisy_samples[:, i]
                col = jnp.where(col < 0, -col, col)
                col = jnp.where(col > 1, 2 - col, col)
                # Clip as final safety (shouldn't be needed after reflection)
                col = jnp.clip(col, 0, 1)
                noisy_samples = noisy_samples.at[:, i].set(col)

        return noisy_samples

    @property
    def steps_per_epoch(self) -> int:
        """Number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class Preprocessor:
    """
    Simplified preprocessing for flow training with Uniform[0, 1] base.

    Since all flows now use Uniform[0, 1] base and scaling is built into
    the training functions, this class only handles:
    1. Clipping data to interior for numerical stability (preprocess)
    2. Transforming flow outputs back to [0, 1] range (inverse)

    Flow output ranges:
    - Standard: [0, 1] (already in correct range)
    - Circular: [0, 2π] (needs scaling back to [0, 1])
    - Mixed: Linear dims in [0, 1], circular dims in [0, 2π]
    """

    def __init__(
        self,
        is_circular: np.ndarray,
        eps: float = EPS,
    ):
        """
        Args:
            is_circular: Boolean mask for circular dimensions
            eps: Epsilon for boundary clipping (default: EPS global constant)
        """
        self.is_circular = is_circular
        self.dim = len(is_circular)
        self.eps = eps

    def preprocess(self, samples: np.ndarray) -> np.ndarray:
        """Clip samples to interior of [0, 1] for numerical stability.

        Args:
            samples: Input data in [0, 1] range

        Returns:
            Samples clipped to [eps, 1-eps]
        """
        return clip_to_interior(samples, self.eps)

    def inverse(self, samples: np.ndarray, mode: str) -> np.ndarray:
        """Transform flow outputs back to [0, 1] range.

        Args:
            samples: Flow output samples
            mode: "standard", "circular", or "mixed"

        Returns:
            Samples in [0, 1] range
        """
        out = samples.copy()

        if mode == "standard":
            # Output already in [0, 1], just clip for safety
            out = np.clip(out, 0, 1)

        elif mode == "circular":
            # Output in [0, 2π], scale back to [0, 1]
            # First wrap to [0, 2π] to handle any overflow
            out = out % (2 * np.pi)
            out = out / (2 * np.pi)

        elif mode == "mixed":
            # ALL dims are in [0, 2π] (we scale everything to [0, 2π] for training)
            # Scale ALL back to [0, 1]
            out = out % (2 * np.pi)  # Wrap to [0, 2π] for safety
            out = out / (2 * np.pi)  # Scale to [0, 1]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return out


# =============================================================================
# Flow Training
# =============================================================================

def prepare_training_data(
    samples: np.ndarray,
    weights: np.ndarray,
    is_circular: np.ndarray,
    training_mode: str,
    add_noise: bool,
    key: PRNGKeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray | None, PRNGKeyArray]:
    """Prepare data for training based on training mode.

    Args:
        samples: Training samples in [0, 1] range
        weights: Importance weights for samples
        is_circular: Boolean array indicating circular dimensions
        training_mode: Either "weighted" or "resample"
        add_noise: If True, add topology-aware noise in resample mode
        key: JAX random key

    Returns:
        Tuple of (prepared_samples, prepared_weights_or_None, updated_key)
        - weighted mode: returns (samples, weights, key)
        - resample mode: returns (resampled_samples, None, key)
    """
    if training_mode == "weighted":
        # Weighted mode: pass samples and weights directly
        # fit_to_data will shuffle at each epoch
        return jnp.array(samples), jnp.array(weights), key

    elif training_mode == "resample":
        # Resample mode: draw samples according to weights
        n_samples = len(samples)
        weights_normalized = weights / weights.sum()

        key, subkey = jr.split(key)
        indices = jr.choice(
            subkey,
            n_samples,
            shape=(n_samples,),  # Same size as original for fair epoch comparison
            p=jnp.array(weights_normalized),
        )
        resampled = jnp.array(samples)[indices]

        # Report resampling efficiency (unique samples / total samples)
        unique_count = len(np.unique(np.array(indices)))
        efficiency = unique_count / n_samples
        print(f"  Resampling efficiency: {unique_count}/{n_samples} = {efficiency:.1%} unique samples")

        # Apply topology-aware noise if enabled
        if add_noise:
            key, subkey = jr.split(key)
            noise_std = WeightedDataLoader.NOISE_STD
            noise = jr.normal(subkey, resampled.shape) * noise_std
            noisy = resampled + noise

            # Handle each dimension according to topology
            for i in range(samples.shape[1]):
                if is_circular[i]:
                    # Circular: wrap via modulo
                    noisy = noisy.at[:, i].set(noisy[:, i] % 1.0)
                else:
                    # Linear: reflect at boundaries
                    col = noisy[:, i]
                    col = jnp.where(col < 0, -col, col)
                    col = jnp.where(col > 1, 2 - col, col)
                    col = jnp.clip(col, 0, 1)
                    noisy = noisy.at[:, i].set(col)
            resampled = noisy

        return resampled, None, key

    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


def create_optimizer(
    learning_rate: float,
    gradient_clip: bool,
) -> optax.GradientTransformation:
    """Create optimizer with optional gradient clipping.

    Args:
        learning_rate: Learning rate for Adam
        gradient_clip: If True, enable gradient norm clipping

    Returns:
        Optax optimizer
    """
    if gradient_clip:
        return optax.chain(
            optax.clip_by_global_norm(1.0),  # Max gradient norm of 1.0
            optax.adam(learning_rate),
        )
    else:
        return optax.adam(learning_rate)


def train_standard_flow(
    samples: np.ndarray,
    weights: np.ndarray,
    key: PRNGKeyArray,
    n_layers: int = 8,
    n_knots: int = 8,
    max_epochs: int = 200,
    max_patience: int = 15,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    training_mode: str = "weighted",
    add_noise: bool = False,
    gradient_clip: bool = False,
) -> tuple:
    """Train standard MAF with Uniform[0,1] base and bounded RQ splines.

    Uses flat prior on [0, 1] for all dimensions with RationalQuadraticSpline
    transformers that have f(0)=0, f(1)=1 and free boundary derivatives.

    Args:
        samples: Training samples in [0, 1] range (will be clipped to interior)
        weights: Importance weights for samples
        key: JAX random key
        n_layers: Number of MAF layers
        n_knots: Number of spline knots
        max_epochs: Maximum training epochs
        max_patience: Early stopping patience
        learning_rate: Learning rate
        batch_size: Batch size
        training_mode: "weighted" or "resample"
        add_noise: Add topology-aware noise in resample mode
        gradient_clip: Enable gradient norm clipping

    Returns:
        Tuple of (trained_flow, losses_dict)
    """
    print("\n" + "=" * 60)
    print("Training STANDARD MAF (Uniform[0,1] base, bounded splines)")
    print(f"  Training mode: {training_mode}")
    if training_mode == "resample":
        print(f"  Noise injection: {add_noise}")
    print(f"  Gradient clipping: {gradient_clip}")
    print("=" * 60)

    dim = samples.shape[1]

    # Clip data to interior for numerical stability with Uniform base
    samples_clipped = clip_to_interior(samples)

    # Prepare training data based on mode
    # For standard flow, all dimensions are linear (not circular)
    is_circular = np.zeros(dim, dtype=bool)
    key, subkey = jr.split(key)
    samples_jax, weights_jax, key = prepare_training_data(
        samples_clipped, weights, is_circular, training_mode, add_noise, subkey
    )

    # Flat prior on [0, 1]
    key, subkey = jr.split(key)
    base_dist = Uniform(minval=jnp.zeros(dim), maxval=jnp.ones(dim))

    # Bounded spline transformer: f(0)=0, f(1)=1, free boundary derivatives
    # boundary_derivatives=None means all derivatives (including at boundaries) are learned
    transformer = RationalQuadraticSpline(
        knots=n_knots,
        interval=(0.0, 1.0),
        boundary_derivatives=None,  # Free boundary derivatives
    )

    flow = masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        flow_layers=n_layers,
        nn_width=64,
        nn_depth=1,
    )

    # Train with appropriate loss function and optimizer
    key, subkey = jr.split(key)

    # Create optimizer with optional gradient clipping
    optimizer = create_optimizer(learning_rate, gradient_clip)

    # Use appropriate loss function based on training mode
    if training_mode == "weighted":
        loss_fn = WeightedMaximumLikelihoodLoss()
        data = (samples_jax, weights_jax)
    else:  # resample mode
        loss_fn = MaximumLikelihoodLoss()
        data = samples_jax  # No weights needed

    trained_flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        data=data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
        show_progress=True,
    )

    print(f"  Final loss: {losses['val'][-1]:.4f}")
    return trained_flow, losses


def train_circular_flow(
    samples: np.ndarray,
    weights: np.ndarray,
    key: PRNGKeyArray,
    n_layers: int = 8,
    num_bins: int = 8,
    max_epochs: int = 200,
    max_patience: int = 15,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    training_mode: str = "weighted",
    add_noise: bool = False,
    gradient_clip: bool = False,
) -> tuple:
    """Train circular MAF with TorusUniform base (flat prior on [0, 2π]).

    Uses TorusUniform (flat on [0, 2π]) as base distribution with
    CircularRationalQuadraticSpline transformers. Data is scaled to [0, 2π]
    for training, and outputs must be scaled back to [0, 1] for plotting.

    Note: TorusUniform[0, 2π] is equivalent to flat prior in [0, 1] after
    the output scaling, so this achieves the "flat prior" objective.

    Args:
        samples: Training samples in [0, 1] range (will be clipped to interior)
        weights: Importance weights for samples
        key: JAX random key
        n_layers: Number of MAF layers
        num_bins: Number of spline bins
        max_epochs: Maximum training epochs
        max_patience: Early stopping patience
        learning_rate: Learning rate
        batch_size: Batch size
        training_mode: "weighted" or "resample"
        add_noise: Add topology-aware noise in resample mode
        gradient_clip: Enable gradient norm clipping

    Returns:
        Tuple of (trained_flow, losses_dict)
    """
    print("\n" + "=" * 60)
    print("Training CIRCULAR MAF (TorusUniform base, all circular)")
    print(f"  Training mode: {training_mode}")
    if training_mode == "resample":
        print(f"  Noise injection: {add_noise}")
    print(f"  Gradient clipping: {gradient_clip}")
    print("=" * 60)

    dim = samples.shape[1]

    # Clip data to interior for numerical stability
    samples_clipped = clip_to_interior(samples)

    # Prepare training data based on mode
    # For circular flow, all dimensions are circular (noise wraps via modulo)
    is_circular = np.ones(dim, dtype=bool)
    key, subkey = jr.split(key)
    samples_prepared, weights_jax, key = prepare_training_data(
        samples_clipped, weights, is_circular, training_mode, add_noise, subkey
    )

    # Scale data to [0, 2π] for training (circular flows operate in [0, 2π])
    # The inverse transform in Preprocessor will scale outputs back to [0, 1]
    samples_jax = samples_prepared * (2 * np.pi)

    # TorusUniform: flat prior on [0, 2π]^dim
    # This is equivalent to flat in [0, 1] after scaling output back
    key, subkey = jr.split(key)
    base_dist = TorusUniform(dim)

    # Standard circular spline transformer (no scaling Chain needed)
    transformer = CircularRationalQuadraticSpline(num_bins=num_bins)

    flow = circular_masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        flow_layers=n_layers,
        nn_width=64,
        nn_depth=1,
    )

    # Train with appropriate loss function and optimizer
    key, subkey = jr.split(key)

    # Create optimizer with optional gradient clipping
    optimizer = create_optimizer(learning_rate, gradient_clip)

    # Use appropriate loss function based on training mode
    if training_mode == "weighted":
        loss_fn = WeightedMaximumLikelihoodLoss()
        data = (samples_jax, weights_jax)
    else:  # resample mode
        loss_fn = MaximumLikelihoodLoss()
        data = samples_jax  # No weights needed

    trained_flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        data=data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
        show_progress=True,
    )

    print(f"  Final loss: {losses['val'][-1]:.4f}")
    return trained_flow, losses


def train_mixed_flow(
    samples: np.ndarray,
    weights: np.ndarray,
    is_circular: np.ndarray,
    key: PRNGKeyArray,
    n_layers: int = 8,
    n_knots: int = 8,
    max_epochs: int = 200,
    max_patience: int = 15,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    training_mode: str = "weighted",
    add_noise: bool = False,
    gradient_clip: bool = False,
) -> tuple:
    """Train mixed MAF with TorusUniform base and topology-aware transformers.

    All dimensions are scaled to [0, 2π] for training, using TorusUniform as base:
    - Linear dims: RationalQuadraticSpline(interval=[0, 2π], boundary_derivatives=None)
    - Circular dims: CircularRationalQuadraticSpline (operates in [0, 2π])

    The flow output for ALL dims is in [0, 2π], which is scaled back to [0, 1].

    Note: This achieves "flat prior in [0, 1]" because TorusUniform[0, 2π] is
    equivalent to flat in [0, 1] after the output scaling back to [0, 1].

    Args:
        samples: Training samples in [0, 1] range (will be clipped to interior)
        weights: Importance weights for samples
        is_circular: Boolean mask indicating circular dimensions
        key: JAX random key
        n_layers: Number of MAF layers
        n_knots: Number of spline knots/bins
        max_epochs: Maximum training epochs
        max_patience: Early stopping patience
        learning_rate: Learning rate
        batch_size: Batch size
        training_mode: "weighted" or "resample"
        add_noise: Add topology-aware noise in resample mode
        gradient_clip: Enable gradient norm clipping

    Returns:
        Tuple of (trained_flow, losses_dict)
    """
    print("\n" + "=" * 60)
    print("Training MIXED MAF (TorusUniform base, topology-aware)")
    print(f"  is_circular: {is_circular}")
    print(f"  Training mode: {training_mode}")
    if training_mode == "resample":
        print(f"  Noise injection: {add_noise}")
    print(f"  Gradient clipping: {gradient_clip}")
    print("=" * 60)

    dim = len(is_circular)

    # Clip data to interior for numerical stability
    samples_clipped = clip_to_interior(samples)

    # Prepare training data based on mode (uses topology-aware noise for resample mode)
    key, subkey = jr.split(key)
    samples_prepared, weights_jax, key = prepare_training_data(
        samples_clipped, weights, is_circular, training_mode, add_noise, subkey
    )

    # Scale ALL dimensions to [0, 2π] for training
    # This simplifies the base distribution (all dims use TorusUniform)
    samples_jax = samples_prepared * (2 * np.pi)
    is_circular_jax = jnp.array(is_circular)

    # TorusUniform: flat prior on [0, 2π]^dim for ALL dimensions
    key, subkey = jr.split(key)
    base_dist = TorusUniform(dim)

    # Create mixed flow with:
    # - Linear bounds [0, 2π] to match data range (with free boundary derivatives)
    # - Circular transformer also operates in [0, 2π]
    flow = mixed_masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        is_circular=is_circular_jax,
        linear_bounds=(0.0, 2 * jnp.pi),  # Bounded to [0, 2π] to match TorusUniform
        linear_boundary_derivatives=None,  # Free boundary derivatives
        linear_transformer_kwargs={"knots": n_knots},
        circular_transformer_kwargs={"num_bins": n_knots},
        flow_layers=n_layers,
        nn_width=64,
        nn_depth=1,
    )

    # Train with appropriate loss function and optimizer
    key, subkey = jr.split(key)

    # Create optimizer with optional gradient clipping
    optimizer = create_optimizer(learning_rate, gradient_clip)

    # Use appropriate loss function based on training mode
    if training_mode == "weighted":
        loss_fn = WeightedMaximumLikelihoodLoss()
        data = (samples_jax, weights_jax)
    else:  # resample mode
        loss_fn = MaximumLikelihoodLoss()
        data = samples_jax  # No weights needed

    trained_flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        data=data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
        show_progress=True,
    )

    print(f"  Final loss: {losses['val'][-1]:.4f}")
    return trained_flow, losses


# =============================================================================
# Visualization
# =============================================================================

def plot_logL_correlation(
    logL_tblite: np.ndarray,
    logL_aims: np.ndarray,
    output_path: Path,
    title: str = "Log-Likelihood Correlation: tblite vs aims",
):
    """
    Plot correlation between tblite and aims log-likelihoods.

    This diagnostic plot helps assess how well tblite approximates aims.
    Points should cluster around the diagonal if both methods agree.

    Args:
        logL_tblite: Log-likelihoods from tblite
        logL_aims: Log-likelihoods from aims
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Scatter plot
    ax = axes[0]
    ax.scatter(logL_tblite, logL_aims, alpha=0.3, s=5)

    # Add diagonal line
    lims = [
        min(logL_tblite.min(), logL_aims.min()),
        max(logL_tblite.max(), logL_aims.max()),
    ]
    ax.plot(lims, lims, 'r--', lw=2, label='y = x')

    # Compute correlation
    corr = np.corrcoef(logL_tblite, logL_aims)[0, 1]
    ax.set_xlabel('logL (tblite)')
    ax.set_ylabel('logL (aims)')
    ax.set_title(f'Scatter (correlation: {corr:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Difference histogram
    ax = axes[1]
    diff = logL_aims - logL_tblite
    ax.hist(diff, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.axvline(diff.mean(), color='g', linestyle='-', lw=2, label=f'Mean: {diff.mean():.2f}')
    ax.set_xlabel('logL(aims) - logL(tblite)')
    ax.set_ylabel('Density')
    ax.set_title(f'Difference Distribution (std: {diff.std():.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_weight_distribution(
    weights: np.ndarray,
    log_w_total: np.ndarray,
    ess: float,
    output_path: Path,
    title: str = "IS Weight Distribution",
):
    """
    Plot the distribution of importance sampling weights.

    This diagnostic plot helps assess the quality of IS reweighting.
    A good reweighting should have:
    - Weights not dominated by a few samples
    - Reasonable ESS (not too low)

    Args:
        weights: Normalized weights (sum to 1)
        log_w_total: Unnormalized log-weights
        ess: Effective sample size
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: Weight histogram (linear scale)
    ax = axes[0]
    ax.hist(weights, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(1.0 / len(weights), color='r', linestyle='--', lw=2,
               label=f'Uniform: {1.0/len(weights):.2e}')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Density')
    ax.set_title('Weight Distribution (linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: Log-weight histogram
    ax = axes[1]
    log_w_norm = np.log(weights + 1e-300)  # Avoid log(0)
    ax.hist(log_w_norm, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(np.log(1.0 / len(weights)), color='r', linestyle='--', lw=2,
               label=f'Uniform: {np.log(1.0/len(weights)):.1f}')
    ax.set_xlabel('log(Weight)')
    ax.set_ylabel('Density')
    ax.set_title('Weight Distribution (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Cumulative weight
    ax = axes[2]
    sorted_weights = np.sort(weights)[::-1]  # Descending
    cumsum = np.cumsum(sorted_weights)
    n_samples = len(weights)
    ax.plot(np.arange(1, n_samples + 1), cumsum, 'b-', lw=2)
    ax.axhline(0.5, color='r', linestyle='--', lw=1, alpha=0.7)
    ax.axhline(0.9, color='orange', linestyle='--', lw=1, alpha=0.7)

    # Find how many samples contribute 50% and 90% of weight
    n_50 = np.searchsorted(cumsum, 0.5) + 1
    n_90 = np.searchsorted(cumsum, 0.9) + 1
    ax.axvline(n_50, color='r', linestyle=':', lw=1, alpha=0.7, label=f'50%: {n_50} samples')
    ax.axvline(n_90, color='orange', linestyle=':', lw=1, alpha=0.7, label=f'90%: {n_90} samples')

    ax.set_xlabel('Number of samples (sorted by weight)')
    ax.set_ylabel('Cumulative weight')
    ax.set_title(f'Cumulative Weight (ESS: {ess:.0f} / {n_samples})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(n_samples, n_90 * 3))  # Zoom in on relevant region

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)
    return fig


def plot_training_losses(
    losses_dict: dict[str, dict],
    output_path: Path,
):
    """Plot training loss curves for all flow types."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'standard': 'C0', 'circular': 'C1', 'mixed': 'C2'}

    for name, losses in losses_dict.items():
        epochs = range(1, len(losses['train']) + 1)
        ax.plot(epochs, losses['val'], label=f'{name} (val)', color=colors.get(name, 'gray'))
        ax.plot(epochs, losses['train'], '--', alpha=0.5, color=colors.get(name, 'gray'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Weighted NLL)')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_marginals_comparison(
    samples_dict: dict[str, np.ndarray],
    weights_dict: dict[str, np.ndarray | None],
    param_names: list[str],
    is_circular: np.ndarray,
    output_path: Path,
):
    """Plot 1D marginal comparisons.

    All samples are expected to be in [0, 1] range (unit hypercube).
    Uses step histograms (outline only) to avoid color overlap.
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(samples_dict)))

    for idx, (name, samples) in enumerate(samples_dict.items()):
        weights = weights_dict.get(name)
        color = colors[idx]

        for i, ax in enumerate(axes):
            # All dims in [0, 1] range
            bins = np.linspace(0, 1, 50)

            # Use step histogram (outline only) to avoid color overlap
            ax.hist(samples[:, i], bins=bins, density=True, histtype='step',
                    weights=weights, color=color, linewidth=1.5,
                    label=name if i == 0 else None)

            # Mark circular vs linear in xlabel
            label_suffix = " (circ)" if is_circular[i] else " (lin)"
            ax.set_xlabel(param_names[i] + label_suffix)
            ax.set_xlim(0, 1)

            if i == 0:
                ax.set_ylabel('Density')

    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('1D Marginal Comparison (Unit Hypercube [0, 1])')
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_corner_comparison(
    samples1: np.ndarray,
    samples2: np.ndarray,
    weights1: np.ndarray | None,
    weights2: np.ndarray | None,
    labels: list[str],
    names: tuple[str, str],
    is_circular: np.ndarray,
    output_path: Path,
    title: str = "",
):
    """Create corner plot comparing two sample sets.

    All samples are expected to be in [0, 1] range (unit hypercube).
    """
    # Add topology suffix to labels
    labels_with_type = [
        f"{l} (circ)" if is_circular[i] else f"{l} (lin)"
        for i, l in enumerate(labels)
    ]

    if HAS_CORNER:
        # All dims in [0, 1] range
        ranges = [(0, 1) for _ in range(len(labels))]

        fig = corner.corner(
            samples1,
            weights=weights1,
            labels=labels_with_type,
            color='C0',
            hist_kwargs={'density': True, 'alpha': 0.6},
            plot_density=True,
            plot_datapoints=False,
            fill_contours=True,
            levels=[0.68, 0.95],
            range=ranges,
            smooth=1.0,
        )

        corner.corner(
            samples2,
            weights=weights2,
            fig=fig,
            color='C1',
            hist_kwargs={'density': True, 'alpha': 0.6},
            plot_density=True,
            plot_datapoints=False,
            fill_contours=True,
            levels=[0.68, 0.95],
            range=ranges,
            smooth=1.0,
        )

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='C0', lw=2, label=names[0]),
            Line2D([0], [0], color='C1', lw=2, label=names[1]),
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

        if title:
            fig.suptitle(title, fontsize=14)

    else:
        # Simple fallback without corner package
        n = len(labels)
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

        # Use [0, 1] range for all dims
        bins = np.linspace(0, 1, 30)

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis('off')
                elif i == j:
                    ax.hist(samples1[:, i], bins=bins, density=True, alpha=0.5,
                            weights=weights1, color='C0')
                    ax.hist(samples2[:, i], bins=bins, density=True, alpha=0.5,
                            weights=weights2, color='C1')
                    ax.set_xlim(0, 1)
                else:
                    ax.scatter(samples1[:, j], samples1[:, i], alpha=0.1, s=1, c='C0')
                    ax.scatter(samples2[:, j], samples2[:, i], alpha=0.1, s=1, c='C1')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                if i == n - 1:
                    ax.set_xlabel(labels_with_type[j])
                if j == 0:
                    ax.set_ylabel(labels_with_type[i])

        if title:
            fig.suptitle(title)
        fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_is_circular(s: str) -> np.ndarray:
    """Parse is_circular string like 'TTFFF' to boolean array."""
    return np.array([c.upper() == 'T' for c in s])


def main():
    parser = argparse.ArgumentParser(
        description="Weighted Posterior Fitting with Mixed Topology Flows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--dead-points",
        type=Path,
        help="Path to PolyChord dead points file",
    )
    data_group.add_argument(
        "--chains-root",
        type=Path,
        help="Path to PolyChord chains root (uses anesthetic)",
    )
    data_group.add_argument(
        "--synthetic",
        type=str,
        choices=list(SYNTHETIC_GENERATORS.keys()),
        help="Use synthetic data generator",
    )
    data_group.add_argument(
        "--list-synthetic",
        action="store_true",
        help="List available synthetic examples",
    )

    # IS reweighting (use with --chains-root)
    parser.add_argument(
        "--is-csv",
        type=Path,
        default=None,
        help="Path to IS CSV file with 'name', 'logL_aims' columns (use with --chains-root)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature in Kelvin for IS reweighting (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--aims-chains-root",
        type=Path,
        default=None,
        help="Path to direct aims PolyChord chains for ground truth comparison",
    )
    parser.add_argument(
        "--min-ess",
        type=float,
        default=100.0,
        help="Minimum ESS threshold for IS reweighting warning (default: 100)",
    )

    # Data configuration
    parser.add_argument(
        "--n-params",
        type=int,
        default=5,
        help="Number of parameters (for real data)",
    )
    parser.add_argument(
        "--is-circular",
        type=str,
        default=None,
        help="Circular dims mask, e.g., 'TTFFF' (T=circular, F=linear)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples (for synthetic data)",
    )

    # Training configuration
    parser.add_argument("--n-layers", type=int, default=8, help="Number of flow layers")
    parser.add_argument("--max-epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training mode configuration
    parser.add_argument(
        "--training-mode",
        type=str,
        default="weighted",
        choices=["weighted", "resample"],
        help="Training mode: 'weighted' uses importance weights in loss, "
             "'resample' draws samples according to weights (default: weighted)",
    )
    parser.add_argument(
        "--resample-noise",
        action="store_true",
        help="Add small Gaussian noise (std=1e-3) to resampled points in resample mode. "
             "Helps prevent overfitting when ESS is low.",
    )
    parser.add_argument(
        "--gradient-clip",
        action="store_true",
        help="Enable gradient norm clipping (max_norm=1.0). "
             "Recommended when using unclipped importance weights.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs_weighted_posterior"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained flow models for later sampling. "
             "Models are saved as .eqx files with accompanying .json metadata.",
    )

    # Flow selection
    parser.add_argument(
        "--flows",
        type=str,
        nargs="+",
        default=["standard", "circular", "mixed"],
        choices=["standard", "circular", "mixed"],
        help="Which flows to train",
    )

    args = parser.parse_args()

    # Handle --list-synthetic
    if args.list_synthetic:
        print("Available synthetic data generators:")
        for name, func in SYNTHETIC_GENERATORS.items():
            print(f"  {name}: {func.__doc__.split(chr(10))[1].strip()}")
        return

    # Validate --is-csv usage
    if args.is_csv and not args.chains_root:
        parser.error("--is-csv requires --chains-root")

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = jr.key(args.seed)

    # Load data
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    if args.dead_points:
        data = load_polychord_dead_points(args.dead_points, args.n_params)
    elif args.chains_root:
        if args.is_csv:
            # IS reweighting mode: combine PolyChord chains with aims energies
            data = load_is_reweighted_data(
                chains_root=args.chains_root,
                is_csv_path=args.is_csv,
                n_params=args.n_params,
                temperature=args.temperature,
            )
        else:
            # Standard anesthetic loading
            data = load_anesthetic_chains(args.chains_root, args.n_params)
    elif args.synthetic:
        key, subkey = jr.split(key)
        data = SYNTHETIC_GENERATORS[args.synthetic](subkey, args.n_samples)
    else:
        raise ValueError("No data source specified")

    # Override is_circular if provided
    if args.is_circular:
        is_circular = parse_is_circular(args.is_circular)
        if len(is_circular) != data.samples.shape[1]:
            raise ValueError(
                f"is_circular length {len(is_circular)} != data dim {data.samples.shape[1]}"
            )
        data = DataResult(
            samples=data.samples,
            weights=data.weights,
            is_circular=is_circular,
            ground_truth=data.ground_truth,
            param_names=data.param_names,
            # Preserve IS diagnostic fields
            logL_tblite=data.logL_tblite,
            logL_aims=data.logL_aims,
            log_w_ns=data.log_w_ns,
            ess=data.ess,
        )

    # Load direct aims samples for ground truth comparison (if provided)
    aims_samples, aims_weights = None, None
    if args.aims_chains_root:
        aims_samples, aims_weights, _ = load_aims_direct_samples(
            args.aims_chains_root, args.n_params
        )

    # IS diagnostic plots and ESS check (only for IS mode)
    if data.logL_tblite is not None and data.logL_aims is not None:
        print("\n" + "-" * 40)
        print("IS DIAGNOSTICS")
        print("-" * 40)

        # Plot logL correlation
        plot_logL_correlation(
            data.logL_tblite,
            data.logL_aims,
            output_path=args.output_dir / "diagnostic_logL_correlation.png",
        )

        # Compute log_w_total for weight distribution plot
        # Use the same formula as load_is_reweighted_data:
        # log_w_imp = log(w_tblite) + logL_aims - logL_tblite
        # where w_tblite is the posterior weight (not prior volume weight)
        # For visualization, we approximate using log_w_ns + logL_tblite as log(w_tblite)
        log_w_tblite_approx = data.log_w_ns + data.logL_tblite
        log_w_total = log_w_tblite_approx + data.logL_aims - data.logL_tblite

        # Plot weight distribution
        plot_weight_distribution(
            data.weights,
            log_w_total,
            data.ess,
            output_path=args.output_dir / "diagnostic_weight_distribution.png",
        )

        # ESS warning check
        if data.ess < args.min_ess:
            print(f"\n  WARNING: ESS ({data.ess:.1f}) is below threshold ({args.min_ess})")
            print("  IS reweighting may have poor overlap between posteriors.")
            print("  Consider: running more aims calculations or adjusting temperature.")

    print(f"\nData shape: {data.samples.shape}")
    print(f"is_circular: {data.is_circular}")
    print(f"Has ground truth: {data.ground_truth is not None}")

    # Training mode configuration
    print(f"\n  Training mode: {args.training_mode}")
    if args.training_mode == "resample":
        print(f"  Noise injection: {args.resample_noise}")
    print(f"  Gradient clipping: {args.gradient_clip}")

    # For visualization, optionally clip weights to reduce outlier influence in plots
    # Training uses unclipped weights - the training mode handles high-variance weights
    weights_clipped = clip_weights(data.weights, percentile=99)  # Only for plotting

    # Verify data is in [0, 1] (expected for PolyChord and our synthetic generators)
    data_min, data_max = data.samples.min(), data.samples.max()
    print(f"  Data range: [{data_min:.4f}, {data_max:.4f}]")
    if data_min < 0 or data_max > 1:
        print("  WARNING: Data outside [0, 1] range - preprocessing assumes unit hypercube")

    # Create preprocessor for inverse transforms (flow output → [0, 1])
    # Training functions handle their own clipping and scaling internally
    preprocessor = Preprocessor(is_circular=data.is_circular)

    # Train flows
    # Note: Training functions receive raw [0, 1] data and handle clipping internally
    # Weights are NOT clipped for training - gradient clipping or resampling handles variance
    flows = {}
    losses_dict = {}

    if "standard" in args.flows:
        key, subkey = jr.split(key)
        flows["standard"], losses_dict["standard"] = train_standard_flow(
            data.samples, data.weights, subkey,  # Use unclipped weights
            n_layers=args.n_layers,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            add_noise=args.resample_noise,
            gradient_clip=args.gradient_clip,
        )

    if "circular" in args.flows:
        key, subkey = jr.split(key)
        flows["circular"], losses_dict["circular"] = train_circular_flow(
            data.samples, data.weights, subkey,  # Use unclipped weights
            n_layers=args.n_layers,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            add_noise=args.resample_noise,
            gradient_clip=args.gradient_clip,
        )

    if "mixed" in args.flows:
        key, subkey = jr.split(key)
        flows["mixed"], losses_dict["mixed"] = train_mixed_flow(
            data.samples, data.weights, data.is_circular, subkey,  # Use unclipped weights
            n_layers=args.n_layers,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            add_noise=args.resample_noise,
            gradient_clip=args.gradient_clip,
        )

    # Save trained models if requested
    if args.save_models and flows:
        print("\n" + "=" * 60)
        print("SAVING TRAINED MODELS")
        print("=" * 60)
        param_names = data.param_names or [f"p{i}" for i in range(data.samples.shape[1])]
        for name, flow in flows.items():
            save_trained_flow(
                flow=flow,
                flow_type=name,
                is_circular=data.is_circular,
                param_names=param_names,
                output_dir=args.output_dir,
            )

    # Generate samples from trained flows
    print("\n" + "=" * 60)
    print("GENERATING SAMPLES FROM TRAINED FLOWS")
    print("=" * 60)

    n_gen = 10000
    flow_samples = {}

    for name, flow in flows.items():
        key, subkey = jr.split(key)
        samples_flow = flow.sample(subkey, (n_gen,))
        samples_flow = np.array(samples_flow)

        # Transform back to [0, 1] using mode-specific inverse
        flow_samples[name] = preprocessor.inverse(samples_flow, mode=name)
        print(f"  {name}: generated {n_gen} samples, range: [{flow_samples[name].min():.4f}, {flow_samples[name].max():.4f}]")

    # Visualization
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    # Training losses
    if losses_dict:
        plot_training_losses(losses_dict, args.output_dir / "training_losses.png")

    # Marginal comparison
    param_names = data.param_names or [f"p{i}" for i in range(data.samples.shape[1])]

    samples_for_plot = {"Original (weighted)": data.samples}
    weights_for_plot = {"Original (weighted)": weights_clipped}

    if data.ground_truth is not None:
        samples_for_plot["Ground Truth"] = data.ground_truth
        weights_for_plot["Ground Truth"] = None

    # Add direct aims samples if available (as ground truth for IS mode)
    if aims_samples is not None:
        samples_for_plot["Direct aims"] = aims_samples
        weights_for_plot["Direct aims"] = aims_weights

    for name, samples in flow_samples.items():
        samples_for_plot[f"Flow ({name})"] = samples
        weights_for_plot[f"Flow ({name})"] = None

    plot_marginals_comparison(
        samples_for_plot,
        weights_for_plot,
        param_names,
        data.is_circular,
        args.output_dir / "marginals_comparison.png",
    )

    # Corner plot: Ground truth vs original weighted samples (if ground truth available)
    if data.ground_truth is not None:
        plot_corner_comparison(
            data.ground_truth,
            data.samples,
            None,
            weights_clipped,
            param_names,
            ("Ground Truth", "Original (weighted)"),
            data.is_circular,
            args.output_dir / "corner_ground_truth_vs_original.png",
            title="Ground Truth vs Original Weighted Samples",
        )

    # Corner plot: IS-reweighted vs direct aims (if aims samples available)
    # Use UNCLIPPED IS weights for fair comparison with direct aims
    # (clipping introduces bias that would make the comparison invalid)
    if aims_samples is not None:
        plot_corner_comparison(
            data.samples,
            aims_samples,
            data.weights,  # Use unclipped IS weights for accurate comparison
            aims_weights,
            param_names,
            ("IS-reweighted", "Direct aims"),
            data.is_circular,
            args.output_dir / "corner_is_vs_direct_aims.png",
            title="IS-reweighted vs Direct aims posterior",
        )

    # Determine reference for flow comparison
    # Priority: direct aims > synthetic ground truth > original weighted
    if aims_samples is not None:
        reference = aims_samples
        reference_weights = aims_weights
        reference_name = "Direct aims"
    elif data.ground_truth is not None:
        reference = data.ground_truth
        reference_weights = None
        reference_name = "Ground Truth"
    else:
        reference = data.samples
        reference_weights = weights_clipped
        reference_name = "Original (weighted)"

    for name, samples in flow_samples.items():
        plot_corner_comparison(
            reference,
            samples,
            reference_weights,
            None,
            param_names,
            (reference_name, f"Flow ({name})"),
            data.is_circular,
            args.output_dir / f"corner_{name}.png",
            title=f"{name.upper()} Flow vs {reference_name}",
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
