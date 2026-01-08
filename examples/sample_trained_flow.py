#!/usr/bin/env python3
"""
Sample from a trained normalizing flow and visualize the posterior.

This script loads a flow model saved by weighted_posterior_fitting.py,
generates samples, and creates posterior visualizations.

Usage:
    # Sample from a saved model and plot
    python sample_trained_flow.py --model-dir ./outputs_weighted_posterior --flow-type mixed

    # Generate more samples
    python sample_trained_flow.py --model-dir ./outputs --flow-type standard --n-samples 50000

    # Save samples to file
    python sample_trained_flow.py --model-dir ./outputs --flow-type mixed --save-samples
"""

import argparse
import json
from pathlib import Path

import jax
# Enable new-style typed PRNG keys (required by flowjax)
jax.config.update("jax_enable_custom_prng", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import PRNGKeyArray

# FlowJAX imports
from flowjax.distributions import Uniform, TorusUniform
from flowjax.bijections import (
    RationalQuadraticSpline,
    CircularRationalQuadraticSpline,
)
from flowjax.flows import (
    masked_autoregressive_flow,
    circular_masked_autoregressive_flow,
    mixed_masked_autoregressive_flow,
)


def create_flow_skeleton(
    flow_type: str,
    dim: int,
    is_circular: np.ndarray,
    key: PRNGKeyArray,
    n_layers: int = 8,
    n_knots: int = 8,
):
    """Create an uninitialized flow with the same architecture as the saved model.

    Args:
        flow_type: Type of flow ("standard", "circular", "mixed")
        dim: Number of dimensions
        is_circular: Boolean array indicating circular dimensions
        key: JAX random key
        n_layers: Number of MAF layers (must match saved model)
        n_knots: Number of spline knots (must match saved model)

    Returns:
        Flow skeleton with same structure as saved model
    """
    if flow_type == "standard":
        base_dist = Uniform(minval=jnp.zeros(dim), maxval=jnp.ones(dim))
        transformer = RationalQuadraticSpline(
            knots=n_knots,
            interval=(0.0, 1.0),
            boundary_derivatives=None,
        )
        flow = masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            transformer=transformer,
            flow_layers=n_layers,
            nn_width=64,
            nn_depth=1,
        )
    elif flow_type == "circular":
        base_dist = TorusUniform(dim)
        transformer = CircularRationalQuadraticSpline(num_bins=n_knots)
        flow = circular_masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            transformer=transformer,
            flow_layers=n_layers,
            nn_width=64,
            nn_depth=1,
        )
    elif flow_type == "mixed":
        base_dist = TorusUniform(dim)
        is_circular_jax = jnp.array(is_circular)
        flow = mixed_masked_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            is_circular=is_circular_jax,
            linear_bounds=(0.0, 2 * jnp.pi),
            linear_boundary_derivatives=None,
            linear_transformer_kwargs={"knots": n_knots},
            circular_transformer_kwargs={"num_bins": n_knots},
            flow_layers=n_layers,
            nn_width=64,
            nn_depth=1,
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")

    return flow


def load_trained_flow(
    model_dir: Path,
    flow_type: str,
    key: PRNGKeyArray,
    n_layers: int = 8,
    n_knots: int = 8,
):
    """Load a trained flow from saved files.

    Args:
        model_dir: Directory containing saved model files
        flow_type: Type of flow to load ("standard", "circular", "mixed")
        key: JAX random key (for creating skeleton)
        n_layers: Number of MAF layers (must match saved model)
        n_knots: Number of spline knots (must match saved model)

    Returns:
        Tuple of (loaded_flow, metadata_dict)
    """
    model_dir = Path(model_dir)
    model_path = model_dir / f"flow_{flow_type}.eqx"
    metadata_path = model_dir / f"flow_{flow_type}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    dim = metadata["dim"]
    is_circular = np.array(metadata["is_circular"])

    # Create skeleton with same architecture
    skeleton = create_flow_skeleton(
        flow_type=flow_type,
        dim=dim,
        is_circular=is_circular,
        key=key,
        n_layers=n_layers,
        n_knots=n_knots,
    )

    # Load trained weights into skeleton
    flow = eqx.tree_deserialise_leaves(model_path, skeleton)

    return flow, metadata


def transform_samples_to_unit(
    samples: np.ndarray,
    flow_type: str,
) -> np.ndarray:
    """Transform flow samples back to [0, 1] unit hypercube.

    Args:
        samples: Raw samples from flow
        flow_type: Type of flow (affects scaling)

    Returns:
        Samples in [0, 1] range
    """
    if flow_type == "standard":
        # Standard flow outputs in [0, 1] already
        return np.clip(samples, 0, 1)
    else:
        # Circular and mixed flows output in [0, 2π], scale to [0, 1]
        samples_scaled = samples / (2 * np.pi)
        return np.clip(samples_scaled, 0, 1)


def plot_corner(
    samples: np.ndarray,
    param_names: list[str],
    is_circular: np.ndarray,
    output_path: Path,
    title: str = "Flow Posterior Samples",
):
    """Create a corner plot of the samples.

    Args:
        samples: Samples array of shape (n_samples, dim)
        param_names: Names for each parameter
        is_circular: Boolean array indicating circular dimensions
        output_path: Path to save the plot
        title: Plot title
    """
    dim = samples.shape[1]

    fig, axes = plt.subplots(dim, dim, figsize=(2.5 * dim, 2.5 * dim))

    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j] if dim > 1 else axes

            if j > i:
                ax.axis('off')
                continue

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, color='steelblue')
                ax.set_xlim(0, 1)
                if i == dim - 1:
                    ax.set_xlabel(param_names[i])
            else:
                # Off-diagonal: 2D scatter/density
                ax.scatter(samples[:, j], samples[:, i], alpha=0.1, s=1, color='steelblue')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                if i == dim - 1:
                    ax.set_xlabel(param_names[j])
                if j == 0:
                    ax.set_ylabel(param_names[i])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved corner plot: {output_path}")
    plt.close(fig)


def plot_marginals(
    samples: np.ndarray,
    param_names: list[str],
    is_circular: np.ndarray,
    output_path: Path,
    title: str = "Flow Posterior Marginals",
):
    """Create marginal distribution plots.

    Args:
        samples: Samples array of shape (n_samples, dim)
        param_names: Names for each parameter
        is_circular: Boolean array indicating circular dimensions
        output_path: Path to save the plot
        title: Plot title
    """
    dim = samples.shape[1]
    n_cols = min(4, dim)
    n_rows = (dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(dim):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        ax.hist(samples[:, i], bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='darkblue', linewidth=0.5)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)

        # Mark if circular
        if is_circular[i]:
            ax.set_title(f"{param_names[i]} (circular)", fontsize=10)
        else:
            ax.set_title(param_names[i], fontsize=10)

    # Hide unused subplots
    for i in range(dim, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved marginals plot: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Sample from a trained normalizing flow and visualize the posterior.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing saved model files (flow_*.eqx and flow_*_metadata.json)",
    )
    parser.add_argument(
        "--flow-type",
        type=str,
        required=True,
        choices=["standard", "circular", "mixed"],
        help="Type of flow to load and sample from",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=8,
        help="Number of MAF layers (must match saved model, default: 8)",
    )
    parser.add_argument(
        "--n-knots",
        type=int,
        default=8,
        help="Number of spline knots (must match saved model, default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as model-dir)",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save generated samples to a .npy file",
    )
    parser.add_argument(
        "--no-corner",
        action="store_true",
        help="Skip corner plot (faster for high dimensions)",
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or args.model_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize random key
    key = jr.PRNGKey(args.seed)

    print("=" * 60)
    print("LOADING TRAINED FLOW")
    print("=" * 60)

    key, subkey = jr.split(key)
    flow, metadata = load_trained_flow(
        model_dir=args.model_dir,
        flow_type=args.flow_type,
        key=subkey,
        n_layers=args.n_layers,
        n_knots=args.n_knots,
    )

    print(f"  Flow type: {metadata['flow_type']}")
    print(f"  Dimensions: {metadata['dim']}")
    print(f"  is_circular: {metadata['is_circular']}")
    print(f"  Parameters: {metadata['param_names']}")

    print("\n" + "=" * 60)
    print("GENERATING SAMPLES")
    print("=" * 60)

    key, subkey = jr.split(key)
    raw_samples = flow.sample(subkey, (args.n_samples,))
    raw_samples = np.array(raw_samples)

    # Transform to [0, 1] unit hypercube
    samples = transform_samples_to_unit(raw_samples, args.flow_type)

    print(f"  Generated {args.n_samples} samples")
    print(f"  Sample range: [{samples.min():.4f}, {samples.max():.4f}]")

    # Save samples if requested
    if args.save_samples:
        samples_path = output_dir / f"samples_{args.flow_type}.npy"
        np.save(samples_path, samples)
        print(f"  Saved samples to: {samples_path}")

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    param_names = metadata['param_names'] or [f"p{i}" for i in range(metadata['dim'])]
    is_circular = np.array(metadata['is_circular'])

    # Marginals plot
    plot_marginals(
        samples=samples,
        param_names=param_names,
        is_circular=is_circular,
        output_path=output_dir / f"sampled_marginals_{args.flow_type}.png",
        title=f"Posterior Marginals ({args.flow_type} flow, n={args.n_samples})",
    )

    # Corner plot (skip for very high dimensions)
    if not args.no_corner:
        if metadata['dim'] <= 10:
            plot_corner(
                samples=samples,
                param_names=param_names,
                is_circular=is_circular,
                output_path=output_dir / f"sampled_corner_{args.flow_type}.png",
                title=f"Posterior Corner Plot ({args.flow_type} flow)",
            )
        else:
            print(f"  Skipping corner plot (dim={metadata['dim']} > 10)")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
