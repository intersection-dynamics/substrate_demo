#!/usr/bin/env python3
"""
gauss_sweep.py

Parameter sweep: vary lambda_G (Gauss constraint strength) and measure:
- Ground state energy
- Antisymmetry score
- Singlet fraction at overlap
- CHSH S parameter (Bell inequality test)
- Gauss energy expectation
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Try normal import first; fall back to the chat env path if needed
try:
    from substrate_engine_3d import TwoFermion3DParams, run_twofermion3d_experiment
except ImportError:
    # Fallback path used in the chat/sandbox environment
    sys.path.insert(0, "/mnt/user-data/uploads")
    from substrate_engine_3d import TwoFermion3DParams, run_twofermion3d_experiment


def run_gauss_sweep(lambda_G_values):
    """Run parameter sweep over lambda_G values."""

    results = {
        "lambda_G": [],
        "E0": [],
        "antisym_score": [],
        "singlet_fraction": [],
        "S_chsh": [],
        "E_gauss": [],
        "overlap_prob": [],
    }

    for i, lambda_G in enumerate(lambda_G_values):
        print(f"\n{'=' * 70}")
        print(f"Running: lambda_G = {lambda_G:.2f} ({i + 1}/{len(lambda_G_values)})")
        print(f"{'=' * 70}")

        # Set up parameters
        params = TwoFermion3DParams(
            Lx=2,
            Ly=2,
            Lz=2,
            J_hop=1.0,
            m=0.1,
            g_defrag=1.0,
            sigma_defrag=1.0,
            lambda_G=lambda_G,
            lambda_S=-1.0,
            lambda_T=0.0,
            J_exch=1.0,
            max_eigsh_iter=5000,
            k_eigs=1,
        )

        # Run experiment
        result = run_twofermion3d_experiment(params)

        # Store results
        results["lambda_G"].append(lambda_G)
        results["E0"].append(result["E0"])
        results["antisym_score"].append(result["antisymmetry"]["antisym_score"])
        results["singlet_fraction"].append(result["overlap"]["singlet_fraction"])
        results["S_chsh"].append(result["S_chsh"])
        results["E_gauss"].append(result["E_gauss"])
        results["overlap_prob"].append(result["overlap"]["overlap_prob"])

        print(
            f"\n[QUICK RESULTS] lambda_G={lambda_G:.2f}: "
            f"S_chsh={result['S_chsh']:.4f}, "
            f"antisym={result['antisymmetry']['antisym_score']:.4f}, "
            f"singlet_frac={result['overlap']['singlet_fraction']:.4f}"
        )

    return results


def plot_results(results):
    """Create visualization of sweep results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Gauss Constraint Strength (λ_G) Parameter Sweep\n"
        "3D Two-Excitation Substrate with Bell Test",
        fontsize=14,
        fontweight="bold",
    )

    lambda_G = np.array(results["lambda_G"])

    # Plot 1: Ground state energy
    ax = axes[0, 0]
    ax.plot(lambda_G, results["E0"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("λ_G (Gauss strength)", fontsize=11)
    ax.set_ylabel("Ground State Energy E₀", fontsize=11)
    ax.set_title("Energy vs Constraint Strength")
    ax.grid(True, alpha=0.3)

    # Plot 2: CHSH S parameter - THE KEY PLOT
    ax = axes[0, 1]
    S_abs = np.abs(results["S_chsh"])
    ax.plot(lambda_G, S_abs, "o-", linewidth=2, markersize=6, color="red", label="|S|")
    ax.axhline(
        y=2.0,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Classical bound (|S| ≤ 2)",
    )
    ax.axhline(
        y=2 * np.sqrt(2),
        color="green",
        linestyle="--",
        linewidth=2,
        label="Tsirelson bound (2√2)",
    )
    ax.fill_between(
        lambda_G,
        2.0,
        2 * np.sqrt(2),
        alpha=0.2,
        color="yellow",
        label="Quantum regime",
    )
    ax.set_xlabel("λ_G (Gauss strength)", fontsize=11)
    ax.set_ylabel("|S| (CHSH parameter)", fontsize=11)
    ax.set_title("Bell Inequality Violation", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3.0])

    # Plot 3: Antisymmetry score
    ax = axes[0, 2]
    ax.plot(
        lambda_G,
        results["antisym_score"],
        "o-",
        linewidth=2,
        markersize=6,
        color="purple",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("λ_G (Gauss strength)", fontsize=11)
    ax.set_ylabel("Antisymmetry Score", fontsize=11)
    ax.set_title("Exchange Antisymmetry")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Plot 4: Singlet fraction at overlap
    ax = axes[1, 0]
    ax.plot(
        lambda_G,
        results["singlet_fraction"],
        "o-",
        linewidth=2,
        markersize=6,
        color="orange",
    )
    ax.set_xlabel("λ_G (Gauss strength)", fontsize=11)
    ax.set_ylabel("Singlet Fraction at Overlap", fontsize=11)
    ax.set_title("Spin Character")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Plot 5: Gauss energy expectation
    ax = axes[1, 1]
    ax.plot(
        lambda_G,
        results["E_gauss"],
        "o-",
        linewidth=2,
        markersize=6,
        color="brown",
    )
    ax.set_xlabel("λ_G (Gauss strength)", fontsize=11)
    ax.set_ylabel("<H_Gauss>", fontsize=11)
    ax.set_title("Gauss Energy Cost")
    ax.grid(True, alpha=0.3)

    # Plot 6: Overlap probability
    ax = axes[1, 2]
    ax.plot(
        lambda_G,
        results["overlap_prob"],
        "o-",
        linewidth=2,
        markersize=6,
        color="teal",
    )
    ax.set_xlabel("λ_G (Gauss strength)", fontsize=11)
    ax.set_ylabel("Spatial Overlap Prob", fontsize=11)
    ax.set_title("P(r₁ = r₂)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Define sweep range
    lambda_G_values = np.linspace(0.0, 10.0, 21)

    print("\n" + "=" * 70)
    print("GAUSS CONSTRAINT PARAMETER SWEEP")
    print("Testing Ben's substrate framework: Does Gauss law create entanglement?")
    print("=" * 70)
    print(f"\nSweeping lambda_G from {lambda_G_values[0]:.1f} to {lambda_G_values[-1]:.1f}")
    print(f"Number of points: {len(lambda_G_values)}")
    print(f"\nFixed parameters:")
    print(f"  Lattice: 2×2×2 (8 sites)")
    print(f"  J_exch = 1.0 (exchange interaction)")
    print(f"  lambda_S = -1.0 (singlet bonus)")
    print(f"  J_hop = 1.0 (hopping)")
    print("\n" + "=" * 70)

    # Run sweep
    results = run_gauss_sweep(lambda_G_values)

    # Create plots
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)
    fig = plot_results(results)

    # Determine output directory relative to this script
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results (plot)
    output_path = output_dir / "gauss_sweep_results.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")

    # Save data (npz)
    data_path = output_dir / "gauss_sweep_data.npz"
    np.savez(str(data_path), **results)
    print(f"✓ Data saved to: {data_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find transition point where |S| crosses 2.0
    S_abs = np.abs(results["S_chsh"])
    quantum_mask = S_abs > 2.0
    if np.any(quantum_mask):
        first_quantum_idx = np.where(quantum_mask)[0][0]
        lambda_G_transition = results["lambda_G"][first_quantum_idx]
        print(
            f"\n✓ Bell inequality violation (|S| > 2) begins at λ_G ≈ {lambda_G_transition:.2f}"
        )

        # Check if it saturates Tsirelson bound
        max_S = np.max(S_abs)
        if max_S > 2.7:  # Close to 2√2 ≈ 2.828
            print(
                f"✓ Maximum |S| = {max_S:.4f} "
                f"(approaches Tsirelson bound 2√2 ≈ 2.828)"
            )
            print("  → Maximal quantum entanglement achieved!")
    else:
        print("\n✗ No Bell inequality violation observed in this parameter range")

    # Check antisymmetry
    if np.all(np.array(results["antisym_score"]) > 0.99):
        print(f"\n✓ Perfect antisymmetry maintained across all λ_G values")

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
