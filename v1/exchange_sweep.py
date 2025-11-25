#!/usr/bin/env python3
"""
exchange_sweep.py

Parameter sweep: vary J_exch (exchange interaction strength) and measure:
- Ground state energy
- Antisymmetry score
- Singlet fraction at overlap
- CHSH S parameter (Bell inequality test)
- Overlap probability

Keep lambda_G fixed at 5.0 to maintain Gauss constraint.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try normal import first; fall back to sandbox path if needed
try:
    from substrate_engine_3d import TwoFermion3DParams, run_twofermion3d_experiment
except ImportError:
    sys.path.insert(0, "/mnt/user-data/uploads")
    from substrate_engine_3d import TwoFermion3DParams, run_twofermion3d_experiment


def run_exchange_sweep(J_exch_values):
    """Run parameter sweep over J_exch values."""

    results = {
        "J_exch": [],
        "E0": [],
        "antisym_score": [],
        "singlet_fraction": [],
        "S_chsh": [],
        "E_gauss": [],
        "overlap_prob": [],
    }

    for i, J_exch in enumerate(J_exch_values):
        print(f"\n{'=' * 70}")
        print(f"Running: J_exch = {J_exch:.2f} ({i + 1}/{len(J_exch_values)})")
        print(f"{'=' * 70}")

        params = TwoFermion3DParams(
            Lx=2,
            Ly=2,
            Lz=2,
            J_hop=1.0,
            m=0.1,
            g_defrag=1.0,
            sigma_defrag=1.0,
            lambda_G=5.0,  # Fixed!
            lambda_S=-1.0,
            lambda_T=0.0,
            J_exch=J_exch,  # Variable!
            max_eigsh_iter=5000,
            k_eigs=1,
        )

        result = run_twofermion3d_experiment(params)

        results["J_exch"].append(J_exch)
        results["E0"].append(result["E0"])
        results["antisym_score"].append(result["antisymmetry"]["antisym_score"])
        results["singlet_fraction"].append(result["overlap"]["singlet_fraction"])
        results["S_chsh"].append(result["S_chsh"])
        results["E_gauss"].append(result["E_gauss"])
        results["overlap_prob"].append(result["overlap"]["overlap_prob"])

        print(
            f"\n[QUICK RESULTS] J_exch={J_exch:.2f}: "
            f"S_chsh={result['S_chsh']:.4f}, "
            f"antisym={result['antisymmetry']['antisym_score']:.4f}, "
            f"singlet_frac={result['overlap']['singlet_fraction']:.4f}"
        )

    return results


def plot_results(results):
    """Create visualization of sweep results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Exchange Interaction Strength (J_exch) Parameter Sweep\n"
        "3D Two-Excitation Substrate with Bell Test (λ_G = 5.0 fixed)",
        fontsize=14,
        fontweight="bold",
    )

    J_exch = np.array(results["J_exch"])

    # Plot 1: Ground state energy
    ax = axes[0, 0]
    ax.plot(J_exch, results["E0"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("J_exch (Exchange strength)", fontsize=11)
    ax.set_ylabel("Ground State Energy E₀", fontsize=11)
    ax.set_title("Energy vs Exchange Strength")
    ax.grid(True, alpha=0.3)

    # Plot 2: CHSH S parameter - THE KEY PLOT
    ax = axes[0, 1]
    S_abs = np.abs(results["S_chsh"])
    ax.plot(
        J_exch,
        S_abs,
        "o-",
        linewidth=2,
        markersize=6,
        color="red",
        label="|S|",
    )
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
        J_exch,
        2.0,
        2 * np.sqrt(2),
        alpha=0.2,
        color="yellow",
        label="Quantum regime",
    )
    ax.set_xlabel("J_exch (Exchange strength)", fontsize=11)
    ax.set_ylabel("|S| (CHSH parameter)", fontsize=11)
    ax.set_title("Bell Inequality Violation", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3.0])

    # Plot 3: Antisymmetry score
    ax = axes[0, 2]
    ax.plot(
        J_exch,
        results["antisym_score"],
        "o-",
        linewidth=2,
        markersize=6,
        color="purple",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("J_exch (Exchange strength)", fontsize=11)
    ax.set_ylabel("Antisymmetry Score", fontsize=11)
    ax.set_title("Exchange Antisymmetry")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Plot 4: Singlet fraction at overlap
    ax = axes[1, 0]
    ax.plot(
        J_exch,
        results["singlet_fraction"],
        "o-",
        linewidth=2,
        markersize=6,
        color="orange",
    )
    ax.set_xlabel("J_exch (Exchange strength)", fontsize=11)
    ax.set_ylabel("Singlet Fraction at Overlap", fontsize=11)
    ax.set_title("Spin Character")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Plot 5: Gauss energy expectation
    ax = axes[1, 1]
    ax.plot(
        J_exch,
        results["E_gauss"],
        "o-",
        linewidth=2,
        markersize=6,
        color="brown",
    )
    ax.set_xlabel("J_exch (Exchange strength)", fontsize=11)
    ax.set_ylabel("<H_Gauss>", fontsize=11)
    ax.set_title("Gauss Energy (should be constant)")
    ax.grid(True, alpha=0.3)

    # Plot 6: Overlap probability
    ax = axes[1, 2]
    ax.plot(
        J_exch,
        results["overlap_prob"],
        "o-",
        linewidth=2,
        markersize=6,
        color="teal",
    )
    ax.set_xlabel("J_exch (Exchange strength)", fontsize=11)
    ax.set_ylabel("Spatial Overlap Prob", fontsize=11)
    ax.set_title("P(r₁ = r₂)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Define sweep range
    J_exch_values = np.linspace(0.0, 2.0, 21)

    print("\n" + "=" * 70)
    print("EXCHANGE INTERACTION PARAMETER SWEEP")
    print("Testing: Is exchange necessary for Bell violation?")
    print("=" * 70)
    print(f"\nSweeping J_exch from {J_exch_values[0]:.1f} to {J_exch_values[-1]:.1f}")
    print(f"Number of points: {len(J_exch_values)}")
    print(f"\nFixed parameters:")
    print(f"  lambda_G = 5.0 (Gauss constraint - FIXED)")
    print(f"  lambda_S = -1.0 (singlet bonus)")
    print(f"  Lattice: 2×2×2 (8 sites)")
    print("\n" + "=" * 70)

    # Run sweep
    results = run_exchange_sweep(J_exch_values)

    # Create plots
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)
    fig = plot_results(results)

    # Output directory relative to this script
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results (plot)
    output_path = output_dir / "exchange_sweep_results.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")

    # Save data (npz)
    data_path = output_dir / "exchange_sweep_data.npz"
    np.savez(str(data_path), **results)
    print(f"✓ Data saved to: {data_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    S_abs = np.abs(results["S_chsh"])

    # Check if Bell violation persists at J_exch = 0
    if S_abs[0] > 2.0:
        print(f"\n✓ Bell violation exists even at J_exch = 0: |S| = {S_abs[0]:.4f}")
        print("  → Exchange interaction is NOT necessary for entanglement!")
    else:
        print(f"\n✗ No Bell violation at J_exch = 0: |S| = {S_abs[0]:.4f}")
        print("  → Exchange interaction IS necessary for entanglement")

        # Find transition
        quantum_mask = S_abs > 2.0
        if np.any(quantum_mask):
            first_quantum_idx = np.where(quantum_mask)[0][0]
            J_transition = results["J_exch"][first_quantum_idx]
            print(f"  → Bell violation begins at J_exch ≈ {J_transition:.2f}")

    # Check antisymmetry trend
    antisym_0 = results["antisym_score"][0]
    antisym_max = results["antisym_score"][-1]
    print(f"\nAntisymmetry: J_exch=0 → {antisym_0:.4f}, J_exch=2.0 → {antisym_max:.4f}")

    if antisym_0 < 0.5 and antisym_max > 0.95:
        print("  → Exchange interaction CREATES antisymmetry!")
    elif antisym_0 > 0.95:
        print("  → Antisymmetry exists even without exchange interaction")

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
