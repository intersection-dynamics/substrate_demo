#!/usr/bin/env python3
"""
information_extraction_test.py

Test: Information flows OUT of substrate INTO classical reality.

Key predictions:
1. S_substrate (full state entropy) should DECREASE over time
2. I_classical (mutual information between particles) should INCREASE over time
3. Energy changes should correlate with extraction rate

Method:
- Start with pure ground state (maximal substrate info, minimal classical info)
- Apply sequential perturbations (force extraction events)
- Measure BOTH substrate and classical information at each step
- Track the flow: substrate → classical
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm  # (currently unused but kept)
from pathlib import Path

# Try project-local import first, then fall back to sandbox path
try:
    from substrate_engine_3d import (
        TwoFermion3DParams,
        build_twofermion3d_hamiltonian,
        decode_basis_3d,
    )
except ImportError:
    sys.path.insert(0, "/mnt/user-data/uploads")
    from substrate_engine_3d import (
        TwoFermion3DParams,
        build_twofermion3d_hamiltonian,
        decode_basis_3d,
    )


def apply_local_perturbation(psi, params, site, strength=0.1, seed=None):
    """Apply local perturbation (same as before)."""
    if seed is not None:
        np.random.seed(seed)

    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    perturbation = np.zeros((dim, dim), dtype=complex)
    rand_phase = np.random.random() * 2 * np.pi

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)

        if r1 == site or r2 == site:
            perturbation[idx, idx] = strength * np.exp(1j * rand_phase)

            for idx2 in range(max(0, idx - 5), min(dim, idx + 5)):
                if idx != idx2:
                    r1_2, s1_2, r2_2, s2_2 = decode_basis_3d(idx2, Ns)
                    if (r1_2 == site or r2_2 == site):
                        coupling = (
                            strength
                            * 0.1
                            * np.random.randn()
                            * np.exp(1j * np.random.random() * 2 * np.pi)
                        )
                        perturbation[idx, idx2] = coupling

    perturbation = (perturbation + perturbation.conj().T) / 2

    from scipy.linalg import expm

    U_pert = expm(-1j * perturbation)
    psi_new = U_pert @ psi

    norm = np.sqrt(float(np.vdot(psi_new, psi_new).real))
    if norm > 0:
        psi_new /= norm

    return psi_new


def substrate_entropy(psi):
    """
    Compute substrate information: entropy of the full quantum state.

    For a pure state |ψ⟩, S = 0 (maximal substrate info, minimal classical info).
    As state becomes mixed, S increases (info leaking to classical).

    Since we have pure state, we measure entropy of reduced density matrix
    to see how much the state has "spread" across the Hilbert space.
    """
    # For pure state, use participation ratio as entropy measure
    # PR = 1/sum(|c_i|^4) measures how many basis states participate

    amplitudes = np.abs(psi) ** 2
    amplitudes = amplitudes[amplitudes > 1e-14]  # Filter numerical zeros

    # Shannon entropy of probability distribution
    S = -np.sum(amplitudes * np.log(amplitudes + 1e-16))

    return float(S)


def compute_single_particle_rdm(psi, params, particle=1):
    """
    Compute reduced density matrix for single particle.
    Traces out the other particle's degrees of freedom.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim_single = Ns * 2

    rho = np.zeros((dim_single, dim_single), dtype=complex)

    dim = Ns * 2 * Ns * 2

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        for idx2 in range(dim):
            r1_p, s1_p, r2_p, s2_p = decode_basis_3d(idx2, Ns)

            if particle == 1:
                # Trace out particle 2
                if r2 == r2_p and s2 == s2_p:
                    i = r1 * 2 + s1
                    j = r1_p * 2 + s1_p
                    rho[i, j] += psi[idx] * np.conj(psi[idx2])
            else:
                # Trace out particle 1
                if r1 == r1_p and s1 == s1_p:
                    i = r2 * 2 + s2
                    j = r2_p * 2 + s2_p
                    rho[i, j] += psi[idx] * np.conj(psi[idx2])

    return rho


def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(ρ log ρ)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    S = -np.sum(eigvals * np.log(eigvals + 1e-16))
    return float(S.real)


def mutual_information(psi, params):
    """
    Compute mutual information I(1:2) between the two particles.

    I(1:2) = S(ρ₁) + S(ρ₂) - S(ρ₁₂)

    This measures classical correlations between particles.
    High I(1:2) = strong classical correlations = info has "emerged" into observables.
    """
    # Get reduced density matrices
    rho1 = compute_single_particle_rdm(psi, params, particle=1)
    rho2 = compute_single_particle_rdm(psi, params, particle=2)

    # Compute individual entropies
    S1 = von_neumann_entropy(rho1)
    S2 = von_neumann_entropy(rho2)

    # For pure state, S(ρ₁₂) = 0
    S12 = 0.0

    # Mutual information
    I = S1 + S2 - S12

    return I, S1, S2


def classical_information(psi, params):
    """
    Measure "classical information" = how much information is accessible
    to local measurements on individual particles.

    This is quantified by mutual information I(1:2).
    """
    I, S1, S2 = mutual_information(psi, params)
    return I, S1, S2


def get_ground_state(params):
    """Get ground state of system."""
    from scipy.sparse.linalg import eigsh

    H = build_twofermion3d_hamiltonian(params)
    evals, evecs = eigsh(H, k=1, which="SA", maxiter=10000)
    psi0 = evecs[:, 0]

    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    E0 = float(evals[0].real)
    return psi0, E0, H


def run_information_extraction_test(params, n_perturbations=20, strength=0.15):
    """
    Main test: Track information flow from substrate to classical observables.
    """

    print("\n" + "=" * 70)
    print("INFORMATION EXTRACTION TEST")
    print("Ben's Hypothesis: Information flows OUT of substrate INTO classical")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Lattice: {params.Lx}×{params.Ly}×{params.Lz}")
    print(f"  Perturbations: {n_perturbations}")
    print(f"  Strength: {strength}")
    print(f"  λ_G (Gauss): {params.lambda_G}")
    print(f"  J_exch: {params.J_exch}")
    print("\n" + "=" * 70)

    # Get ground state
    print("\n[1/4] Computing ground state...")
    psi, E0, H = get_ground_state(params)
    print(f"      Ground state energy: {E0:.6f}")

    # Initial measurements
    print("\n[2/4] Measuring initial information content...")
    S_sub_0 = substrate_entropy(psi)
    I_class_0, S1_0, S2_0 = classical_information(psi, params)

    print(f"      S_substrate (full state): {S_sub_0:.6f}")
    print(f"      I_classical (mutual info): {I_class_0:.6f}")
    print(f"      S₁ (particle 1): {S1_0:.6f}")
    print(f"      S₂ (particle 2): {S2_0:.6f}")

    # Storage
    results = {
        "step": [0],
        "energy": [E0],
        "S_substrate": [S_sub_0],
        "I_classical": [I_class_0],
        "S1": [S1_0],
        "S2": [S2_0],
        "delta_S_substrate": [0.0],
        "delta_I_classical": [0.0],
        "delta_energy": [0.0],
    }

    print(f"\n[3/4] Applying {n_perturbations} perturbations and tracking info flow...")

    Ns = params.Lx * params.Ly * params.Lz

    for step in range(1, n_perturbations + 1):
        # Apply perturbation
        site = np.random.randint(0, Ns)
        psi = apply_local_perturbation(psi, params, site, strength=strength, seed=step)

        # Measure energy
        E = float(np.vdot(psi, H @ psi).real)
        delta_E = E - results["energy"][-1]

        # Measure substrate information (full state)
        S_sub = substrate_entropy(psi)
        delta_S_sub = S_sub - results["S_substrate"][-1]

        # Measure classical information (mutual info)
        I_class, S1, S2 = classical_information(psi, params)
        delta_I_class = I_class - results["I_classical"][-1]

        # Store results
        results["step"].append(step)
        results["energy"].append(E)
        results["S_substrate"].append(S_sub)
        results["I_classical"].append(I_class)
        results["S1"].append(S1)
        results["S2"].append(S2)
        results["delta_S_substrate"].append(delta_S_sub)
        results["delta_I_classical"].append(delta_I_class)
        results["delta_energy"].append(delta_E)

        if step % 5 == 0 or step == 1:
            print(f"\n  Step {step}/{n_perturbations}:")
            print(f"    S_substrate: {S_sub:.6f} (Δ = {delta_S_sub:+.6f})")
            print(f"    I_classical: {I_class:.6f} (Δ = {delta_I_class:+.6f})")
            print(f"    Energy: {E:.6f} (Δ = {delta_E:+.6f})")

    print(f"\n[4/4] Analyzing results...")

    # Overall trends
    total_delta_S_sub = results["S_substrate"][-1] - results["S_substrate"][0]
    total_delta_I_class = results["I_classical"][-1] - results["I_classical"][0]
    total_delta_E = results["energy"][-1] - results["energy"][0]

    print(f"\n  Overall changes:")
    print(f"    ΔS_substrate (total): {total_delta_S_sub:+.6f}")
    print(f"    ΔI_classical (total): {total_delta_I_class:+.6f}")
    print(f"    ΔEnergy (total): {total_delta_E:+.6f}")

    print(f"\n  Ben's Predictions:")
    print(f"    1. S_substrate should DECREASE (info leaving)")
    print(f"    2. I_classical should INCREASE (info arriving)")
    print(f"    3. Energy cost correlated with extraction rate")

    # Check predictions
    print(f"\n  Prediction Check:")

    if total_delta_S_sub < 0:
        print(f"    ✓ S_substrate DECREASED by {abs(total_delta_S_sub):.6f}")
        print(f"      → Information LEFT the substrate")
    else:
        print(f"    ✗ S_substrate INCREASED by {total_delta_S_sub:.6f}")
        print(f"      → Information ENTERED the substrate")

    if total_delta_I_class > 0:
        print(f"    ✓ I_classical INCREASED by {total_delta_I_class:.6f}")
        print(f"      → Classical correlations strengthened")
    else:
        print(f"    ✗ I_classical DECREASED by {abs(total_delta_I_class):.6f}")
        print(f"      → Classical correlations weakened")

    # Compute correlation between extraction and energy
    delta_S_subs = np.array(results["delta_S_substrate"][1:])
    delta_I_classs = np.array(results["delta_I_classical"][1:])
    delta_Es = np.array(results["delta_energy"][1:])

    if len(delta_I_classs) > 2:
        # Correlation: info extraction rate vs energy cost
        corr_extraction_energy = np.corrcoef(
            delta_I_classs, np.abs(delta_Es)
        )[0, 1]

        print(f"\n  Correlation (ΔI_classical, |ΔE|): {corr_extraction_energy:.3f}")

        if corr_extraction_energy > 0.3:
            print(f"    ✓ Positive correlation: Energy correlated with extraction")
        else:
            print(f"    ✗ Weak correlation")

    return results


def plot_results(results, params):
    """Visualize information flow test."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Information Extraction Test: Substrate → Classical Flow\n"
        f"Lattice {params.Lx}×{params.Ly}×{params.Lz}, λ_G={params.lambda_G}, J_exch={params.J_exch}",
        fontsize=14,
        fontweight="bold",
    )

    steps = results["step"]

    # Plot 1: Substrate entropy (should decrease)
    ax = axes[0, 0]
    ax.plot(steps, results["S_substrate"], "o-", linewidth=2, markersize=6, color="red")
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("S_substrate (Full State Entropy)")
    ax.set_title("Substrate Information\n(Should DECREASE)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add trend arrow
    if results["S_substrate"][-1] < results["S_substrate"][0]:
        ax.annotate(
            "",
            xy=(len(steps) - 1, results["S_substrate"][-1]),
            xytext=(0, results["S_substrate"][0]),
            arrowprops=dict(arrowstyle="->", color="green", lw=2, alpha=0.5),
        )
        ax.text(
            len(steps) / 2,
            np.mean(results["S_substrate"]),
            "Info leaving ✓",
            fontsize=10,
            color="green",
            ha="center",
        )

    # Plot 2: Classical mutual information (should increase)
    ax = axes[0, 1]
    ax.plot(
        steps,
        results["I_classical"],
        "o-",
        linewidth=2,
        markersize=6,
        color="blue",
    )
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("I_classical (Mutual Information)")
    ax.set_title("Classical Information\n(Should INCREASE)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add trend arrow
    if results["I_classical"][-1] > results["I_classical"][0]:
        ax.annotate(
            "",
            xy=(len(steps) - 1, results["I_classical"][-1]),
            xytext=(0, results["I_classical"][0]),
            arrowprops=dict(arrowstyle="->", color="green", lw=2, alpha=0.5),
        )
        ax.text(
            len(steps) / 2,
            np.mean(results["I_classical"]),
            "Info arriving ✓",
            fontsize=10,
            color="green",
            ha="center",
        )

    # Plot 3: Energy evolution
    ax = axes[0, 2]
    ax.plot(
        steps,
        results["energy"],
        "o-",
        linewidth=2,
        markersize=6,
        color="purple",
    )
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy (Extraction Cost)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Info flow diagram (KEY PLOT)
    ax = axes[1, 0]
    ax2 = ax.twinx()

    line1 = ax.plot(
        steps,
        results["S_substrate"],
        "o-",
        linewidth=2,
        markersize=6,
        color="red",
        label="S_substrate (left)",
    )
    line2 = ax2.plot(
        steps,
        results["I_classical"],
        "s-",
        linewidth=2,
        markersize=6,
        color="blue",
        label="I_classical (right)",
    )

    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("S_substrate", color="red")
    ax2.set_ylabel("I_classical", color="blue")
    ax.tick_params(axis="y", labelcolor="red")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax.set_title("Information Flow: Substrate → Classical", fontweight="bold")
    ax.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")

    # Plot 5: Extraction rate vs energy cost
    ax = axes[1, 1]
    delta_I = results["delta_I_classical"][1:]
    delta_E = np.abs(results["delta_energy"][1:])

    ax.scatter(delta_I, delta_E, s=50, alpha=0.6, color="green")
    ax.set_xlabel("ΔI_classical (Extraction Rate)")
    ax.set_ylabel("|ΔE| (Energy Cost)")
    ax.set_title("Energy Cost vs Extraction Rate", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Trend line
    if len(delta_I) > 2:
        z = np.polyfit(delta_I, delta_E, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(delta_I), max(delta_I), 100)
        ax.plot(
            x_line,
            p(x_line),
            "r--",
            linewidth=2,
            alpha=0.5,
            label=f"slope={z[0]:.2f}",
        )
        ax.legend()

    # Plot 6: Individual particle entropies
    ax = axes[1, 2]
    ax.plot(
        steps,
        results["S1"],
        "o-",
        linewidth=2,
        markersize=4,
        color="cyan",
        label="S₁ (particle 1)",
        alpha=0.7,
    )
    ax.plot(
        steps,
        results["S2"],
        "s-",
        linewidth=2,
        markersize=4,
        color="orange",
        label="S₂ (particle 2)",
        alpha=0.7,
    )
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("Single-Particle Entropy")
    ax.set_title("Local Information Content")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Use fermionic regime with constraints
    params = TwoFermion3DParams(
        Lx=2,
        Ly=2,
        Lz=2,
        J_hop=1.0,
        m=0.1,
        g_defrag=1.0,
        sigma_defrag=1.0,
        lambda_G=5.0,  # Strong Gauss constraint
        lambda_S=-1.0,
        lambda_T=0.0,
        J_exch=1.0,  # Fermionic regime
        max_eigsh_iter=10000,
        k_eigs=1,
    )

    np.random.seed(42)
    results = run_information_extraction_test(
        params,
        n_perturbations=20,
        strength=0.15,
    )

    # Plot
    fig = plot_results(results, params)

    # Output directory relative to this script
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    output_path = output_dir / "information_extraction_test.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")

    # Save data
    data_path = output_dir / "information_extraction_data.npz"
    np.savez(str(data_path), **results)
    print(f"✓ Data saved to: {data_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
