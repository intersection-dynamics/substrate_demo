#!/usr/bin/env python3
"""
sequential_perturbation_test.py

Test hypothesis: 
- Redundant information (high overlap with history) → energy increase (thermalization)
- Novel information (low overlap with history) → structural change (new patterns)

Method:
1. Start with ground state of two-fermion 3D substrate
2. Apply sequential perturbations (simulating photon absorptions)
3. Track: energy, entropy, overlap with history subspace
4. Measure correlation between redundancy and energy vs structure
"""

import sys
import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import from local project first; fall back to sandbox path if needed
try:
    from substrate_engine_3d import (
        TwoFermion3DParams,
        build_twofermion3d_hamiltonian,
        encode_basis_3d,
        decode_basis_3d,
    )
except ImportError:
    sys.path.insert(0, "/mnt/user-data/uploads")
    from substrate_engine_3d import (
        TwoFermion3DParams,
        build_twofermion3d_hamiltonian,
        encode_basis_3d,
        decode_basis_3d,
    )


def get_ground_state(params):
    """Get ground state of the system."""
    from scipy.sparse.linalg import eigsh

    H = build_twofermion3d_hamiltonian(params)
    evals, evecs = eigsh(H, k=1, which="SA", maxiter=10000)
    psi0 = evecs[:, 0]

    # Normalize
    norm = np.sqrt(float(np.vdot(psi0, psi0).real))
    if norm > 0:
        psi0 /= norm

    E0 = float(evals[0].real)
    return psi0, E0, H


def apply_local_perturbation(psi, params, site, strength=0.1, seed=None):
    """
    Apply a local random unitary perturbation at a specific site.
    This simulates a 'photon' interacting with the local degrees of freedom.
    """
    if seed is not None:
        np.random.seed(seed)

    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    # Create a local perturbation operator
    # Acts on all basis states where particle 1 or 2 is at 'site'
    perturbation = np.zeros((dim, dim), dtype=complex)

    # Random hermitian matrix for the perturbation
    rand_phase = np.random.random() * 2 * np.pi

    for idx in range(dim):
        # Decode to find particle positions
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)

        if r1 == site or r2 == site:
            # Apply phase shift and small random unitary
            perturbation[idx, idx] = strength * np.exp(1j * rand_phase)

            # Small off-diagonal coupling
            for idx2 in range(max(0, idx - 5), min(dim, idx + 5)):
                if idx != idx2:
                    r1_2, s1_2, r2_2, s2_2 = decode_basis_3d(idx2, Ns)
                    if r1_2 == site or r2_2 == site:
                        coupling = (
                            strength
                            * 0.1
                            * np.random.randn()
                            * np.exp(1j * np.random.random() * 2 * np.pi)
                        )
                        perturbation[idx, idx2] = coupling

    # Make hermitian
    perturbation = (perturbation + perturbation.conj().T) / 2

    # Apply perturbation: |ψ'⟩ = exp(-iHₚₑᵣₜ)|ψ⟩
    from scipy.linalg import expm

    U_pert = expm(-1j * perturbation)

    psi_new = U_pert @ psi

    # Normalize
    norm = np.sqrt(float(np.vdot(psi_new, psi_new).real))
    if norm > 0:
        psi_new /= norm

    return psi_new


def compute_reduced_density_matrix(psi, params, particle=1):
    """
    Compute reduced density matrix for one particle by tracing out the other.
    This measures the local information content.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    # For particle 1: trace out particle 2
    # rho_1[r1,s1; r1',s1'] = sum_{r2,s2} psi(r1,s1,r2,s2) * psi*(r1',s1',r2,s2)

    dim_single = Ns * 2
    rho = np.zeros((dim_single, dim_single), dtype=complex)

    dim = Ns * 2 * Ns * 2

    for idx in range(dim):
        r1, s1, r2, s2 = decode_basis_3d(idx, Ns)
        for idx2 in range(dim):
            r1_p, s1_p, r2_p, s2_p = decode_basis_3d(idx2, Ns)

            # Only sum if particle 2 coordinates match
            if r2 == r2_p and s2 == s2_p:
                i = r1 * 2 + s1
                j = r1_p * 2 + s1_p
                rho[i, j] += psi[idx] * np.conj(psi[idx2])

    return rho


def von_neumann_entropy(rho):
    """Compute von Neumann entropy S = -Tr(rho log rho)."""
    # Get eigenvalues
    eigvals = np.linalg.eigvalsh(rho)

    # Filter out numerical zeros
    eigvals = eigvals[eigvals > 1e-12]

    # Compute entropy
    S = -np.sum(eigvals * np.log(eigvals + 1e-16))

    return float(S.real)


def fidelity(psi1, psi2):
    """Compute fidelity F = |⟨ψ₁|ψ₂⟩|²."""
    overlap = np.vdot(psi1, psi2)
    return float(np.abs(overlap) ** 2)


def overlap_with_subspace(psi, subspace_states):
    """
    Compute overlap of |ψ⟩ with subspace spanned by states in subspace_states.
    Returns |P_subspace|ψ⟩|² where P is projector onto subspace.
    """
    if len(subspace_states) == 0:
        return 0.0

    # Orthogonalize subspace states (Gram-Schmidt)
    ortho_states = []
    for state in subspace_states:
        # Orthogonalize against previous states
        orth_state = state.copy()
        for prev_state in ortho_states:
            orth_state -= np.vdot(prev_state, orth_state) * prev_state

        # Normalize
        norm = np.sqrt(float(np.vdot(orth_state, orth_state).real))
        if norm > 1e-10:
            orth_state /= norm
            ortho_states.append(orth_state)

    # Compute projection
    projection = np.zeros_like(psi)
    for basis_state in ortho_states:
        projection += np.vdot(basis_state, psi) * basis_state

    # Return norm squared of projection
    overlap = np.sqrt(float(np.vdot(projection, projection).real))

    return overlap**2


def run_sequential_perturbation_test(
    params, n_perturbations=10, perturbation_strength=0.1
):
    """
    Main test: apply sequential perturbations and track observables.
    """

    print("\n" + "=" * 70)
    print("SEQUENTIAL PERTURBATION TEST")
    print("Testing: Redundancy → Energy, Novelty → Structure")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Lattice: {params.Lx}×{params.Ly}×{params.Lz}")
    print(f"  Perturbations: {n_perturbations}")
    print(f"  Strength: {perturbation_strength}")
    print(f"  λ_G (Gauss): {params.lambda_G}")
    print(f"  J_exch: {params.J_exch}")
    print("=" * 70)

    # Get ground state
    print("\n[1/4] Computing ground state...")
    psi, E0, H = get_ground_state(params)
    print(f"      Ground state energy: {E0:.6f}")

    # Storage for results
    results = {
        "step": [0],
        "energy": [E0],
        "entropy": [],
        "overlap_with_history": [0.0],  # First state has no history
        "fidelity_to_previous": [1.0],
        "delta_energy": [0.0],
        "delta_entropy": [0.0],
    }

    # Initial entropy
    rho_0 = compute_reduced_density_matrix(psi, params, particle=1)
    S_0 = von_neumann_entropy(rho_0)
    results["entropy"].append(S_0)
    results["delta_entropy"][0] = 0.0

    print(f"      Initial entropy: {S_0:.6f}")

    # History subspace (states we've seen before)
    history_states = [psi.copy()]

    # Apply sequential perturbations
    print(f"\n[2/4] Applying {n_perturbations} sequential perturbations...")

    Ns = params.Lx * params.Ly * params.Lz

    for step in range(1, n_perturbations + 1):
        print(f"\n  Step {step}/{n_perturbations}:")

        # Apply perturbation at random site
        site = np.random.randint(0, Ns)
        psi_prev = psi.copy()
        psi = apply_local_perturbation(
            psi, params, site, strength=perturbation_strength, seed=step
        )

        # Compute overlap with history subspace (redundancy measure)
        overlap_hist = overlap_with_subspace(psi, history_states)

        # Compute energy
        E = float(np.vdot(psi, H @ psi).real)
        delta_E = E - results["energy"][-1]

        # Compute entropy (structure measure)
        rho = compute_reduced_density_matrix(psi, params, particle=1)
        S = von_neumann_entropy(rho)
        delta_S = S - results["entropy"][-1]

        # Compute fidelity to previous state
        F = fidelity(psi, psi_prev)

        # Store results
        results["step"].append(step)
        results["energy"].append(E)
        results["entropy"].append(S)
        results["overlap_with_history"].append(overlap_hist)
        results["fidelity_to_previous"].append(F)
        results["delta_energy"].append(delta_E)
        results["delta_entropy"].append(delta_S)

        # Add to history
        history_states.append(psi.copy())

        print(f"    Overlap with history: {overlap_hist:.4f}")
        print(f"    ΔE = {delta_E:+.6f}, ΔS = {delta_S:+.6f}")
        print(f"    Fidelity to prev: {F:.6f}")

    print("\n[3/4] Analysis...")

    # Compute correlations
    overlaps = np.array(results["overlap_with_history"][1:])  # Exclude first
    delta_Es = np.array(results["delta_energy"][1:])
    delta_Ss = np.array(results["delta_entropy"][1:])

    # Correlation: overlap vs energy change
    if len(overlaps) > 2:
        corr_E = np.corrcoef(overlaps, np.abs(delta_Es))[0, 1]
        corr_S = np.corrcoef(overlaps, np.abs(delta_Ss))[0, 1]

        print(f"\n  Correlation (overlap, |ΔE|): {corr_E:.3f}")
        print(f"  Correlation (overlap, |ΔS|): {corr_S:.3f}")

        print(f"\n  Ben's Hypothesis Predictions:")
        print(f"    High overlap → large ΔE (redundant → energy)")
        print(f"    Low overlap → large ΔS (novel → structure)")

        if corr_E > 0.3:
            print(f"    ✓ Positive correlation (overlap, |ΔE|) supports hypothesis")
        else:
            print(f"    ✗ Weak correlation (overlap, |ΔE|)")

        if corr_S < -0.3:
            print(f"    ✓ Negative correlation (overlap, |ΔS|) supports hypothesis")
        else:
            print(f"    ✗ Weak anti-correlation (overlap, |ΔS|)")

    return results


def plot_results(results, params):
    """Create visualization of sequential perturbation test."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Sequential Perturbation Test: Information → Energy Trade\n"
        f"Lattice {params.Lx}×{params.Ly}×{params.Lz}, "
        f"λ_G={params.lambda_G}, J_exch={params.J_exch}",
        fontsize=14,
        fontweight="bold",
    )

    steps = results["step"]

    # Plot 1: Energy evolution
    ax = axes[0, 0]
    ax.plot(steps, results["energy"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")
    ax.grid(True, alpha=0.3)

    # Plot 2: Entropy evolution
    ax = axes[0, 1]
    ax.plot(steps, results["entropy"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("Von Neumann Entropy")
    ax.set_title("Entropy Evolution (Structure)")
    ax.grid(True, alpha=0.3)

    # Plot 3: Overlap with history
    ax = axes[0, 2]
    ax.plot(
        steps,
        results["overlap_with_history"],
        "o-",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("Overlap with History Subspace")
    ax.set_title("Redundancy Measure")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 4: ΔE vs overlap (KEY PLOT)
    ax = axes[1, 0]
    overlaps = results["overlap_with_history"][1:]
    delta_Es = np.abs(results["delta_energy"][1:])
    ax.scatter(overlaps, delta_Es, s=50, alpha=0.6)
    ax.set_xlabel("Overlap with History (Redundancy)")
    ax.set_ylabel("|ΔE| (Energy Change)")
    ax.set_title("Hypothesis: Redundancy → Energy", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(overlaps) > 2:
        z = np.polyfit(overlaps, delta_Es, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(
            x_line,
            p(x_line),
            "r--",
            linewidth=2,
            alpha=0.5,
            label=f"Trend: slope={z[0]:.3f}",
        )
        ax.legend()

    # Plot 5: ΔS vs overlap (KEY PLOT)
    ax = axes[1, 1]
    delta_Ss = np.abs(results["delta_entropy"][1:])
    ax.scatter(overlaps, delta_Ss, s=50, alpha=0.6)
    ax.set_xlabel("Overlap with History (Redundancy)")
    ax.set_ylabel("|ΔS| (Entropy Change)")
    ax.set_title("Hypothesis: Novelty → Structure", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(overlaps) > 2:
        z = np.polyfit(overlaps, delta_Ss, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(
            x_line,
            p(x_line),
            "b--",
            linewidth=2,
            alpha=0.5,
            label=f"Trend: slope={z[0]:.3f}",
        )
        ax.legend()

    # Plot 6: Fidelity to previous state
    ax = axes[1, 2]
    ax.plot(
        steps,
        results["fidelity_to_previous"],
        "o-",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Perturbation Step")
    ax.set_ylabel("Fidelity to Previous State")
    ax.set_title("State Similarity")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Set up parameters (fermionic regime with strong constraints)
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

    # Run test
    np.random.seed(42)  # For reproducibility
    results = run_sequential_perturbation_test(
        params, n_perturbations=15, perturbation_strength=0.2
    )

    # Plot
    print("\n[4/4] Creating visualization...")
    fig = plot_results(results, params)

    # Output directory relative to this script
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save figure
    output_path = output_dir / "sequential_perturbation_test.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")

    # Save data
    data_path = output_dir / "sequential_perturbation_data.npz"
    np.savez(str(data_path), **results)
    print(f"✓ Data saved to: {data_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
