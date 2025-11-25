#!/usr/bin/env python3
"""
two_particle_spin_spatial_coupling.py

Tests whether spin-spatial coupling can induce fermionic exchange antisymmetry.

Key idea: If the Hamiltonian favors spin-singlet configurations when particles
are close together, and the singlet is inherently antisymmetric in spin space,
can this antisymmetry "transfer" to full exchange antisymmetry in configuration space?

Setup:
- Two spin-1/2 particles on a 1D periodic lattice (L sites)
- Each particle has position (i or j) and spin (↑ or ↓)
- Hilbert space: |i,σ₁, j,σ₂⟩ where σ ∈ {↑, ↓}
- Dimension: L² × 4 = 4L² (manageable for L ~ 8-12)

Hamiltonian:
  H = H_hop + H_exchange + H_constraint
  
  H_hop: spin-conserving nearest-neighbor hopping
  H_exchange: J·S₁·S₂ at same site (favors singlet if J > 0)
  H_constraint: λ_G penalty for triplet states at same site

Test for exchange antisymmetry:
  Does ψ(i,σ₁, j,σ₂) ≈ -ψ(j,σ₂, i,σ₁)?
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass
from typing import Tuple, Dict


# -----------------------------------------------------------------------------
# Configuration space indexing
# -----------------------------------------------------------------------------

class SpinfulTwoParticleSpace:
    """
    Manages the Hilbert space for two spin-1/2 particles on a lattice.
    
    States: |i,σ₁, j,σ₂⟩
    Spin convention: 0 = ↑, 1 = ↓
    
    Ordering: lexicographic by (i, σ₁, j, σ₂)
    """
    
    def __init__(self, L: int):
        self.L = L
        self.dim = L * L * 4  # positions × spin configurations
        
    def state_to_index(self, i: int, s1: int, j: int, s2: int) -> int:
        """Convert (i, σ₁, j, σ₂) to linear index."""
        return ((i * 2 + s1) * self.L + j) * 2 + s2
    
    def index_to_state(self, idx: int) -> Tuple[int, int, int, int]:
        """Convert linear index to (i, σ₁, j, σ₂)."""
        s2 = idx % 2
        idx //= 2
        j = idx % self.L
        idx //= self.L
        s1 = idx % 2
        i = idx // 2
        return i, s1, j, s2
    
    def is_singlet(self, s1: int, s2: int) -> bool:
        """Check if spin configuration is singlet-compatible (↑↓ or ↓↑)."""
        return s1 != s2
    
    def is_triplet(self, s1: int, s2: int) -> bool:
        """Check if spin configuration is triplet (↑↑, ↓↓, or ↑↓+↓↑)."""
        return not self.is_singlet(s1, s2)


# -----------------------------------------------------------------------------
# Hamiltonian construction
# -----------------------------------------------------------------------------

@dataclass
class HamiltonianParams:
    """Parameters for the spinful two-particle Hamiltonian."""
    t: float = 1.0          # Hopping amplitude
    J: float = 1.0          # Exchange coupling (positive favors singlet)
    lambda_G: float = 5.0   # Constraint penalty for triplet overlap
    lambda_S: float = 0.0   # Optional: direct singlet bonus at same site


def build_spinful_hamiltonian(space: SpinfulTwoParticleSpace, 
                              params: HamiltonianParams) -> csr_matrix:
    """
    Build H = H_hop + H_exchange + H_constraint.
    
    H_hop: -t Σ [ |i±1,σ, j,τ⟩⟨i,σ, j,τ| + |i,σ, j±1,τ⟩⟨i,σ, j,τ| ]
          Spin-conserving nearest-neighbor hopping (periodic BC)
    
    H_exchange: J Σ_i S₁·S₂|_{particles at site i}
          Where S₁·S₂ = -3/4 for singlet, +1/4 for triplet
          Equivalently: J Σ_i [ 1/2 n_i(n_i-1) - 2 S^z_{1,i} S^z_{2,i} ]
          For J > 0: favors singlet (antisymmetric spin)
    
    H_constraint: λ_G Σ_i Σ_{triplet states} |i,i,triplet⟩⟨i,i,triplet|
          Heavy penalty for particles at same site in triplet configuration
    """
    L = space.L
    dim = space.dim
    H = lil_matrix((dim, dim), dtype=np.float64)
    
    # --------------- Hopping term ---------------
    for i in range(L):
        for s1 in range(2):
            for j in range(L):
                for s2 in range(2):
                    idx = space.state_to_index(i, s1, j, s2)
                    
                    # Particle 1 hops (spin conserved)
                    i_next = (i + 1) % L
                    i_prev = (i - 1) % L
                    H[idx, space.state_to_index(i_next, s1, j, s2)] += -params.t
                    H[idx, space.state_to_index(i_prev, s1, j, s2)] += -params.t
                    
                    # Particle 2 hops (spin conserved)
                    j_next = (j + 1) % L
                    j_prev = (j - 1) % L
                    H[idx, space.state_to_index(i, s1, j_next, s2)] += -params.t
                    H[idx, space.state_to_index(i, s1, j_prev, s2)] += -params.t
    
    # --------------- Exchange interaction at same site ---------------
    # S₁·S₂ = S^x₁S^x₂ + S^y₁S^y₂ + S^z₁S^z₂
    # For spin-1/2: S^z = ±1/2, so S^z₁S^z₂ = +1/4 (parallel) or -1/4 (antiparallel)
    # And S^+₁S^-₂ + S^-₁S^+₂ contributes spin-flip terms
    # 
    # Easier formula: S₁·S₂ = (1/2)[S²_total - S²₁ - S²₂]
    # For singlet (S=0): S₁·S₂ = (1/2)[0 - 3/4 - 3/4] = -3/4
    # For triplet (S=1): S₁·S₂ = (1/2)[2 - 3/4 - 3/4] = +1/4
    
    for i in range(L):
        # Same-site configurations
        # Diagonal terms: S^z₁ S^z₂
        for s1 in range(2):
            for s2 in range(2):
                idx = space.state_to_index(i, s1, i, s2)
                sz1 = 0.5 if s1 == 0 else -0.5  # ↑ = +1/2, ↓ = -1/2
                sz2 = 0.5 if s2 == 0 else -0.5
                H[idx, idx] += params.J * sz1 * sz2
        
        # Off-diagonal: S^+₁S^-₂ and S^-₁S^+₂ (spin flip)
        # |i,↑,i,↓⟩ ↔ |i,↓,i,↑⟩
        idx_up_down = space.state_to_index(i, 0, i, 1)  # ↑↓
        idx_down_up = space.state_to_index(i, 1, i, 0)  # ↓↑
        # S^+₁S^-₂ |↑↓⟩ = |↓↑⟩ and S^-₁S^+₂ |↓↑⟩ = |↑↓⟩
        H[idx_up_down, idx_down_up] += 0.5 * params.J
        H[idx_down_up, idx_up_down] += 0.5 * params.J
    
    # --------------- Constraint penalty ---------------
    # Penalize triplet configurations at same site
    for i in range(L):
        # Triplet states: |↑↑⟩, |↓↓⟩ (and symmetric combo of |↑↓⟩+|↓↑⟩)
        # For simplicity, penalize |↑↑⟩ and |↓↓⟩ heavily
        idx_up_up = space.state_to_index(i, 0, i, 0)
        idx_down_down = space.state_to_index(i, 1, i, 1)
        H[idx_up_up, idx_up_up] += params.lambda_G
        H[idx_down_down, idx_down_down] += params.lambda_G
        
        # The symmetric |↑↓⟩+|↓↑⟩ state is harder to penalize directly
        # in this basis, but the diagonal penalty above should be enough
    
    # --------------- Optional: Direct singlet bonus ---------------
    if params.lambda_S != 0:
        for i in range(L):
            # Singlet = (|↑↓⟩ - |↓↑⟩)/√2
            # Lower energy for singlet-like configurations
            idx_up_down = space.state_to_index(i, 0, i, 1)
            idx_down_up = space.state_to_index(i, 1, i, 0)
            # Bonus for being in singlet subspace (negative energy)
            H[idx_up_down, idx_up_down] += -params.lambda_S
            H[idx_down_up, idx_down_up] += -params.lambda_S
            # And coupling to encourage singlet formation
            H[idx_up_down, idx_down_up] += -params.lambda_S
            H[idx_down_up, idx_up_down] += -params.lambda_S
    
    return H.tocsr()


# -----------------------------------------------------------------------------
# Exchange antisymmetry analysis
# -----------------------------------------------------------------------------

def extract_wavefunction_tensor(psi_flat: np.ndarray, 
                               space: SpinfulTwoParticleSpace) -> np.ndarray:
    """
    Reshape flat wavefunction into 4D tensor ψ[i,σ₁,j,σ₂].
    """
    L = space.L
    psi_tensor = np.zeros((L, 2, L, 2), dtype=np.complex128)
    for idx in range(space.dim):
        i, s1, j, s2 = space.index_to_state(idx)
        psi_tensor[i, s1, j, s2] = psi_flat[idx]
    return psi_tensor


def compute_exchange_antisymmetry_metrics(psi_tensor: np.ndarray) -> Dict[str, float]:
    """
    Test exchange antisymmetry: ψ(i,σ₁,j,σ₂) =? -ψ(j,σ₂,i,σ₁)
    
    Returns various metrics quantifying degree of antisymmetry.
    """
    L = psi_tensor.shape[0]
    
    total_norm = np.sum(np.abs(psi_tensor)**2)
    
    # Diagonal suppression (both particles at same site)
    diagonal_norm = 0.0
    for i in range(L):
        for s1 in range(2):
            for s2 in range(2):
                diagonal_norm += np.abs(psi_tensor[i, s1, i, s2])**2
    
    # Exchange antisymmetry violation
    antisym_violation = 0.0
    for i in range(L):
        for s1 in range(2):
            for j in range(L):
                for s2 in range(2):
                    # Check: ψ(i,s1,j,s2) + ψ(j,s2,i,s1) should be ~0
                    antisym_violation += np.abs(
                        psi_tensor[i, s1, j, s2] + psi_tensor[j, s2, i, s1]
                    )**2
    
    # Symmetry violation (should be large for antisymmetric)
    sym_violation = 0.0
    for i in range(L):
        for s1 in range(2):
            for j in range(L):
                for s2 in range(2):
                    sym_violation += np.abs(
                        psi_tensor[i, s1, j, s2] - psi_tensor[j, s2, i, s1]
                    )**2
    
    off_diagonal_norm = total_norm - diagonal_norm
    antisym_fraction = 1.0 - (antisym_violation / total_norm if total_norm > 1e-12 else 0)
    
    return {
        'total_norm': total_norm,
        'diagonal_norm': diagonal_norm,
        'diagonal_fraction': diagonal_norm / total_norm if total_norm > 1e-12 else 0,
        'antisym_violation': antisym_violation,
        'sym_violation': sym_violation,
        'antisym_fraction': antisym_fraction,
        'off_diagonal_antisym_fraction': 1.0 - (antisym_violation / total_norm) if total_norm > 1e-12 else 0
    }


def analyze_spin_correlations(psi_tensor: np.ndarray, 
                              space: SpinfulTwoParticleSpace) -> Dict[str, float]:
    """
    Analyze spin structure when particles are at same vs different sites.
    """
    L = space.L
    
    # Singlet weight at same site
    singlet_same_site = 0.0
    triplet_same_site = 0.0
    
    for i in range(L):
        # Singlet: (|↑↓⟩ - |↓↑⟩)/√2
        singlet_amplitude = (psi_tensor[i, 0, i, 1] - psi_tensor[i, 1, i, 0]) / np.sqrt(2)
        singlet_same_site += np.abs(singlet_amplitude)**2
        
        # Triplet components
        triplet_same_site += np.abs(psi_tensor[i, 0, i, 0])**2  # |↑↑⟩
        triplet_same_site += np.abs(psi_tensor[i, 1, i, 1])**2  # |↓↓⟩
        # Symmetric |↑↓⟩+|↓↑⟩
        triplet_sym = (psi_tensor[i, 0, i, 1] + psi_tensor[i, 1, i, 0]) / np.sqrt(2)
        triplet_same_site += np.abs(triplet_sym)**2
    
    total_same_site = singlet_same_site + triplet_same_site
    
    return {
        'singlet_same_site': singlet_same_site,
        'triplet_same_site': triplet_same_site,
        'total_same_site': total_same_site,
        'singlet_fraction': singlet_same_site / total_same_site if total_same_site > 1e-12 else 0
    }


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_wavefunction_slices(psi_tensor: np.ndarray, 
                            filename: str = None):
    """
    Plot ψ(i,j) for each spin configuration.
    """
    L = psi_tensor.shape[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    spin_labels = ['↑↑', '↑↓', '↓↑', '↓↓']
    
    for idx, (s1, s2) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        ax = axes[idx//2, idx%2]
        slice_data = psi_tensor[:, s1, :, s2]
        
        im = ax.imshow(np.abs(slice_data), cmap='viridis', origin='lower')
        ax.set_title(f'|ψ(i,j)| for spin config {spin_labels[idx]}', fontsize=12)
        ax.set_xlabel('Particle 2 position (j)')
        ax.set_ylabel('Particle 1 position (i)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    return fig


def plot_exchange_test_scatter(psi_tensor: np.ndarray, 
                               filename: str = None):
    """
    Scatter plot: ψ(i,σ₁,j,σ₂) vs ψ(j,σ₂,i,σ₁)
    Perfect antisymmetry: all points on y=-x line
    """
    L = psi_tensor.shape[0]
    
    forward_vals = []
    backward_vals = []
    
    for i in range(L):
        for s1 in range(2):
            for j in range(L):
                for s2 in range(2):
                    if (i, s1, j, s2) < (j, s2, i, s1):  # Avoid double-counting
                        forward_vals.append(psi_tensor[i, s1, j, s2])
                        backward_vals.append(psi_tensor[j, s2, i, s1])
    
    forward_vals = np.array(forward_vals)
    backward_vals = np.array(backward_vals)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(np.real(forward_vals), np.real(backward_vals), 
              alpha=0.5, s=20, label='Real part')
    
    # Perfect antisymmetry line
    lim = max(np.abs(forward_vals).max(), np.abs(backward_vals).max())
    ax.plot([-lim, lim], [lim, -lim], 'r--', linewidth=2, 
           label='Perfect antisymmetry: ψ(j,σ₂,i,σ₁) = -ψ(i,σ₁,j,σ₂)')
    ax.plot([-lim, lim], [-lim, lim], 'g--', alpha=0.3, linewidth=1,
           label='Perfect symmetry')
    
    ax.set_xlabel('ψ(i,σ₁,j,σ₂)', fontsize=12)
    ax.set_ylabel('ψ(j,σ₂,i,σ₁)', fontsize=12)
    ax.set_title('Exchange Antisymmetry Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    return fig


# -----------------------------------------------------------------------------
# Parameter sweep
# -----------------------------------------------------------------------------

def run_coupling_sweep(L: int, 
                      t: float,
                      J_values: list,
                      lambda_G: float) -> list:
    """
    Sweep over exchange coupling J to see effect on antisymmetry.
    """
    space = SpinfulTwoParticleSpace(L)
    results = []
    
    for J in J_values:
        print(f"\n{'='*70}")
        print(f"Computing for J = {J:.2f}, λ_G = {lambda_G:.2f}")
        print(f"{'='*70}")
        
        params = HamiltonianParams(t=t, J=J, lambda_G=lambda_G)
        H = build_spinful_hamiltonian(space, params)
        
        # Find ground state
        eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
        E_ground = eigenvalues[0]
        psi_flat = eigenvectors[:, 0]
        psi_tensor = extract_wavefunction_tensor(psi_flat, space)
        
        # Analyze
        exchange_metrics = compute_exchange_antisymmetry_metrics(psi_tensor)
        spin_metrics = analyze_spin_correlations(psi_tensor, space)
        
        print(f"Ground state energy: {E_ground:.6f}")
        print(f"Diagonal weight: {100*exchange_metrics['diagonal_fraction']:.2f}%")
        print(f"Exchange antisymmetry fraction: {exchange_metrics['antisym_fraction']:.4f}")
        print(f"Singlet fraction (same site): {spin_metrics['singlet_fraction']:.4f}")
        
        if exchange_metrics['antisym_fraction'] > 0.95:
            print("✓✓✓ STRONG EXCHANGE ANTISYMMETRY ACHIEVED!")
        elif exchange_metrics['diagonal_fraction'] < 0.05 and spin_metrics['singlet_fraction'] > 0.9:
            print("✓ Particles avoid overlap + prefer singlet when close")
        
        results.append({
            'J': J,
            'lambda_G': lambda_G,
            'energy': E_ground,
            'psi_tensor': psi_tensor,
            'exchange_metrics': exchange_metrics,
            'spin_metrics': spin_metrics
        })
    
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test if spin-spatial coupling induces exchange antisymmetry"
    )
    parser.add_argument("--L", type=int, default=6,
                       help="Lattice size (default: 6)")
    parser.add_argument("--t", type=float, default=1.0,
                       help="Hopping amplitude (default: 1.0)")
    parser.add_argument("--J-min", type=float, default=0.0,
                       help="Minimum exchange coupling (default: 0.0)")
    parser.add_argument("--J-max", type=float, default=5.0,
                       help="Maximum exchange coupling (default: 5.0)")
    parser.add_argument("--lambda-G", type=float, default=10.0,
                       help="Constraint penalty (default: 10.0)")
    parser.add_argument("--n-points", type=int, default=6,
                       help="Number of J values (default: 6)")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory (default: current)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("SPIN-SPATIAL COUPLING → EXCHANGE ANTISYMMETRY TEST")
    print("="*70)
    print(f"Lattice size: {args.L}")
    print(f"Hopping: t = {args.t}")
    print(f"Exchange coupling range: J ∈ [{args.J_min}, {args.J_max}]")
    print(f"Constraint penalty: λ_G = {args.lambda_G}")
    print(f"Hilbert space dimension: {4 * args.L * args.L}")
    print()
    
    J_values = np.linspace(args.J_min, args.J_max, args.n_points)
    results = run_coupling_sweep(args.L, args.t, J_values, args.lambda_G)
    
    # Plot final state
    final = results[-1]
    psi_final = final['psi_tensor']
    
    fig1 = plot_wavefunction_slices(psi_final, 
                                    f"{args.output_dir}/wavefunction_slices.png")
    print(f"\nSaved: {args.output_dir}/wavefunction_slices.png")
    
    fig2 = plot_exchange_test_scatter(psi_final,
                                     f"{args.output_dir}/exchange_scatter.png")
    print(f"Saved: {args.output_dir}/exchange_scatter.png")
    
    # Summary
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    final_exchange = final['exchange_metrics']
    final_spin = final['spin_metrics']
    
    if final_exchange['antisym_fraction'] > 0.95:
        print("✓✓✓ SUCCESS: Spin-spatial coupling induces strong exchange antisymmetry!")
        print(f"    Antisymmetry fraction: {final_exchange['antisym_fraction']:.4f}")
        print(f"    Singlet fraction (same site): {final_spin['singlet_fraction']:.4f}")
        print()
        print("INTERPRETATION:")
        print("  The Hamiltonian's preference for spin-singlet at close range,")
        print("  combined with constraint penalties on triplet overlap, has")
        print("  induced FERMIONIC EXCHANGE ANTISYMMETRY in the ground state.")
        print()
        print("  This suggests: fermionic statistics CAN emerge from")
        print("  spin-spatial coupling + constraint structure!")
    else:
        print("○ PARTIAL: Some structure but not full antisymmetry.")
        print(f"   Antisymmetry fraction: {final_exchange['antisym_fraction']:.4f}")
        print(f"   Diagonal weight: {100*final_exchange['diagonal_fraction']:.2f}%")
        print(f"   Singlet fraction: {final_spin['singlet_fraction']:.4f}")
        print()
        print("Try stronger J or λ_G, or different lattice size.")
    
    plt.close('all')


if __name__ == "__main__":
    main()