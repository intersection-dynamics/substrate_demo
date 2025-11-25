"""
Dynamical Stability of Fermionic Sector
=======================================
Test that once a system is in an antisymmetric state,
unitary evolution keeps it there.

Key question: Is the antisymmetric sector closed under dynamics?

For true fermions, the answer must be YES - you can't evolve
from fermionic to bosonic statistics.
"""

import numpy as np
from scipy.linalg import expm
from spin_statistics_engine import (
    SpinfulParticleSpace,
    build_spinful_hamiltonian,
    find_ground_states,
    full_analysis,
    analyze_spin_state,
    compute_exchange_eigenvalue
)
import argparse
import json
import os
from datetime import datetime


def create_antisymmetric_state(space: SpinfulParticleSpace) -> np.ndarray:
    """
    Create a pure antisymmetric state (Slater determinant).
    
    For 2 particles with single-particle states φ_a, φ_b:
    |ψ⟩ = (1/√2)[φ_a(1)φ_b(2) - φ_a(2)φ_b(1)]
    
    This is antisymmetric under FULL exchange (position + spin).
    """
    psi = np.zeros(space.dim, dtype=complex)
    
    # Single-particle states: φ_a = |site=0, ↑⟩, φ_b = |site=1, ↓⟩
    # Antisymmetric: |0,↑;1,↓⟩ - |1,↓;0,↑⟩
    # In our notation: (x1,s1,x2,s2)
    
    state1 = (0, 0, 1, 1)  # particle 1: (site=0, spin=↑), particle 2: (site=1, spin=↓)
    state2 = (1, 1, 0, 0)  # particle 1: (site=1, spin=↓), particle 2: (site=0, spin=↑)
    
    idx1 = space.state_to_idx.get(state1)
    idx2 = space.state_to_idx.get(state2)
    
    if idx1 is not None and idx2 is not None:
        psi[idx1] = 1.0 / np.sqrt(2)
        psi[idx2] = -1.0 / np.sqrt(2)  # Minus sign for antisymmetry
    
    # Verify it's antisymmetric
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    
    return psi


def create_symmetric_state(space: SpinfulParticleSpace) -> np.ndarray:
    """
    Create a pure symmetric (bosonic) state.
    
    For 2 particles: symmetric under full exchange
    |ψ⟩ = (1/√2)[φ_a(1)φ_b(2) + φ_a(2)φ_b(1)]
    """
    psi = np.zeros(space.dim, dtype=complex)
    
    # Use same single-particle states but with + sign
    state1 = (0, 0, 1, 1)  # particle 1: (site=0, spin=↑), particle 2: (site=1, spin=↓)
    state2 = (1, 1, 0, 0)  # particle 1: (site=1, spin=↓), particle 2: (site=0, spin=↑)
    
    idx1 = space.state_to_idx.get(state1)
    idx2 = space.state_to_idx.get(state2)
    
    if idx1 is not None and idx2 is not None:
        psi[idx1] = 1.0 / np.sqrt(2)
        psi[idx2] = 1.0 / np.sqrt(2)  # Plus sign for symmetry
    
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    
    return psi


def create_mixed_state(space: SpinfulParticleSpace) -> np.ndarray:
    """
    Create a state with mixed symmetry.
    
    Superposition of symmetric and antisymmetric parts.
    """
    psi_anti = create_antisymmetric_state(space)
    psi_sym = create_symmetric_state(space)
    
    # 50-50 superposition
    psi = (psi_anti + psi_sym) / np.sqrt(2)
    return psi / np.linalg.norm(psi)


def evolve_and_track(psi0: np.ndarray, H: np.ndarray, times: np.ndarray,
                     space: SpinfulParticleSpace) -> dict:
    """
    Evolve state and track symmetry properties over time.
    """
    results = {
        'times': times.tolist(),
        'antisym_weight': [],
        'exchange_eigenvalue': [],
        'singlet_weight': [],
        'norm': []
    }
    
    H_dense = H.toarray() if hasattr(H, 'toarray') else H
    
    psi = psi0.copy()
    
    for i, t in enumerate(times):
        if i > 0:
            dt = times[i] - times[i-1]
            U = expm(-1j * H_dense * dt)
            psi = U @ psi
        
        # Normalize (should stay 1, but numerical drift)
        psi = psi / np.linalg.norm(psi)
        
        analysis = full_analysis(psi, space)
        spin = analyze_spin_state(psi, space)
        
        results['antisym_weight'].append(analysis['antisymmetric_weight'])
        results['exchange_eigenvalue'].append(float(np.real(analysis['exchange_eigenvalue'])))
        results['singlet_weight'].append(spin['singlet_weight'])
        results['norm'].append(float(np.linalg.norm(psi)))
    
    return results


def test_sector_stability(n_sites: int = 4, t: float = 1.0, U: float = 5.0,
                          J_spin: float = 2.0, t_max: float = 20.0,
                          n_steps: int = 100) -> dict:
    """
    Main test: verify antisymmetric and symmetric sectors are stable.
    """
    print(f"\n{'='*70}")
    print("DYNAMICAL STABILITY TEST")
    print(f"{'='*70}")
    print(f"\nParameters: n_sites={n_sites}, t={t}, U={U}, J_spin={J_spin}")
    print(f"Evolution: t_max={t_max}, steps={n_steps}")
    
    space = SpinfulParticleSpace(n_sites, n_particles=2)
    print(f"Hilbert space dimension: {space.dim}")
    
    H = build_spinful_hamiltonian(space, t=t, U=U, J_spin=J_spin)
    times = np.linspace(0, t_max, n_steps)
    
    results = {
        'parameters': {
            'n_sites': n_sites, 't': t, 'U': U, 'J_spin': J_spin,
            't_max': t_max, 'n_steps': n_steps
        },
        'tests': {}
    }
    
    # Test 1: Pure antisymmetric initial state
    print("\n" + "-"*70)
    print("TEST 1: Pure ANTISYMMETRIC initial state")
    print("-"*70)
    
    psi_anti = create_antisymmetric_state(space)
    init_analysis = full_analysis(psi_anti, space)
    print(f"Initial exchange eigenvalue: {np.real(init_analysis['exchange_eigenvalue']):.6f}")
    print(f"Initial antisym weight: {init_analysis['antisymmetric_weight']:.6f}")
    
    evol_anti = evolve_and_track(psi_anti, H, times, space)
    
    # Check stability
    anti_stable = all(w > 0.99 for w in evol_anti['antisym_weight'])
    exch_stable = all(abs(e + 1) < 0.01 for e in evol_anti['exchange_eigenvalue'])
    
    print(f"\nAfter evolution to t={t_max}:")
    print(f"  Final exchange eigenvalue: {evol_anti['exchange_eigenvalue'][-1]:.6f}")
    print(f"  Final antisym weight: {evol_anti['antisym_weight'][-1]:.6f}")
    print(f"  Min antisym weight: {min(evol_anti['antisym_weight']):.6f}")
    print(f"  Max exchange deviation from -1: {max(abs(e + 1) for e in evol_anti['exchange_eigenvalue']):.6f}")
    
    if anti_stable and exch_stable:
        print("\n  ✓ ANTISYMMETRIC SECTOR IS STABLE")
    else:
        print("\n  ✗ ANTISYMMETRIC SECTOR LEAKED")
    
    results['tests']['antisymmetric'] = {
        'initial_exchange': float(np.real(init_analysis['exchange_eigenvalue'])),
        'evolution': evol_anti,
        'stable': anti_stable and exch_stable
    }
    
    # Test 2: Pure symmetric initial state
    print("\n" + "-"*70)
    print("TEST 2: Pure SYMMETRIC initial state")
    print("-"*70)
    
    psi_sym = create_symmetric_state(space)
    init_analysis = full_analysis(psi_sym, space)
    print(f"Initial exchange eigenvalue: {np.real(init_analysis['exchange_eigenvalue']):.6f}")
    print(f"Initial symmetric weight: {init_analysis['symmetric_weight']:.6f}")
    
    evol_sym = evolve_and_track(psi_sym, H, times, space)
    
    # Check stability (should stay symmetric, exchange = +1)
    sym_stable = all(w < 0.01 for w in evol_sym['antisym_weight'])  # antisym should stay ~0
    exch_sym_stable = all(abs(e - 1) < 0.01 for e in evol_sym['exchange_eigenvalue'])
    
    print(f"\nAfter evolution to t={t_max}:")
    print(f"  Final exchange eigenvalue: {evol_sym['exchange_eigenvalue'][-1]:.6f}")
    print(f"  Final antisym weight: {evol_sym['antisym_weight'][-1]:.6f}")
    print(f"  Max antisym weight: {max(evol_sym['antisym_weight']):.6f}")
    
    if sym_stable and exch_sym_stable:
        print("\n  ✓ SYMMETRIC SECTOR IS STABLE")
    else:
        print("\n  ✗ SYMMETRIC SECTOR LEAKED")
    
    results['tests']['symmetric'] = {
        'initial_exchange': float(np.real(init_analysis['exchange_eigenvalue'])),
        'evolution': evol_sym,
        'stable': sym_stable and exch_sym_stable
    }
    
    # Test 3: Mixed initial state - should oscillate but not mix sectors
    print("\n" + "-"*70)
    print("TEST 3: MIXED initial state (superposition of sectors)")
    print("-"*70)
    
    psi_mixed = create_mixed_state(space)
    init_analysis = full_analysis(psi_mixed, space)
    print(f"Initial exchange eigenvalue: {np.real(init_analysis['exchange_eigenvalue']):.6f}")
    print(f"Initial antisym weight: {init_analysis['antisymmetric_weight']:.6f}")
    
    evol_mixed = evolve_and_track(psi_mixed, H, times, space)
    
    # For mixed state, the antisym/sym weights should oscillate
    # but the TOTAL should be conserved
    print(f"\nAfter evolution to t={t_max}:")
    print(f"  Exchange eigenvalue range: [{min(evol_mixed['exchange_eigenvalue']):.4f}, {max(evol_mixed['exchange_eigenvalue']):.4f}]")
    print(f"  Antisym weight range: [{min(evol_mixed['antisym_weight']):.4f}, {max(evol_mixed['antisym_weight']):.4f}]")
    
    results['tests']['mixed'] = {
        'initial_exchange': float(np.real(init_analysis['exchange_eigenvalue'])),
        'evolution': evol_mixed
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = results['tests']['antisymmetric']['stable'] and results['tests']['symmetric']['stable']
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nThe fermionic (antisymmetric) and bosonic (symmetric) sectors")
        print("are DYNAMICALLY STABLE under Hamiltonian evolution.")
        print("\nThis means:")
        print("  - Once a particle is fermionic, it stays fermionic")
        print("  - Statistics are conserved quantities")
        print("  - The exchange eigenvalue is a constant of motion")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Check the detailed output above.")
    
    results['all_passed'] = all_passed
    
    return results


def run_parameter_sweep(output_dir: str = None):
    """
    Sweep over parameters to verify stability holds generally.
    """
    print("\n" + "="*70)
    print("PARAMETER SWEEP: STABILITY ACROSS PARAMETER SPACE")
    print("="*70)
    
    results = []
    
    # Sweep over J_spin
    for J in [0.5, 1.0, 2.0, 5.0]:
        for U in [0.0, 2.0, 5.0, 10.0]:
            print(f"\nTesting J_spin={J}, U={U}...")
            
            space = SpinfulParticleSpace(4, n_particles=2)
            H = build_spinful_hamiltonian(space, t=1.0, U=U, J_spin=J)
            
            # Quick stability check
            psi = create_antisymmetric_state(space)
            times = np.linspace(0, 10, 50)
            evol = evolve_and_track(psi, H, times, space)
            
            min_antisym = min(evol['antisym_weight'])
            max_dev = max(abs(e + 1) for e in evol['exchange_eigenvalue'])
            
            stable = min_antisym > 0.99 and max_dev < 0.01
            
            results.append({
                'J_spin': J,
                'U': U,
                'min_antisym': min_antisym,
                'max_exchange_deviation': max_dev,
                'stable': stable
            })
            
            status = "✓" if stable else "✗"
            print(f"  {status} min_antisym={min_antisym:.4f}, max_dev={max_dev:.6f}")
    
    # Summary
    n_stable = sum(1 for r in results if r['stable'])
    print(f"\n{n_stable}/{len(results)} parameter combinations showed stable antisymmetric sector")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test dynamical stability of fermionic sector')
    parser.add_argument('--n-sites', type=int, default=4, help='Number of lattice sites')
    parser.add_argument('--t', type=float, default=1.0, help='Hopping amplitude')
    parser.add_argument('--U', type=float, default=5.0, help='On-site repulsion')
    parser.add_argument('--J-spin', type=float, default=2.0, help='Spin-spin coupling')
    parser.add_argument('--t-max', type=float, default=20.0, help='Max evolution time')
    parser.add_argument('--n-steps', type=int, default=100, help='Number of time steps')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--tag', type=str, default='', help='Tag for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"_{args.tag}" if args.tag else ""
    
    if args.sweep:
        results = run_parameter_sweep(args.output_dir)
        output_file = os.path.join(args.output_dir, f'stability_sweep{tag}_{timestamp}.json')
    else:
        results = test_sector_stability(
            n_sites=args.n_sites,
            t=args.t,
            U=args.U,
            J_spin=args.J_spin,
            t_max=args.t_max,
            n_steps=args.n_steps
        )
        output_file = os.path.join(args.output_dir, f'stability_test{tag}_{timestamp}.json')
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()