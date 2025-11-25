#!/usr/bin/env python
"""
test_evolved_state_monogamy.py

Test monogamy on the FINAL state from time evolution.
This is what actually matters - does the evolved state have monogamous entanglement?
"""

import numpy as np
import argparse
import os

try:
    from qutip import Qobj, entropy_vn
    QUTIP_AVAILABLE = True
except ImportError:
    print("[ERROR] QuTiP required")
    import sys
    sys.exit(1)

def load_evolved_state(results_dir):
    """
    Re-run evolution to get final state.
    (Could save/load state, but this is simpler for now)
    """
    import json
    
    # Load parameters
    json_file = os.path.join(results_dir, "entanglement_evolution.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Results not found: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    params = data['parameters']
    
    # Need to re-evolve to get the actual state
    # Import the evolver
    import sys
    sys.path.insert(0, '.')
    from quantum_substrate_evolution import QuantumSubstrateEvolver
    
    print(f"[LOAD] Re-evolving to get final state...")
    print(f"  Winding: {params['winding']}")
    print(f"  Lattice: {params['Nx']}×{params['Ny']}")
    
    evolver = QuantumSubstrateEvolver(params['Nx'], params['Ny'])
    evolver.build_hamiltonian(
        J_nn=params['J_nn'],
        J_defrag=params['J_defrag'],
        mass=params['mass']
    )
    
    # Initialize same way
    if params['pattern_type'] == 'single_vortex':
        evolver.init_state('single_vortex',
                          x0=params['Nx']/2,
                          y0=params['Ny']/2,
                          winding=params['winding'])
    
    # Get times
    times = data['times']
    
    # Evolve
    result = evolver.evolve(times[-1], times[1] - times[0], measurement_times=times)
    
    # Return final state
    final_state = result.states[-1]
    
    print(f"[LOAD] Final state obtained")
    print(f"  Dimension: {final_state.shape}")
    
    return final_state, data['pattern_sites'], data['env_sites'], params

def split_environment(env_sites, Nx, Ny):
    """
    Split environment into two disjoint regions.
    E1: left half, E2: right half
    """
    env1 = [s for s in env_sites if (s % Ny) < Ny // 2]
    env2 = [s for s in env_sites if (s % Ny) >= Ny // 2]
    
    return env1, env2

def compute_mutual_information(state, region_A, region_B):
    """
    Compute I(A:B) = S(A) + S(B) - S(AB)
    """
    # von Neumann entropy for each region
    rho_A = state.ptrace(region_A)
    rho_B = state.ptrace(region_B)
    rho_AB = state.ptrace(list(set(region_A) | set(region_B)))
    
    S_A = entropy_vn(rho_A)
    S_B = entropy_vn(rho_B)
    S_AB = entropy_vn(rho_AB)
    
    I_AB = S_A + S_B - S_AB
    
    return I_AB, S_A, S_B, S_AB

def test_monogamy(state, pattern_sites, env_sites, params):
    """
    Test if pattern has monogamous entanglement with environment.
    """
    print("\n" + "=" * 70)
    print("MONOGAMY TEST")
    print("=" * 70)
    
    # Split environment
    env1, env2 = split_environment(env_sites, params['Nx'], params['Ny'])
    
    print(f"\nRegions:")
    print(f"  Pattern: {len(pattern_sites)} sites: {pattern_sites}")
    print(f"  Env1 (left): {len(env1)} sites: {env1}")
    print(f"  Env2 (right): {len(env2)} sites: {env2}")
    
    # Pattern entropy
    rho_P = state.ptrace(pattern_sites)
    S_P = entropy_vn(rho_P)
    
    print(f"\nPattern entropy: S(P) = {S_P:.6f}")
    
    # Maximum possible for pattern
    S_max = len(pattern_sites) * np.log(2)
    print(f"Maximum possible: S_max = {S_max:.6f}")
    print(f"Entanglement fraction: {S_P/S_max:.1%}")
    
    # Mutual information with each environment
    print(f"\nMutual Information:")
    
    I_PE1, S_P_check, S_E1, S_PE1 = compute_mutual_information(state, pattern_sites, env1)
    print(f"  I(P:E1) = {I_PE1:.6f}")
    print(f"    S(P) = {S_P_check:.6f}, S(E1) = {S_E1:.6f}, S(P+E1) = {S_PE1:.6f}")
    
    I_PE2, _, S_E2, S_PE2 = compute_mutual_information(state, pattern_sites, env2)
    print(f"  I(P:E2) = {I_PE2:.6f}")
    print(f"    S(P) = {S_P:.6f}, S(E2) = {S_E2:.6f}, S(P+E2) = {S_PE2:.6f}")
    
    # Monogamy test
    print(f"\nMonogamy Test:")
    print(f"  I(P:E1) + I(P:E2) = {I_PE1 + I_PE2:.6f}")
    print(f"  Bound (2×S(P)) = {2*S_P:.6f}")
    print(f"  Satisfies inequality: {I_PE1 + I_PE2 <= 2*S_P + 0.01}")
    
    # Is it monogamous?
    # If I(P:E1) ≈ S(P), pattern is maximally entangled with E1
    # Then I(P:E2) should be ≈ 0 by monogamy
    
    ratio_E1 = I_PE1 / S_P if S_P > 0.01 else 0
    ratio_E2 = I_PE2 / S_P if S_P > 0.01 else 0
    
    print(f"\nEntanglement Distribution:")
    print(f"  Fraction with E1: {ratio_E1:.1%}")
    print(f"  Fraction with E2: {ratio_E2:.1%}")
    
    # Monogamy criterion: >80% with one environment, <20% with other
    monogamous = (ratio_E1 > 0.8 and ratio_E2 < 0.2) or (ratio_E2 > 0.8 and ratio_E1 < 0.2)
    
    print(f"\n{'='*70}")
    if monogamous:
        print("✓ MONOGAMOUS ENTANGLEMENT DETECTED")
        print("  Pattern is primarily entangled with ONE environment region")
    elif S_P > 1.0:
        print("⚠ HIGH ENTANGLEMENT BUT NOT MONOGAMOUS")
        print("  Pattern is entangled with BOTH environment regions")
    else:
        print("✗ LOW ENTANGLEMENT")
        print("  Pattern is not strongly entangled with environment")
    print(f"{'='*70}\n")
    
    return {
        'S_P': S_P,
        'I_PE1': I_PE1,
        'I_PE2': I_PE2,
        'ratio_E1': ratio_E1,
        'ratio_E2': ratio_E2,
        'monogamous': monogamous,
        'winding': params['winding']
    }

def main():
    parser = argparse.ArgumentParser(description="Test monogamy on evolved states")
    parser.add_argument("results_dir", type=str, help="Results directory from evolution")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing Monogamy on Evolved State")
    print("=" * 70)
    
    # Load evolved state
    final_state, pattern_sites, env_sites, params = load_evolved_state(args.results_dir)
    
    # Test monogamy
    results = test_monogamy(final_state, pattern_sites, env_sites, params)
    
    # Save results
    import json
    output_file = os.path.join(args.results_dir, "monogamy_test_evolved.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()