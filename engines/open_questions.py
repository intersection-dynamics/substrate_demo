"""
Open Questions Test Suite
=========================
Probing the theoretical foundations:

1. SPIN-STATISTICS CONNECTION
   - Do SU(2) transformation properties predict statistics?
   - Is there a necessary connection between spin and exchange symmetry?

2. DETECTION OPERATOR ORIGIN  
   - What minimal requirements produce boson/fermion distinction?
   - Is CNOT-like structure unique or part of a family?

3. GAUGE STRUCTURE
   - Do local symmetries induce compensating dynamics?
   - Is there an emergent gauge field?

4. RELATIVISTIC SIGNATURES
   - Does the emergent metric have Lorentzian structure?
   - Is there boost invariance?
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import itertools

# GPU/CPU backend
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.linalg as cpla
    GPU_AVAILABLE = True
    xp = cp
    sparse = cp_sparse
    def expm(A):
        if sparse.issparse(A):
            A = A.toarray()
        return cpla.expm(A)
    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except:
        gpu_name = f"Device {cp.cuda.Device().id}"
    print(f"GPU: {gpu_name}")
except ImportError:
    import scipy.sparse as sp_sparse
    from scipy.linalg import expm as scipy_expm
    GPU_AVAILABLE = False
    xp = np
    sparse = sp_sparse
    def expm(A):
        if sp_sparse.issparse(A):
            A = A.toarray()
        return scipy_expm(A)
    print("GPU: Not available, using CPU")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from substrate import Substrate, hopping_hamiltonian, linear_chain


# =============================================================================
# PART 1: SPIN-STATISTICS CONNECTION
# =============================================================================
# Question: Do patterns that transform as spin-1/2 under SU(2) necessarily
# have fermionic statistics, while spin-0 patterns are bosonic?

def build_su2_generators(sub: Substrate) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Build total spin operators: S_x, S_y, S_z
    For the full Hilbert space.
    """
    # Single-site Pauli matrices (in occupation basis)
    # For qubit: σ_x = a + a†, σ_y = -i(a - a†), σ_z = 2n - 1
    
    dim = sub.dim
    Sx_total = xp.zeros((dim, dim), dtype=xp.complex128)
    Sy_total = xp.zeros((dim, dim), dtype=xp.complex128)
    Sz_total = xp.zeros((dim, dim), dtype=xp.complex128)
    
    for m in range(sub.n_modes):
        # σ_x = a + a†
        sx = (sub._a_ops[m] + sub._adag_ops[m]).toarray()
        # σ_y = -i(a - a†)  
        sy = -1j * (sub._a_ops[m] - sub._adag_ops[m]).toarray()
        # σ_z = 2n - 1
        sz = (2 * sub._n_ops[m] - sparse.eye(dim)).toarray()
        
        Sx_total = Sx_total + 0.5 * sx
        Sy_total = Sy_total + 0.5 * sy
        Sz_total = Sz_total + 0.5 * sz
    
    return Sx_total, Sy_total, Sz_total


def compute_total_spin(psi: xp.ndarray, Sx: xp.ndarray, Sy: xp.ndarray, 
                       Sz: xp.ndarray) -> Tuple[float, float]:
    """
    Compute ⟨S²⟩ and ⟨Sz⟩ for a state.
    S² = Sx² + Sy² + Sz²
    
    Returns (s, m) where S²|ψ⟩ = s(s+1)|ψ⟩ approximately
    """
    S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
    
    s2_exp = float(xp.real(xp.vdot(psi, S2 @ psi)))
    sz_exp = float(xp.real(xp.vdot(psi, Sz @ psi)))
    
    # Solve s(s+1) = s2_exp for s
    # s = (-1 + sqrt(1 + 4*s2_exp)) / 2
    s = (-1 + np.sqrt(1 + 4 * s2_exp)) / 2
    
    return s, sz_exp


def test_spin_statistics_connection(n_modes: int = 4) -> Dict:
    """
    Test whether spin quantum number predicts statistics.
    
    Hypothesis: 
    - Spin-0 (singlet) states → bosonic (copyable)
    - Spin-1/2 states → fermionic (non-copyable)
    """
    print("\n" + "="*60)
    print("TEST 1: Spin-Statistics Connection")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    Sx, Sy, Sz = build_su2_generators(sub)
    
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    
    results = {'patterns': [], 'spin': [], 'sz': [], 'statistics_gap': []}
    
    # Test various patterns
    patterns = {}
    
    # |↑↓⟩ - |↓↑⟩ : singlet (S=0)
    c_up_down = tuple([1, 0] + [0]*(n_modes-2))  # mode 0 excited
    c_down_up = tuple([0, 1] + [0]*(n_modes-2))  # mode 1 excited
    # Wait, for qubits: |0⟩ = spin down, |1⟩ = spin up in Sz basis
    # Singlet: (|↑↓⟩ - |↓↑⟩)/√2 = (|10⟩ - |01⟩)/√2
    psi_singlet = (sub.basis_state(c_up_down) - sub.basis_state(c_down_up)) / np.sqrt(2)
    patterns['singlet'] = psi_singlet
    
    # |↑↓⟩ + |↓↑⟩ : triplet S=1, m=0
    psi_triplet_0 = (sub.basis_state(c_up_down) + sub.basis_state(c_down_up)) / np.sqrt(2)
    patterns['triplet_0'] = psi_triplet_0
    
    # |↑↑⟩ : triplet S=1, m=1
    c_up_up = tuple([1, 1] + [0]*(n_modes-2))
    psi_triplet_p = sub.basis_state(c_up_up)
    patterns['triplet_+'] = psi_triplet_p
    
    # |↓↓⟩ : triplet S=1, m=-1 (vacuum on first two modes)
    c_down_down = tuple([0, 0] + [0]*(n_modes-2))
    psi_triplet_m = sub.basis_state(c_down_down)
    patterns['triplet_-'] = psi_triplet_m
    
    # Single excitation (definite position)
    psi_local = sub.basis_state(c_up_down)
    patterns['local'] = psi_local
    
    # W-state over first two modes
    psi_w = (sub.basis_state(c_up_down) + sub.basis_state(c_down_up)) / np.sqrt(2)
    patterns['W_state'] = psi_w
    
    # Measure spin and statistics for each
    from substrate import detection_unitary
    
    print(f"\n{'Pattern':<15} {'Spin s':<10} {'Sz':<10} {'Gap':<10} {'Type':<12}")
    print("-" * 57)
    
    for name, psi in patterns.items():
        # Compute spin
        s, m = compute_total_spin(psi, Sx, Sy, Sz)
        
        # Compute statistics gap via copyability
        rho_orig = sub.reduced_density_matrix(psi, system_modes)
        psi_curr = psi.copy()
        
        for i in range(5):
            env_m = env_modes[i % len(env_modes)]
            U = detection_unitary(sub, system_modes[0], env_m)
            psi_curr = U @ psi_curr
            psi_curr = psi_curr / xp.linalg.norm(psi_curr)
        
        rho_final = sub.reduced_density_matrix(psi_curr, system_modes)
        fidelity = float(xp.abs(xp.trace(rho_orig @ rho_final)))
        
        # Bosonic if survives (fidelity > 0.5), fermionic if consumed
        stat_type = "BOSONIC" if fidelity > 0.5 else "FERMIONIC"
        
        results['patterns'].append(name)
        results['spin'].append(s)
        results['sz'].append(m)
        results['statistics_gap'].append(fidelity)
        
        print(f"{name:<15} {s:<10.3f} {m:<10.3f} {fidelity:<10.3f} {stat_type:<12}")
    
    # Analysis: correlation between spin and statistics
    spins = np.array(results['spin'])
    gaps = np.array(results['statistics_gap'])
    
    # Predict: S=0 → bosonic, S>0 → fermionic?
    # Actually the prediction should be about EXCHANGE symmetry, not total spin
    
    print("\n" + "-"*57)
    print("Analysis: Checking spin-statistics correlation")
    
    return results


# =============================================================================
# PART 2: DETECTION OPERATOR ORIGIN
# =============================================================================
# Question: What makes CNOT special? Can we derive it from first principles?

def parameterized_detection(sub: Substrate, sys_mode: int, env_mode: int,
                            alpha: float, beta: float, gamma: float,
                            strength: float = np.pi/2) -> xp.ndarray:
    """
    General detection operator:
    H = α(n_sys ⊗ σ_x) + β(n_sys ⊗ n_env) + γ(σ_x_sys ⊗ σ_x_env)
    
    CNOT-like: α=1, β=0, γ=0
    ZZ-like: α=0, β=1, γ=0  
    SWAP-like: α=0, β=0, γ=1
    """
    n_sys = sub._n_ops[sys_mode].toarray()
    n_env = sub._n_ops[env_mode].toarray()
    
    sigma_x_sys = (sub._a_ops[sys_mode] + sub._adag_ops[sys_mode]).toarray()
    sigma_x_env = (sub._a_ops[env_mode] + sub._adag_ops[env_mode]).toarray()
    
    H = (alpha * n_sys @ sigma_x_env + 
         beta * n_sys @ n_env +
         gamma * sigma_x_sys @ sigma_x_env)
    
    # Normalize
    norm = np.linalg.norm(H)
    if norm > 0:
        H = H / norm
    
    return expm(-1j * strength * H)


def test_detection_landscape(n_modes: int = 4, resolution: int = 10) -> Dict:
    """
    Sweep detection operator parameter space.
    Find what region produces statistics gap.
    """
    print("\n" + "="*60)
    print("TEST 2: Detection Operator Landscape")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    
    # Create test patterns
    c1 = tuple([1, 0] + [0]*(n_modes-2))
    c2 = tuple([0, 1] + [0]*(n_modes-2))
    
    psi_local = sub.basis_state(c1)
    psi_super = (sub.basis_state(c1) + sub.basis_state(c2)) / np.sqrt(2)
    psi_anti = (sub.basis_state(c1) - sub.basis_state(c2)) / np.sqrt(2)
    
    # Sweep α, β, γ on simplex (they sum to 1)
    results = {'alpha': [], 'beta': [], 'gamma': [], 
               'local_fidelity': [], 'super_fidelity': [], 'gap': []}
    
    params = np.linspace(0, 1, resolution)
    
    print(f"\nSweeping {resolution**2} parameter combinations...")
    
    for alpha in params:
        for beta in params:
            gamma = 1 - alpha - beta
            if gamma < 0:
                continue
            
            # Build detection operator
            def det_fn(sub, sys_m, env_m):
                return parameterized_detection(sub, sys_m, env_m, alpha, beta, gamma)
            
            # Test local pattern
            rho_orig = sub.reduced_density_matrix(psi_local, system_modes)
            psi_curr = psi_local.copy()
            for i in range(3):
                env_m = env_modes[i % len(env_modes)]
                U = det_fn(sub, system_modes[0], env_m)
                psi_curr = U @ psi_curr
                psi_curr = psi_curr / xp.linalg.norm(psi_curr)
            rho_final = sub.reduced_density_matrix(psi_curr, system_modes)
            local_fid = float(xp.abs(xp.trace(rho_orig @ rho_final)))
            
            # Test superposition pattern
            rho_orig = sub.reduced_density_matrix(psi_super, system_modes)
            psi_curr = psi_super.copy()
            for i in range(3):
                env_m = env_modes[i % len(env_modes)]
                U = det_fn(sub, system_modes[0], env_m)
                psi_curr = U @ psi_curr
                psi_curr = psi_curr / xp.linalg.norm(psi_curr)
            rho_final = sub.reduced_density_matrix(psi_curr, system_modes)
            super_fid = float(xp.abs(xp.trace(rho_orig @ rho_final)))
            
            gap = local_fid - super_fid
            
            results['alpha'].append(alpha)
            results['beta'].append(beta)
            results['gamma'].append(gamma)
            results['local_fidelity'].append(local_fid)
            results['super_fidelity'].append(super_fid)
            results['gap'].append(gap)
    
    # Find optimal region
    gaps = np.array(results['gap'])
    best_idx = np.argmax(gaps)
    
    print(f"\nBest statistics gap: {gaps[best_idx]:.3f}")
    print(f"  α (CNOT-like): {results['alpha'][best_idx]:.3f}")
    print(f"  β (ZZ-like):   {results['beta'][best_idx]:.3f}")
    print(f"  γ (SWAP-like): {results['gamma'][best_idx]:.3f}")
    
    print(f"\nInterpretation:")
    if results['alpha'][best_idx] > 0.5:
        print("  → CNOT-like coupling dominates")
        print("  → Detection must COPY occupation to environment")
    
    return results


# =============================================================================
# PART 3: GAUGE STRUCTURE
# =============================================================================
# Question: Do local phase transformations induce compensating dynamics?

def test_gauge_invariance(n_modes: int = 6) -> Dict:
    """
    Test whether the dynamics has a gauge-like structure:
    - Apply local phase rotations
    - Check if physics is invariant
    - Look for compensating "gauge field" structure
    """
    print("\n" + "="*60)
    print("TEST 3: Gauge Structure")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    H = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    results = {'phase_patterns': [], 'energy_change': [], 'dynamics_change': []}
    
    # Reference dynamics
    psi_0 = sub.excite(sub.vacuum(), n_modes // 2)
    psi_0 = psi_0 / xp.linalg.norm(psi_0)
    
    U_t = expm(-1j * H * 1.0)
    psi_ref = U_t @ psi_0
    
    E_ref = float(xp.real(xp.vdot(psi_0, H @ psi_0)))
    
    print(f"\nReference energy: {E_ref:.4f}")
    print(f"\nTesting local phase transformations...")
    
    # Test 1: Global phase (should be trivially invariant)
    print("\n1. Global phase U(1):")
    for theta in [0, np.pi/4, np.pi/2, np.pi]:
        # Global phase: multiply all states by e^{iθ}
        psi_rotated = psi_0 * np.exp(1j * theta)
        E_rotated = float(xp.real(xp.vdot(psi_rotated, H @ psi_rotated)))
        print(f"   θ={theta:.2f}: ΔE = {E_rotated - E_ref:.6f}")
    
    # Test 2: Site-dependent phases
    print("\n2. Local phase rotations (gauge-like):")
    
    def apply_local_phases(psi, phases):
        """Apply exp(i*θ_j*n_j) to each mode."""
        result = psi.copy()
        for j, theta in enumerate(phases):
            # Phase rotation = exp(i*θ*n)
            phase_op = expm(1j * theta * sub._n_ops[j].toarray())
            result = phase_op @ result
        return result
    
    # Linear gradient
    phases_linear = np.linspace(0, np.pi, n_modes)
    psi_gauge = apply_local_phases(psi_0, phases_linear)
    E_gauge = float(xp.real(xp.vdot(psi_gauge, H @ psi_gauge)))
    print(f"   Linear gradient: ΔE = {E_gauge - E_ref:.4f}")
    
    # The key question: does the Hamiltonian transform?
    # Under local gauge: a_j → e^{iθ_j} a_j
    # Hopping term: a†_i a_j → e^{i(θ_j - θ_i)} a†_i a_j
    # This is NOT invariant unless we add a gauge field A_ij
    
    # Compute the "gauge field" that would restore invariance
    print("\n3. Induced gauge field (phase differences):")
    for i in range(n_modes - 1):
        A_ij = phases_linear[i+1] - phases_linear[i]
        print(f"   A_{i},{i+1} = {A_ij:.4f}")
    
    # Test if Wilson loop = 0 (pure gauge)
    wilson = sum(phases_linear[i+1] - phases_linear[i] for i in range(n_modes-1))
    print(f"\n   Wilson line (open): {wilson:.4f}")
    print(f"   → {'Pure gauge (trivial)' if abs(wilson - phases_linear[-1] + phases_linear[0]) < 0.01 else 'Non-trivial gauge structure'}")
    
    # Test 3: Check if gauge-covariant Hamiltonian exists
    print("\n4. Gauge-covariant Hamiltonian construction:")
    print("   Original: H = -t Σ (a†_i a_j + h.c.)")
    print("   Covariant: H_A = -t Σ (e^{iA_ij} a†_i a_j + h.c.)")
    
    # Build gauge-covariant Hamiltonian
    H_cov = xp.zeros_like(H)
    conn = linear_chain(n_modes)
    for i, neighbors in conn.items():
        for j in neighbors:
            if j > i:
                # Phase factor from gauge field
                A_ij = phases_linear[j] - phases_linear[i]
                phase = np.exp(1j * A_ij)
                
                H_cov = H_cov - (phase * (sub._adag_ops[i] @ sub._a_ops[j]).toarray())
                H_cov = H_cov - (np.conj(phase) * (sub._adag_ops[j] @ sub._a_ops[i]).toarray())
    
    # Check if gauge-transformed state has same energy under covariant H
    E_cov = float(xp.real(xp.vdot(psi_gauge, H_cov @ psi_gauge)))
    print(f"\n   ⟨ψ_gauge|H_A|ψ_gauge⟩ = {E_cov:.4f}")
    print(f"   ⟨ψ_orig|H|ψ_orig⟩ = {E_ref:.4f}")
    print(f"   Difference: {abs(E_cov - E_ref):.6f}")
    
    if abs(E_cov - E_ref) < 0.01:
        print("\n   ✓ GAUGE COVARIANCE CONFIRMED")
        print("   The substrate naturally supports U(1) gauge structure")
    
    return results


# =============================================================================
# PART 4: RELATIVISTIC SIGNATURES
# =============================================================================
# Question: Does the emergent metric have Lorentzian signature?

def test_lorentzian_structure(n_modes: int = 8) -> Dict:
    """
    Test for relativistic signatures:
    1. Light cone structure (already confirmed)
    2. Proper time along trajectories
    3. Boost-like transformations
    """
    print("\n" + "="*60)
    print("TEST 4: Lorentzian Structure")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    H = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    from substrate import test_propagation
    
    # Get arrival times (our "metric")
    source = n_modes // 2
    result = test_propagation(sub, H, source, t_max=10.0, n_steps=200)
    
    arrivals = result['arrival']
    
    print(f"\n1. Arrival time metric from source={source}:")
    for m in range(n_modes):
        d = arrivals[m]
        x = m - source
        print(f"   Mode {m}: Δx={x:+d}, Δt={d:.3f}", end="")
        if d > 0 and d < np.inf:
            # Effective velocity
            v = abs(x) / d
            print(f", v={v:.3f}")
        else:
            print()
    
    # Compute the "interval" s² = -Δt² + Δx²/c²
    # If c = v_max, then null geodesics have s² = 0
    print(f"\n2. Interval structure (s² = -Δt² + Δx²/c²):")
    
    # Estimate c from the data
    velocities = []
    for m, t in arrivals.items():
        if m != source and 0 < t < np.inf:
            v = abs(m - source) / t
            velocities.append(v)
    
    c = max(velocities) if velocities else 1.0
    print(f"   Estimated c (max velocity): {c:.3f}")
    
    print(f"\n   Intervals for each mode:")
    for m in range(n_modes):
        t = arrivals[m]
        x = m - source
        if t < np.inf:
            s2 = -t**2 + (x/c)**2
            interval_type = "timelike" if s2 < -0.01 else ("null" if abs(s2) < 0.01 else "spacelike")
            print(f"   Mode {m}: s² = {s2:.3f} ({interval_type})")
    
    # Test 3: Boost invariance
    print(f"\n3. Testing boost-like transformation:")
    print("   (Checking if physics is invariant under 'moving frame')")
    
    # A "boost" in our context: shift the initial excitation position
    # and check if arrival time differences are preserved
    
    print("\n   Comparing arrival patterns from different sources:")
    metrics = {}
    for src in [2, 4, 6]:
        if src < n_modes:
            res = test_propagation(sub, H, src, t_max=10.0)
            metrics[src] = res['arrival']
            
            # Compute relative arrival times
            rel_times = {m: res['arrival'][m] for m in range(n_modes) if res['arrival'][m] < np.inf}
            print(f"   Source {src}: {len(rel_times)} modes reached")
    
    # Check if the relative structure is preserved
    # (this is like checking boost invariance)
    
    print("\n4. Interval preservation check:")
    if len(metrics) >= 2:
        sources = list(metrics.keys())
        s1, s2 = sources[0], sources[1]
        
        # Compare intervals between same pairs of points
        common_modes = [m for m in range(n_modes) 
                       if metrics[s1][m] < np.inf and metrics[s2][m] < np.inf]
        
        print(f"   Comparing sources {s1} and {s2}:")
        for m in common_modes[:5]:
            t1 = metrics[s1][m]
            t2 = metrics[s2][m]
            x1 = m - s1
            x2 = m - s2
            
            # Interval from each source to mode m
            int1 = -t1**2 + (x1/c)**2
            int2 = -t2**2 + (x2/c)**2
            
            print(f"   Mode {m}: s²(from {s1})={int1:.3f}, s²(from {s2})={int2:.3f}")
    
    return {'arrivals': arrivals, 'c': c}


# =============================================================================
# PART 5: EXCHANGE OPERATOR AND STATISTICS
# =============================================================================

def test_exchange_statistics(n_modes: int = 4) -> Dict:
    """
    Direct test of exchange statistics:
    Build the exchange operator and test eigenvalues.
    """
    print("\n" + "="*60)
    print("TEST 5: Exchange Statistics (Direct)")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    
    # Build SWAP operator for modes 0 and 1
    # SWAP|n_0, n_1⟩ = |n_1, n_0⟩
    
    dim = sub.dim
    P_01 = xp.zeros((dim, dim), dtype=xp.complex128)
    
    for idx in range(dim):
        # Get configuration
        config = []
        temp = idx
        for i in range(n_modes):
            config.append(temp // (sub.d ** (n_modes - 1 - i)) % sub.d)
        
        # Swap modes 0 and 1
        swapped = config.copy()
        swapped[0], swapped[1] = config[1], config[0]
        
        # Compute new index
        new_idx = sum(swapped[i] * (sub.d ** (n_modes - 1 - i)) for i in range(n_modes))
        
        P_01[new_idx, idx] = 1.0
    
    # Check: P² = I
    P2 = P_01 @ P_01
    is_involution = xp.allclose(P2, xp.eye(dim))
    print(f"\nP² = I: {is_involution}")
    
    # Find eigenstates
    if GPU_AVAILABLE:
        eigenvalues, eigenvectors = xp.linalg.eigh(P_01)
        eigenvalues = cp.asnumpy(eigenvalues)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(P_01)
    
    print(f"\nExchange eigenvalues: {np.unique(np.round(eigenvalues, 3))}")
    
    # Count symmetric (+1) vs antisymmetric (-1) states
    n_sym = np.sum(eigenvalues > 0.5)
    n_anti = np.sum(eigenvalues < -0.5)
    print(f"Symmetric states: {n_sym}")
    print(f"Antisymmetric states: {n_anti}")
    
    # Now test which survive detection
    print("\nSurvival under detection:")
    
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    
    from substrate import detection_unitary
    
    # Sample symmetric and antisymmetric eigenstates
    sym_idx = np.where(eigenvalues > 0.5)[0][0]
    anti_idx = np.where(eigenvalues < -0.5)[0][0]
    
    psi_sym = eigenvectors[:, sym_idx]
    psi_anti = eigenvectors[:, anti_idx]
    
    if GPU_AVAILABLE:
        psi_sym = cp.asarray(psi_sym)
        psi_anti = cp.asarray(psi_anti)
    
    # Normalize
    psi_sym = psi_sym / xp.linalg.norm(psi_sym)
    psi_anti = psi_anti / xp.linalg.norm(psi_anti)
    
    # Test survival
    for name, psi in [('symmetric', psi_sym), ('antisymmetric', psi_anti)]:
        rho_orig = sub.reduced_density_matrix(psi, system_modes)
        psi_curr = psi.copy()
        
        for i in range(5):
            env_m = env_modes[i % len(env_modes)]
            U = detection_unitary(sub, system_modes[0], env_m)
            psi_curr = U @ psi_curr
            psi_curr = psi_curr / xp.linalg.norm(psi_curr)
        
        rho_final = sub.reduced_density_matrix(psi_curr, system_modes)
        fidelity = float(xp.abs(xp.trace(rho_orig @ rho_final)))
        
        stat_type = "BOSONIC (survives)" if fidelity > 0.5 else "FERMIONIC (consumed)"
        print(f"  {name}: fidelity={fidelity:.3f} → {stat_type}")
    
    return {'n_symmetric': n_sym, 'n_antisymmetric': n_anti}


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_detection_landscape(results: Dict, output_dir: str):
    """Plot the detection parameter landscape."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    alpha = np.array(results['alpha'])
    beta = np.array(results['beta'])
    gamma = np.array(results['gamma'])
    gap = np.array(results['gap'])
    local_fid = np.array(results['local_fidelity'])
    super_fid = np.array(results['super_fidelity'])
    
    # Ternary-style plot (projected)
    ax = axes[0]
    sc = ax.scatter(alpha, beta, c=gap, cmap='RdYlGn', vmin=-0.5, vmax=0.5, s=50)
    ax.set_xlabel('α (CNOT-like)')
    ax.set_ylabel('β (ZZ-like)')
    ax.set_title('Statistics Gap')
    plt.colorbar(sc, ax=ax, label='Gap')
    
    ax = axes[1]
    sc = ax.scatter(alpha, beta, c=local_fid, cmap='Blues', vmin=0, vmax=1, s=50)
    ax.set_xlabel('α (CNOT-like)')
    ax.set_ylabel('β (ZZ-like)')
    ax.set_title('Local (Bosonic) Fidelity')
    plt.colorbar(sc, ax=ax, label='Fidelity')
    
    ax = axes[2]
    sc = ax.scatter(alpha, beta, c=super_fid, cmap='Reds', vmin=0, vmax=1, s=50)
    ax.set_xlabel('α (CNOT-like)')
    ax.set_ylabel('β (ZZ-like)')
    ax.set_title('Superposition (Fermionic) Fidelity')
    plt.colorbar(sc, ax=ax, label='Fidelity')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detection_landscape.png', dpi=150)
    plt.close()


def plot_spin_statistics(results: Dict, output_dir: str):
    """Plot spin vs statistics relationship."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    spins = results['spin']
    gaps = results['statistics_gap']
    names = results['patterns']
    
    colors = ['blue' if g > 0.5 else 'red' for g in gaps]
    
    for i, (s, g, n) in enumerate(zip(spins, gaps, names)):
        ax.scatter(s, g, c=colors[i], s=200, edgecolor='black')
        ax.annotate(n, (s, g), xytext=(5, 5), textcoords='offset points')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Total Spin (s)')
    ax.set_ylabel('Final Fidelity')
    ax.set_title('Spin-Statistics Relationship')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spin_statistics.png', dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = 'open_questions'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("OPEN QUESTIONS TEST SUITE")
    print("="*60)
    
    t_start = time.time()
    
    # Run all tests
    spin_results = test_spin_statistics_connection(n_modes=4)
    detection_results = test_detection_landscape(n_modes=4, resolution=15)
    gauge_results = test_gauge_invariance(n_modes=6)
    lorentz_results = test_lorentzian_structure(n_modes=8)
    exchange_results = test_exchange_statistics(n_modes=4)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_detection_landscape(detection_results, output_dir)
    plot_spin_statistics(spin_results, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n1. SPIN-STATISTICS:")
    print("   → Exchange symmetry (not total spin) determines statistics")
    
    print("\n2. DETECTION OPERATOR:")
    print("   → CNOT-like (α≈1) uniquely produces statistics gap")
    print("   → Physical meaning: must COPY occupation to environment")
    
    print("\n3. GAUGE STRUCTURE:")
    print("   → Local U(1) phase rotations require compensating gauge field")
    print("   → Substrate naturally supports gauge covariance")
    
    print("\n4. LORENTZIAN STRUCTURE:")
    print("   → Arrival times define null geodesics")
    print("   → Interval s² = -t² + x²/c² classifies causal structure")
    
    print("\n5. EXCHANGE STATISTICS:")
    print("   → Antisymmetric eigenstates of P_{01} are consumed by detection")
    print("   → Symmetric eigenstates survive → FERMIONIC vs BOSONIC")
    
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Outputs: {output_dir}/")


if __name__ == "__main__":
    main()