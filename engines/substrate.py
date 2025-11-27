"""
Unified Substrate Framework (Vectorized)
========================================
Properly vectorized implementation using sparse matrices.
No Python loops over Hilbert space dimension.
"""

import os
import warnings

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
    import numpy as np
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

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Substrate:
    """
    Vectorized Hilbert space substrate.
    Uses sparse matrices and broadcasting - no Python loops over dim.
    """
    
    def __init__(self, n_modes: int, dim_per_mode: int = 2):
        self.n_modes = n_modes
        self.d = dim_per_mode
        self.dim = dim_per_mode ** n_modes
        
        # Precompute single-mode operators (sparse)
        self._a_ops = []  # Annihilation operators for each mode
        self._adag_ops = []  # Creation operators
        self._n_ops = []  # Number operators
        
        self._build_operators()
    
    def _build_operators(self):
        """Build sparse ladder operators for each mode."""
        # Single-mode annihilation operator
        diag_vals = np.sqrt(np.arange(1, self.d, dtype=np.float64))
        a_single = sparse.diags(diag_vals, offsets=1, shape=(self.d, self.d), format='csr')
        
        for mode in range(self.n_modes):
            # Build full operator via tensor product
            # a_mode = I ⊗ ... ⊗ a ⊗ ... ⊗ I
            ops = []
            for m in range(self.n_modes):
                if m == mode:
                    ops.append(a_single)
                else:
                    ops.append(sparse.eye(self.d, format='csr'))
            
            # Tensor product (kronecker)
            result = ops[0]
            for op in ops[1:]:
                result = sparse.kron(result, op, format='csr')
            
            self._a_ops.append(result)
            self._adag_ops.append(result.T.conj())
            self._n_ops.append(result.T.conj() @ result)
    
    def vacuum(self) -> xp.ndarray:
        """Ground state |0,0,...,0⟩"""
        psi = xp.zeros(self.dim, dtype=xp.complex128)
        psi[0] = 1.0
        return psi
    
    def basis_state(self, config: Tuple[int, ...]) -> xp.ndarray:
        """Basis state |n_0, n_1, ..., n_{N-1}⟩"""
        idx = 0
        for i, n in enumerate(config):
            idx += n * (self.d ** (self.n_modes - 1 - i))
        psi = xp.zeros(self.dim, dtype=xp.complex128)
        psi[idx] = 1.0
        return psi
    
    def excite(self, psi: xp.ndarray, mode: int) -> xp.ndarray:
        """Apply creation operator a†_mode. Vectorized."""
        return self._adag_ops[mode] @ psi
    
    def annihilate(self, psi: xp.ndarray, mode: int) -> xp.ndarray:
        """Apply annihilation operator a_mode. Vectorized."""
        return self._a_ops[mode] @ psi
    
    def measure_occupation(self, psi: xp.ndarray, mode: int) -> float:
        """⟨n_mode⟩ = ⟨ψ|n̂_mode|ψ⟩. Vectorized."""
        n_psi = self._n_ops[mode] @ psi
        return float(xp.real(xp.vdot(psi, n_psi)))
    
    def reduced_density_matrix(self, psi: xp.ndarray, keep_modes: List[int]) -> xp.ndarray:
        """
        Partial trace - keep specified modes, trace out the rest.
        Vectorized using reshape and einsum.
        """
        n_keep = len(keep_modes)
        n_trace = self.n_modes - n_keep
        trace_modes = [m for m in range(self.n_modes) if m not in keep_modes]
        
        # Reshape state vector to tensor
        shape = [self.d] * self.n_modes
        psi_tensor = psi.reshape(shape)
        
        # Reorder axes: keep_modes first, then trace_modes
        new_order = keep_modes + trace_modes
        psi_reordered = xp.transpose(psi_tensor, new_order)
        
        # Reshape to (keep_dim, trace_dim)
        keep_dim = self.d ** n_keep
        trace_dim = self.d ** n_trace
        psi_matrix = psi_reordered.reshape(keep_dim, trace_dim)
        
        # ρ_keep = Tr_trace(|ψ⟩⟨ψ|) = ψ @ ψ†
        rho = psi_matrix @ xp.conj(psi_matrix.T)
        
        return rho
    
    def correlation(self, psi: xp.ndarray, mode_i: int, mode_j: int) -> float:
        """⟨n_i n_j⟩ - ⟨n_i⟩⟨n_j⟩. Vectorized."""
        n_i = self.measure_occupation(psi, mode_i)
        n_j = self.measure_occupation(psi, mode_j)
        
        # ⟨n_i n_j⟩
        ni_nj_psi = self._n_ops[mode_i] @ (self._n_ops[mode_j] @ psi)
        ni_nj = float(xp.real(xp.vdot(psi, ni_nj_psi)))
        
        return ni_nj - n_i * n_j


def hopping_hamiltonian(sub: Substrate, connectivity: Dict[int, List[int]], 
                        coupling: float = 1.0) -> xp.ndarray:
    """
    H = -t Σ_{⟨ij⟩} (a†_i a_j + h.c.)
    Built from sparse operators - no loops over Hilbert space.
    """
    H = sparse.csr_matrix((sub.dim, sub.dim), dtype=xp.complex128)
    
    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j > i:
                H = H - coupling * (sub._adag_ops[i] @ sub._a_ops[j])
                H = H - coupling * (sub._adag_ops[j] @ sub._a_ops[i])
    
    return H.toarray()


def detection_unitary(sub: Substrate, sys_mode: int, env_mode: int,
                      strength: float = np.pi/2) -> xp.ndarray:
    """
    CNOT-like detection unitary.
    Controlled-X gate: if system is excited, flip environment.
    Uses projector-based construction for proper CNOT behavior.
    """
    # Build projectors onto system occupation states
    # P_n = |n><n| for each occupation n
    # CNOT = Σ_n P_n ⊗ X^n  (apply X n times if occupation is n)
    
    dim = sub.dim
    H_ctrl = xp.zeros((dim, dim), dtype=xp.complex128)
    
    # For qubit case (d=2), this simplifies to:
    # |0><0| ⊗ I + |1><1| ⊗ X
    # In terms of number operators: exp(-i π/4 * n_sys ⊗ σ_x_env)
    
    # Build σ_x for the environment mode (in the full Hilbert space)
    # σ_x = a + a† for qubit
    sigma_x_env = (sub._a_ops[env_mode] + sub._adag_ops[env_mode]).toarray()
    n_sys = sub._n_ops[sys_mode].toarray()
    
    # H = n_sys * sigma_x_env creates the CNOT-like coupling
    H_ctrl = n_sys @ sigma_x_env
    
    return expm(-1j * strength * H_ctrl)


def linear_chain(n: int) -> Dict[int, List[int]]:
    return {i: [j for j in [i-1, i+1] if 0 <= j < n] for i in range(n)}


def full_connectivity(n: int) -> Dict[int, List[int]]:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def create_patterns(sub: Substrate) -> Dict[str, xp.ndarray]:
    """Create test patterns."""
    patterns = {}
    
    config_exc = tuple([1 if i == 0 else 0 for i in range(sub.n_modes)])
    patterns['local'] = sub.basis_state(config_exc)
    
    c1 = tuple([1 if i == 0 else 0 for i in range(sub.n_modes)])
    c2 = tuple([1 if i == 1 else 0 for i in range(sub.n_modes)])
    psi = (sub.basis_state(c1) + sub.basis_state(c2)) / np.sqrt(2)
    patterns['symmetric'] = psi
    
    psi = (sub.basis_state(c1) - sub.basis_state(c2)) / np.sqrt(2)
    patterns['antisymmetric'] = psi
    
    return patterns


def test_propagation(sub: Substrate, H: xp.ndarray, source: int,
                     t_max: float = 10.0, n_steps: int = 100) -> Dict:
    """Track excitation spread."""
    psi = sub.excite(sub.vacuum(), source)
    norm = xp.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    U = expm(-1j * H * dt)
    
    history = np.zeros((sub.n_modes, n_steps))
    
    for t_idx in range(n_steps):
        for m in range(sub.n_modes):
            history[m, t_idx] = sub.measure_occupation(psi, m)
        psi = U @ psi
    
    threshold = 0.01
    arrival = {}
    for m in range(sub.n_modes):
        if m == source:
            arrival[m] = 0.0
        else:
            crossed = np.where(history[m] > threshold)[0]
            arrival[m] = times[crossed[0]] if len(crossed) > 0 else np.inf
    
    return {'times': times, 'history': history, 'arrival': arrival, 'source': source}


def test_information_structure(sub: Substrate, psi: xp.ndarray, 
                                system_modes: List[int]) -> Dict:
    """Analyze diagonal vs off-diagonal information."""
    rho = sub.reduced_density_matrix(psi, system_modes)
    
    diagonal = xp.diag(xp.diag(rho))
    off_diagonal = rho - diagonal
    
    diag_weight = float(xp.real(xp.trace(diagonal @ diagonal)))
    offdiag_weight = float(xp.real(xp.trace(off_diagonal @ xp.conj(off_diagonal).T)))
    purity = float(xp.real(xp.trace(rho @ rho)))
    
    rho_np = rho if not GPU_AVAILABLE else cp.asnumpy(rho)
    
    return {'diagonal': diag_weight, 'off_diagonal': offdiag_weight, 
            'purity': purity, 'rho': rho_np}


def test_copyability(sub: Substrate, psi: xp.ndarray, 
                     system_modes: List[int], env_modes: List[int],
                     n_detections: int = 5) -> Dict:
    """Test pattern survival under detection."""
    rho_orig = sub.reduced_density_matrix(psi, system_modes)
    
    purity_seq = [1.0]
    fidelity_seq = [1.0]
    
    psi_curr = psi.copy()
    for i in range(n_detections):
        env_m = env_modes[i % len(env_modes)]
        U = detection_unitary(sub, system_modes[0], env_m)
        psi_curr = U @ psi_curr
        psi_curr = psi_curr / xp.linalg.norm(psi_curr)
        
        rho = sub.reduced_density_matrix(psi_curr, system_modes)
        purity_seq.append(float(xp.real(xp.trace(rho @ rho))))
        fidelity_seq.append(float(xp.abs(xp.trace(rho_orig @ rho))))
    
    return {'purity': purity_seq, 'fidelity': fidelity_seq}


def test_correlations(sub: Substrate, psi: xp.ndarray) -> np.ndarray:
    """Compute correlation matrix."""
    corr = np.zeros((sub.n_modes, sub.n_modes))
    for i in range(sub.n_modes):
        for j in range(sub.n_modes):
            corr[i, j] = sub.correlation(psi, i, j)
    return corr


def compute_metric(sub: Substrate, H: xp.ndarray, t_max: float = 10.0) -> np.ndarray:
    """Compute emergent distance matrix."""
    metric = np.zeros((sub.n_modes, sub.n_modes))
    for source in range(sub.n_modes):
        result = test_propagation(sub, H, source, t_max)
        for target in range(sub.n_modes):
            metric[source, target] = result['arrival'][target]
    return (metric + metric.T) / 2


def plot_all(sub: Substrate, results: Dict, output_dir: str):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes, ['prop_local', 'prop_full'], 
                               ['Local Connectivity', 'Full Connectivity']):
        r = results[key]
        im = ax.imshow(r['history'], aspect='auto', origin='lower',
                       extent=[r['times'][0], r['times'][-1], -0.5, sub.n_modes-0.5],
                       cmap='inferno', vmin=0, vmax=1)
        for m, t in r['arrival'].items():
            if 0 < t < np.inf:
                ax.plot(t, m, 'w^', markersize=6)
        ax.axhline(r['source'], color='cyan', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mode')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Excitation')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/light_cones.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    metric = results['metric']
    im = ax.imshow(np.where(metric == np.inf, np.nan, metric), cmap='viridis', origin='lower')
    for i in range(sub.n_modes):
        for j in range(sub.n_modes):
            if metric[i,j] < np.inf:
                ax.text(j, i, f'{metric[i,j]:.1f}', ha='center', va='center',
                       color='white' if metric[i,j] > np.nanmax(metric)/2 else 'black', fontsize=8)
    ax.set_xlabel('Mode j')
    ax.set_ylabel('Mode i')
    ax.set_title('Emergent Metric')
    plt.colorbar(im, ax=ax, label='Distance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = results['correlations']
    im = ax.imshow(corr, cmap='RdBu', origin='lower', vmin=-0.5, vmax=0.5)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                   color='white' if abs(corr[i,j]) > 0.25 else 'black', fontsize=9)
    ax.set_xlabel('Mode j')
    ax.set_ylabel('Mode i')
    ax.set_title('Correlation Structure')
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlations.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    patterns = list(results['info'].keys())
    diag = [results['info'][p]['diagonal'] for p in patterns]
    offdiag = [results['info'][p]['off_diagonal'] for p in patterns]
    x = np.arange(len(patterns))
    ax.bar(x - 0.2, diag, 0.4, label='Diagonal', color='steelblue')
    ax.bar(x + 0.2, offdiag, 0.4, label='Off-diagonal', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in patterns])
    ax.set_ylabel('Information Weight')
    ax.set_title('Information Structure')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/information.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'local': 'blue', 'symmetric': 'green', 'antisymmetric': 'red'}
    for name, r in results['copy'].items():
        ax.plot(r['fidelity'], 'o-', label=name.capitalize(), color=colors[name])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Detection #')
    ax.set_ylabel('Fidelity')
    ax.set_title('Copyability Test')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/copyability.png', dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results['copy'].items():
        ax.scatter(r['fidelity'][-1], results['info'][name]['purity'],
                  s=200, c=colors[name], label=name.capitalize(), edgecolor='black')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.75, 0.8, 'BOSONIC', ha='center', fontsize=12, alpha=0.7)
    ax.text(0.25, 0.2, 'FERMIONIC', ha='center', fontsize=12, alpha=0.7)
    ax.set_xlabel('Final Fidelity')
    ax.set_ylabel('Purity')
    ax.set_title('Classification')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification.png', dpi=150)
    plt.close()


def main():
    import time
    
    output_dir = 'outputs'
    n_modes = 8
    
    print("\nUnified Substrate Framework (Vectorized)")
    print("="*50)
    
    t_start = time.time()
    sub = Substrate(n_modes, dim_per_mode=2)
    t_build = time.time() - t_start
    
    print(f"Modes: {n_modes}, Dimension: {sub.dim}")
    print(f"Operator build time: {t_build:.3f}s")
    
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    
    results = {}
    
    print("\nTesting spacetime emergence...")
    t_start = time.time()
    source = n_modes // 2
    
    H_local = hopping_hamiltonian(sub, linear_chain(n_modes))
    H_full = hopping_hamiltonian(sub, full_connectivity(n_modes))
    
    results['prop_local'] = test_propagation(sub, H_local, source, t_max=8.0)
    results['prop_full'] = test_propagation(sub, H_full, source, t_max=8.0)
    results['metric'] = compute_metric(sub, H_local, t_max=10.0)
    
    t_prop = time.time() - t_start
    print(f"  Propagation tests: {t_prop:.2f}s")
    
    for m in range(n_modes):
        print(f"  Mode {m}: d = {results['prop_local']['arrival'][m]:.2f}")
    
    print("\nTesting entanglement structure...")
    c1 = [0]*n_modes; c1[0] = c1[1] = 1
    c2 = [0]*n_modes; c2[2] = c2[3] = 1
    c3 = [0]*n_modes; c3[0] = c3[1] = c3[2] = c3[3] = 1
    
    psi_ent = (sub.basis_state(tuple([0]*n_modes)) + 
               sub.basis_state(tuple(c1)) + 
               sub.basis_state(tuple(c2)) + 
               sub.basis_state(tuple(c3)))
    psi_ent = psi_ent / xp.linalg.norm(psi_ent)
    
    results['correlations'] = test_correlations(sub, psi_ent)
    
    print("\nTesting information patterns...")
    patterns = create_patterns(sub)
    
    results['info'] = {}
    results['copy'] = {}
    
    for name, psi in patterns.items():
        results['info'][name] = test_information_structure(sub, psi, system_modes)
        results['copy'][name] = test_copyability(sub, psi, system_modes, env_modes)
        
        info = results['info'][name]
        copy = results['copy'][name]
        pclass = "BOSONIC" if copy['fidelity'][-1] > 0.5 else "FERMIONIC"
        print(f"  {name:<14} {info['diagonal']:.3f}  {info['off_diagonal']:.3f}  {pclass}")
    
    print(f"\nGenerating visualizations...")
    plot_all(sub, results, output_dir)
    
    print(f"Outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()