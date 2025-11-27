"""
Universality Test Suite
=======================
The critical question: Do emergent phenomena depend on implementation details,
or are they inevitable consequences of the axioms?

Tests across:
- Local dimension: d = 2, 3, 4
- Hamiltonians: Hopping, Heisenberg, XY, random local
- Detection: CNOT-like, SWAP-like, random coupling
- Lattices: Chain, ring, square, honeycomb, random graph
- Patterns: Various quantum superpositions

Prediction: Light cones, metrics, and boson/fermion distinction are INVARIANT
under all variations that preserve locality + unitarity.
"""

import os
import time
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
import json

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


# =============================================================================
# SUBSTRATE CLASS (supports arbitrary local dimension)
# =============================================================================

class Substrate:
    """Hilbert space substrate with arbitrary local dimension."""
    
    def __init__(self, n_modes: int, dim_per_mode: int = 2):
        self.n_modes = n_modes
        self.d = dim_per_mode
        self.dim = dim_per_mode ** n_modes
        
        self._a_ops = []
        self._adag_ops = []
        self._n_ops = []
        self._build_operators()
    
    def _build_operators(self):
        """Build sparse ladder operators."""
        diag_vals = np.sqrt(np.arange(1, self.d, dtype=np.float64))
        a_single = sparse.diags(diag_vals, offsets=1, shape=(self.d, self.d), format='csr')
        
        for mode in range(self.n_modes):
            ops = [sparse.eye(self.d, format='csr') for _ in range(self.n_modes)]
            ops[mode] = a_single
            
            result = ops[0]
            for op in ops[1:]:
                result = sparse.kron(result, op, format='csr')
            
            self._a_ops.append(result)
            self._adag_ops.append(result.T.conj())
            self._n_ops.append(result.T.conj() @ result)
    
    def vacuum(self) -> xp.ndarray:
        psi = xp.zeros(self.dim, dtype=xp.complex128)
        psi[0] = 1.0
        return psi
    
    def basis_state(self, config: Tuple[int, ...]) -> xp.ndarray:
        idx = sum(n * (self.d ** (self.n_modes - 1 - i)) for i, n in enumerate(config))
        psi = xp.zeros(self.dim, dtype=xp.complex128)
        psi[idx] = 1.0
        return psi
    
    def excite(self, psi: xp.ndarray, mode: int) -> xp.ndarray:
        return self._adag_ops[mode] @ psi
    
    def measure_occupation(self, psi: xp.ndarray, mode: int) -> float:
        n_psi = self._n_ops[mode] @ psi
        return float(xp.real(xp.vdot(psi, n_psi)))
    
    def reduced_density_matrix(self, psi: xp.ndarray, keep_modes: List[int]) -> xp.ndarray:
        n_keep = len(keep_modes)
        trace_modes = [m for m in range(self.n_modes) if m not in keep_modes]
        
        shape = [self.d] * self.n_modes
        psi_tensor = psi.reshape(shape)
        
        new_order = keep_modes + trace_modes
        psi_reordered = xp.transpose(psi_tensor, new_order)
        
        keep_dim = self.d ** n_keep
        trace_dim = self.d ** (self.n_modes - n_keep)
        psi_matrix = psi_reordered.reshape(keep_dim, trace_dim)
        
        return psi_matrix @ xp.conj(psi_matrix.T)


# =============================================================================
# LATTICE GEOMETRIES
# =============================================================================

def lattice_chain(n: int) -> Dict[int, List[int]]:
    """Linear chain with open boundaries."""
    return {i: [j for j in [i-1, i+1] if 0 <= j < n] for i in range(n)}

def lattice_ring(n: int) -> Dict[int, List[int]]:
    """Ring (periodic boundaries)."""
    return {i: [(i-1) % n, (i+1) % n] for i in range(n)}

def lattice_square(n: int) -> Dict[int, List[int]]:
    """Square lattice (as close as possible for n sites)."""
    side = int(np.sqrt(n))
    if side * side != n:
        side = int(np.ceil(np.sqrt(n)))
    
    conn = {i: [] for i in range(n)}
    for i in range(n):
        row, col = i // side, i % side
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < side and 0 <= nc < side:
                j = nr * side + nc
                if j < n:
                    conn[i].append(j)
    return conn

def lattice_honeycomb(n: int) -> Dict[int, List[int]]:
    """Honeycomb-like (each site connects to 2-3 neighbors)."""
    conn = {i: [] for i in range(n)}
    for i in range(n):
        # Alternate connectivity pattern
        if i % 2 == 0:
            neighbors = [i-1, i+1, i+2]
        else:
            neighbors = [i-2, i-1, i+1]
        conn[i] = [j for j in neighbors if 0 <= j < n]
    return conn

def lattice_random(n: int, avg_degree: float = 2.5, seed: int = 42) -> Dict[int, List[int]]:
    """Random graph with specified average degree."""
    np.random.seed(seed)
    conn = {i: [] for i in range(n)}
    
    # Erdos-Renyi-like
    p = avg_degree / (n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                conn[i].append(j)
                conn[j].append(i)
    
    # Ensure connected
    for i in range(n - 1):
        if i + 1 not in conn[i]:
            conn[i].append(i + 1)
            conn[i + 1].append(i)
    
    return conn

def lattice_full(n: int) -> Dict[int, List[int]]:
    """Fully connected (for comparison - should break causality)."""
    return {i: [j for j in range(n) if j != i] for i in range(n)}


LATTICES = {
    'chain': lattice_chain,
    'ring': lattice_ring,
    'square': lattice_square,
    'honeycomb': lattice_honeycomb,
    'random': lattice_random,
    'full': lattice_full,
}


# =============================================================================
# HAMILTONIANS
# =============================================================================

def hamiltonian_hopping(sub: Substrate, conn: Dict[int, List[int]], 
                        coupling: float = 1.0) -> xp.ndarray:
    """H = -t Σ (a†_i a_j + h.c.)"""
    H = sparse.csr_matrix((sub.dim, sub.dim), dtype=np.complex128)
    for i, neighbors in conn.items():
        for j in neighbors:
            if j > i:
                H = H - coupling * (sub._adag_ops[i] @ sub._a_ops[j])
                H = H - coupling * (sub._adag_ops[j] @ sub._a_ops[i])
    return H.toarray()


def hamiltonian_heisenberg(sub: Substrate, conn: Dict[int, List[int]],
                           coupling: float = 1.0) -> xp.ndarray:
    """H = J Σ (S_i · S_j) = J Σ (S+_i S-_j + S-_i S+_j + 2 Sz_i Sz_j)
    For qubits: S+ = a†, S- = a, Sz = n - 1/2
    """
    H = sparse.csr_matrix((sub.dim, sub.dim), dtype=np.complex128)
    
    # Sz = n - 1/2 (but we'll use n since constant shift doesn't matter)
    for i, neighbors in conn.items():
        for j in neighbors:
            if j > i:
                # S+_i S-_j + S-_i S+_j
                H = H + coupling * (sub._adag_ops[i] @ sub._a_ops[j])
                H = H + coupling * (sub._a_ops[i] @ sub._adag_ops[j])
                # 2 Sz_i Sz_j (using n_i n_j)
                H = H + 2 * coupling * (sub._n_ops[i] @ sub._n_ops[j])
    
    return H.toarray()


def hamiltonian_xy(sub: Substrate, conn: Dict[int, List[int]],
                   coupling: float = 1.0) -> xp.ndarray:
    """H = J Σ (S+_i S-_j + S-_i S+_j) - XY model without Sz terms."""
    H = sparse.csr_matrix((sub.dim, sub.dim), dtype=np.complex128)
    for i, neighbors in conn.items():
        for j in neighbors:
            if j > i:
                H = H + coupling * (sub._adag_ops[i] @ sub._a_ops[j])
                H = H + coupling * (sub._a_ops[i] @ sub._adag_ops[j])
    return H.toarray()


def hamiltonian_random_local(sub: Substrate, conn: Dict[int, List[int]],
                             coupling: float = 1.0, seed: int = 42) -> xp.ndarray:
    """Random Hermitian local interactions (respects connectivity)."""
    np.random.seed(seed)
    H = sparse.csr_matrix((sub.dim, sub.dim), dtype=np.complex128)
    
    for i, neighbors in conn.items():
        for j in neighbors:
            if j > i:
                # Random coefficients
                c1 = np.random.normal(0, coupling)
                c2 = np.random.normal(0, coupling)
                c3 = np.random.normal(0, coupling)
                
                # Random combination of hopping-like terms
                H = H + c1 * (sub._adag_ops[i] @ sub._a_ops[j])
                H = H + c1 * (sub._a_ops[i] @ sub._adag_ops[j])  # Hermitian
                H = H + c2 * (sub._n_ops[i] @ sub._n_ops[j])
                H = H + c3 * (sub._adag_ops[i] @ sub._adag_ops[j] + 
                              sub._a_ops[i] @ sub._a_ops[j])
    
    return H.toarray()


HAMILTONIANS = {
    'hopping': hamiltonian_hopping,
    'heisenberg': hamiltonian_heisenberg,
    'xy': hamiltonian_xy,
    'random': hamiltonian_random_local,
}


# =============================================================================
# DETECTION MECHANISMS
# =============================================================================

def detection_cnot(sub: Substrate, sys_mode: int, env_mode: int,
                   strength: float = np.pi/2) -> xp.ndarray:
    """CNOT-like: controlled flip."""
    sigma_x = (sub._a_ops[env_mode] + sub._adag_ops[env_mode]).toarray()
    n_sys = sub._n_ops[sys_mode].toarray()
    H = n_sys @ sigma_x
    return expm(-1j * strength * H)


def detection_swap(sub: Substrate, sys_mode: int, env_mode: int,
                   strength: float = np.pi/4) -> xp.ndarray:
    """SWAP-like: exchange excitations."""
    H = (sub._adag_ops[sys_mode] @ sub._a_ops[env_mode] +
         sub._a_ops[sys_mode] @ sub._adag_ops[env_mode]).toarray()
    return expm(-1j * strength * H)


def detection_zz(sub: Substrate, sys_mode: int, env_mode: int,
                 strength: float = np.pi/2) -> xp.ndarray:
    """ZZ-like: phase coupling."""
    H = (sub._n_ops[sys_mode] @ sub._n_ops[env_mode]).toarray()
    return expm(-1j * strength * H)


def detection_random(sub: Substrate, sys_mode: int, env_mode: int,
                     strength: float = np.pi/2, seed: int = None) -> xp.ndarray:
    """Random unitary coupling between system and environment."""
    if seed is not None:
        np.random.seed(seed + sys_mode * 100 + env_mode)
    
    # Random Hermitian combination
    c1 = np.random.normal(0, 1)
    c2 = np.random.normal(0, 1)
    c3 = np.random.normal(0, 1)
    
    H = (c1 * (sub._adag_ops[sys_mode] @ sub._a_ops[env_mode] +
               sub._a_ops[sys_mode] @ sub._adag_ops[env_mode]) +
         c2 * sub._n_ops[sys_mode] @ sub._n_ops[env_mode] +
         c3 * sub._n_ops[sys_mode] @ (sub._a_ops[env_mode] + sub._adag_ops[env_mode])).toarray()
    
    # Normalize
    H = H / (np.linalg.norm(H) + 1e-10)
    return expm(-1j * strength * H)


DETECTIONS = {
    'cnot': detection_cnot,
    'swap': detection_swap,
    'zz': detection_zz,
    'random': detection_random,
}


# =============================================================================
# PATTERN GENERATORS
# =============================================================================

def pattern_local(sub: Substrate, mode: int = 0) -> xp.ndarray:
    """Single excitation at specified mode."""
    config = tuple([1 if i == mode else 0 for i in range(sub.n_modes)])
    return sub.basis_state(config)


def pattern_symmetric(sub: Substrate, modes: List[int] = None) -> xp.ndarray:
    """Symmetric superposition over modes."""
    if modes is None:
        modes = [0, 1]
    psi = xp.zeros(sub.dim, dtype=xp.complex128)
    for m in modes:
        config = tuple([1 if i == m else 0 for i in range(sub.n_modes)])
        psi = psi + sub.basis_state(config)
    return psi / xp.linalg.norm(psi)


def pattern_antisymmetric(sub: Substrate, modes: List[int] = None) -> xp.ndarray:
    """Antisymmetric superposition (alternating signs)."""
    if modes is None:
        modes = [0, 1]
    psi = xp.zeros(sub.dim, dtype=xp.complex128)
    for i, m in enumerate(modes):
        config = tuple([1 if j == m else 0 for j in range(sub.n_modes)])
        sign = (-1) ** i
        psi = psi + sign * sub.basis_state(config)
    return psi / xp.linalg.norm(psi)


def pattern_random_superposition(sub: Substrate, n_terms: int = 3, seed: int = 42) -> xp.ndarray:
    """Random superposition of basis states."""
    np.random.seed(seed)
    psi = xp.zeros(sub.dim, dtype=xp.complex128)
    
    modes = np.random.choice(sub.n_modes, size=min(n_terms, sub.n_modes), replace=False)
    phases = np.random.uniform(0, 2*np.pi, len(modes))
    
    for m, phase in zip(modes, phases):
        config = tuple([1 if i == m else 0 for i in range(sub.n_modes)])
        psi = psi + np.exp(1j * phase) * sub.basis_state(config)
    
    return psi / xp.linalg.norm(psi)


def pattern_w_state(sub: Substrate) -> xp.ndarray:
    """W state: equal superposition of all single excitations."""
    psi = xp.zeros(sub.dim, dtype=xp.complex128)
    for m in range(sub.n_modes):
        config = tuple([1 if i == m else 0 for i in range(sub.n_modes)])
        psi = psi + sub.basis_state(config)
    return psi / xp.linalg.norm(psi)


PATTERNS = {
    'local': lambda sub: pattern_local(sub, 0),
    'symmetric': lambda sub: pattern_symmetric(sub, [0, 1]),
    'antisymmetric': lambda sub: pattern_antisymmetric(sub, [0, 1]),
    'random': lambda sub: pattern_random_superposition(sub, 3, 42),
    'w_state': pattern_w_state,
}


# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

def test_propagation(sub: Substrate, H: xp.ndarray, source: int,
                     t_max: float = 10.0, n_steps: int = 100) -> Dict:
    """Track excitation spread."""
    psi = sub.excite(sub.vacuum(), source)
    norm = xp.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    else:
        return {'arrival': {m: np.inf for m in range(sub.n_modes)}}
    
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
    
    return {'arrival': arrival, 'history': history, 'times': times}


def measure_light_cone_quality(sub: Substrate, H: xp.ndarray, 
                                conn: Dict[int, List[int]]) -> float:
    """Measure how well light cone structure emerges."""
    source = sub.n_modes // 2
    result = test_propagation(sub, H, source)
    
    # Compute graph distance
    def bfs_distance(start):
        dist = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in conn.get(node, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist
    
    graph_dist = bfs_distance(source)
    
    distances, times = [], []
    for m, t in result['arrival'].items():
        if m != source and t < np.inf and m in graph_dist:
            distances.append(graph_dist[m])
            times.append(t)
    
    if len(distances) < 2:
        return 0.0
    
    if np.std(distances) > 0 and np.std(times) > 0:
        return max(0, np.corrcoef(distances, times)[0, 1])
    return 0.0


def measure_metric_quality(sub: Substrate, H: xp.ndarray) -> float:
    """Measure triangle inequality satisfaction."""
    n = sub.n_modes
    metric = np.zeros((n, n))
    
    for source in range(n):
        result = test_propagation(sub, H, source)
        for target in range(n):
            metric[source, target] = result['arrival'][target]
    metric = (metric + metric.T) / 2
    
    violations, total = 0, 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len({i, j, k}) == 3:
                    if all(metric[a, b] < np.inf for a, b in [(i,k), (i,j), (j,k)]):
                        total += 1
                        if metric[i, k] > metric[i, j] + metric[j, k] + 1e-6:
                            violations += 1
    
    return 1.0 - violations / total if total > 0 else 0.0


def measure_statistics_gap(sub: Substrate, detection_fn: Callable,
                           system_modes: List[int], env_modes: List[int],
                           n_detections: int = 5) -> Tuple[float, Dict]:
    """Measure boson/fermion distinction via copyability."""
    results = {}
    
    for pattern_name in ['local', 'symmetric', 'antisymmetric']:
        psi = PATTERNS[pattern_name](sub)
        rho_orig = sub.reduced_density_matrix(psi, system_modes)
        
        psi_curr = psi.copy()
        fidelities = [1.0]
        
        for i in range(n_detections):
            env_m = env_modes[i % len(env_modes)]
            try:
                U = detection_fn(sub, system_modes[0], env_m)
                psi_curr = U @ psi_curr
                psi_curr = psi_curr / xp.linalg.norm(psi_curr)
                
                rho = sub.reduced_density_matrix(psi_curr, system_modes)
                fidelities.append(float(xp.abs(xp.trace(rho_orig @ rho))))
            except:
                fidelities.append(fidelities[-1])
        
        results[pattern_name] = fidelities[-1]
    
    bosonic = results.get('local', 0)
    fermionic = (results.get('symmetric', 0) + results.get('antisymmetric', 0)) / 2
    gap = bosonic - fermionic
    
    return max(0, min(1, gap)), results


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Single test configuration."""
    n_modes: int
    dim_per_mode: int
    lattice: str
    hamiltonian: str
    detection: str
    
    def __str__(self):
        return f"d={self.dim_per_mode}, {self.lattice}, {self.hamiltonian}, {self.detection}"


@dataclass
class TestResult:
    """Results from a single test."""
    config: TestConfig
    light_cone_quality: float
    metric_quality: float
    statistics_gap: float
    pattern_fidelities: Dict[str, float]
    runtime: float
    
    def overall(self) -> float:
        return (self.light_cone_quality + self.metric_quality + self.statistics_gap) / 3
    
    def is_universal(self, threshold: float = 0.4) -> bool:
        """Check if emergent structure appears."""
        return (self.light_cone_quality > threshold and 
                self.metric_quality > threshold and
                self.statistics_gap > threshold)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_single_test(config: TestConfig, verbose: bool = False) -> TestResult:
    """Run a single configuration test."""
    t_start = time.time()
    
    # Build substrate
    sub = Substrate(config.n_modes, config.dim_per_mode)
    
    # Build lattice
    conn = LATTICES[config.lattice](config.n_modes)
    
    # Build Hamiltonian
    H = HAMILTONIANS[config.hamiltonian](sub, conn)
    
    # Detection function
    det_fn = DETECTIONS[config.detection]
    
    # Define system/environment
    system_modes = [0, 1]
    env_modes = list(range(2, config.n_modes))
    
    # Measure
    lc_quality = measure_light_cone_quality(sub, H, conn)
    metric_quality = measure_metric_quality(sub, H)
    stats_gap, pattern_results = measure_statistics_gap(sub, det_fn, system_modes, env_modes)
    
    runtime = time.time() - t_start
    
    result = TestResult(
        config=config,
        light_cone_quality=lc_quality,
        metric_quality=metric_quality,
        statistics_gap=stats_gap,
        pattern_fidelities=pattern_results,
        runtime=runtime
    )
    
    if verbose:
        print(f"  {config}: LC={lc_quality:.3f}, M={metric_quality:.3f}, "
              f"S={stats_gap:.3f}, t={runtime:.2f}s")
    
    return result


def run_universality_test(
    n_modes: int = 6,
    dimensions: List[int] = [2, 3],
    lattices: List[str] = ['chain', 'ring', 'square', 'random'],
    hamiltonians: List[str] = ['hopping', 'heisenberg', 'xy', 'random'],
    detections: List[str] = ['cnot', 'swap', 'zz', 'random'],
    output_dir: str = 'universality_results'
) -> List[TestResult]:
    """Run full universality test suite."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all configurations
    configs = []
    for d, lat, ham, det in itertools.product(dimensions, lattices, hamiltonians, detections):
        configs.append(TestConfig(n_modes, d, lat, ham, det))
    
    print(f"\nUniversality Test Suite")
    print("=" * 60)
    print(f"Configurations: {len(configs)}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Lattices: {lattices}")
    print(f"  Hamiltonians: {hamiltonians}")
    print(f"  Detections: {detections}")
    print(f"  Modes: {n_modes}")
    print()
    
    results = []
    t_start = time.time()
    
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {config}...", end=" ", flush=True)
        try:
            result = run_single_test(config)
            results.append(result)
            status = "✓" if result.is_universal() else "✗"
            print(f"{status} LC={result.light_cone_quality:.2f}, "
                  f"M={result.metric_quality:.2f}, S={result.statistics_gap:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    total_time = time.time() - t_start
    
    # Analyze results
    analyze_results(results, output_dir)
    
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}/")
    
    return results


def analyze_results(results: List[TestResult], output_dir: str):
    """Analyze and visualize universality results."""
    
    # Summary statistics
    n_total = len(results)
    n_universal = sum(1 for r in results if r.is_universal())
    
    print(f"\n{'='*60}")
    print("UNIVERSALITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total tests: {n_total}")
    print(f"Universal: {n_universal} ({100*n_universal/n_total:.1f}%)")
    
    # Group by dimension
    by_dim = {}
    for r in results:
        d = r.config.dim_per_mode
        if d not in by_dim:
            by_dim[d] = []
        by_dim[d].append(r)
    
    print(f"\nBy local dimension:")
    for d, rs in sorted(by_dim.items()):
        lc = np.mean([r.light_cone_quality for r in rs])
        m = np.mean([r.metric_quality for r in rs])
        s = np.mean([r.statistics_gap for r in rs])
        print(f"  d={d}: LC={lc:.3f} ± {np.std([r.light_cone_quality for r in rs]):.3f}, "
              f"M={m:.3f}, S={s:.3f}")
    
    # Group by lattice
    by_lattice = {}
    for r in results:
        lat = r.config.lattice
        if lat not in by_lattice:
            by_lattice[lat] = []
        by_lattice[lat].append(r)
    
    print(f"\nBy lattice:")
    for lat, rs in sorted(by_lattice.items()):
        lc = np.mean([r.light_cone_quality for r in rs])
        m = np.mean([r.metric_quality for r in rs])
        s = np.mean([r.statistics_gap for r in rs])
        u = sum(1 for r in rs if r.is_universal()) / len(rs) * 100
        print(f"  {lat:12s}: LC={lc:.3f}, M={m:.3f}, S={s:.3f}, Universal={u:.0f}%")
    
    # Group by Hamiltonian
    by_ham = {}
    for r in results:
        ham = r.config.hamiltonian
        if ham not in by_ham:
            by_ham[ham] = []
        by_ham[ham].append(r)
    
    print(f"\nBy Hamiltonian:")
    for ham, rs in sorted(by_ham.items()):
        lc = np.mean([r.light_cone_quality for r in rs])
        m = np.mean([r.metric_quality for r in rs])
        s = np.mean([r.statistics_gap for r in rs])
        u = sum(1 for r in rs if r.is_universal()) / len(rs) * 100
        print(f"  {ham:12s}: LC={lc:.3f}, M={m:.3f}, S={s:.3f}, Universal={u:.0f}%")
    
    # Group by detection
    by_det = {}
    for r in results:
        det = r.config.detection
        if det not in by_det:
            by_det[det] = []
        by_det[det].append(r)
    
    print(f"\nBy detection:")
    for det, rs in sorted(by_det.items()):
        lc = np.mean([r.light_cone_quality for r in rs])
        m = np.mean([r.metric_quality for r in rs])
        s = np.mean([r.statistics_gap for r in rs])
        u = sum(1 for r in rs if r.is_universal()) / len(rs) * 100
        print(f"  {det:12s}: LC={lc:.3f}, M={m:.3f}, S={s:.3f}, Universal={u:.0f}%")
    
    # Failures analysis
    failures = [r for r in results if not r.is_universal()]
    if failures:
        print(f"\nFailed configurations ({len(failures)}):")
        for r in failures[:10]:  # Show first 10
            print(f"  {r.config}: LC={r.light_cone_quality:.2f}, "
                  f"M={r.metric_quality:.2f}, S={r.statistics_gap:.2f}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    
    # Visualizations
    plot_universality_heatmap(results, output_dir)
    plot_metric_distributions(results, output_dir)
    plot_invariance_test(results, output_dir)
    
    # Save raw data
    save_results_json(results, output_dir)


def plot_universality_heatmap(results: List[TestResult], output_dir: str):
    """Heatmap of universality across configurations."""
    lattices = sorted(set(r.config.lattice for r in results))
    hamiltonians = sorted(set(r.config.hamiltonian for r in results))
    
    # Average over dimensions and detections
    heatmap = np.zeros((len(lattices), len(hamiltonians)))
    counts = np.zeros_like(heatmap)
    
    for r in results:
        i = lattices.index(r.config.lattice)
        j = hamiltonians.index(r.config.hamiltonian)
        heatmap[i, j] += r.overall()
        counts[i, j] += 1
    
    heatmap = heatmap / np.maximum(counts, 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(hamiltonians)))
    ax.set_xticklabels(hamiltonians, rotation=45, ha='right')
    ax.set_yticks(range(len(lattices)))
    ax.set_yticklabels(lattices)
    
    for i in range(len(lattices)):
        for j in range(len(hamiltonians)):
            ax.text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center',
                   color='white' if heatmap[i,j] < 0.5 else 'black')
    
    ax.set_xlabel('Hamiltonian')
    ax.set_ylabel('Lattice')
    ax.set_title('Overall Stability (averaged over d and detection)')
    plt.colorbar(im, ax=ax, label='Overall Score')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap.png', dpi=150)
    plt.close()


def plot_metric_distributions(results: List[TestResult], output_dir: str):
    """Distribution of each metric."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = [
        ('light_cone_quality', 'Light Cone Quality'),
        ('metric_quality', 'Metric Quality'),
        ('statistics_gap', 'Statistics Gap'),
    ]
    
    for ax, (attr, title) in zip(axes, metrics):
        values = [getattr(r, attr) for r in results]
        
        # Separate by lattice type (local vs full)
        local_vals = [getattr(r, attr) for r in results if r.config.lattice != 'full']
        full_vals = [getattr(r, attr) for r in results if r.config.lattice == 'full']
        
        ax.hist(local_vals, bins=20, alpha=0.7, label='Local lattices', color='steelblue')
        if full_vals:
            ax.hist(full_vals, bins=20, alpha=0.7, label='Full connectivity', color='coral')
        
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel(title)
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.suptitle('Metric Distributions Across All Configurations', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributions.png', dpi=150)
    plt.close()


def plot_invariance_test(results: List[TestResult], output_dir: str):
    """Test invariance: do metrics depend on microscopic details?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # By dimension
    ax = axes[0, 0]
    dims = sorted(set(r.config.dim_per_mode for r in results))
    for metric, label, color in [('light_cone_quality', 'LC', 'blue'),
                                  ('metric_quality', 'Metric', 'green'),
                                  ('statistics_gap', 'Stats', 'red')]:
        means = [np.mean([getattr(r, metric) for r in results if r.config.dim_per_mode == d]) 
                 for d in dims]
        stds = [np.std([getattr(r, metric) for r in results if r.config.dim_per_mode == d])
                for d in dims]
        ax.errorbar(dims, means, yerr=stds, marker='o', label=label, capsize=3)
    ax.set_xlabel('Local Dimension')
    ax.set_ylabel('Quality')
    ax.set_title('Invariance under Dimension')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # By Hamiltonian
    ax = axes[0, 1]
    hams = sorted(set(r.config.hamiltonian for r in results))
    x = np.arange(len(hams))
    width = 0.25
    for i, (metric, label, color) in enumerate([('light_cone_quality', 'LC', 'blue'),
                                                 ('metric_quality', 'Metric', 'green'),
                                                 ('statistics_gap', 'Stats', 'red')]):
        means = [np.mean([getattr(r, metric) for r in results 
                         if r.config.hamiltonian == h and r.config.lattice != 'full']) 
                 for h in hams]
        ax.bar(x + i*width, means, width, label=label, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(hams, rotation=45, ha='right')
    ax.set_ylabel('Quality')
    ax.set_title('Invariance under Hamiltonian')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # By detection
    ax = axes[1, 0]
    dets = sorted(set(r.config.detection for r in results))
    x = np.arange(len(dets))
    for i, (metric, label, color) in enumerate([('light_cone_quality', 'LC', 'blue'),
                                                 ('metric_quality', 'Metric', 'green'),
                                                 ('statistics_gap', 'Stats', 'red')]):
        means = [np.mean([getattr(r, metric) for r in results 
                         if r.config.detection == d and r.config.lattice != 'full'])
                 for d in dets]
        ax.bar(x + i*width, means, width, label=label, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(dets, rotation=45, ha='right')
    ax.set_ylabel('Quality')
    ax.set_title('Invariance under Detection')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # By lattice
    ax = axes[1, 1]
    lats = sorted(set(r.config.lattice for r in results))
    x = np.arange(len(lats))
    for i, (metric, label, color) in enumerate([('light_cone_quality', 'LC', 'blue'),
                                                 ('metric_quality', 'Metric', 'green'),
                                                 ('statistics_gap', 'Stats', 'red')]):
        means = [np.mean([getattr(r, metric) for r in results if r.config.lattice == l])
                 for l in lats]
        ax.bar(x + i*width, means, width, label=label, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(lats, rotation=45, ha='right')
    ax.set_ylabel('Quality')
    ax.set_title('Dependence on Lattice (expect FULL to fail)')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('UNIVERSALITY TEST: Metrics should be INVARIANT\n'
                 '(except for full connectivity which breaks locality)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/invariance.png', dpi=150)
    plt.close()


def save_results_json(results: List[TestResult], output_dir: str):
    """Save results to JSON."""
    data = []
    for r in results:
        data.append({
            'config': {
                'n_modes': int(r.config.n_modes),
                'dim_per_mode': int(r.config.dim_per_mode),
                'lattice': r.config.lattice,
                'hamiltonian': r.config.hamiltonian,
                'detection': r.config.detection,
            },
            'light_cone_quality': float(r.light_cone_quality),
            'metric_quality': float(r.metric_quality),
            'statistics_gap': float(r.statistics_gap),
            'pattern_fidelities': {k: float(v) for k, v in r.pattern_fidelities.items()},
            'overall': float(r.overall()),
            'is_universal': bool(r.is_universal()),
            'runtime': float(r.runtime),
        })
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    results = run_universality_test(
        n_modes=6,
        dimensions=[2, 3],
        lattices=['chain', 'ring', 'square', 'honeycomb', 'random', 'full'],
        hamiltonians=['hopping', 'heisenberg', 'xy', 'random'],
        detections=['cnot', 'swap', 'zz', 'random'],
        output_dir='universality_results'
    )
    
    # Final verdict
    local_results = [r for r in results if r.config.lattice != 'full']
    full_results = [r for r in results if r.config.lattice == 'full']
    
    local_universal = sum(1 for r in local_results if r.is_universal()) / len(local_results)
    full_universal = sum(1 for r in full_results if r.is_universal()) / len(full_results) if full_results else 0
    
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    print(f"LOCAL lattices: {100*local_universal:.1f}% universal")
    print(f"FULL connectivity: {100*full_universal:.1f}% universal")
    
    if local_universal > 0.8 and full_universal < 0.3:
        print("\n✓ UNIVERSALITY CONFIRMED")
        print("  Emergent structure appears across all local configurations")
        print("  and breaks down (as predicted) without locality.")
    elif local_universal > 0.5:
        print("\n~ PARTIAL UNIVERSALITY")
        print("  Emergent structure appears in most configurations")
        print("  Some microscopic details may matter.")
    else:
        print("\n✗ UNIVERSALITY NOT CONFIRMED")
        print("  Emergent structure depends on implementation details.")


if __name__ == "__main__":
    main()