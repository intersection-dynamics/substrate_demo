"""
Noise Robustness Analysis (Vectorized)
======================================
1. What TYPE of noise breaks it?
2. Does threshold depend on SIZE?
3. Are there noise-RESISTANT structures?
"""

import os
import time
import numpy as np

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

from dataclasses import dataclass
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from substrate import (
    Substrate, hopping_hamiltonian, linear_chain, full_connectivity,
    test_propagation, test_copyability, create_patterns
)


@dataclass
class StabilityMetrics:
    light_cone_quality: float
    metric_quality: float
    statistics_gap: float
    
    def overall(self) -> float:
        return (self.light_cone_quality + self.metric_quality + self.statistics_gap) / 3


# =============================================================================
# NOISE FUNCTIONS
# =============================================================================

def add_magnitude_noise(H, level: float):
    """Random magnitude perturbations."""
    noise = xp.asarray(np.random.normal(0, level, H.shape))
    noise = (noise + noise.T) / 2
    mask = xp.abs(H) > 1e-10
    return H + noise * mask


def add_phase_noise(H, level: float):
    """Random phase rotations."""
    phases = xp.asarray(np.random.uniform(-level * np.pi, level * np.pi, H.shape))
    phases = (phases - phases.T) / 2
    phase_factors = xp.exp(1j * phases)
    result = H * phase_factors
    diag_real = xp.real(xp.diag(H))
    result = result - xp.diag(xp.diag(result)) + xp.diag(diag_real)
    return result


def add_sparse_noise(H, level: float, sparsity: float = 0.1):
    """Noise on fraction of elements."""
    mask = xp.asarray(np.random.random(H.shape) < sparsity)
    noise = xp.asarray(np.random.normal(0, level, H.shape))
    noise = (noise + noise.T) / 2
    if GPU_AVAILABLE:
        cp.fill_diagonal(noise, 0)
    else:
        np.fill_diagonal(noise, 0)
    return H + noise * mask


def add_structured_noise(H, level: float):
    """Noise only where connections exist."""
    mask = xp.abs(H) > 1e-10
    noise = xp.asarray(np.random.normal(0, level, H.shape))
    noise = (noise + noise.T) / 2
    return H + noise * mask


def add_diagonal_noise(H, level: float):
    """On-site disorder only."""
    n = H.shape[0]
    disorder = xp.diag(xp.asarray(np.random.uniform(-level, level, n)))
    return H + disorder


# =============================================================================
# EVALUATION
# =============================================================================

def measure_light_cone_quality(arrival_times: Dict[int, float], source: int) -> float:
    distances, times = [], []
    for m, t in arrival_times.items():
        if t < np.inf and m != source:
            distances.append(abs(m - source))
            times.append(t)
    if len(distances) < 2:
        return 0.0
    if np.std(distances) > 0 and np.std(times) > 0:
        return max(0, np.corrcoef(distances, times)[0, 1])
    return 0.0


def measure_metric_quality(sub: Substrate, H, t_max: float = 10.0) -> float:
    metric = np.zeros((sub.n_modes, sub.n_modes))
    for source in range(sub.n_modes):
        result = test_propagation(sub, H, source, t_max)
        for target in range(sub.n_modes):
            metric[source, target] = result['arrival'][target]
    metric = (metric + metric.T) / 2
    
    n = metric.shape[0]
    violations, total = 0, 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    if all(metric[a, b] < np.inf for a, b in [(i,k), (i,j), (j,k)]):
                        total += 1
                        if metric[i, k] > metric[i, j] + metric[j, k] + 1e-6:
                            violations += 1
    return 1.0 - violations / total if total > 0 else 0.0


def measure_statistics_gap(sub: Substrate, system_modes: List[int], env_modes: List[int]) -> float:
    patterns = create_patterns(sub)
    copy_results = {}
    for name, psi in patterns.items():
        copy_results[name] = test_copyability(sub, psi, system_modes, env_modes, n_detections=5)
    
    bosonic = copy_results.get('local', {}).get('fidelity', [0])[-1]
    fermionic = [copy_results[n]['fidelity'][-1] for n in ['symmetric', 'antisymmetric'] if n in copy_results]
    return max(0, min(1, bosonic - np.mean(fermionic))) if fermionic else 0.0


def evaluate(sub: Substrate, H, system_modes: List[int], env_modes: List[int]) -> StabilityMetrics:
    source = sub.n_modes // 2
    prop = test_propagation(sub, H, source, t_max=10.0)
    return StabilityMetrics(
        light_cone_quality=measure_light_cone_quality(prop['arrival'], source),
        metric_quality=measure_metric_quality(sub, H),
        statistics_gap=measure_statistics_gap(sub, system_modes, env_modes)
    )


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_noise_types(n_modes: int = 8, n_levels: int = 12, n_trials: int = 3):
    """Compare noise types."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Noise Type Comparison")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    H_base = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    noise_levels = np.linspace(0, 0.5, n_levels)
    noise_types = {
        'magnitude': add_magnitude_noise,
        'phase': add_phase_noise,
        'sparse': lambda H, l: add_sparse_noise(H, l, 0.1),
        'structured': add_structured_noise,
        'diagonal': add_diagonal_noise,
    }
    
    results = {name: {'levels': [], 'stability': [], 'std': []} for name in noise_types}
    
    for level in noise_levels:
        print(f"  Level {level:.3f}...", end=" ", flush=True)
        for name, noise_fn in noise_types.items():
            stabilities = []
            for _ in range(n_trials):
                H = noise_fn(H_base.copy(), level)
                stabilities.append(evaluate(sub, H, system_modes, env_modes).overall())
            results[name]['levels'].append(level)
            results[name]['stability'].append(np.mean(stabilities))
            results[name]['std'].append(np.std(stabilities))
        print("done")
    
    return results


def experiment_size_scaling(sizes: List[int] = None, n_levels: int = 10, n_trials: int = 3):
    """Size dependence of noise threshold."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Size Scaling")
    print("="*60)
    
    if sizes is None:
        sizes = [4, 5, 6, 7, 8]
    
    noise_levels = np.linspace(0, 0.3, n_levels)
    results = {}
    
    for n_modes in sizes:
        print(f"\n  Size {n_modes} modes (dim={2**n_modes})...")
        sub = Substrate(n_modes, dim_per_mode=2)
        system_modes = [0, 1]
        env_modes = list(range(2, n_modes))
        H_base = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
        
        results[n_modes] = {'levels': [], 'stability': [], 'threshold': None}
        
        for level in noise_levels:
            stabilities = [evaluate(sub, add_magnitude_noise(H_base.copy(), level), 
                                    system_modes, env_modes).overall() for _ in range(n_trials)]
            results[n_modes]['levels'].append(level)
            results[n_modes]['stability'].append(np.mean(stabilities))
        
        for l, s in zip(results[n_modes]['levels'], results[n_modes]['stability']):
            if s < 0.5:
                results[n_modes]['threshold'] = l
                break
    
    return results


def experiment_structures(n_modes: int = 8, n_levels: int = 12, n_trials: int = 3):
    """Compare Hamiltonian structures."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Noise-Resistant Structures")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    
    def redundant_chain(n):
        return {i: [j for j in range(n) if 0 < abs(i-j) <= 2] for i in range(n)}
    
    def ring(n):
        return {i: [(i-1) % n, (i+1) % n] for i in range(n)}
    
    def ladder(n):
        n_rungs = n // 2
        conn = {}
        for i in range(n):
            rung, leg = i // 2, i % 2
            neighbors = [rung * 2 + (1 - leg)]  # Rung partner
            if rung > 0:
                neighbors.append((rung - 1) * 2 + leg)
            if rung < n_rungs - 1:
                neighbors.append((rung + 1) * 2 + leg)
            conn[i] = neighbors
        return conn
    
    structures = {
        'chain': linear_chain(n_modes),
        'ring': ring(n_modes),
        'redundant': redundant_chain(n_modes),
        'ladder': ladder(n_modes),
    }
    
    noise_levels = np.linspace(0, 0.4, n_levels)
    results = {name: {'levels': [], 'stability': [], 'std': []} for name in structures}
    
    for level in noise_levels:
        print(f"  Level {level:.3f}...", end=" ", flush=True)
        for name, connectivity in structures.items():
            H_base = hopping_hamiltonian(sub, connectivity, coupling=1.0)
            stabilities = [evaluate(sub, add_magnitude_noise(H_base.copy(), level),
                                    system_modes, env_modes).overall() for _ in range(n_trials)]
            results[name]['levels'].append(level)
            results[name]['stability'].append(np.mean(stabilities))
            results[name]['std'].append(np.std(stabilities))
        print("done")
    
    return results


def experiment_metric_breakdown(n_modes: int = 8, n_levels: int = 12, n_trials: int = 3):
    """Which metric fails first?"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Metric Breakdown")
    print("="*60)
    
    sub = Substrate(n_modes, dim_per_mode=2)
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    H_base = hopping_hamiltonian(sub, linear_chain(n_modes), coupling=1.0)
    
    noise_levels = np.linspace(0, 0.3, n_levels)
    results = {'levels': [], 
               'light_cone': [], 'light_cone_std': [],
               'metric': [], 'metric_std': [],
               'statistics': [], 'statistics_std': []}
    
    for level in noise_levels:
        print(f"  Level {level:.3f}...", end=" ", flush=True)
        lc, m, s = [], [], []
        for _ in range(n_trials):
            H = add_magnitude_noise(H_base.copy(), level)
            metrics = evaluate(sub, H, system_modes, env_modes)
            lc.append(metrics.light_cone_quality)
            m.append(metrics.metric_quality)
            s.append(metrics.statistics_gap)
        
        results['levels'].append(level)
        results['light_cone'].append(np.mean(lc))
        results['light_cone_std'].append(np.std(lc))
        results['metric'].append(np.mean(m))
        results['metric_std'].append(np.std(m))
        results['statistics'].append(np.mean(s))
        results['statistics_std'].append(np.std(s))
        print("done")
    
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_noise_types(results: Dict, filename: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'magnitude': 'red', 'phase': 'blue', 'sparse': 'green', 
              'structured': 'orange', 'diagonal': 'purple'}
    for name, data in results.items():
        ax.plot(data['levels'], data['stability'], 'o-', label=name.capitalize(), 
                color=colors[name], linewidth=2)
        ax.fill_between(data['levels'], 
                        np.array(data['stability']) - np.array(data['std']),
                        np.array(data['stability']) + np.array(data['std']),
                        alpha=0.2, color=colors[name])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Overall Stability')
    ax.set_title('Stability vs Noise Type')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_size_scaling(results: Dict, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cmap = plt.cm.viridis
    sizes = sorted(results.keys())
    for i, n in enumerate(sizes):
        axes[0].plot(results[n]['levels'], results[n]['stability'], 'o-',
                     label=f'N={n}', color=cmap(i / len(sizes)), linewidth=2)
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Stability')
    axes[0].set_title('Stability vs Size')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    
    valid = [(n, results[n]['threshold']) for n in sizes if results[n]['threshold']]
    if valid:
        ns, ts = zip(*valid)
        axes[1].plot(ns, ts, 'o-', linewidth=2, markersize=10, color='steelblue')
        if len(ns) >= 3:
            slope, intercept = np.polyfit(np.log(ns), np.log(ts), 1)
            axes[1].plot(ns, np.exp(intercept) * np.array(ns)**slope, '--', 
                         color='red', label=f't ~ N^{slope:.2f}')
            axes[1].legend()
    axes[1].set_xlabel('System Size (N)')
    axes[1].set_ylabel('Noise Threshold')
    axes[1].set_title('Critical Noise vs Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_structures(results: Dict, filename: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'chain': 'blue', 'ring': 'green', 'redundant': 'red', 'ladder': 'orange'}
    for name, data in results.items():
        ax.plot(data['levels'], data['stability'], 'o-', label=name.capitalize(),
                color=colors.get(name, 'gray'), linewidth=2)
        ax.fill_between(data['levels'],
                        np.array(data['stability']) - np.array(data['std']),
                        np.array(data['stability']) + np.array(data['std']),
                        alpha=0.2, color=colors.get(name, 'gray'))
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Stability')
    ax.set_title('Stability vs Structure')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_breakdown(results: Dict, filename: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color in [('light_cone', 'Light Cone', 'blue'),
                               ('metric', 'Metric', 'green'),
                               ('statistics', 'Statistics Gap', 'red')]:
        vals = np.array(results[key])
        std = np.array(results[f'{key}_std'])
        ax.plot(results['levels'], vals, 'o-', label=label, color=color, linewidth=2)
        ax.fill_between(results['levels'], vals - std, vals + std, alpha=0.2, color=color)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Quality')
    ax.set_title('Individual Metrics vs Noise')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = 'noise_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nNoise Robustness Analysis")
    print("="*60)
    
    t_start = time.time()
    
    noise_results = experiment_noise_types(n_modes=8, n_levels=10, n_trials=3)
    plot_noise_types(noise_results, f'{output_dir}/noise_types.png')
    
    size_results = experiment_size_scaling(sizes=[4, 5, 6, 7, 8], n_levels=8, n_trials=3)
    plot_size_scaling(size_results, f'{output_dir}/size_scaling.png')
    
    struct_results = experiment_structures(n_modes=8, n_levels=10, n_trials=3)
    plot_structures(struct_results, f'{output_dir}/structures.png')
    
    breakdown_results = experiment_metric_breakdown(n_modes=8, n_levels=10, n_trials=3)
    plot_breakdown(breakdown_results, f'{output_dir}/metric_breakdown.png')
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    
    print("\n1. NOISE TYPE THRESHOLDS:")
    for name, data in noise_results.items():
        thresh = next((l for l, s in zip(data['levels'], data['stability']) if s < 0.5), None)
        print(f"   {name:12s}: {thresh:.3f}" if thresh else f"   {name:12s}: > max")
    
    print("\n2. SIZE SCALING:")
    for n, data in sorted(size_results.items()):
        t = data['threshold']
        print(f"   N={n}: {t:.3f}" if t else f"   N={n}: > max")
    
    print("\n3. STRUCTURE @ noise=0.2:")
    for name, data in struct_results.items():
        s = next((s for l, s in zip(data['levels'], data['stability']) if l >= 0.19), None)
        if s:
            print(f"   {name:12s}: {s:.3f}")
    
    print("\n4. METRIC FAILURE ORDER:")
    for key in ['light_cone', 'metric', 'statistics']:
        thresh = next((l for l, s in zip(breakdown_results['levels'], breakdown_results[key]) if s < 0.5), None)
        print(f"   {key:12s}: {thresh:.3f}" if thresh else f"   {key:12s}: > max")
    
    print(f"\nTime: {time.time() - t_start:.1f}s")
    print(f"Outputs: {output_dir}/")

    # ──────────────────────────────────────────────────────────────
    # QUICK λ* THRESHOLD EXTRACTION (added Dec 2025)
    # ──────────────────────────────────────────────────────────────
    print("\nλ* THRESHOLD CHECK (phase-sensitive vs amplitude noise)")
    levels = breakdown_results['levels']
    stat_gap = np.array(breakdown_results['statistics'])

    idx = np.where(stat_gap < 0.5)[0]
    if len(idx) > 0:
        critical_noise = levels[idx[0]]
        print(f"Statistics Gap falls below 0.5 at noise level ≈ {critical_noise:.4f}")
        print(f"→ Measured critical λ ≈ {critical_noise:.4f} (in equal-amplitude limit)")
        print(f"→ Analytic two-qubit bound gives λ* = ¼ + √2/4 ≈ 0.2707")
    else:
        print("Statistics Gap never dropped below 0.5 in this sweep!")
    # ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()