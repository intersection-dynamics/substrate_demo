"""
Stability Basin Analysis (Vectorized)
=====================================
Sweep parameters to find where emergent phenomena break down.
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
    Substrate, hopping_hamiltonian, detection_unitary,
    linear_chain, full_connectivity,
    test_propagation, test_copyability, create_patterns
)


@dataclass
class StabilityMetrics:
    light_cone_quality: float
    metric_quality: float
    statistics_gap: float
    propagation_speed: float
    
    def overall(self) -> float:
        return (self.light_cone_quality + self.metric_quality + self.statistics_gap) / 3


def measure_light_cone_quality(arrival_times: Dict[int, float], source: int) -> float:
    distances, times = [], []
    for mode, t in arrival_times.items():
        if t < np.inf and mode != source:
            distances.append(abs(mode - source))
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
    if not fermionic:
        return 0.0
    return max(0, min(1, bosonic - np.mean(fermionic)))


def measure_propagation_speed(arrival_times: Dict[int, float], source: int) -> float:
    speeds = [abs(m - source) / t for m, t in arrival_times.items() if 0 < t < np.inf and m != source]
    return np.mean(speeds) if speeds else 0.0


def evaluate(sub: Substrate, H, system_modes: List[int], env_modes: List[int]) -> StabilityMetrics:
    source = sub.n_modes // 2
    prop = test_propagation(sub, H, source, t_max=10.0)
    return StabilityMetrics(
        light_cone_quality=measure_light_cone_quality(prop['arrival'], source),
        metric_quality=measure_metric_quality(sub, H),
        statistics_gap=measure_statistics_gap(sub, system_modes, env_modes),
        propagation_speed=measure_propagation_speed(prop['arrival'], source)
    )


# Perturbations (work on dense arrays after Hamiltonian is built)
def add_disorder(H, strength: float):
    n = H.shape[0]
    disorder = xp.diag(xp.asarray(np.random.uniform(-strength, strength, n)))
    return H + disorder


def add_noise(H, noise_level: float):
    noise = xp.asarray(np.random.normal(0, noise_level, H.shape))
    noise = (noise + noise.T) / 2
    if GPU_AVAILABLE:
        cp.fill_diagonal(noise, 0)
    else:
        np.fill_diagonal(noise, 0)
    return H + noise


def perturb_connectivity(n_modes: int, locality: float) -> Dict[int, List[int]]:
    local = linear_chain(n_modes)
    full = full_connectivity(n_modes)
    result = {i: [] for i in range(n_modes)}
    for i in range(n_modes):
        for j in full[i]:
            if abs(i - j) == 1 or np.random.random() > locality:
                if j not in result[i]:
                    result[i].append(j)
    return result


# Sweeps
def sweep_coupling(sub: Substrate, base_coupling: float, perturbations: np.ndarray,
                   system_modes: List[int], env_modes: List[int]) -> Dict:
    results = {'perturbation': [], 'metrics': []}
    for p in perturbations:
        H = hopping_hamiltonian(sub, linear_chain(sub.n_modes), base_coupling * (1 + p))
        results['perturbation'].append(p)
        results['metrics'].append(evaluate(sub, H, system_modes, env_modes))
    return results


def sweep_disorder(sub: Substrate, coupling: float, disorder_strengths: np.ndarray,
                   system_modes: List[int], env_modes: List[int]) -> Dict:
    results = {'disorder': [], 'metrics': []}
    H_base = hopping_hamiltonian(sub, linear_chain(sub.n_modes), coupling)
    for d in disorder_strengths:
        H = add_disorder(H_base.copy(), d)
        results['disorder'].append(d)
        results['metrics'].append(evaluate(sub, H, system_modes, env_modes))
    return results


def sweep_locality(sub: Substrate, coupling: float, locality_values: np.ndarray,
                   system_modes: List[int], env_modes: List[int]) -> Dict:
    results = {'locality': [], 'metrics': []}
    for loc in locality_values:
        H = hopping_hamiltonian(sub, perturb_connectivity(sub.n_modes, loc), coupling)
        results['locality'].append(loc)
        results['metrics'].append(evaluate(sub, H, system_modes, env_modes))
    return results


def sweep_noise(sub: Substrate, coupling: float, noise_levels: np.ndarray,
                system_modes: List[int], env_modes: List[int]) -> Dict:
    results = {'noise': [], 'metrics': []}
    H_base = hopping_hamiltonian(sub, linear_chain(sub.n_modes), coupling)
    for n in noise_levels:
        H = add_noise(H_base.copy(), n)
        results['noise'].append(n)
        results['metrics'].append(evaluate(sub, H, system_modes, env_modes))
    return results


# Plotting
def plot_sweep(results: Dict, param_name: str, filename: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = results[param_name]
    for ax, (metric, title) in zip(axes.flatten(), 
            [('light_cone_quality', 'Light Cone'), ('metric_quality', 'Metric'),
             ('statistics_gap', 'Statistics Gap'), ('overall', 'Overall')]):
        y = [m.overall() if metric == 'overall' else getattr(m, metric) for m in results['metrics']]
        ax.plot(x, y, 'o-', linewidth=2, markersize=6)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Quality')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_combined(all_results: Dict, filename: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    configs = [('coupling', 'perturbation'), ('disorder', 'disorder'), 
               ('locality', 'locality'), ('noise', 'noise')]
    for ax, (name, param) in zip(axes.flatten(), configs):
        if name in all_results:
            x = all_results[name][param]
            y = [m.overall() for m in all_results[name]['metrics']]
            ax.plot(x, y, 'o-', linewidth=2, color='steelblue')
            ax.fill_between(x, 0, y, alpha=0.3)
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel(name.capitalize())
            ax.set_ylabel('Overall Stability')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_phase_diagram(sub: Substrate, coupling_range: np.ndarray, disorder_range: np.ndarray,
                       system_modes: List[int], env_modes: List[int], filename: str):
    stability = np.zeros((len(disorder_range), len(coupling_range)))
    total = len(disorder_range) * len(coupling_range)
    for i, d in enumerate(disorder_range):
        for j, c in enumerate(coupling_range):
            H = hopping_hamiltonian(sub, linear_chain(sub.n_modes), c)
            H = add_disorder(H, d)
            stability[i, j] = evaluate(sub, H, system_modes, env_modes).overall()
        print(f"  Phase diagram: {(i+1)*len(coupling_range)}/{total}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(stability, origin='lower', aspect='auto',
                   extent=[coupling_range[0], coupling_range[-1],
                           disorder_range[0], disorder_range[-1]],
                   cmap='RdYlGn', vmin=0, vmax=1)
    ax.contour(coupling_range, disorder_range, stability, levels=[0.5], colors='black', linewidths=2)
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Disorder')
    ax.set_title('Stability Phase Diagram')
    plt.colorbar(im, ax=ax, label='Overall Stability')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    output_dir = 'stability_results'
    os.makedirs(output_dir, exist_ok=True)
    
    n_modes = 8
    sub = Substrate(n_modes, dim_per_mode=2)
    system_modes = [0, 1]
    env_modes = list(range(2, n_modes))
    
    print(f"\nStability Basin Analysis")
    print("="*50)
    print(f"Substrate: {n_modes} modes, dim={sub.dim}")
    
    all_results = {}
    t_start = time.time()
    
    print("\nRunning coupling sweep...")
    all_results['coupling'] = sweep_coupling(sub, 1.0, np.linspace(-0.9, 2.0, 12), system_modes, env_modes)
    plot_sweep(all_results['coupling'], 'perturbation', f'{output_dir}/sweep_coupling.png')
    
    print("Running disorder sweep...")
    all_results['disorder'] = sweep_disorder(sub, 1.0, np.linspace(0, 3.0, 12), system_modes, env_modes)
    plot_sweep(all_results['disorder'], 'disorder', f'{output_dir}/sweep_disorder.png')
    
    print("Running locality sweep...")
    all_results['locality'] = sweep_locality(sub, 1.0, np.linspace(0, 1.0, 10), system_modes, env_modes)
    plot_sweep(all_results['locality'], 'locality', f'{output_dir}/sweep_locality.png')
    
    print("Running noise sweep...")
    all_results['noise'] = sweep_noise(sub, 1.0, np.linspace(0, 1.0, 12), system_modes, env_modes)
    plot_sweep(all_results['noise'], 'noise', f'{output_dir}/sweep_noise.png')
    
    print("\nGenerating combined plot...")
    plot_combined(all_results, f'{output_dir}/combined_sweeps.png')
    
    print("Computing phase diagram...")
    plot_phase_diagram(sub, np.linspace(0.2, 2.0, 8), np.linspace(0, 2.0, 8),
                       system_modes, env_modes, f'{output_dir}/phase_diagram.png')
    
    print(f"\nTime: {time.time() - t_start:.1f}s")
    print(f"\n{'='*50}\nRESULTS\n{'='*50}")
    
    for name, results in all_results.items():
        param = [k for k in results.keys() if k != 'metrics'][0]
        overall = [m.overall() for m in results['metrics']]
        stable = [p for p, o in zip(results[param], overall) if o > 0.5]
        if stable:
            print(f"\n{name.upper()}: Stable [{min(stable):.2f}, {max(stable):.2f}], baseline={overall[0]:.3f}")
        else:
            print(f"\n{name.upper()}: No stable region")
    
    print(f"\nOutputs: {output_dir}/")


if __name__ == "__main__":
    main()