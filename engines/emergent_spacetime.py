"""
Emergent Spacetime from No-Signaling
====================================
Axioms: Hilbert Space Realism, Unitary Evolution, Classical Emergence
Constraint: No-signaling

Distance emerges from information propagation time.
Spacetime is the structure enforcing no-signaling.
"""

import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class Substrate:
    """Quantum modes with no assumed spatial structure."""
    
    def __init__(self, n_modes: int, dim: int = 2):
        self.n_modes = n_modes
        self.d = dim
        self.dim_total = dim ** n_modes
        
    def vacuum(self) -> np.ndarray:
        psi = np.zeros(self.dim_total, dtype=complex)
        psi[0] = 1.0
        return psi
    
    def index_to_config(self, idx: int) -> Tuple[int, ...]:
        config = []
        for i in range(self.n_modes):
            config.append(idx // (self.d ** (self.n_modes - 1 - i)) % self.d)
        return tuple(config)
    
    def config_to_index(self, config: Tuple[int, ...]) -> int:
        idx = 0
        for i, n in enumerate(config):
            idx += n * (self.d ** (self.n_modes - 1 - i))
        return idx
    
    def excite(self, psi: np.ndarray, mode: int) -> np.ndarray:
        result = np.zeros_like(psi)
        for idx in range(self.dim_total):
            config = self.index_to_config(idx)
            if config[mode] < self.d - 1:
                new_config = list(config)
                new_config[mode] += 1
                new_idx = self.config_to_index(tuple(new_config))
                result[new_idx] += psi[idx] * np.sqrt(config[mode] + 1)
        return result
    
    def measure(self, psi: np.ndarray, mode: int) -> float:
        exp_val = 0.0
        for idx in range(self.dim_total):
            config = self.index_to_config(idx)
            exp_val += np.abs(psi[idx])**2 * config[mode]
        return exp_val


def build_hamiltonian(substrate: Substrate, connectivity: Dict[int, List[int]], 
                      coupling: float = 1.0) -> np.ndarray:
    """H = coupling * Σ_{connected (i,j)} (a†_i a_j + h.c.)"""
    dim = substrate.dim_total
    H = np.zeros((dim, dim), dtype=complex)
    
    for i, neighbors in connectivity.items():
        for j in neighbors:
            if j > i:
                for idx in range(dim):
                    config = list(substrate.index_to_config(idx))
                    
                    if config[j] > 0 and config[i] < substrate.d - 1:
                        coeff = np.sqrt(config[j]) * np.sqrt(config[i] + 1)
                        new_config = config.copy()
                        new_config[j] -= 1
                        new_config[i] += 1
                        H[substrate.config_to_index(tuple(new_config)), idx] += coupling * coeff
                    
                    if config[i] > 0 and config[j] < substrate.d - 1:
                        coeff = np.sqrt(config[i]) * np.sqrt(config[j] + 1)
                        new_config = config.copy()
                        new_config[i] -= 1
                        new_config[j] += 1
                        H[substrate.config_to_index(tuple(new_config)), idx] += coupling * coeff
    
    return H


def linear_chain(n: int) -> Dict[int, List[int]]:
    return {i: [j for j in [i-1, i+1] if 0 <= j < n] for i in range(n)}


def fully_connected(n: int) -> Dict[int, List[int]]:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def propagate(substrate: Substrate, H: np.ndarray, source: int,
              t_max: float = 10.0, n_steps: int = 100) -> Dict:
    """Track information spread from source mode."""
    psi = substrate.excite(substrate.vacuum(), source)
    psi = psi / np.linalg.norm(psi)
    
    times = np.linspace(0, t_max, n_steps)
    dt = times[1] - times[0]
    U = expm(-1j * H * dt)
    
    history = np.zeros((substrate.n_modes, n_steps))
    
    for t_idx in range(n_steps):
        for mode in range(substrate.n_modes):
            history[mode, t_idx] = substrate.measure(psi, mode)
        psi = U @ psi
    
    threshold = 0.01
    arrival = {}
    for mode in range(substrate.n_modes):
        if mode == source:
            arrival[mode] = 0.0
        else:
            crossed = np.where(history[mode] > threshold)[0]
            arrival[mode] = times[crossed[0]] if len(crossed) > 0 else np.inf
    
    return {'times': times, 'history': history, 'arrival': arrival, 'source': source}


def compute_metric(substrate: Substrate, H: np.ndarray, t_max: float = 10.0) -> np.ndarray:
    """Compute emergent distance matrix from propagation."""
    n = substrate.n_modes
    metric = np.zeros((n, n))
    
    for source in range(n):
        result = propagate(substrate, H, source, t_max)
        for target in range(n):
            metric[source, target] = result['arrival'][target]
    
    return (metric + metric.T) / 2


def plot_light_cone(result: Dict, title: str, filename: str):
    """Visualize light cone structure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    history = result['history']
    times = result['times']
    n_modes = history.shape[0]
    source = result['source']
    
    im = ax.imshow(history, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, n_modes-0.5],
                   cmap='inferno', vmin=0, vmax=1)
    
    for mode, t_arr in result['arrival'].items():
        if 0 < t_arr < np.inf:
            ax.plot(t_arr, mode, 'w^', markersize=8)
    
    ax.axhline(source, color='cyan', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mode', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(range(n_modes))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Excitation', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_metric(metric: np.ndarray, title: str, filename: str):
    """Visualize emergent distance matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    n = metric.shape[0]
    metric_vis = np.where(metric == np.inf, np.nan, metric)
    
    im = ax.imshow(metric_vis, cmap='viridis', origin='lower')
    
    ax.set_xlabel('Mode j', fontsize=12)
    ax.set_ylabel('Mode i', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    
    for i in range(n):
        for j in range(n):
            val = metric[i, j]
            if val < np.inf:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color='white' if val > np.nanmax(metric_vis)/2 else 'black', fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (arrival time)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_comparison(result_local: Dict, result_full: Dict, filename: str):
    """Compare local vs full connectivity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, result, title in zip(axes, [result_local, result_full], 
                                  ['Local Connectivity', 'Full Connectivity']):
        history = result['history']
        times = result['times']
        n_modes = history.shape[0]
        
        im = ax.imshow(history, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], -0.5, n_modes-0.5],
                       cmap='inferno', vmin=0, vmax=1)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mode', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_yticks(range(n_modes))
        plt.colorbar(im, ax=ax, label='Excitation')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_entanglement(correlations: np.ndarray, filename: str):
    """Visualize entanglement/correlation structure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    n = correlations.shape[0]
    
    im = ax.imshow(correlations, cmap='RdBu', origin='lower', vmin=-0.5, vmax=0.5)
    
    ax.set_xlabel('Mode j', fontsize=12)
    ax.set_ylabel('Mode i', fontsize=12)
    ax.set_title('Correlation Structure', fontsize=14)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    
    for i in range(n):
        for j in range(n):
            val = correlations[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color='white' if abs(val) > 0.25 else 'black', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Emergent Spacetime")
    print("="*40)
    
    n_modes = 8
    substrate = Substrate(n_modes, dim=2)
    source = n_modes // 2
    
    print(f"Modes: {n_modes}, Source: {source}")
    
    # Local vs full connectivity
    H_local = build_hamiltonian(substrate, linear_chain(n_modes))
    H_full = build_hamiltonian(substrate, fully_connected(n_modes))
    
    result_local = propagate(substrate, H_local, source, t_max=8.0)
    result_full = propagate(substrate, H_full, source, t_max=8.0)
    
    # Metric
    metric = compute_metric(substrate, H_local, t_max=10.0)
    
    # Entanglement structure (4-mode demo)
    n_small = 4
    sub_small = Substrate(n_small, dim=2)
    psi = np.zeros(sub_small.dim_total, dtype=complex)
    psi[sub_small.config_to_index((0,0,0,0))] = 0.5
    psi[sub_small.config_to_index((0,0,1,1))] = 0.5
    psi[sub_small.config_to_index((1,1,0,0))] = 0.5
    psi[sub_small.config_to_index((1,1,1,1))] = 0.5
    psi = psi / np.linalg.norm(psi)
    
    correlations = np.zeros((n_small, n_small))
    for i in range(n_small):
        for j in range(n_small):
            n_i = sub_small.measure(psi, i)
            n_j = sub_small.measure(psi, j)
            n_ij = sum(np.abs(psi[idx])**2 * sub_small.index_to_config(idx)[i] * 
                       sub_small.index_to_config(idx)[j] for idx in range(sub_small.dim_total))
            correlations[i, j] = n_ij - n_i * n_j
    
    # Generate plots
    plot_light_cone(result_local, 'Light Cone (Local)', f'{output_dir}/light_cone_local.png')
    plot_light_cone(result_full, 'Light Cone (Full)', f'{output_dir}/light_cone_full.png')
    plot_comparison(result_local, result_full, f'{output_dir}/connectivity_comparison.png')
    plot_metric(metric, 'Emergent Metric', f'{output_dir}/emergent_metric.png')
    plot_entanglement(correlations, f'{output_dir}/entanglement_structure.png')
    
    print(f"\nArrival times (d from source):")
    for m in range(n_modes):
        print(f"  {m}: {result_local['arrival'][m]:.2f}")
    
    print(f"\nOutputs: {output_dir}/")


if __name__ == "__main__":
    main()