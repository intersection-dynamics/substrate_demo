"""
Information Patterns in Hilbert Space
=====================================
Axioms: Hilbert Space Realism, Unitary Evolution, Classical Emergence

Fermions: Non-copyable information (off-diagonal structure)
Bosons: Copyable information (diagonal structure)
Exclusion emerges from information-theoretic constraints.
"""

import numpy as np
from scipy.linalg import expm
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class Substrate:
    """Hilbert space partitioned into system + environment."""
    
    def __init__(self, n_sys: int, n_env: int, dim: int = 2):
        self.n_sys = n_sys
        self.n_env = n_env
        self.n_total = n_sys + n_env
        self.d = dim
        self.dim_sys = dim ** n_sys
        self.dim_env = dim ** n_env
        self.dim_total = dim ** self.n_total
        
    def vacuum(self) -> np.ndarray:
        psi = np.zeros(self.dim_total, dtype=complex)
        psi[0] = 1.0
        return psi
    
    def index_to_config(self, idx: int) -> Tuple[int, ...]:
        config = []
        for i in range(self.n_total):
            config.append(idx // (self.d ** (self.n_total - 1 - i)) % self.d)
        return tuple(config)
    
    def config_to_index(self, config: Tuple[int, ...]) -> int:
        idx = 0
        for i, n in enumerate(config):
            idx += n * (self.d ** (self.n_total - 1 - i))
        return idx
    
    def partial_trace_env(self, rho: np.ndarray) -> np.ndarray:
        """Trace out environment."""
        rho_sys = np.zeros((self.dim_sys, self.dim_sys), dtype=complex)
        
        for i_sys in range(self.dim_sys):
            for j_sys in range(self.dim_sys):
                for k_env in range(self.dim_env):
                    i_full = i_sys * self.dim_env + k_env
                    j_full = j_sys * self.dim_env + k_env
                    rho_sys[i_sys, j_sys] += rho[i_full, j_full]
        
        return rho_sys


def create_patterns(substrate: Substrate) -> Dict[str, np.ndarray]:
    """Create different pattern types."""
    patterns = {}
    
    # Local excitation (classical/diagonal)
    psi = substrate.vacuum()
    config = [0] * substrate.n_total
    config[0] = 1
    psi = np.zeros(substrate.dim_total, dtype=complex)
    psi[substrate.config_to_index(tuple(config))] = 1.0
    patterns['local'] = psi
    
    # Symmetric superposition
    psi = np.zeros(substrate.dim_total, dtype=complex)
    config1 = [0] * substrate.n_total
    config2 = [0] * substrate.n_total
    config1[0] = 1
    config2[1] = 1
    psi[substrate.config_to_index(tuple(config1))] = 1/np.sqrt(2)
    psi[substrate.config_to_index(tuple(config2))] = 1/np.sqrt(2)
    patterns['symmetric'] = psi
    
    # Antisymmetric superposition
    psi = np.zeros(substrate.dim_total, dtype=complex)
    psi[substrate.config_to_index(tuple(config1))] = 1/np.sqrt(2)
    psi[substrate.config_to_index(tuple(config2))] = -1/np.sqrt(2)
    patterns['antisymmetric'] = psi
    
    # Bell state (entangled)
    psi = np.zeros(substrate.dim_total, dtype=complex)
    config1 = [0] * substrate.n_total
    config2 = [0] * substrate.n_total
    config1[0], config1[1] = 0, 1
    config2[0], config2[1] = 1, 0
    psi[substrate.config_to_index(tuple(config1))] = 1/np.sqrt(2)
    psi[substrate.config_to_index(tuple(config2))] = 1/np.sqrt(2)
    patterns['bell'] = psi
    
    return patterns


def build_detection_unitary(substrate: Substrate, sys_mode: int, env_mode: int,
                            strength: float = np.pi/4) -> np.ndarray:
    """Detection = CNOT-like coupling between system and environment."""
    dim = substrate.dim_total
    H = np.zeros((dim, dim), dtype=complex)
    
    for idx in range(dim):
        config = list(substrate.index_to_config(idx))
        if config[sys_mode] > 0:
            config_flip = config.copy()
            config_flip[env_mode] = (config[env_mode] + 1) % substrate.d
            idx_flip = substrate.config_to_index(tuple(config_flip))
            H[idx_flip, idx] += config[sys_mode]
            H[idx, idx_flip] += config[sys_mode]
    
    return expm(-1j * strength * H)


def analyze_pattern(substrate: Substrate, psi: np.ndarray) -> Dict:
    """Analyze information structure of a pattern."""
    rho = np.outer(psi, np.conj(psi))
    rho_sys = substrate.partial_trace_env(rho)
    
    # Diagonal vs off-diagonal content
    diagonal = np.diag(np.diag(rho_sys))
    off_diagonal = rho_sys - diagonal
    
    diag_weight = np.real(np.trace(diagonal @ diagonal))
    offdiag_weight = np.real(np.trace(off_diagonal @ np.conj(off_diagonal).T))
    
    # Purity
    purity = np.real(np.trace(rho_sys @ rho_sys))
    
    return {
        'diagonal': diag_weight,
        'off_diagonal': offdiag_weight,
        'purity': purity,
        'rho_sys': rho_sys
    }


def test_copyability(substrate: Substrate, psi: np.ndarray, n_detections: int = 5) -> Dict:
    """Test if pattern survives multiple detections."""
    rho_orig = np.outer(psi, np.conj(psi))
    rho_sys_orig = substrate.partial_trace_env(rho_orig)
    
    purity_history = [1.0]
    fidelity_history = [1.0]
    
    psi_current = psi.copy()
    
    for i in range(n_detections):
        env_mode = substrate.n_sys + (i % substrate.n_env)
        U = build_detection_unitary(substrate, sys_mode=0, env_mode=env_mode)
        psi_current = U @ psi_current
        psi_current = psi_current / np.linalg.norm(psi_current)
        
        rho = np.outer(psi_current, np.conj(psi_current))
        rho_sys = substrate.partial_trace_env(rho)
        
        purity = np.real(np.trace(rho_sys @ rho_sys))
        fidelity = np.abs(np.trace(rho_sys_orig @ rho_sys))
        
        purity_history.append(purity)
        fidelity_history.append(fidelity)
    
    return {
        'purity': purity_history,
        'fidelity': fidelity_history,
        'final_purity': purity_history[-1],
        'final_fidelity': fidelity_history[-1]
    }


def plot_pattern_structure(patterns: Dict[str, np.ndarray], substrate: Substrate, filename: str):
    """Visualize density matrix structure for each pattern."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, (name, psi) in zip(axes, patterns.items()):
        analysis = analyze_pattern(substrate, psi)
        rho = np.abs(analysis['rho_sys'])
        
        im = ax.imshow(rho, cmap='Blues', vmin=0)
        ax.set_title(f'{name.capitalize()}\nDiag: {analysis["diagonal"]:.2f}, '
                    f'Off-diag: {analysis["off_diagonal"]:.2f}', fontsize=11)
        ax.set_xlabel('j')
        ax.set_ylabel('i')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Pattern Density Matrices (|ρ_sys|)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_copyability(results: Dict[str, Dict], filename: str):
    """Visualize copyability test results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'local': 'blue', 'symmetric': 'green', 'antisymmetric': 'red', 'bell': 'purple'}
    
    for name, result in results.items():
        axes[0].plot(result['purity'], 'o-', label=name, color=colors[name])
        axes[1].plot(result['fidelity'], 's-', label=name, color=colors[name])
    
    axes[0].set_xlabel('Detection #')
    axes[0].set_ylabel('Purity')
    axes[0].set_title('Purity Under Repeated Detection')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Detection #')
    axes[1].set_ylabel('Fidelity')
    axes[1].set_title('Fidelity Under Repeated Detection')
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_information_content(patterns: Dict, substrate: Substrate, filename: str):
    """Bar chart of diagonal vs off-diagonal information."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(patterns.keys())
    diag_vals = []
    offdiag_vals = []
    
    for name, psi in patterns.items():
        analysis = analyze_pattern(substrate, psi)
        diag_vals.append(analysis['diagonal'])
        offdiag_vals.append(analysis['off_diagonal'])
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, diag_vals, width, label='Diagonal (classical)', color='steelblue')
    bars2 = ax.bar(x + width/2, offdiag_vals, width, label='Off-diagonal (quantum)', color='coral')
    
    ax.set_xlabel('Pattern Type', fontsize=12)
    ax.set_ylabel('Information Weight', fontsize=12)
    ax.set_title('Information Structure: Diagonal vs Off-Diagonal', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_classification(results: Dict, filename: str):
    """Scatter plot classifying patterns by copyability."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'local': 'blue', 'symmetric': 'green', 'antisymmetric': 'red', 'bell': 'purple'}
    
    for name, result in results.items():
        ax.scatter(result['final_fidelity'], result['final_purity'], 
                  s=200, c=colors[name], label=name.capitalize(), edgecolor='black')
    
    # Classification regions
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.text(0.75, 0.75, 'BOSONIC\n(copyable)', ha='center', fontsize=12, alpha=0.7)
    ax.text(0.25, 0.25, 'FERMIONIC\n(consumed)', ha='center', fontsize=12, alpha=0.7)
    
    ax.set_xlabel('Final Fidelity', fontsize=12)
    ax.set_ylabel('Final Purity', fontsize=12)
    ax.set_title('Pattern Classification After Detection', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Information Patterns")
    print("="*40)
    
    substrate = Substrate(n_sys=2, n_env=3, dim=2)
    print(f"System: {substrate.n_sys} modes, Environment: {substrate.n_env} modes")
    print(f"Total dim: {substrate.dim_total}")
    
    # Create patterns
    patterns = create_patterns(substrate)
    
    # Analyze
    print("\nPattern analysis:")
    print(f"{'Pattern':<15} {'Diagonal':<10} {'Off-diag':<10} {'Type'}")
    print("-"*50)
    
    for name, psi in patterns.items():
        analysis = analyze_pattern(substrate, psi)
        ptype = "CLASSICAL" if analysis['diagonal'] > analysis['off_diagonal'] else "QUANTUM"
        print(f"{name:<15} {analysis['diagonal']:<10.3f} {analysis['off_diagonal']:<10.3f} {ptype}")
    
    # Copyability tests
    copy_results = {}
    for name, psi in patterns.items():
        copy_results[name] = test_copyability(substrate, psi, n_detections=5)
    
    print("\nCopyability (after 5 detections):")
    print(f"{'Pattern':<15} {'Fidelity':<10} {'Purity':<10} {'Class'}")
    print("-"*50)
    
    for name, result in copy_results.items():
        pclass = "COPYABLE" if result['final_fidelity'] > 0.5 else "CONSUMED"
        print(f"{name:<15} {result['final_fidelity']:<10.3f} {result['final_purity']:<10.3f} {pclass}")
    
    # Generate plots
    plot_pattern_structure(patterns, substrate, f'{output_dir}/pattern_structure.png')
    plot_copyability(copy_results, f'{output_dir}/copyability_test.png')
    plot_information_content(patterns, substrate, f'{output_dir}/information_content.png')
    plot_classification(copy_results, f'{output_dir}/pattern_classification.png')
    
    print(f"\nOutputs: {output_dir}/")


if __name__ == "__main__":
    main()