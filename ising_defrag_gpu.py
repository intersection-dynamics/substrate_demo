#!/usr/bin/env python3
"""
2D Ising model with emergent defrag gravity.

Evolution: Monte Carlo (Metropolis-Hastings) with defrag potential coupling

Hamiltonian:
    H = -J Σ⟨i,j⟩ σ_i σ_j - Σ_i (h + g·Φ_i) σ_i

where:
    J = nearest-neighbor coupling
    h = external field
    g = defrag coupling strength
    Φ solves: ∇²Φ = M - ⟨M⟩  (M = local magnetization)

This implements the complete framework:
    Spin configuration → Magnetization → Defrag potential → Feedback
"""

import numpy as np_cpu  # Keep NumPy for plotting/saving
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# GPU detection
try:
    import cupy as cp
    import cupyx.scipy.fft as cpfft
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected - using GPU acceleration")
except ImportError:
    import numpy as cp
    import scipy.fft as cpfft
    GPU_AVAILABLE = False
    print("✗ GPU not available - using CPU (NumPy)")


class IsingDefragGPU:
    """
    2D Ising model with emergent defrag gravity.
    GPU-accelerated via CuPy.
    """
    
    def __init__(self, L=64, dx=1.0, J=1.0, h=0.0, T=2.0, 
                 g_defrag=0.5, coarse_grain_size=1):
        """
        Args:
            L: Lattice size (L×L)
            dx: Spatial resolution (for defrag potential)
            J: Nearest-neighbor coupling strength
            h: External magnetic field
            T: Temperature
            g_defrag: Defrag coupling strength
            coarse_grain_size: Coarsening for magnetization field (1=no coarsening)
        """
        self.L = L
        self.dx = dx
        self.J = J
        self.h = h
        self.T = T
        self.beta = 1.0 / T  # Inverse temperature
        self.g_defrag = g_defrag
        self.coarse_grain_size = coarse_grain_size
        
        # Precompute k² for Poisson solver
        self.k2 = self._make_k2()
        
        # Precompute energy lookup table for Metropolis
        # ΔE = -2 σ_i (J·sum_neighbors + h_eff)
        # sum_neighbors ∈ {-4, -2, 0, 2, 4}
        self._precompute_metropolis_tables()
        
        print(f"\nInitialized IsingDefragGPU:")
        print(f"  Lattice: {L}×{L}")
        print(f"  Backend: {'CuPy (GPU)' if GPU_AVAILABLE else 'NumPy (CPU)'}")
        print(f"  J: {J}, T: {T} (β: {self.beta:.3f})")
        print(f"  h: {h}, g_defrag: {g_defrag}")
        print(f"  Coarse-graining: {coarse_grain_size}")
    
    def _make_k2(self):
        """Compute k² grid for Fourier space operators."""
        kx = 2*cp.pi*cp.fft.fftfreq(self.L, d=self.dx)
        ky = 2*cp.pi*cp.fft.fftfreq(self.L, d=self.dx)
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0  # Avoid division by zero
        return k2
    
    def _precompute_metropolis_tables(self):
        """Precompute acceptance probabilities for Metropolis."""
        # For each possible neighbor sum and field value, compute flip probability
        # This is an optimization - can look up instead of computing exp each time
        self.neighbor_sums = cp.array([-4, -2, 0, 2, 4])
        # We'll compute probabilities on the fly since field is spatially varying
    
    def create_random_spins(self, seed=42):
        """Create random spin configuration."""
        if GPU_AVAILABLE:
            cp.random.seed(seed)
            spins = 2 * cp.random.randint(0, 2, (self.L, self.L)) - 1
        else:
            np_cpu.random.seed(seed)
            spins = 2 * cp.random.randint(0, 2, (self.L, self.L)) - 1
        return spins.astype(cp.int8)  # Save memory
    
    def create_ordered_spins(self, state='up'):
        """Create fully ordered configuration."""
        if state == 'up':
            return cp.ones((self.L, self.L), dtype=cp.int8)
        elif state == 'down':
            return -cp.ones((self.L, self.L), dtype=cp.int8)
        elif state == 'domain':
            # Create two domains
            spins = cp.ones((self.L, self.L), dtype=cp.int8)
            spins[:, :self.L//2] = -1
            return spins
        else:
            raise ValueError(f"Unknown state: {state}")
    
    def create_noise_spins(self, flip_prob=0.1, seed=42):
        """Create mostly ordered state with random flips."""
        if GPU_AVAILABLE:
            cp.random.seed(seed)
        else:
            np_cpu.random.seed(seed)
        
        spins = cp.ones((self.L, self.L), dtype=cp.int8)
        mask = cp.random.rand(self.L, self.L) < flip_prob
        spins[mask] = -1
        return spins
    
    def compute_magnetization_field(self, spins):
        """
        Compute local magnetization field (potentially coarse-grained).
        
        Returns M(x,y) where M is average spin in local region.
        """
        M = spins.astype(cp.float32)
        
        if self.coarse_grain_size > 1:
            # Coarse-grain by averaging over blocks
            cg = self.coarse_grain_size
            L_cg = self.L // cg
            M_cg = cp.zeros((L_cg, L_cg), dtype=cp.float32)
            
            for i in range(L_cg):
                for j in range(L_cg):
                    M_cg[i, j] = cp.mean(M[i*cg:(i+1)*cg, j*cg:(j+1)*cg])
            
            # Upsample back to full grid
            M = cp.repeat(cp.repeat(M_cg, cg, axis=0), cg, axis=1)
        
        return M
    
    def solve_defrag_potential(self, spins):
        """
        Solve ∇²Φ = s where s = M - ⟨M⟩
        
        M is local magnetization (coarse-grained spin configuration).
        Returns Φ with -1/r structure for domain walls.
        """
        # Source term: magnetization fluctuation
        M = self.compute_magnetization_field(spins)
        s = M - cp.mean(M)
        
        # Solve in Fourier space: Φ_k = -s_k / k²
        s_k = cpfft.fft2(s)
        Phi_k = -s_k / self.k2
        Phi_k[0, 0] = 0.0  # Zero mean
        
        Phi = cp.real(cpfft.ifft2(Phi_k))
        
        return Phi
    
    def compute_neighbor_sum(self, spins):
        """
        Compute sum of nearest neighbors for each spin.
        
        Returns array of same shape as spins with values in {-4, -2, 0, 2, 4}.
        Uses periodic boundary conditions.
        """
        # Roll in each direction and sum
        neighbor_sum = (
            cp.roll(spins, 1, axis=0) +   # up
            cp.roll(spins, -1, axis=0) +   # down
            cp.roll(spins, 1, axis=1) +    # left
            cp.roll(spins, -1, axis=1)     # right
        )
        return neighbor_sum
    
    def compute_local_fields(self, Phi):
        """
        Compute effective local field h_eff = h + g·Φ.
        """
        return self.h + self.g_defrag * Phi
    
    def compute_energy_change(self, spins, neighbor_sum, h_eff):
        """
        Compute energy change ΔE for flipping each spin.
        
        ΔE = -2 σ_i (J·sum_neighbors + h_eff)
        
        Returns array of energy changes.
        """
        return -2 * spins * (self.J * neighbor_sum + h_eff)
    
    def metropolis_sweep(self, spins, Phi):
        """
        Perform one Metropolis Monte Carlo sweep.
        
        Checkerboard decomposition for parallelization:
        - Update all odd sites (i+j odd)
        - Then update all even sites (i+j even)
        
        Returns updated spins.
        """
        h_eff = self.compute_local_fields(Phi)
        
        # Checkerboard masks
        i, j = cp.meshgrid(cp.arange(self.L), cp.arange(self.L), indexing='ij')
        odd_mask = ((i + j) % 2 == 1)
        even_mask = ((i + j) % 2 == 0)
        
        # Update odd sites
        spins = self._metropolis_update_subset(spins, h_eff, odd_mask)
        
        # Update even sites
        spins = self._metropolis_update_subset(spins, h_eff, even_mask)
        
        return spins
    
    def _metropolis_update_subset(self, spins, h_eff, mask):
        """Update spins at sites where mask is True."""
        # Compute neighbor sum
        neighbor_sum = self.compute_neighbor_sum(spins)
        
        # Compute energy change for flipping
        dE = self.compute_energy_change(spins, neighbor_sum, h_eff)
        
        # Metropolis criterion: accept if exp(-β·ΔE) > random
        # Equivalent: accept if ΔE < 0 OR random < exp(-β·ΔE)
        accept_prob = cp.minimum(1.0, cp.exp(-self.beta * dE))
        random_vals = cp.random.rand(self.L, self.L)
        
        # Flip spins where accepted AND in mask
        flip_mask = (random_vals < accept_prob) & mask
        spins = spins.copy()  # Don't modify in place
        spins[flip_mask] = -spins[flip_mask]
        
        return spins
    
    def compute_energy(self, spins, Phi):
        """
        Compute total energy of configuration.
        
        E = E_Ising + E_defrag
        
        where:
            E_Ising = -J Σ⟨i,j⟩ σ_i σ_j - h Σ σ_i
            E_defrag = -g Σ_i Φ_i σ_i
        """
        # Ising energy (nearest neighbor)
        neighbor_sum = self.compute_neighbor_sum(spins)
        E_nn = -self.J * cp.sum(spins * neighbor_sum) / 2  # Divide by 2 (double counting)
        
        # External field energy
        E_field = -self.h * cp.sum(spins)
        
        E_Ising = E_nn + E_field
        
        # Defrag energy
        E_defrag = -self.g_defrag * cp.sum(Phi * spins)
        
        return E_Ising, E_defrag
    
    def compute_defrag_binding_energy(self, spins, Phi):
        """
        Compute defrag binding energy: E_bind = (1/2) Σ Φ·s
        
        where s = M - ⟨M⟩ is the source term.
        Negative values indicate bound configurations.
        """
        M = self.compute_magnetization_field(spins)
        s = M - cp.mean(M)
        E_bind = 0.5 * cp.sum(Phi * s) * self.dx**2
        return E_bind
    
    def compute_diagnostics(self, spins, Phi):
        """
        Compute all relevant diagnostics.
        """
        # Convert to float for calculations
        spins_f = spins.astype(cp.float32)
        
        # Magnetization
        M = cp.mean(spins_f)
        M_abs = cp.abs(M)
        
        # Energy components
        E_Ising, E_defrag = self.compute_energy(spins, Phi)
        E_total = E_Ising + E_defrag
        
        # Defrag binding energy
        E_bind = self.compute_defrag_binding_energy(spins, Phi)
        
        # Domain structure
        # Count domain walls (sites where neighbors disagree)
        neighbor_sum = self.compute_neighbor_sum(spins)
        n_walls = cp.sum(cp.abs(neighbor_sum) < 4) // 2  # Divide by 2 (double counting)
        
        # Magnetization variance (measures domain size)
        M_field = self.compute_magnetization_field(spins)
        var_M = cp.var(M_field)
        
        # Defrag potential statistics
        Phi_max = cp.max(cp.abs(Phi))
        Phi_var = cp.var(Phi)
        
        # Convert to CPU for output
        return {
            'M': float(cp.asnumpy(M)) if GPU_AVAILABLE else float(M),
            'M_abs': float(cp.asnumpy(M_abs)) if GPU_AVAILABLE else float(M_abs),
            'E_Ising': float(cp.asnumpy(E_Ising)) if GPU_AVAILABLE else float(E_Ising),
            'E_defrag': float(cp.asnumpy(E_defrag)) if GPU_AVAILABLE else float(E_defrag),
            'E_total': float(cp.asnumpy(E_total)) if GPU_AVAILABLE else float(E_total),
            'E_bind': float(cp.asnumpy(E_bind)) if GPU_AVAILABLE else float(E_bind),
            'n_walls': int(cp.asnumpy(n_walls)) if GPU_AVAILABLE else int(n_walls),
            'var_M': float(cp.asnumpy(var_M)) if GPU_AVAILABLE else float(var_M),
            'Phi_max': float(cp.asnumpy(Phi_max)) if GPU_AVAILABLE else float(Phi_max),
            'Phi_var': float(cp.asnumpy(Phi_var)) if GPU_AVAILABLE else float(Phi_var)
        }
    
    def run_evolution(self, spins_init, n_sweeps=1000, 
                     snapshot_interval=100, output_dir='output_ising',
                     save_snapshots=True):
        """
        Run Monte Carlo evolution with self-consistent defrag potential.
        
        Args:
            spins_init: Initial spin configuration
            n_sweeps: Number of MC sweeps
            snapshot_interval: Save snapshot every N sweeps
            output_dir: Output directory
            save_snapshots: Whether to save visualization snapshots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        spins = spins_init.copy()
        
        diagnostics = []
        
        print(f"\nStarting MC evolution...")
        print(f"  Sweeps: {n_sweeps}")
        print(f"  Snapshot interval: {snapshot_interval}")
        print(f"  Temperature: {self.T}")
        
        for sweep in range(n_sweeps + 1):
            if sweep % snapshot_interval == 0 or sweep == n_sweeps:
                # Compute diagnostics
                Phi = self.solve_defrag_potential(spins)
                diag = self.compute_diagnostics(spins, Phi)
                diag['sweep'] = sweep
                diagnostics.append(diag)
                
                print(f"Sweep {sweep:4d}: M={diag['M']:+.4f}, "
                      f"E_bind={diag['E_bind']:+.4e}, "
                      f"walls={diag['n_walls']:4d}")
                
                # Save snapshot
                if save_snapshots:
                    self.save_snapshot(spins, Phi, sweep, output_dir)
            
            # MC sweep
            if sweep < n_sweeps:
                Phi = self.solve_defrag_potential(spins)
                spins = self.metropolis_sweep(spins, Phi)
        
        # Save diagnostics
        df = pd.DataFrame(diagnostics)
        df.to_csv(output_dir / 'diagnostics.csv', index=False)
        print(f"\nSaved diagnostics to {output_dir / 'diagnostics.csv'}")
        
        # Create diagnostic plots
        self.plot_diagnostics(df, output_dir)
        
        return df
    
    def save_snapshot(self, spins, Phi, sweep, output_dir):
        """Save visualization of current state."""
        # Transfer to CPU for plotting
        if GPU_AVAILABLE:
            spins_cpu = cp.asnumpy(spins)
            Phi_cpu = cp.asnumpy(Phi)
        else:
            spins_cpu = spins
            Phi_cpu = Phi
        
        M_field = self.compute_magnetization_field(spins)
        if GPU_AVAILABLE:
            M_cpu = cp.asnumpy(M_field)
        else:
            M_cpu = M_field
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Spin configuration
        im0 = axes[0].imshow(spins_cpu, cmap='bwr', origin='lower',
                            extent=[0, self.L*self.dx, 0, self.L*self.dx],
                            vmin=-1, vmax=1)
        axes[0].set_title(f'Spin Configuration (sweep {sweep})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, ticks=[-1, 0, 1])
        
        # Defrag potential
        im1 = axes[1].imshow(Phi_cpu, cmap='RdBu_r', origin='lower',
                            extent=[0, self.L*self.dx, 0, self.L*self.dx])
        axes[1].set_title('Defrag Potential Φ')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Magnetization field
        im2 = axes[2].imshow(M_cpu, cmap='RdBu', origin='lower',
                            extent=[0, self.L*self.dx, 0, self.L*self.dx],
                            vmin=-1, vmax=1)
        axes[2].set_title('Magnetization Field M')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'snapshot_{sweep:05d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_diagnostics(self, df, output_dir):
        """Create diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Magnetization
        axes[0, 0].plot(df['sweep'], df['M'], label='M')
        axes[0, 0].plot(df['sweep'], df['M_abs'], label='|M|', linestyle='--')
        axes[0, 0].set_xlabel('Sweep')
        axes[0, 0].set_ylabel('Magnetization')
        axes[0, 0].set_title('Order Parameter')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Energy components
        axes[0, 1].plot(df['sweep'], df['E_Ising'], label='E_Ising')
        axes[0, 1].plot(df['sweep'], df['E_defrag'], label='E_defrag')
        axes[0, 1].plot(df['sweep'], df['E_total'], label='E_total', linestyle='--')
        axes[0, 1].set_xlabel('Sweep')
        axes[0, 1].set_ylabel('Energy')
        axes[0, 1].set_title('Energy Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Binding energy
        axes[0, 2].plot(df['sweep'], df['E_bind'])
        axes[0, 2].set_xlabel('Sweep')
        axes[0, 2].set_ylabel('E_bind')
        axes[0, 2].set_title('Defrag Binding Energy')
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Domain walls
        axes[1, 0].plot(df['sweep'], df['n_walls'])
        axes[1, 0].set_xlabel('Sweep')
        axes[1, 0].set_ylabel('Number of walls')
        axes[1, 0].set_title('Domain Wall Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Magnetization variance
        axes[1, 1].plot(df['sweep'], df['var_M'])
        axes[1, 1].set_xlabel('Sweep')
        axes[1, 1].set_ylabel('Var(M)')
        axes[1, 1].set_title('Magnetization Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Defrag potential variance
        axes[1, 2].plot(df['sweep'], df['Phi_var'])
        axes[1, 2].set_xlabel('Sweep')
        axes[1, 2].set_ylabel('Var(Φ)')
        axes[1, 2].set_title('Defrag Potential Variance')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved diagnostic plots to {output_dir / 'diagnostics.png'}")


def main():
    """
    Run test cases:
    1. Ordered state → equilibrium
    2. Random state → structure formation
    3. Domain wall dynamics
    """
    
    print("="*70)
    print("ISING MODEL + DEFRAG GRAVITY")
    print("="*70)
    
    # Create simulator
    sim = IsingDefragGPU(
        L=64,
        dx=1.0,
        J=1.0,
        h=0.0,
        T=2.0,           # Near critical temperature (T_c ≈ 2.269)
        g_defrag=0.5,
        coarse_grain_size=1
    )
    
    # ===== TEST 1: Random State → Equilibrium =====
    print("\n" + "="*70)
    print("TEST 1: RANDOM STATE → EQUILIBRIUM")
    print("="*70)
    
    spins_random = sim.create_random_spins(seed=42)
    
    print(f"\nInitial magnetization: {cp.mean(spins_random.astype(cp.float32)):.4f}")
    
    df_random = sim.run_evolution(
        spins_random,
        n_sweeps=1000,
        snapshot_interval=100,
        output_dir='output_ising/test1_random'
    )
    
    # ===== TEST 2: Domain Wall Dynamics =====
    print("\n" + "="*70)
    print("TEST 2: DOMAIN WALL DYNAMICS")
    print("="*70)
    
    spins_domain = sim.create_ordered_spins(state='domain')
    
    print(f"\nInitial configuration: Two domains")
    print(f"Initial magnetization: {cp.mean(spins_domain.astype(cp.float32)):.4f}")
    
    df_domain = sim.run_evolution(
        spins_domain,
        n_sweeps=1000,
        snapshot_interval=100,
        output_dir='output_ising/test2_domain'
    )
    
    # ===== TEST 3: Noisy Ordered State =====
    print("\n" + "="*70)
    print("TEST 3: NOISY ORDERED STATE → COARSENING")
    print("="*70)
    
    spins_noisy = sim.create_noise_spins(flip_prob=0.2, seed=42)
    
    print(f"\nInitial magnetization: {cp.mean(spins_noisy.astype(cp.float32)):.4f}")
    print("(80% up, 20% down - should coarsen)")
    
    df_noisy = sim.run_evolution(
        spins_noisy,
        n_sweeps=1000,
        snapshot_interval=100,
        output_dir='output_ising/test3_noisy'
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nTest 1 (Random → Equilibrium):")
    print(f"  Initial |M|: {abs(df_random['M'].iloc[0]):.4f}")
    print(f"  Final |M|:   {abs(df_random['M'].iloc[-1]):.4f}")
    print(f"  Final walls: {df_random['n_walls'].iloc[-1]}")
    
    print("\nTest 2 (Domain Wall):")
    print(f"  Initial walls: {df_domain['n_walls'].iloc[0]}")
    print(f"  Final walls:   {df_domain['n_walls'].iloc[-1]}")
    print(f"  E_bind change: {df_domain['E_bind'].iloc[-1] - df_domain['E_bind'].iloc[0]:.2e}")
    
    print("\nTest 3 (Noisy → Coarsening):")
    print(f"  Initial |M|: {abs(df_noisy['M'].iloc[0]):.4f}")
    print(f"  Final |M|:   {abs(df_noisy['M'].iloc[-1]):.4f}")
    print(f"  Ordering: {abs(df_noisy['M'].iloc[-1]) / abs(df_noisy['M'].iloc[0]):.2f}x")
    
    print("\n" + "="*70)
    print("All outputs saved to output_ising/")
    print("="*70)


if __name__ == '__main__':
    main()