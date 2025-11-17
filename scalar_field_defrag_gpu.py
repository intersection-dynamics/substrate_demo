#!/usr/bin/env python3
"""
scalar_field_defrag_gpu.py

GPU-accelerated quantum scalar field evolution with emergent defrag gravity.

Auto-detects CuPy (GPU) or falls back to NumPy (CPU).
Expected speedup: 50-100× on RTX 3080 for 64×64 grids.

Evolution equation:
    i ∂ψ/∂t = [H_substrate + g·Φ_defrag]ψ

where:
    H_substrate = -∇²/(2m) + V(ψ)  # Substrate Hamiltonian
    Φ_defrag solves: ∇²Φ = |ψ|² - ⟨|ψ|²⟩  # Defrag potential
    g = coupling strength
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


class ScalarFieldDefragGPU:
    """
    2D complex scalar field with emergent defrag gravity.
    GPU-accelerated via CuPy.
    """
    
    def __init__(self, L=64, dx=1.0, dt=0.005, g_defrag=0.5, 
                 v=1.0, lambda_param=1.0):
        """
        Args:
            L: Grid size (L×L)
            dx: Spatial resolution
            dt: Time step
            g_defrag: Defrag coupling strength
            v: Vacuum expectation value
            lambda_param: Self-interaction strength
        """
        self.L = L
        self.dx = dx
        self.dt = dt
        self.g_defrag = g_defrag
        self.v = v
        self.lambda_param = lambda_param
        
        # Precompute k² for operators (on GPU if available)
        self.k2 = self._make_k2()
        
        print(f"\nInitialized ScalarFieldDefragGPU:")
        print(f"  Grid: {L}×{L}")
        print(f"  Backend: {'CuPy (GPU)' if GPU_AVAILABLE else 'NumPy (CPU)'}")
        print(f"  dt: {dt}")
        print(f"  g_defrag: {g_defrag}")
        print(f"  v: {v}, λ: {lambda_param}")
    
    def _make_k2(self):
        """Compute k² grid for Fourier space operators."""
        kx = 2*cp.pi*cp.fft.fftfreq(self.L, d=self.dx)
        ky = 2*cp.pi*cp.fft.fftfreq(self.L, d=self.dx)
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0  # Avoid division by zero
        return k2
    
    def create_vortex(self, x_center=None, y_center=None, r_core=10.0):
        """Create vortex configuration."""
        if x_center is None:
            x_center = self.L // 2
        if y_center is None:
            y_center = self.L // 2
        
        i = cp.arange(self.L)[:, None]
        j = cp.arange(self.L)[None, :]
        
        x = (i - x_center) * self.dx
        y = (j - y_center) * self.dx
        
        r = cp.sqrt(x**2 + y**2)
        theta = cp.arctan2(y, x)
        
        # Vortex profile: ρ = tanh(r/r_core) * e^(iθ)
        rho = cp.tanh(r / r_core)
        psi = rho * cp.exp(1j * theta)
        
        return psi
    
    def create_uniform_noise(self, mean=1.0, noise_amp=0.05, seed=42):
        """Create uniform state with small noise."""
        if GPU_AVAILABLE:
            cp.random.seed(seed)
            psi = mean * cp.ones((self.L, self.L), dtype=complex)
            psi += noise_amp * (cp.random.randn(self.L, self.L) + 
                               1j * cp.random.randn(self.L, self.L))
        else:
            np_cpu.random.seed(seed)
            psi = mean * cp.ones((self.L, self.L), dtype=complex)
            psi += noise_amp * (cp.random.randn(self.L, self.L) + 
                               1j * cp.random.randn(self.L, self.L))
        return psi
    
    def solve_defrag_potential(self, psi):
        """
        Solve ∇²Φ = s where s = |ψ|² - ⟨|ψ|²⟩
        
        This is the defrag principle: potential from density fluctuations.
        Returns Φ with -1/r structure.
        """
        # Source term: density fluctuation
        rho = cp.abs(psi)**2
        s = rho - cp.mean(rho)
        
        # Solve in Fourier space: Φ_k = -s_k / k²
        s_k = cpfft.fft2(s)
        Phi_k = -s_k / self.k2
        Phi_k[0, 0] = 0.0  # Zero mean
        
        Phi = cp.real(cpfft.ifft2(Phi_k))
        
        return Phi
    
    def kinetic_term(self, psi):
        """
        Kinetic energy: -∇²ψ/(2m)
        
        Computed in Fourier space.
        """
        psi_k = cpfft.fft2(psi)
        lap_psi_k = -self.k2 * psi_k / 2.0  # m = 1
        lap_psi = cpfft.ifft2(lap_psi_k)
        return -lap_psi
    
    def potential_term(self, psi):
        """
        Mexican hat potential: V = λ(|ψ|² - v²)²
        
        Returns V·ψ
        """
        rho = cp.abs(psi)**2
        V = self.lambda_param * (rho - self.v**2)**2
        return V * psi
    
    def substrate_hamiltonian(self, psi):
        """
        H_substrate·ψ = [-∇²/(2m) + V(ψ)]ψ
        """
        return self.kinetic_term(psi) + self.potential_term(psi)
    
    def total_hamiltonian(self, psi, Phi_defrag):
        """
        H_total·ψ = [H_substrate + g·Φ_defrag]ψ
        """
        H_sub = self.substrate_hamiltonian(psi)
        H_defrag = self.g_defrag * Phi_defrag * psi
        return H_sub + H_defrag
    
    def evolve_step_rk4(self, psi):
        """
        RK4 time integration of i∂ψ/∂t = H·ψ
        
        Self-consistent: Φ computed from ψ at each stage.
        """
        dt = self.dt
        
        # Stage 1
        Phi1 = self.solve_defrag_potential(psi)
        k1 = -1j * self.total_hamiltonian(psi, Phi1)
        
        # Stage 2
        psi2 = psi + 0.5*dt*k1
        Phi2 = self.solve_defrag_potential(psi2)
        k2 = -1j * self.total_hamiltonian(psi2, Phi2)
        
        # Stage 3
        psi3 = psi + 0.5*dt*k2
        Phi3 = self.solve_defrag_potential(psi3)
        k3 = -1j * self.total_hamiltonian(psi3, Phi3)
        
        # Stage 4
        psi4 = psi + dt*k3
        Phi4 = self.solve_defrag_potential(psi4)
        k4 = -1j * self.total_hamiltonian(psi4, Phi4)
        
        # Combine
        psi_new = psi + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize to conserve probability
        norm = cp.sqrt(cp.sum(cp.abs(psi_new)**2) * self.dx**2)
        psi_new = psi_new / norm * cp.sqrt(cp.sum(cp.abs(psi)**2) * self.dx**2)
        
        return psi_new, Phi4
    
    def compute_diagnostics(self, psi, Phi):
        """
        Compute all relevant diagnostics.
        """
        rho = cp.abs(psi)**2
        s = rho - cp.mean(rho)
        
        # Substrate energy
        H_sub_psi = self.substrate_hamiltonian(psi)
        E_substrate = cp.real(cp.sum(cp.conj(psi) * H_sub_psi) * self.dx**2)
        
        # Defrag binding energy (should be negative for bound states)
        E_defrag = 0.5 * cp.sum(Phi * s) * self.dx**2
        
        # Total energy
        E_total = E_substrate + self.g_defrag * E_defrag
        
        # Localization measures
        participation_ratio = 1.0 / (cp.sum(rho**2) * self.dx**2)
        max_rho = cp.max(rho)
        var_rho = cp.var(rho)
        
        # Norm (should be conserved)
        norm = cp.sum(rho) * self.dx**2
        
        # Convert to CPU for output
        return {
            'E_substrate': float(cp.asnumpy(E_substrate)) if GPU_AVAILABLE else float(E_substrate),
            'E_defrag': float(cp.asnumpy(E_defrag)) if GPU_AVAILABLE else float(E_defrag),
            'E_total': float(cp.asnumpy(E_total)) if GPU_AVAILABLE else float(E_total),
            'participation_ratio': float(cp.asnumpy(participation_ratio)) if GPU_AVAILABLE else float(participation_ratio),
            'max_rho': float(cp.asnumpy(max_rho)) if GPU_AVAILABLE else float(max_rho),
            'var_rho': float(cp.asnumpy(var_rho)) if GPU_AVAILABLE else float(var_rho),
            'norm': float(cp.asnumpy(norm)) if GPU_AVAILABLE else float(norm)
        }
    
    def compute_power_spectrum(self, psi):
        """
        Compute power spectrum P(k) = |ψ_k|².
        Returns k_radial and P_radial (azimuthally averaged).
        """
        rho = cp.abs(psi)**2
        s = rho - cp.mean(rho)
        
        # FFT
        s_k = cpfft.fft2(s)
        P_k = cp.abs(s_k)**2
        
        # Radial average
        kx = 2*cp.pi*cp.fft.fftfreq(self.L, d=self.dx)
        ky = 2*cp.pi*cp.fft.fftfreq(self.L, d=self.dx)
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        k_mag = cp.sqrt(KX**2 + KY**2)
        
        # Bin by k magnitude
        k_bins = cp.linspace(0, cp.max(k_mag), 20)
        P_radial = cp.zeros(len(k_bins)-1)
        k_radial = cp.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if cp.sum(mask) > 0:
                P_radial[i] = cp.mean(P_k[mask])
                k_radial[i] = cp.mean(k_mag[mask])
        
        # Convert to CPU
        if GPU_AVAILABLE:
            k_radial = cp.asnumpy(k_radial)
            P_radial = cp.asnumpy(P_radial)
        
        return k_radial, P_radial
    
    def run_evolution(self, psi_init, n_steps=500, 
                     snapshot_interval=50, output_dir='output_defrag',
                     save_snapshots=True):
        """
        Run full self-consistent evolution.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        psi = psi_init.copy()
        
        diagnostics = []
        
        print(f"\nStarting evolution...")
        print(f"  Steps: {n_steps}")
        print(f"  Snapshot interval: {snapshot_interval}")
        
        for step in range(n_steps + 1):
            if step % snapshot_interval == 0 or step == n_steps:
                # Compute diagnostics
                Phi = self.solve_defrag_potential(psi)
                diag = self.compute_diagnostics(psi, Phi)
                diag['step'] = step
                diag['time'] = step * self.dt
                diagnostics.append(diag)
                
                print(f"Step {step:4d}: E_defrag={diag['E_defrag']:+.4e}, "
                      f"max_ρ={diag['max_rho']:.4f}, "
                      f"var={diag['var_rho']:.4e}")
                
                # Save snapshot
                if save_snapshots:
                    self.save_snapshot(psi, Phi, step, output_dir)
            
            # Evolve
            if step < n_steps:
                psi, Phi = self.evolve_step_rk4(psi)
        
        # Save diagnostics
        df = pd.DataFrame(diagnostics)
        df.to_csv(output_dir / 'diagnostics.csv', index=False)
        print(f"\nSaved diagnostics to {output_dir / 'diagnostics.csv'}")
        
        # Create diagnostic plots
        self.plot_diagnostics(df, output_dir)
        
        return df
    
    def save_snapshot(self, psi, Phi, step, output_dir):
        """Save visualization of current state."""
        # Transfer to CPU for plotting
        if GPU_AVAILABLE:
            psi_cpu = cp.asnumpy(psi)
            Phi_cpu = cp.asnumpy(Phi)
        else:
            psi_cpu = psi
            Phi_cpu = Phi
        
        rho = np_cpu.abs(psi_cpu)**2
        phase = np_cpu.angle(psi_cpu)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Density
        im0 = axes[0].imshow(rho, cmap='hot', origin='lower',
                            extent=[0, self.L*self.dx, 0, self.L*self.dx])
        axes[0].set_title(f'Density |ψ|² (step {step})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Defrag potential
        im1 = axes[1].imshow(Phi_cpu, cmap='RdBu_r', origin='lower',
                            extent=[0, self.L*self.dx, 0, self.L*self.dx])
        axes[1].set_title('Defrag Potential Φ')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Phase
        im2 = axes[2].imshow(phase, cmap='twilight', origin='lower',
                            extent=[0, self.L*self.dx, 0, self.L*self.dx],
                            vmin=-np_cpu.pi, vmax=np_cpu.pi)
        axes[2].set_title('Phase arg(ψ)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'snapshot_{step:05d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_diagnostics(self, df, output_dir):
        """Create diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Energy components
        axes[0, 0].plot(df['time'], df['E_substrate'], label='E_substrate')
        axes[0, 0].plot(df['time'], df['E_defrag'], label='E_defrag')
        axes[0, 0].plot(df['time'], df['E_total'], label='E_total', linestyle='--')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].set_title('Energy Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Binding energy (should be negative)
        axes[0, 1].plot(df['time'], df['E_defrag'])
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('E_defrag')
        axes[0, 1].set_title('Defrag Binding Energy (negative = bound)')
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max density
        axes[0, 2].plot(df['time'], df['max_rho'])
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('max(ρ)')
        axes[0, 2].set_title('Maximum Density')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Variance
        axes[1, 0].plot(df['time'], df['var_rho'])
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Var(ρ)')
        axes[1, 0].set_title('Density Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Participation ratio
        axes[1, 1].plot(df['time'], df['participation_ratio'])
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('PR')
        axes[1, 1].set_title('Participation Ratio (higher = more spread)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Norm conservation
        axes[1, 2].plot(df['time'], df['norm'])
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('∫|ψ|²')
        axes[1, 2].set_title('Norm Conservation')
        axes[1, 2].axhline(y=df['norm'].iloc[0], color='r', 
                          linestyle='--', alpha=0.5, label='Initial')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved diagnostic plots to {output_dir / 'diagnostics.png'}")


def main():
    """
    Run test cases:
    1. Static vortex (validate binding energy)
    2. Noise → structure formation
    """
    
    print("="*70)
    print("SCALAR FIELD + DEFRAG GRAVITY (GPU-ACCELERATED)")
    print("="*70)
    
    # Create simulator
    sim = ScalarFieldDefragGPU(
        L=64,
        dx=1.0,
        dt=0.005,
        g_defrag=0.3,  # Moderate coupling
        v=1.0,
        lambda_param=0.5
    )
    
    # ===== TEST 1: Static Vortex =====
    print("\n" + "="*70)
    print("TEST 1: STATIC VORTEX (validate binding energy)")
    print("="*70)
    
    psi_vortex = sim.create_vortex(r_core=8.0)
    Phi_vortex = sim.solve_defrag_potential(psi_vortex)
    
    rho = cp.abs(psi_vortex)**2
    s = rho - cp.mean(rho)
    E_bind = 0.5 * cp.sum(Phi_vortex * s) * sim.dx**2
    E_bind_cpu = float(cp.asnumpy(E_bind)) if GPU_AVAILABLE else float(E_bind)
    
    print(f"\nStatic vortex binding energy: {E_bind_cpu:.6e}")
    print("(Should be negative - indicates bound state)")
    
    # Short evolution to see if it's stable
    print("\nEvolving vortex for 200 steps...")
    df_vortex = sim.run_evolution(
        psi_vortex, 
        n_steps=200, 
        snapshot_interval=50,
        output_dir='output_defrag_gpu/test1_vortex'
    )
    
    # ===== TEST 2: Noise → Structure =====
    print("\n" + "="*70)
    print("TEST 2: UNIFORM NOISE → STRUCTURE FORMATION")
    print("="*70)
    
    psi_noise = sim.create_uniform_noise(mean=1.0, noise_amp=0.1)
    
    initial_max = float(cp.asnumpy(cp.max(cp.abs(psi_noise)**2))) if GPU_AVAILABLE else float(cp.max(cp.abs(psi_noise)**2))
    initial_var = float(cp.asnumpy(cp.var(cp.abs(psi_noise)**2))) if GPU_AVAILABLE else float(cp.var(cp.abs(psi_noise)**2))
    
    print("\nInitial state:")
    print(f"  max(ρ) = {initial_max:.4f}")
    print(f"  var(ρ) = {initial_var:.6e}")
    
    print("\nEvolving noise for 500 steps...")
    df_noise = sim.run_evolution(
        psi_noise,
        n_steps=500,
        snapshot_interval=50,
        output_dir='output_defrag_gpu/test2_noise'
    )
    
    print("\nFinal state:")
    print(f"  max(ρ) = {df_noise['max_rho'].iloc[-1]:.4f}")
    print(f"  var(ρ) = {df_noise['var_rho'].iloc[-1]:.6e}")
    print(f"  E_defrag = {df_noise['E_defrag'].iloc[-1]:.6e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nTest 1 (Vortex):")
    print(f"  Initial E_defrag: {df_vortex['E_defrag'].iloc[0]:.6e}")
    print(f"  Final E_defrag:   {df_vortex['E_defrag'].iloc[-1]:.6e}")
    print(f"  Change: {df_vortex['E_defrag'].iloc[-1] - df_vortex['E_defrag'].iloc[0]:.6e}")
    
    print("\nTest 2 (Noise → Structure):")
    print(f"  Initial max(ρ): {df_noise['max_rho'].iloc[0]:.4f}")
    print(f"  Final max(ρ):   {df_noise['max_rho'].iloc[-1]:.4f}")
    print(f"  Growth factor: {df_noise['max_rho'].iloc[-1] / df_noise['max_rho'].iloc[0]:.2f}x")
    
    if df_noise['max_rho'].iloc[-1] > 1.5 * df_noise['max_rho'].iloc[0]:
        print("\n✓ STRUCTURE FORMATION OBSERVED!")
        print("  Defrag gravity creates localization from uniform noise")
    
    if df_vortex['E_defrag'].iloc[-1] < -0.01:
        print("\n✓ VORTEX IS BOUND!")
        print("  Negative binding energy confirms attractive defrag potential")
    
    print("\n" + "="*70)
    print("All outputs saved to output_defrag_gpu/")
    print("="*70)


if __name__ == '__main__':
    main()