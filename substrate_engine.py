#!/usr/bin/env python
"""
substrate_engine.py

Production-ready implementation of the Hilbert Substrate Framework.
Simulates 2+1D Dirac spinors coupled to Maxwell fields and a non-local
"Defrag" potential (Emergent Gravity).

References to "The Substrate Framework":
- Axiom 2: Unitary Evolution via Corrected Leapfrog
- Sec 4.1: Maxwell Structure (Mr. Magnetic)
- Sec 5.1: Defrag Potential (1/r^2 interaction)
- Sec 5.3.1: Fermionic Exclusion via topological repulsion
"""

import os
import argparse

import time
import numpy as np

# Try importing CuPy for GPU acceleration
try:
    import cupy as xp
    BACKEND = "cupy"
    import cupyx.scipy.fft as xp_fft
except ImportError:
    import numpy as xp
    import numpy.fft as xp_fft
    BACKEND = "numpy"

class SubstrateEngine:
    def __init__(self, Nx, Ny, dx, dt, m, c, q, G_defrag, interaction_epsilon):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dt = dt
        self.m  = m
        self.c  = c
        self.q  = q
        
        # Defrag Potential strength & regularization
        self.G  = G_defrag
        self.eps = interaction_epsilon
        
        # Spatial Grid
        x = xp.arange(Nx) * dx
        y = xp.arange(Ny) * dx
        self.X, self.Y = xp.meshgrid(x, y)
        
        # Dirac matrices (2x2 representation)
        self._build_dirac_matrices()
        
        # Precompute Defrag Kernel in Fourier space
        self._build_interaction_kernel()
        
        # State Variables (Previous, Current, Next for Leapfrog)
        # Shape: (2, Ny, Nx)
        self.Psi_prev = None
        self.Psi_curr = xp.zeros((2, Ny, Nx), dtype=xp.complex128)
        
        # EM Fields
        self.Ex = xp.zeros((Ny, Nx), dtype=xp.float64)
        self.Ey = xp.zeros((Ny, Nx), dtype=xp.float64)
        self.Bz = xp.zeros((Ny, Nx), dtype=xp.float64)

    def _build_dirac_matrices(self):
        """Constructs 2+1D Dirac Algebra matrices."""
        # Pauli matrices
        sigma_x = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
        sigma_y = xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
        sigma_z = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)
        
        # Dirac representations
        self.gamma0 = sigma_z
        self.alpha1 = self.gamma0 @ (1j * sigma_y)  # +sigma_x
        self.alpha2 = self.gamma0 @ (-1j * sigma_x) # +sigma_y
        self.beta   = self.gamma0

    def _build_interaction_kernel(self):
        """
        Pre-computes the 1/r^2 kernel in Frequency domain for efficient convolution.
        Implements Sec 5.1.
        """
        if self.G == 0:
            self.kernel_hat = None
            return

        # Centered coordinates for kernel construction
        cx = self.Nx * self.dx / 2
        cy = self.Ny * self.dx / 2
        
        # Distance squared
        r2 = (self.X - cx)**2 + (self.Y - cy)**2
        
        # Defrag Kernel (gravity-like): -1/(r^2 + eps)
        kernel = -1.0 / (r2 + self.eps)
        
        # Normalize for stability
        kernel /= (xp.sum(xp.abs(kernel)) * self.dx * self.dx)

        # Shift so r=0 is at the origin for FFT convolution
        kernel = xp_fft.fftshift(kernel)

        # FFT kernel
        self.kernel_hat = xp_fft.fft2(kernel)

    def init_vortex_system(self, vortices):
        """
        Initialize spinor with multiple topological defects.
        vortices: list of dicts {'x': float, 'y': float, 'w': int, 'sigma': float}
        Implements Sec 5.3.1.
        """
        psi = xp.zeros((2, self.Ny, self.Nx), dtype=xp.complex128)
        
        for v in vortices:
            x0 = v['x']
            y0 = v['y']
            w  = v['w']
            sigma = v['sigma']
            
            dx = self.X - x0
            dy = self.Y - y0
            r  = xp.sqrt(dx**2 + dy**2) + 1e-12
            
            theta = xp.arctan2(dy, dx)
            phase = xp.exp(1j * w * theta)
            
            envelope = xp.exp(- (r**2) / (2 * sigma**2))
            
            psi_vortex = xp.zeros_like(psi)
            psi_vortex[0] = envelope * phase
            
            psi += psi_vortex
            
        norm = xp.sqrt(xp.sum(xp.abs(psi)**2) * self.dx**2)
        psi /= norm
        
        self.Psi_curr = psi.copy()

    def _compute_currents(self, Psi):
        r"""
        Compute probability density and current:
            rho = Psi^\dagger Psi
            J  = c * Psi^\dagger alpha Psi
        """
        psi_flat = Psi.reshape(2, -1)
        psi_dag  = xp.conj(psi_flat.T)
        
        rho_flat = xp.sum(psi_flat * xp.conj(psi_flat), axis=0).real
        
        Jx_flat = xp.sum(psi_dag * (self.alpha1 @ psi_flat).T, axis=1).real
        Jy_flat = xp.sum(psi_dag * (self.alpha2 @ psi_flat).T, axis=1).real
        
        rho = rho_flat.reshape(self.Ny, self.Nx)
        Jx  = self.c * Jx_flat.reshape(self.Ny, self.Nx)
        Jy  = self.c * Jy_flat.reshape(self.Ny, self.Nx)
        
        return rho, Jx, Jy

    def _compute_defrag_potential(self, rho):
        """Compute V_defrag(x) via FFT convolution."""
        if self.G == 0 or self.kernel_hat is None:
            return 0.0

        rho_hat = xp_fft.fft2(rho)
        pot_hat = rho_hat * self.kernel_hat
        V = xp_fft.ifft2(pot_hat).real

        return self.G * V

    def measure_observables(self):
        """Compute max density and core separation."""
        rho = xp.sum(xp.abs(self.Psi_curr)**2, axis=0)

        max_rho = xp.max(rho)

        rho_flat = rho.ravel()
        idx_sorted = xp.argsort(rho_flat)

        if BACKEND == "cupy":
            idx_sorted_np = xp.asnumpy(idx_sorted)
        else:
            idx_sorted_np = idx_sorted

        i1 = int(idx_sorted_np[-1])
        i2 = int(idx_sorted_np[-2])

        y1, x1 = divmod(i1, self.Nx)
        y2, x2 = divmod(i2, self.Nx)

        dx_pix = x2 - x1
        dy_pix = y2 - y1

        Nx = self.Nx
        Ny = self.Ny

        if dx_pix > Nx / 2:
            dx_pix -= Nx
        if dx_pix < -Nx / 2:
            dx_pix += Nx
        if dy_pix > Ny / 2:
            dy_pix -= Ny
        if dy_pix < -Ny / 2:
            dy_pix += Ny

        core_sep = ((dx_pix * self.dx)**2 + (dy_pix * self.dx)**2) ** 0.5

        if BACKEND == "cupy":
            max_rho_val = float(xp.asnumpy(max_rho))
        else:
            max_rho_val = float(max_rho)

        core_sep_val = float(core_sep)
        core_positions = (
            (x1 * self.dx, y1 * self.dx),
            (x2 * self.dx, y2 * self.dx),
        )

        return {
            "max_rho": max_rho_val,
            "core_separation": core_sep_val,
            "core_positions": core_positions,
        }

    def _apply_hamiltonian(self, Psi, Ex, Ey, Bz):
        """Apply Dirac Hamiltonian + Defrag potential."""
        dPsi_dx = (xp.roll(Psi, -1, axis=2) - xp.roll(Psi, 1, axis=2)) / (2*self.dx)
        dPsi_dy = (xp.roll(Psi, -1, axis=1) - xp.roll(Psi, 1, axis=1)) / (2*self.dx)
        
        term1 = -1j * self.c * (self.alpha1 @ dPsi_dx.reshape(2, -1) +
                                self.alpha2 @ dPsi_dy.reshape(2, -1))
        term1 = term1.reshape(2, self.Ny, self.Nx)
        
        term2 = self.m * self.c**2 * (self.beta @ Psi.reshape(2, -1))
        term2 = term2.reshape(2, self.Ny, self.Nx)
        
        rho, _, _ = self._compute_currents(Psi)
        V_defrag = self._compute_defrag_potential(rho)
        term3 = V_defrag * Psi
        
        H_Psi = term1 + term2 + term3
        return H_Psi

    def step(self):
        """One leapfrog step."""
        rho, Jx, Jy = self._compute_currents(self.Psi_curr)
        
        curl_E = (
            xp.roll(self.Ey, -1, axis=1) - xp.roll(self.Ey, 1, axis=1) -
            xp.roll(self.Ex, -1, axis=0) + xp.roll(self.Ex, 1, axis=0)
        ) / (2*self.dx)
        self.Bz = self.Bz - self.dt * curl_E
        
        dB_dy = (xp.roll(self.Bz, -1, axis=0) - xp.roll(self.Bz, 1, axis=0)) / (2*self.dx)
        dB_dx = (xp.roll(self.Bz, -1, axis=1) - xp.roll(self.Bz, 1, axis=1)) / (2*self.dx)
        
        self.Ex = self.Ex + self.dt * (self.c**2 * dB_dy - Jx)
        self.Ey = self.Ey + self.dt * (-self.c**2 * dB_dx - Jy)
        
        H_Psi = self._apply_hamiltonian(self.Psi_curr, self.Ex, self.Ey, self.Bz)
        
        Psi_next = self.Psi_prev - 2j * self.dt * H_Psi
        
        self.Psi_prev = self.Psi_curr
        self.Psi_curr = Psi_next

    def save_snapshot(self, step, out_dir):
        """Save compressed state, including the full complex spinor."""
        fname = os.path.join(out_dir, f"substrate_{step:05d}.npz")
        
        rho = xp.sum(xp.abs(self.Psi_curr)**2, axis=0)
        
        if BACKEND == "cupy":
            rho_np = xp.asnumpy(rho)
            Psi_np = xp.asnumpy(self.Psi_curr)
        else:
            rho_np = rho
            Psi_np = self.Psi_curr
        
        np.savez_compressed(fname, rho=rho_np, Psi=Psi_np, t=step*self.dt)

def main():
    parser = argparse.ArgumentParser(description="Substrate Framework: Exclusion Engine")
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--dx", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--G", type=float, default=50.0, help="Defrag potential strength")
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="output_exclusion")
    parser.add_argument("--winding", type=int, default=1)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--obs_interval", type=int, default=50,
                        help="Number of steps between observable logging")

    args = parser.parse_args()
    
    print(f"--- Hilbert Substrate Engine ---")
    print(f"Grid: {args.Nx} x {args.Nx}, dx={args.dx}, dt={args.dt}")
    print(f"Mass={args.mass}, G={args.G}, Backend={BACKEND}")

    # Prepare output
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    obs_path = os.path.join(args.out, "observables.csv")

    with open(obs_path, "w") as f_obs:
        f_obs.write("step,time,max_rho,core_sep,x1,y1,x2,y2\n")

        engine = SubstrateEngine(args.Nx, args.Nx, args.dx, args.dt, 
                                 args.mass, c=1.0, q=1.0, 
                                 G_defrag=args.G, interaction_epsilon=1.0)
        
        center = args.Nx * args.dx / 2
        
        offset = args.offset if args.offset is not None else args.Nx * args.dx / 5
        
        print(f"[SETUP] Initializing vortices at ±{offset} from center (sep={2*offset})")
        
        vortices = [
            {'x': center - offset, 'y': center, 'w': args.winding, 'sigma': 4.0},
            {'x': center + offset, 'y': center, 'w': args.winding, 'sigma': 4.0}
        ]
        
        engine.init_vortex_system(vortices)
        
        H_0 = engine._apply_hamiltonian(engine.Psi_curr, engine.Ex, engine.Ey, engine.Bz)
        engine.Psi_prev = engine.Psi_curr + 1j * args.dt * H_0
        
        for step in range(args.steps + 1):
            engine.step()
            
            if step % args.obs_interval == 0:
                norm = xp.sqrt(xp.sum(xp.abs(engine.Psi_curr)**2) * args.dx**2)
                engine.Psi_curr /= norm
                engine.Psi_prev  /= norm

                obs = engine.measure_observables()
                max_rho = obs["max_rho"]
                core_sep = obs["core_separation"]
                (x1, y1), (x2, y2) = obs["core_positions"]

                t = step * args.dt

                print(
                    f"Step {step}: t={t:.3f}, "
                    f"MaxRho={max_rho:.4f}, Sep={core_sep:.3f}, "
                    f"Core1=({x1:.1f},{y1:.1f}) Core2=({x2:.1f},{y2:.1f})"
                )

                f_obs.write(
                    f"{step},{t:.6f},{max_rho:.8e},{core_sep:.8e},"
                    f"{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}\n"
                )
                f_obs.flush()

                engine.save_snapshot(step, args.out)
        
        print(f"\nSimulation complete. Output in: {args.out}/")

if __name__ == "__main__":
    main()
