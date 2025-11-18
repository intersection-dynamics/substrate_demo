#!/usr/bin/env python3
"""
coupled_scalar_gauge.py

Test coupling between scalar field sector and gauge field sector.

Scalar sector:
  - Complex field psi with covariant kinetic term |D psi|^2 
  - D = nabla - iq A (covariant derivative)
  - Defrag self-interaction via Poisson potential

Gauge sector:
  - Real gauge potentials ax, ay
  - Electric fields ex, ey
  - Dynamics: d ax/dt = ex, d ex/dt = c^2 nabla^2 ax - jx
  - Current source: jx = q Im(psi* D_x psi)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def get_xp(use_gpu: bool):
    if use_gpu and CUPY_AVAILABLE:
        print("✓ GPU (CuPy) detected - using GPU")
        return cp, True
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print("⚠ Requested GPU, but CuPy not available. Falling back to NumPy.")
        else:
            print("✓ Using NumPy (CPU)")
        return np, False


class CoupledScalarGauge2D:
    """Coupled scalar + gauge system in 2D."""
    
    def __init__(
        self,
        L: int,
        dx: float = 1.0,
        dt: float = 0.01,
        c: float = 1.0,
        q: float = 0.1,
        g_defrag: float = 1.0,
        use_gpu: bool = True,
    ):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.c = c
        self.q = q
        self.g_defrag = g_defrag
        
        self.xp, self.is_gpu = get_xp(use_gpu)
        
        xp = self.xp
        
        self.psi = xp.zeros((L, L), dtype=xp.complex128)
        self.ax = xp.zeros((L, L), dtype=xp.float64)
        self.ay = xp.zeros((L, L), dtype=xp.float64)
        self.ex = xp.zeros((L, L), dtype=xp.float64)
        self.ey = xp.zeros((L, L), dtype=xp.float64)
    
    # ---------------- spatial helpers ----------------
    
    def roll(self, arr, shift, axis):
        return self.xp.roll(arr, shift=shift, axis=axis)
    
    def grad_x(self, field):
        return (self.roll(field, -1, axis=0) - self.roll(field, +1, axis=0)) / (
            2.0 * self.dx
        )
    
    def grad_y(self, field):
        return (self.roll(field, -1, axis=1) - self.roll(field, +1, axis=1)) / (
            2.0 * self.dx
        )
    
    def laplacian(self, field):
        xp = self.xp
        dx2 = self.dx ** 2
        f_ip = self.roll(field, -1, axis=0)
        f_im = self.roll(field, +1, axis=0)
        f_jp = self.roll(field, -1, axis=1)
        f_jm = self.roll(field, +1, axis=1)
        return (f_ip + f_im + f_jp + f_jm - 4.0 * field) / dx2
    
    # ---------------- scalar sector helpers ----------------
    
    def covariant_gradient_sq(self, psi, ax, ay):
        """
        Return |D psi|^2 = |(∂x - i q ax) psi|^2 + |(∂y - i q ay) psi|^2
        """
        xp = self.xp
        dx = self.dx
        q = self.q
        
        # central differences
        psi_ip = xp.roll(psi, -1, axis=0)
        psi_im = xp.roll(psi, +1, axis=0)
        psi_jp = xp.roll(psi, -1, axis=1)
        psi_jm = xp.roll(psi, +1, axis=1)
        
        dpsi_dx = (psi_ip - psi_im) / (2.0 * dx)
        dpsi_dy = (psi_jp - psi_jm) / (2.0 * dx)
        
        D_x_psi = dpsi_dx - 1j * q * ax * psi
        D_y_psi = dpsi_dy - 1j * q * ay * psi
        
        return xp.abs(D_x_psi)**2 + xp.abs(D_y_psi)**2
    
    def gauge_current(self, psi, ax, ay):
        """Compute gauge current: j = q Im(psi* D psi)."""
        xp = self.xp
        dx = self.dx
        q = self.q
        
        # x-direction
        psi_ip = xp.roll(psi, -1, axis=0)
        psi_im = xp.roll(psi, +1, axis=0)
        dpsi_dx = (psi_ip - psi_im) / (2.0 * dx)
        D_x_psi = dpsi_dx - 1j * q * ax * psi
        jx = q * xp.imag(xp.conjugate(psi) * D_x_psi)
        
        # y-direction
        psi_jp = xp.roll(psi, -1, axis=1)
        psi_jm = xp.roll(psi, +1, axis=1)
        dpsi_dy = (psi_jp - psi_jm) / (2.0 * dx)
        D_y_psi = dpsi_dy - 1j * q * ay * psi
        jy = q * xp.imag(xp.conjugate(psi) * D_y_psi)
        
        return jx, jy
    
    def solve_defrag_potential(self, rho):
        """
        Solve Poisson equation ∇^2 φ = -rho with periodic BC using FFT.
        """
        xp = self.xp
        L = self.L
        dx = self.dx
        
        # FFT of rho
        rho_k = xp.fft.fftn(rho)
        
        kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        KX, KY = xp.meshgrid(kx, ky, indexing="ij")
        k2 = KX**2 + KY**2
        
        # avoid k=0
        denom = -k2
        denom[0, 0] = 1.0
        
        phi_k = rho_k / denom
        phi_k[0, 0] = 0.0
        
        phi = xp.fft.ifftn(phi_k).real
        return phi
    
    # ---------------- RHS functions ----------------
    
    def rhs_scalar(self, psi, ax, ay):
        """Right-hand side for scalar field: dpsi/dt."""
        xp = self.xp
        q = self.q
        
        # Laplacian term (free kinetic)
        lap_psi = self.laplacian(psi)
        
        # Covariant gradient term for coupling
        dx = self.dx
        psi_ip = xp.roll(psi, -1, axis=0)
        psi_im = xp.roll(psi, +1, axis=0)
        psi_jp = xp.roll(psi, -1, axis=1)
        psi_jm = xp.roll(psi, +1, axis=1)
        
        dpsi_dx = (psi_ip - psi_im) / (2.0 * dx)
        dpsi_dy = (psi_jp - psi_jm) / (2.0 * dx)
        
        # approximate minimal coupling expansion of |D psi|^2
        gauge_term = -1j * q * (ax * dpsi_dx + ay * dpsi_dy)
        gauge_term += -1j * q * (dpsi_dx * ax + dpsi_dy * ay)
        gauge_term += -(q**2) * (ax**2 + ay**2) * psi
        
        # defrag potential
        rho = xp.abs(psi)**2
        phi = self.solve_defrag_potential(rho)
        V_defrag = self.g_defrag * phi
        
        H_psi = -0.5 * lap_psi + gauge_term + V_defrag * psi
        dpsi_dt = -1j * H_psi
        return dpsi_dt
    
    def rhs_gauge(self, ax, ay, ex, ey, psi):
        """Right-hand side for gauge fields."""
        jx, jy = self.gauge_current(psi, ax, ay)
        
        lap_ax = self.laplacian(ax)
        lap_ay = self.laplacian(ay)
        
        dax_dt = ex
        day_dt = ey
        dex_dt = (self.c**2) * lap_ax - jx
        dey_dt = (self.c**2) * lap_ay - jy
        
        return dax_dt, day_dt, dex_dt, dey_dt
    
    # ---------------- time stepping ----------------
    
    def step_rk4(self):
        """RK4 step for coupled system."""
        dt = self.dt
        
        psi0 = self.psi
        ax0, ay0 = self.ax, self.ay
        ex0, ey0 = self.ex, self.ey
        
        # k1
        k1_psi = self.rhs_scalar(psi0, ax0, ay0)
        k1_ax, k1_ay, k1_ex, k1_ey = self.rhs_gauge(ax0, ay0, ex0, ey0, psi0)
        
        # k2
        psi_2 = psi0 + 0.5 * dt * k1_psi
        ax_2, ay_2 = ax0 + 0.5 * dt * k1_ax, ay0 + 0.5 * dt * k1_ay
        ex_2, ey_2 = ex0 + 0.5 * dt * k1_ex, ey0 + 0.5 * dt * k1_ey
        
        k2_psi = self.rhs_scalar(psi_2, ax_2, ay_2)
        k2_ax, k2_ay, k2_ex, k2_ey = self.rhs_gauge(ax_2, ay_2, ex_2, ey_2, psi_2)
        
        # k3
        psi_3 = psi0 + 0.5 * dt * k2_psi
        ax_3, ay_3 = ax0 + 0.5 * dt * k2_ax, ay0 + 0.5 * dt * k2_ay
        ex_3, ey_3 = ex0 + 0.5 * dt * k2_ex, ey0 + 0.5 * dt * k2_ey
        
        k3_psi = self.rhs_scalar(psi_3, ax_3, ay_3)
        k3_ax, k3_ay, k3_ex, k3_ey = self.rhs_gauge(ax_3, ay_3, ex_3, ey_3, psi_3)
        
        # k4
        psi_4 = psi0 + dt * k3_psi
        ax_4, ay_4 = ax0 + dt * k3_ax, ay0 + dt * k3_ay
        ex_4, ey_4 = ex0 + dt * k3_ex, ey0 + dt * k3_ey
        
        k4_psi = self.rhs_scalar(psi_4, ax_4, ay_4)
        k4_ax, k4_ay, k4_ex, k4_ey = self.rhs_gauge(ax_4, ay_4, ex_4, ey_4, psi_4)
        
        # Update
        self.psi = psi0 + (dt/6.0) * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi)
        self.ax = ax0 + (dt/6.0) * (k1_ax + 2*k2_ax + 2*k3_ax + k4_ax)
        self.ay = ay0 + (dt/6.0) * (k1_ay + 2*k2_ay + 2*k3_ay + k4_ay)
        self.ex = ex0 + (dt/6.0) * (k1_ex + 2*k2_ex + 2*k3_ex + k4_ex)
        self.ey = ey0 + (dt/6.0) * (k1_ey + 2*k2_ey + 2*k3_ey + k4_ey)
    
    def step_symplectic(self):
        """Symplectic-style kick–drift–kick step for the coupled system.

        - First half-kick gauge momenta (ex, ey) using currents and Laplacian.
        - Drift gauge coordinates (ax, ay) for a full step using the half-updated momenta.
        - Evolve psi with a 2nd-order midpoint method in the updated gauge background.
        - Second half-kick of (ex, ey) using the new psi.
        """
        dt = self.dt

        # Copy current state
        psi0 = self.psi
        ax0, ay0 = self.ax, self.ay
        ex0, ey0 = self.ex, self.ey

        # --- 1) First half-kick for gauge momenta ---
        _, _, dex_dt0, dey_dt0 = self.rhs_gauge(ax0, ay0, ex0, ey0, psi0)
        ex_half = ex0 + 0.5 * dt * dex_dt0
        ey_half = ey0 + 0.5 * dt * dey_dt0

        # --- 2) Drift: update gauge coordinates using half-updated momenta ---
        ax_mid = ax0 + dt * ex_half
        ay_mid = ay0 + dt * ey_half

        # --- 3) Midpoint update for psi in the new gauge background ---
        # k1: derivative at (psi0, A_mid)
        k1_psi = self.rhs_scalar(psi0, ax_mid, ay_mid)
        psi_mid = psi0 + 0.5 * dt * k1_psi

        # k2: derivative at mid-step
        k2_psi = self.rhs_scalar(psi_mid, ax_mid, ay_mid)
        psi_new = psi0 + dt * k2_psi

        # --- 4) Second half-kick for gauge momenta using updated psi ---
        _, _, dex_dt1, dey_dt1 = self.rhs_gauge(ax_mid, ay_mid, ex_half, ey_half, psi_new)
        ex_new = ex_half + 0.5 * dt * dex_dt1
        ey_new = ey_half + 0.5 * dt * dey_dt1

        # Commit updates
        self.psi = psi_new
        self.ax, self.ay = ax_mid, ay_mid
        self.ex, self.ey = ex_new, ey_new
    
    # ---------------- energy diagnostic ----------------
    
    def total_energy(self):
        xp = self.xp
        
        rho = xp.abs(self.psi)**2
        phi = self.solve_defrag_potential(rho)
        
        E_scalar_kin = 0.5 * xp.sum(self.covariant_gradient_sq(self.psi, self.ax, self.ay))
        E_defrag = 0.5 * self.g_defrag * xp.sum(rho * phi)
        
        E_gauge_elec = 0.5 * xp.sum(self.ex**2 + self.ey**2)
        
        grad_ax = xp.gradient(self.ax, self.dx)
        grad_ay = xp.gradient(self.ay, self.dx)
        grad_sq = sum(g**2 for g in grad_ax) + sum(g**2 for g in grad_ay)
        E_gauge_mag = 0.5 * (self.c**2) * xp.sum(grad_sq)
        
        E_tot = E_scalar_kin + E_defrag + E_gauge_elec + E_gauge_mag
        
        return float(E_tot.get()) if self.is_gpu else float(E_tot)
    
    def to_numpy(self):
        """Convert to numpy."""
        if self.is_gpu:
            return (cp.asnumpy(self.psi), cp.asnumpy(self.ax), 
                    cp.asnumpy(self.ay), cp.asnumpy(self.ex), cp.asnumpy(self.ey))
        return (np.array(self.psi), np.array(self.ax), 
                np.array(self.ay), np.array(self.ex), np.array(self.ey))


# ---------------- initialization helpers ----------------

def init_scalar_noise(sim: CoupledScalarGauge2D, amp: float = 0.1):
    xp = sim.xp
    rs = xp.random.RandomState(1234)
    noise_real = rs.randn(sim.L, sim.L)
    noise_imag = rs.randn(sim.L, sim.L)
    sim.psi = amp * (noise_real + 1j * noise_imag)
    print(f"Seeded scalar with complex noise, amp={amp}")


# ---------------- main driver ----------------

def run_coupled_sim(L, dx, dt, c, q, g_defrag, amp_scalar, n_steps, sample_interval, out_prefix, use_gpu):
    out_dir = Path(f"{out_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sim = CoupledScalarGauge2D(L=L, dx=dx, dt=dt, c=c, q=q, g_defrag=g_defrag, use_gpu=use_gpu)
    
    print("=" * 72)
    print(" Coupled Scalar-Gauge Simulation ")
    print("=" * 72)
    print(f"L={L}, dx={dx}, dt={dt}, c={c}, q={q}, g_defrag={g_defrag}")
    print(f"amp_scalar={amp_scalar}, n_steps={n_steps}, sample_interval={sample_interval}")
    print("=" * 72)
    
    init_scalar_noise(sim, amp=amp_scalar)
    E0 = sim.total_energy()
    print(f"[INIT] E0 = {E0:.6e}")
    
    times = []
    energies = []
    
    for step in range(n_steps + 1):
        t = step * dt
        
        if step % sample_interval == 0:
            Etot = sim.total_energy()
            times.append(t)
            energies.append(Etot)
            print(f"[DIAG] step={step:6d}, t={t:8.3f}, E={Etot:.6e}")
            
            psi_np, ax_np, ay_np, ex_np, ey_np = sim.to_numpy()
            rho = np.abs(psi_np)**2
            gauge_mag = np.sqrt(ax_np**2 + ay_np**2)
            
            snap_path = out_dir / f"{out_prefix}_snap_{step:06d}.npz"
            np.savez(
                snap_path,
                psi=psi_np,
                rho=rho,
                ax=ax_np,
                ay=ay_np,
                ex=ex_np,
                ey=ey_np,
                gauge_mag=gauge_mag,
                time=t,
            )
        
        if step < n_steps:
            # use symplectic integrator
            sim.step_symplectic()
    
    times = np.array(times)
    energies = np.array(energies)
    
    csv_path = out_dir / f"{out_prefix}_energies.csv"
    np.savetxt(csv_path, np.column_stack([times, energies]), delimiter=",", header="time,energy", comments="")
    print(f"[RESULT] Saved energies -> {csv_path}")
    
    plt.figure(figsize=(8, 4))
    plt.plot(times, energies)
    plt.xlabel("time")
    plt.ylabel("Total energy")
    plt.title("Coupled System Energy")
    plt.tight_layout()
    png_path = out_dir / f"{out_prefix}_energies.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"[RESULT] Saved energy plot -> {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Coupled scalar-gauge test.")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--g_defrag", type=float, default=0.5)
    parser.add_argument("--amp_scalar", type=float, default=0.05)
    parser.add_argument("--n_steps", type=int, default=4000)
    parser.add_argument("--sample_interval", type=int, default=20)
    parser.add_argument("--out_prefix", type=str, default="coupled")
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    
    run_coupled_sim(
        L=args.L, dx=args.dx, dt=args.dt, c=args.c, q=args.q,
        g_defrag=args.g_defrag, amp_scalar=args.amp_scalar,
        n_steps=args.n_steps, sample_interval=args.sample_interval,
        out_prefix=args.out_prefix, use_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
