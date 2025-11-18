#!/usr/bin/env python3
"""
scalar_field_defrag_gpu.py

GPU-accelerated quantum scalar field evolution with emergent defrag gravity.

Now upgraded to act as a small "measurement lab":

- Core simulator: ScalarFieldDefragGPU (unchanged API)
- Global diagnostics: energies, norm, localization, power spectrum
- Snapshot outputs:
    * PNGs: density |psi|^2, defrag potential Phi, phase arg(psi)
    * NPZs: psi, phi, and key parameters (dx, dy, v, lambda_param, g_defrag)
- Proton suite:
    * Evolve uniform noise into structure
    * Detect proton-like lumps per snapshot
    * Save lump catalog (positions, masses, radii, etc.) to CSV
    * Compute radial "pressure" profile for the best proton-like lump

This gives you a consistent substrate to probe:
- "Strong-like" behavior: lump morphology, binding / clustering
- "Weak-like" hooks: future tunneling / identity-change observables
- Proton pressure: radial mechanical proxy from the energy density.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi

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


# -----------------------------------------------------------------------------
# Core simulator
# -----------------------------------------------------------------------------

class ScalarFieldDefragGPU:
    """
    2D complex scalar field with emergent defrag gravity.
    GPU-accelerated via CuPy where available.

    Evolution equation (natural units, m = 1):

        i ∂ψ/∂t = [H_substrate + g·Φ_defrag] ψ

    where:
        H_substrate = -∇²/(2m) + V(ψ)      # Substrate Hamiltonian
        V(ψ) = λ (|ψ|² - v²)²              # Mexican-hat self-interaction
        Φ_defrag solves: ∇²Φ = |ψ|² - ⟨|ψ|²⟩
        g = g_defrag                       # coupling strength
    """

    def __init__(self,
                 L: int = 64,
                 dx: float = 1.0,
                 dt: float = 0.005,
                 g_defrag: float = 0.5,
                 v: float = 1.0,
                 lambda_param: float = 1.0):
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

    # ----------------------------
    # Operators
    # ----------------------------

    def _make_k2(self):
        """Compute k² grid for Fourier space operators."""
        kx = 2 * cp.pi * cp.fft.fftfreq(self.L, d=self.dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(self.L, d=self.dx)
        KX, KY = cp.meshgrid(kx, ky, indexing="ij")
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0  # Avoid division by zero; we will zero this mode manually.
        return k2

    # ----------------------------
    # Initial conditions
    # ----------------------------

    def create_vortex(self,
                      x_center: Optional[int] = None,
                      y_center: Optional[int] = None,
                      r_core: float = 10.0):
        """Create a single vortex configuration."""
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

    def create_uniform_noise(self,
                             mean: float = 1.0,
                             noise_amp: float = 0.05,
                             seed: int = 42):
        """Create uniform state with small complex noise."""
        if GPU_AVAILABLE:
            cp.random.seed(seed)
            psi = mean * cp.ones((self.L, self.L), dtype=complex)
            psi += noise_amp * (
                cp.random.randn(self.L, self.L)
                + 1j * cp.random.randn(self.L, self.L)
            )
        else:
            np.random.seed(seed)
            psi = mean * cp.ones((self.L, self.L), dtype=complex)
            psi += noise_amp * (
                cp.random.randn(self.L, self.L)
                + 1j * cp.random.randn(self.L, self.L)
            )
        return psi

    # ----------------------------
    # Defrag potential
    # ----------------------------

    def solve_defrag_potential(self, psi):
        """
        Solve ∇²Φ = s where s = |ψ|² - ⟨|ψ|²⟩

        This is the defrag principle: potential from density fluctuations.
        Returns Φ with -1/r structure (up to lattice effects).
        """
        rho = cp.abs(psi) ** 2
        s = rho - cp.mean(rho)

        # Solve in Fourier space: Φ_k = -s_k / k²
        s_k = cpfft.fft2(s)
        Phi_k = -s_k / self.k2
        Phi_k[0, 0] = 0.0  # Zero mean mode

        Phi = cp.real(cpfft.ifft2(Phi_k))
        return Phi

    # ----------------------------
    # Hamiltonian pieces
    # ----------------------------

    def kinetic_term(self, psi):
        """
        Kinetic energy operator: -∇²ψ/(2m), with m=1.

        Implemented in Fourier space.
        """
        psi_k = cpfft.fft2(psi)
        lap_psi_k = -self.k2 * psi_k / 2.0  # m = 1
        lap_psi = cpfft.ifft2(lap_psi_k)
        # The Schrödinger term is -(1/2)∇² ψ, so kinetic_term returns that.
        return -lap_psi

    def potential_term(self, psi):
        """
        Mexican-hat potential: V = λ(|ψ|² - v²)²

        Returns the action of V on ψ, i.e. V(ψ) ψ.
        """
        rho = cp.abs(psi) ** 2
        V = self.lambda_param * (rho - self.v**2) ** 2
        return V * psi

    def substrate_hamiltonian(self, psi):
        """H_substrate ψ = [-∇²/(2m) + V(ψ)] ψ"""
        return self.kinetic_term(psi) + self.potential_term(psi)

    def total_hamiltonian(self, psi, Phi_defrag):
        """H_total ψ = [H_substrate + g·Φ_defrag] ψ"""
        H_sub = self.substrate_hamiltonian(psi)
        H_defrag = self.g_defrag * Phi_defrag * psi
        return H_sub + H_defrag

    # ----------------------------
    # Time integration (RK4)
    # ----------------------------

    def evolve_step_rk4(self, psi):
        """
        RK4 time integration of i ∂ψ/∂t = H ψ

        Self-consistent: Φ is recomputed from ψ at each RK stage.
        """
        dt = self.dt

        # Stage 1
        Phi1 = self.solve_defrag_potential(psi)
        k1 = -1j * self.total_hamiltonian(psi, Phi1)

        # Stage 2
        psi2 = psi + 0.5 * dt * k1
        Phi2 = self.solve_defrag_potential(psi2)
        k2 = -1j * self.total_hamiltonian(psi2, Phi2)

        # Stage 3
        psi3 = psi + 0.5 * dt * k2
        Phi3 = self.solve_defrag_potential(psi3)
        k3 = -1j * self.total_hamiltonian(psi3, Phi3)

        # Stage 4
        psi4 = psi + dt * k3
        Phi4 = self.solve_defrag_potential(psi4)
        k4 = -1j * self.total_hamiltonian(psi4, Phi4)

        # Combine
        psi_new = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Renormalize to keep ∫|ψ|² approximately constant
        norm_old = cp.sum(cp.abs(psi) ** 2) * self.dx**2
        norm_new = cp.sum(cp.abs(psi_new) ** 2) * self.dx**2
        psi_new *= cp.sqrt(norm_old / norm_new)

        return psi_new, Phi4

    # ----------------------------
    # Diagnostics (global)
    # ----------------------------

    def compute_diagnostics(self, psi, Phi):
        """
        Compute global diagnostics:
        - E_substrate
        - E_defrag
        - E_total
        - participation_ratio
        - max_rho, var_rho
        - norm
        """
        rho = cp.abs(psi) ** 2
        s = rho - cp.mean(rho)

        # Substrate energy: <ψ|H_sub|ψ>
        H_sub_psi = self.substrate_hamiltonian(psi)
        E_substrate = cp.real(cp.sum(cp.conj(psi) * H_sub_psi) * self.dx**2)

        # Defrag energy (binding): 0.5 ∑ s Φ dx^2
        E_defrag = 0.5 * cp.sum(Phi * s) * self.dx**2

        # Total energy
        E_total = E_substrate + self.g_defrag * E_defrag

        # Localization measures
        participation_ratio = 1.0 / (cp.sum(rho**2) * self.dx**2)
        max_rho = cp.max(rho)
        var_rho = cp.var(rho)

        # Norm (∫|ψ|² dx)
        norm = cp.sum(rho) * self.dx**2

        # Move to CPU scalars
        def to_cpu(x):
            return float(cp.asnumpy(x)) if GPU_AVAILABLE else float(x)

        return {
            "E_substrate": to_cpu(E_substrate),
            "E_defrag": to_cpu(E_defrag),
            "E_total": to_cpu(E_total),
            "participation_ratio": to_cpu(participation_ratio),
            "max_rho": to_cpu(max_rho),
            "var_rho": to_cpu(var_rho),
            "norm": to_cpu(norm),
        }

    def compute_power_spectrum(self, psi):
        """
        Compute power spectrum P(k) = |δρ_k|² with δρ = |ψ|² - ⟨|ψ|²⟩.
        Returns:
            k_radial, P_radial (azimuthally averaged)
        """
        rho = cp.abs(psi) ** 2
        s = rho - cp.mean(rho)

        s_k = cpfft.fft2(s)
        P_k = cp.abs(s_k) ** 2

        kx = 2 * cp.pi * cp.fft.fftfreq(self.L, d=self.dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(self.L, d=self.dx)
        KX, KY = cp.meshgrid(kx, ky, indexing="ij")
        k_mag = cp.sqrt(KX**2 + KY**2)

        n_bins = 20
        k_bins = cp.linspace(0, cp.max(k_mag), n_bins + 1)
        P_radial = cp.zeros(n_bins)
        k_radial = cp.zeros(n_bins)

        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            if cp.sum(mask) > 0:
                P_radial[i] = cp.mean(P_k[mask])
                k_radial[i] = cp.mean(k_mag[mask])

        if GPU_AVAILABLE:
            k_radial = cp.asnumpy(k_radial)
            P_radial = cp.asnumpy(P_radial)

        return k_radial, P_radial

    # ----------------------------
    # Snapshot saving (PNG + NPZ)
    # ----------------------------

    def save_snapshot(self, psi, Phi, step: int, output_dir: Path):
        """
        Save visualization AND raw fields for this step.

        Outputs:
            snapshot_<step>.png   (density, Phi, phase)
            snapshot_<step>.npz   (psi, phi, dx, dy, v, lambda_param, g_defrag)
        """
        # Transfer to CPU
        if GPU_AVAILABLE:
            psi_cpu = cp.asnumpy(psi)
            Phi_cpu = cp.asnumpy(Phi)
        else:
            psi_cpu = psi
            Phi_cpu = Phi

        # --- Raw NPZ (for proton pressure / strong / weak analysis) ---
        np.savez(
            output_dir / f"snapshot_{step:05d}.npz",
            psi=psi_cpu,
            phi=Phi_cpu,
            dx=self.dx,
            dy=self.dx,
            v=self.v,
            lambda_param=self.lambda_param,
            g_defrag=self.g_defrag,
        )

        # --- PNG visualization ---
        rho = np.abs(psi_cpu) ** 2
        phase = np.angle(psi_cpu)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Density
        im0 = axes[0].imshow(
            rho,
            cmap="hot",
            origin="lower",
            extent=[0, self.L * self.dx, 0, self.L * self.dx],
        )
        axes[0].set_title(f"Density |ψ|² (step {step})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # Defrag potential
        im1 = axes[1].imshow(
            Phi_cpu,
            cmap="RdBu_r",
            origin="lower",
            extent=[0, self.L * self.dx, 0, self.L * self.dx],
        )
        axes[1].set_title("Defrag Potential Φ")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Phase
        im2 = axes[2].imshow(
            phase,
            cmap="twilight",
            origin="lower",
            extent=[0, self.L * self.dx, 0, self.L * self.dx],
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[2].set_title("Phase arg(ψ)")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        fig_path = output_dir / f"snapshot_{step:05d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------
    # Plotting diagnostics
    # ----------------------------

    @staticmethod
    def plot_diagnostics(df: pd.DataFrame, output_dir: Path):
        """Create diagnostic plots from the diagnostics DataFrame."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Energy components
        axes[0, 0].plot(df["time"], df["E_substrate"], label="E_substrate")
        axes[0, 0].plot(df["time"], df["E_defrag"], label="E_defrag")
        axes[0, 0].plot(
            df["time"], df["E_total"], label="E_total", linestyle="--"
        )
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Energy")
        axes[0, 0].set_title("Energy Components")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Binding energy
        axes[0, 1].plot(df["time"], df["E_defrag"])
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("E_defrag")
        axes[0, 1].set_title("Defrag Binding Energy")
        axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)

        # Max density
        axes[0, 2].plot(df["time"], df["max_rho"])
        axes[0, 2].set_xlabel("Time")
        axes[0, 2].set_ylabel("max(ρ)")
        axes[0, 2].set_title("Maximum Density")
        axes[0, 2].grid(True, alpha=0.3)

        # Variance
        axes[1, 0].plot(df["time"], df["var_rho"])
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Var(ρ)")
        axes[1, 0].set_title("Density Variance")
        axes[1, 0].grid(True, alpha=0.3)

        # Participation ratio
        axes[1, 1].plot(df["time"], df["participation_ratio"])
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("PR")
        axes[1, 1].set_title("Participation Ratio")
        axes[1, 1].grid(True, alpha=0.3)

        # Norm conservation
        axes[1, 2].plot(df["time"], df["norm"])
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("∫|ψ|²")
        axes[1, 2].set_title("Norm Conservation")
        axes[1, 2].axhline(
            y=df["norm"].iloc[0],
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Initial",
        )
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        diag_path = output_dir / "diagnostics.png"
        plt.savefig(diag_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved diagnostics plot to {diag_path}")

    # ----------------------------
    # Generic evolution driver
    # ----------------------------

    def run_evolution(self,
                      psi_init,
                      n_steps: int = 500,
                      snapshot_interval: int = 50,
                      output_dir: str = "output_defrag",
                      save_snapshots: bool = True) -> pd.DataFrame:
        """
        Run self-consistent evolution with diagnostics.

        Returns:
            diagnostics DataFrame.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        psi = psi_init.copy()
        diagnostics: List[Dict[str, float]] = []

        print("\n[RUN] Starting evolution...")
        print(f"  Steps: {n_steps}")
        print(f"  Snapshot interval: {snapshot_interval}")

        for step in range(n_steps + 1):
            if step % snapshot_interval == 0 or step == n_steps:
                Phi = self.solve_defrag_potential(psi)
                diag = self.compute_diagnostics(psi, Phi)
                diag["step"] = step
                diag["time"] = step * self.dt
                diagnostics.append(diag)

                print(
                    f"[STEP {step:4d}] E_defrag={diag['E_defrag']:+.4e}, "
                    f"max_ρ={diag['max_rho']:.4f}, var={diag['var_rho']:.4e}"
                )

                if save_snapshots:
                    self.save_snapshot(psi, Phi, step, output_dir)

            if step < n_steps:
                psi, _ = self.evolve_step_rk4(psi)

        df = pd.DataFrame(diagnostics)
        csv_path = output_dir / "diagnostics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved diagnostics to {csv_path}")

        self.plot_diagnostics(df, output_dir)
        return df


# -----------------------------------------------------------------------------
# Lump detection and proton pressure analysis (CPU-side)
# -----------------------------------------------------------------------------

@dataclass
class LumpRecord:
    frame_index: int
    time: float
    id_local: int
    x: float
    y: float
    radius: float
    mass: float
    peak_rho: float
    mean_rho: float
    area: float
    n_pixels: int


def find_lumps_in_density(
    rho: np.ndarray,
    dx: float,
    frame_index: int,
    time: float,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
) -> List[LumpRecord]:
    """
    Detect localized high-density lumps in a 2D density field.

    Args:
        rho: 2D array of |psi|^2 (on CPU)
        dx:  lattice spacing
        frame_index, time: identifiers
        sigma_threshold: mask = rho > mean + sigma_threshold*std
        min_pixels: min connected pixels per lump

    Returns:
        List of LumpRecord.
    """
    assert rho.ndim == 2, "rho must be 2D"
    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, n_labels = ndi.label(mask, structure=structure)

    lumps: List[LumpRecord] = []
    voxel_area = dx * dx

    for lab in range(1, n_labels + 1):
        region = (labeled == lab)
        n_pix = int(region.sum())
        if n_pix < min_pixels:
            continue

        ys, xs = np.nonzero(region)
        x_mean = xs.mean()
        y_mean = ys.mean()

        # RMS radius
        x_phys = (xs - x_mean) * dx
        y_phys = (ys - y_mean) * dx
        r = np.sqrt(x_phys**2 + y_phys**2)
        radius = float(np.sqrt(np.mean(r**2)))

        rho_region = rho[region]
        mass = float(rho_region.sum() * voxel_area)
        peak = float(rho_region.max())
        mean = float(rho_region.mean())
        area = float(n_pix * voxel_area)

        lumps.append(
            LumpRecord(
                frame_index=frame_index,
                time=time,
                id_local=lab,
                x=float(x_mean * dx),
                y=float(y_mean * dx),
                radius=radius,
                mass=mass,
                peak_rho=peak,
                mean_rho=mean,
                area=area,
                n_pixels=n_pix,
            )
        )

    return lumps


def find_lump_center_from_rho(rho: np.ndarray) -> Tuple[float, float]:
    """
    Simple center estimate: start from max pixel, refine with centroid in small window.

    Returns center in index coordinates (cx, cy).
    """
    ny, nx = rho.shape
    max_idx = np.unravel_index(np.argmax(rho), rho.shape)
    y0, x0 = max_idx

    window = 5
    half = window // 2
    x_min = max(0, x0 - half)
    x_max = min(nx, x0 + half + 1)
    y_min = max(0, y0 - half)
    y_max = min(ny, y0 + half + 1)

    sub_rho = rho[y_min:y_max, x_min:x_max]
    ys, xs = np.indices(sub_rho.shape)

    total = sub_rho.sum()
    if total <= 0:
        return float(x0), float(y0)

    cx_local = (xs * sub_rho).sum() / total
    cy_local = (ys * sub_rho).sum() / total

    cx = x_min + cx_local
    cy = y_min + cy_local
    return float(cx), float(cy)


def compute_local_energies(
    psi: np.ndarray,
    phi: np.ndarray,
    dx: float,
    dy: float,
    v: float,
    lambda_param: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local energy densities on CPU grid:

        e_grad   = 0.5 |∇ψ|^2
        e_pot    = λ (|ψ|^2 - v^2)^2
        e_defrag = 0.5 ρ φ    with ρ = |ψ|^2
        p_local  = e_grad - (e_pot + e_defrag)

    Returns:
        e_grad, e_pot, e_defrag, p_local
    """
    rho = np.abs(psi) ** 2

    # Gradients via np.gradient
    dpsi_dy, dpsi_dx = np.gradient(psi, dy, dx, edge_order=2)
    grad_sq = np.abs(dpsi_dx) ** 2 + np.abs(dpsi_dy) ** 2
    e_grad = 0.5 * grad_sq

    e_pot = lambda_param * (rho - v**2) ** 2
    e_defrag = 0.5 * rho * phi
    p_local = e_grad - (e_pot + e_defrag)

    return e_grad, e_pot, e_defrag, p_local


def radial_profile(
    field: np.ndarray,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    nbins: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radial profile of a 2D field around (cx, cy) in index space.

    Returns:
        r_centers, f_mean, f_std, n_points
    """
    ny, nx = field.shape
    y_indices, x_indices = np.indices(field.shape)

    x_phys = (x_indices - cx) * dx
    y_phys = (y_indices - cy) * dy
    r = np.sqrt(x_phys**2 + y_phys**2).ravel()
    values = field.ravel()

    r_max = r.max()
    nbins = max(1, nbins)

    bin_edges = np.linspace(0.0, r_max, nbins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    f_mean = np.zeros(nbins, dtype=float)
    f_std = np.zeros(nbins, dtype=float)
    n_points = np.zeros(nbins, dtype=int)

    bin_indices = np.digitize(r, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    for b in range(nbins):
        mask = bin_indices == b
        if not np.any(mask):
            f_mean[b] = np.nan
            f_std[b] = np.nan
            n_points[b] = 0
        else:
            vals = values[mask]
            f_mean[b] = np.mean(vals)
            f_std[b] = np.std(vals)
            n_points[b] = int(mask.sum())

    return r_centers, f_mean, f_std, n_points


def save_profile_csv(
    path: Path,
    r_centers: np.ndarray,
    f_mean: np.ndarray,
    f_std: np.ndarray,
    n_points: np.ndarray,
):
    with path.open("w") as f:
        f.write("r,f_mean,f_std,n_points\n")
        for r, m, s, n in zip(r_centers, f_mean, f_std, n_points):
            f.write(f"{r},{m},{s},{n}\n")


def plot_pressure_profile(
    out_path: Path,
    r: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    title: str = "Radial Pressure Profile (proton-like lump)",
):
    mask = np.isfinite(p_mean)
    r_plot = r[mask]
    p_plot = p_mean[mask]
    e_plot = p_std[mask]

    plt.figure()
    plt.errorbar(r_plot, p_plot, yerr=e_plot, fmt="o-", capsize=3)
    plt.xlabel("r (physical units)")
    plt.ylabel("pressure proxy p(r)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved pressure profile plot to {out_path}")


# -----------------------------------------------------------------------------
# Proton suite driver
# -----------------------------------------------------------------------------

def run_proton_suite(
    L: int = 128,
    dx: float = 1.0,
    dt: float = 0.005,
    g_defrag: float = 1.0,
    v: float = 1.0,
    lambda_param: float = 0.5,
    n_steps: int = 2000,
    snapshot_interval: int = 50,
    init_mean: float = 1.0,
    init_noise_amp: float = 0.1,
    init_seed: int = 42,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
    nbins_pressure: int = 50,
    output_dir: str = "proton_suite_output",
):
    """
    Main "proton suite":

    1. Evolve uniform noise into structure (ScalarFieldDefragGPU).
    2. At each snapshot:
       - compute global diagnostics
       - save PNG + NPZ snapshot
       - detect lumps and append to a lump catalog
    3. After the run:
       - save diagnostics.csv, diagnostics.png
       - save lumps.csv (strong-force / clustering handle)
       - pick the heaviest lump seen and compute a radial pressure profile
         from the final NPZ snapshot it appears in.

    This gives you:
        - global thermodynamics
        - lump phenomenology (proto-strong sector)
        - proton pressure profile (massive internal stress).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" PROTON SUITE – ScalarFieldDefragGPU ")
    print("=" * 70)
    print(f"L={L}, dx={dx}, dt={dt}")
    print(f"g_defrag={g_defrag}, v={v}, lambda={lambda_param}")
    print(f"n_steps={n_steps}, snapshot_interval={snapshot_interval}")
    print(f"init_mean={init_mean}, init_noise_amp={init_noise_amp}, seed={init_seed}")
    print(f"sigma_threshold={sigma_threshold}, min_pixels={min_pixels}")
    print("=" * 70)

    sim = ScalarFieldDefragGPU(
        L=L,
        dx=dx,
        dt=dt,
        g_defrag=g_defrag,
        v=v,
        lambda_param=lambda_param,
    )

    psi = sim.create_uniform_noise(
        mean=init_mean,
        noise_amp=init_noise_amp,
        seed=init_seed,
    )

    diagnostics: List[Dict[str, float]] = []
    all_lumps: List[LumpRecord] = []

    best_lump: Optional[LumpRecord] = None
    best_lump_snapshot_step: Optional[int] = None

    for step in range(n_steps + 1):
        t = step * dt

        if step % snapshot_interval == 0 or step == n_steps:
            Phi = sim.solve_defrag_potential(psi)
            diag = sim.compute_diagnostics(psi, Phi)
            diag["step"] = step
            diag["time"] = t
            diagnostics.append(diag)

            print(
                f"[SNAP] step={step:5d}, t={t:.3f}, "
                f"E_defrag={diag['E_defrag']:+.4e}, "
                f"max_ρ={diag['max_rho']:.4f}"
            )

            # Save snapshot (PNG + NPZ)
            sim.save_snapshot(psi, Phi, step, out_dir)

            # Lump detection
            if GPU_AVAILABLE:
                psi_cpu = cp.asnumpy(psi)
            else:
                psi_cpu = psi
            rho = np.abs(psi_cpu) ** 2

            lumps = find_lumps_in_density(
                rho,
                dx=dx,
                frame_index=step,
                time=t,
                sigma_threshold=sigma_threshold,
                min_pixels=min_pixels,
            )
            all_lumps.extend(lumps)
            print(f"        → found {len(lumps)} lumps")

            # Track the heaviest lump seen so far (for pressure profile)
            if lumps:
                local_best = max(lumps, key=lambda L_: L_.mass)
                if best_lump is None or local_best.mass > best_lump.mass:
                    best_lump = local_best
                    best_lump_snapshot_step = step

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    # ---- Save diagnostics ----
    df_diag = pd.DataFrame(diagnostics)
    diag_csv = out_dir / "diagnostics.csv"
    df_diag.to_csv(diag_csv, index=False)
    print(f"[RESULT] Saved diagnostics to {diag_csv}")
    ScalarFieldDefragGPU.plot_diagnostics(df_diag, out_dir)

    # ---- Save lump catalog ----
    if all_lumps:
        df_lumps = pd.DataFrame([asdict(l) for l in all_lumps])
        lumps_csv = out_dir / "lumps.csv"
        df_lumps.to_csv(lumps_csv, index=False)
        print(f"[RESULT] Saved lump catalog to {lumps_csv}")
    else:
        df_lumps = None
        print("[WARN] No lumps detected during run; lump catalog not created.")

    # ---- Proton pressure profile for the best lump ----
    if best_lump is not None and best_lump_snapshot_step is not None:
        snap_npz = out_dir / f"snapshot_{best_lump_snapshot_step:05d}.npz"
        if not snap_npz.exists():
            print(
                f"[WARN] Best lump snapshot NPZ {snap_npz} not found; "
                f"skipping pressure profile."
            )
        else:
            print(
                f"[INFO] Computing pressure profile from {snap_npz} "
                f"for best lump at step={best_lump_snapshot_step}"
            )
            data = np.load(snap_npz)
            psi_snap = data["psi"]
            phi_snap = data["phi"]
            dx_snap = float(data.get("dx", dx))
            dy_snap = float(data.get("dy", dx))
            v_snap = float(data.get("v", v))
            lambda_snap = float(data.get("lambda_param", lambda_param))

            # local energies and pressure proxy
            _, _, _, p_local = compute_local_energies(
                psi_snap,
                phi_snap,
                dx=dx_snap,
                dy=dy_snap,
                v=v_snap,
                lambda_param=lambda_snap,
            )

            # center: use density-based center around this lump
            rho_snap = np.abs(psi_snap) ** 2
            cx_idx = best_lump.x / dx_snap
            cy_idx = best_lump.y / dy_snap

            r_centers, p_mean, p_std, n_pts = radial_profile(
                p_local,
                cx=cx_idx,
                cy=cy_idx,
                dx=dx_snap,
                dy=dy_snap,
                nbins=nbins_pressure,
            )

            # Save CSV and PNG
            pressure_csv = out_dir / "proton_pressure_profile.csv"
            pressure_png = out_dir / "proton_pressure_profile.png"

            save_profile_csv(pressure_csv, r_centers, p_mean, p_std, n_pts)
            print(f"[RESULT] Saved proton pressure profile CSV to {pressure_csv}")

            plot_pressure_profile(
                pressure_png,
                r_centers,
                p_mean,
                p_std,
                title=(
                    "Radial Pressure Profile (best proton-like lump)\n"
                    f"step={best_lump_snapshot_step}, mass={best_lump.mass:.3e}"
                ),
            )
    else:
        print("[WARN] No best lump identified; no pressure profile computed.")

    print("=" * 70)
    print(" Proton suite complete ")
    print(f" Results in: {out_dir}")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Old demo mode (kept for convenience)
# -----------------------------------------------------------------------------

def run_demo_tests():
    """
    Original demo:
    1. Static vortex binding energy + short evolution.
    2. Noise → structure formation (with basic diagnostics).
    """
    print("=" * 70)
    print("SCALAR FIELD + DEFRAG GRAVITY (GPU-ACCELERATED) – DEMO MODE")
    print("=" * 70)

    sim = ScalarFieldDefragGPU(
        L=64,
        dx=1.0,
        dt=0.005,
        g_defrag=0.3,
        v=1.0,
        lambda_param=0.5,
    )

    # ----- TEST 1: Static vortex -----
    print("\n" + "=" * 70)
    print("TEST 1: STATIC VORTEX (validate binding energy)")
    print("=" * 70)

    psi_vortex = sim.create_vortex(r_core=8.0)
    Phi_vortex = sim.solve_defrag_potential(psi_vortex)

    rho = cp.abs(psi_vortex) ** 2
    s = rho - cp.mean(rho)
    E_bind = 0.5 * cp.sum(Phi_vortex * s) * sim.dx**2
    E_bind_cpu = float(cp.asnumpy(E_bind)) if GPU_AVAILABLE else float(E_bind)

    print(f"\nStatic vortex binding energy: {E_bind_cpu:.6e}")
    print("(Should be negative - indicates bound state)")

    print("\nEvolving vortex for 200 steps...")
    sim.run_evolution(
        psi_vortex,
        n_steps=200,
        snapshot_interval=50,
        output_dir="output_defrag_gpu/demo_vortex",
    )

    # ----- TEST 2: Noise → structure -----
    print("\n" + "=" * 70)
    print("TEST 2: UNIFORM NOISE → STRUCTURE FORMATION")
    print("=" * 70)

    psi_noise = sim.create_uniform_noise(mean=1.0, noise_amp=0.1)

    if GPU_AVAILABLE:
        initial_max = float(cp.asnumpy(cp.max(cp.abs(psi_noise) ** 2)))
        initial_var = float(cp.asnumpy(cp.var(cp.abs(psi_noise) ** 2)))
    else:
        initial_max = float(cp.max(cp.abs(psi_noise) ** 2))
        initial_var = float(cp.var(cp.abs(psi_noise) ** 2))

    print("\nInitial state:")
    print(f"  max(ρ) = {initial_max:.4f}")
    print(f"  var(ρ) = {initial_var:.6e}")

    print("\nEvolving noise for 500 steps...")
    df_noise = sim.run_evolution(
        psi_noise,
        n_steps=500,
        snapshot_interval=50,
        output_dir="output_defrag_gpu/demo_noise",
    )

    print("\nFinal state:")
    print(f"  max(ρ) = {df_noise['max_rho'].iloc[-1]:.4f}")
    print(f"  var(ρ) = {df_noise['var_rho'].iloc[-1]:.6e}")
    print(f"  E_defrag = {df_noise['E_defrag'].iloc[-1]:.6e}")

    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print("\nTest 1 (Vortex):")
    print(f"  Initial E_defrag: {df_noise['E_defrag'].iloc[0]:.6e}")
    print(f"  Final E_defrag:   {df_noise['E_defrag'].iloc[-1]:.6e}")
    print("\nTest 2 (Noise → Structure):")
    print(f"  Initial max(ρ): {df_noise['max_rho'].iloc[0]:.4f}")
    print(f"  Final max(ρ):   {df_noise['max_rho'].iloc[-1]:.4f}")
    print(
        f"  Growth factor: "
        f"{df_noise['max_rho'].iloc[-1] / df_noise['max_rho'].iloc[0]:.2f}x"
    )

    print("\nAll demo outputs saved under output_defrag_gpu/")
    print("=" * 70)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ScalarFieldDefragGPU core + proton measurement suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="suite",
        choices=["suite", "demo"],
        help="Run the full proton measurement suite or the original demo tests.",
    )

    # Proton suite CLI params
    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--g_defrag", type=float, default=1.0, help="Defrag coupling.")
    parser.add_argument("--v", type=float, default=1.0, help="Vacuum expectation value.")
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=0.5,
        help="Self-interaction strength in Mexican-hat potential.",
    )
    parser.add_argument("--n_steps", type=int, default=2000, help="Number of time steps.")
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=50,
        help="Snapshot interval (steps) for suite.",
    )
    parser.add_argument(
        "--init_mean", type=float, default=1.0, help="Initial mean amplitude."
    )
    parser.add_argument(
        "--init_noise_amp",
        type=float,
        default=0.1,
        help="Initial noise amplitude around mean.",
    )
    parser.add_argument(
        "--init_seed", type=int, default=42, help="Random seed for initial condition."
    )
    parser.add_argument(
        "--sigma_threshold",
        type=float,
        default=2.0,
        help="Density threshold in units of sigma for lump detection.",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=8,
        help="Minimum pixels in a lump for detection.",
    )
    parser.add_argument(
        "--nbins_pressure",
        type=int,
        default=50,
        help="Radial bins for pressure profile.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="proton_suite_output",
        help="Output directory for suite.",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo_tests()
    else:
        run_proton_suite(
            L=args.L,
            dx=args.dx,
            dt=args.dt,
            g_defrag=args.g_defrag,
            v=args.v,
            lambda_param=args.lambda_param,
            n_steps=args.n_steps,
            snapshot_interval=args.snapshot_interval,
            init_mean=args.init_mean,
            init_noise_amp=args.init_noise_amp,
            init_seed=args.init_seed,
            sigma_threshold=args.sigma_threshold,
            min_pixels=args.min_pixels,
            nbins_pressure=args.nbins_pressure,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
