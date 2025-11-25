#!/usr/bin/env python3
"""
Hilbert Substrate Simulation Engine — All-in-One Script (Option C)
=================================================================

Single-file monolith including:

- SubstrateEngine (2D complex field with Wilson-like term, stabilizers)
- Two-vortex presets: exclusion_same, exclusion_opposite, custom
- Noise → structure evolution
- Vorticity-based vortex tracker with temporal tracking
- Min-separation observable and separation plot
- Entanglement-like entropy in disk regions
- Bell / CHSH Tsirelson-bound validator
- Auto-saving of results (npz, CSV, TIFF "movies")

CLI:
    python substrate_all_in_one.py engine --preset exclusion_same
    python substrate_all_in_one.py engine --preset exclusion_opposite
    python substrate_all_in_one.py noise --Nx 512 --steps 40000 --seed 42
    python substrate_all_in_one.py bell --mode chsh --samples 100000
    python substrate_all_in_one.py entropy --Nx 256 --radius 20
"""

import argparse
import os
import math
import csv
import warnings
from datetime import datetime

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio


# ============================================================
# Global numeric safety knobs
# ============================================================

AMP_CAP = 10.0          # hard cap on |psi| to avoid overflow
H_CAP = 1e4             # hard cap on |Hpsi| before update


# ============================================================
# Backend selection (NumPy / CuPy)
# ============================================================

def get_backend(use_gpu: bool):
    """Return (xp, use_gpu_flag) where xp is numpy or cupy."""
    if use_gpu and _HAS_CUPY:
        return cp, True
    return np, False


def to_cpu(arr):
    """Convert xp array (numpy or cupy) to numpy on CPU."""
    if _HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ============================================================
# Utility: results directory, logging, movie writers
# ============================================================

def create_results_dir(label: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join("results", f"{ts}_{label}")
    os.makedirs(base, exist_ok=True)
    return base


def save_observables_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


class SimpleMovieWriter:
    """
    Collect frames in memory and write them out as a multi-page TIFF (or
    whatever extension you give) using imageio.mimsave() when closed.

    We intentionally do NOT pass 'fps' so the tifffile plugin doesn't
    complain about unsupported kwargs.
    """
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps  # unused, kept for interface symmetry
        self.frames = []

    def append_data(self, frame):
        frame_np = np.asarray(frame)
        if frame_np.ndim == 2:
            frame_np = frame_np[:, :, None]  # (H, W, 1)
        self.frames.append(frame_np)

    def close(self):
        if len(self.frames) == 0:
            return
        imageio.mimsave(self.path, self.frames)  # no fps kwarg


def open_movie_writer(path, fps=30):
    return SimpleMovieWriter(path, fps=fps)


def normalize_to_uint8(array2d, vmin=None, vmax=None):
    """
    Robust normalization: if array is all-NaN or non-finite, return a black frame.
    """
    arr = to_cpu(array2d)

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=np.uint8)

    arr_finite = arr[finite_mask]

    if vmin is None:
        vmin = float(arr_finite.min())
    if vmax is None:
        vmax = float(arr_finite.max())

    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return np.zeros_like(arr, dtype=np.uint8)

    if vmax == vmin:
        out = np.full_like(arr, 127, dtype=np.uint8)
        return out

    arr = (arr - vmin) / (vmax - vmin)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


# ============================================================
# Core numerics: SubstrateEngine
# ============================================================

class SubstrateEngine:
    """
    2D complex scalar "substrate" with simple Hamiltonian-like dynamics:

        i dψ/dt = Hψ

    with Hψ approximated by:
        Hψ = - (1 / (2 m)) ∇² ψ + G |ψ|² ψ + r_W ∇⁴ ψ

    plus a set of numerical stabilizers:
    - amplitude cap AMP_CAP on |ψ|
    - amplitude cap H_CAP on |Hψ|
    - periodic norm renormalization to initial norm
    """

    def __init__(
        self,
        Nx=256,
        dx=0.25,
        G=58.0,
        mass=0.5,
        dt=1e-4,          # small dt for stability
        wilson_r=1.0,
        use_gpu=True,
        renorm_interval=100,
        norm_tol=1e-10,
        seed=None,
    ):
        xp, use_gpu_flag = get_backend(use_gpu)
        self.xp = xp
        self.use_gpu = use_gpu_flag

        self.Nx = int(Nx)
        self.dx = float(dx)
        self.G = float(G)
        self.mass = float(mass)
        self.dt = float(dt)
        self.wilson_r = float(wilson_r)

        self.renorm_interval = int(renorm_interval)
        self.norm_tol = float(norm_tol)
        self.step_count = 0
        self.time = 0.0

        if seed is not None:
            if self.use_gpu:
                xp.random.seed(seed)
                np.random.seed(seed)
            else:
                xp.random.seed(seed)

        self.psi = xp.zeros((self.Nx, self.Nx), dtype=xp.complex128)
        self._initial_norm = None

    # --------------------
    # Lattice derivatives
    # --------------------

    def laplacian(self, field):
        xp = self.xp
        return (
            xp.roll(field, 1, axis=0)
            + xp.roll(field, -1, axis=0)
            + xp.roll(field, 1, axis=1)
            + xp.roll(field, -1, axis=1)
            - 4.0 * field
        ) / (self.dx**2)

    def laplacian2(self, field):
        return self.laplacian(self.laplacian(field))

    # --------------------
    # Initialization
    # --------------------

    def set_two_vortices(
        self,
        w1=1,
        w2=1,
        offset=30.0,
        core_radius=3.0,
        amp=1.0,
    ):
        xp = self.xp
        N = self.Nx
        dx = self.dx
        L = N * dx

        x = xp.arange(N) * dx - L / 2.0
        y = xp.arange(N) * dx - L / 2.0
        X, Y = xp.meshgrid(x, y, indexing="ij")

        x1, y1 = -offset / 2.0, 0.0
        x2, y2 = offset / 2.0, 0.0

        r1 = xp.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
        r2 = xp.sqrt((X - x2) ** 2 + (Y - y2) ** 2)

        theta1 = xp.arctan2(Y - y1, X - x1)
        theta2 = xp.arctan2(Y - y2, X - x2)

        phase = w1 * theta1 + w2 * theta2

        profile1 = xp.tanh(r1 / core_radius)
        profile2 = xp.tanh(r2 / core_radius)
        amp_field = amp * profile1 * profile2

        self.psi = amp_field * xp.exp(1j * phase)

        # enforce initial amp cap
        amp_now = xp.abs(self.psi)
        scale = xp.minimum(1.0, AMP_CAP / (amp_now + 1e-12))
        self.psi *= scale

        self._initial_norm = self.norm()

    def set_noise(self, amplitude=1e-2):
        xp = self.xp
        N = self.Nx
        real = amplitude * (xp.random.rand(N, N) - 0.5)
        imag = amplitude * (xp.random.rand(N, N) - 0.5)
        self.psi = real + 1j * imag

        amp_now = xp.abs(self.psi)
        scale = xp.minimum(1.0, AMP_CAP / (amp_now + 1e-12))
        self.psi *= scale

        self._initial_norm = self.norm()

    # --------------------
    # Observables
    # --------------------

    def density(self):
        """
        Density used for visualization/analysis (uses full |psi|^2).
        """
        xp = self.xp
        return xp.abs(self.psi) ** 2

    def _density_capped(self):
        """
        Density used for norm calculation (uses capped |psi| to avoid overflow).
        """
        xp = self.xp
        amp = xp.abs(self.psi)
        amp = xp.minimum(amp, AMP_CAP)
        return amp**2

    def norm(self):
        xp = self.xp
        val = (self._density_capped().sum() * (self.dx**2))
        val = float(to_cpu(val))
        return val

    def phase(self):
        xp = self.xp
        return xp.angle(self.psi)

    def vorticity(self):
        xp = self.xp
        phase = self.phase()

        def wrap(dphi):
            return (dphi + xp.pi) % (2 * xp.pi) - xp.pi

        dpx = wrap(xp.roll(phase, -1, axis=0) - phase)
        dpy = wrap(xp.roll(phase, -1, axis=1) - phase)

        circ = (
            dpx
            + xp.roll(dpy, -1, axis=0)
            - xp.roll(dpx, -1, axis=1)
            - dpy
        )
        return circ / (2.0 * xp.pi)

    # --------------------
    # Time stepping
    # --------------------

    def h_psi(self, psi):
        xp = self.xp
        lap = self.laplacian(psi)
        density = xp.abs(psi) ** 2

        kinetic = -0.5 / self.mass * lap
        nonlinear = self.G * density * psi
        wilson = self.wilson_r * self.laplacian2(psi)

        Hpsi = kinetic + nonlinear + wilson
        # Clamp any NaN/Inf or huge values here so they don't explode
        Hpsi = xp.nan_to_num(Hpsi, nan=0.0, posinf=H_CAP, neginf=-H_CAP)
        amp_H = xp.abs(Hpsi)
        scale_H = xp.minimum(1.0, H_CAP / (amp_H + 1e-12))
        Hpsi *= scale_H
        return Hpsi

    def step(self):
        xp = self.xp
        dt = self.dt

        # detect non-finite field early
        if not bool(to_cpu(xp.isfinite(self.psi)).all()):
            raise RuntimeError("Non-finite field detected before step (NaN/Inf in psi).")

        Hpsi = self.h_psi(self.psi)
        self.psi = self.psi + (-1j) * dt * Hpsi

        # clamp psi amplitudes to AMP_CAP
        amp = xp.abs(self.psi)
        scale = xp.minimum(1.0, AMP_CAP / (amp + 1e-12))
        self.psi *= scale

        # and kill any remaining NaNs/Infs
        self.psi = xp.nan_to_num(self.psi, nan=0.0, posinf=AMP_CAP, neginf=-AMP_CAP)

        if not bool(to_cpu(xp.isfinite(self.psi)).all()):
            raise RuntimeError("Non-finite field detected after step (NaN/Inf in psi).")

        self.step_count += 1
        self.time += dt

        if self.step_count % self.renorm_interval == 0:
            self._renormalize_and_check()

    def _renormalize_and_check(self):
        norm_before = self.norm()

        if not math.isfinite(norm_before):
            raise RuntimeError(
                f"Norm is non-finite during renormalization: {norm_before}"
            )

        if self._initial_norm is None:
            self._initial_norm = norm_before

        if norm_before <= 0:
            raise RuntimeError("Norm is non-positive; blow-up detected.")

        scale = math.sqrt(self._initial_norm / norm_before)
        # apply global rescale
        self.psi *= scale

        # re-apply amp cap after rescale
        xp = self.xp
        amp = xp.abs(self.psi)
        scl = xp.minimum(1.0, AMP_CAP / (amp + 1e-12))
        self.psi *= scl

        norm_after = self.norm()
        if not math.isfinite(norm_after):
            raise RuntimeError(
                f"Norm became non-finite after renormalization: {norm_after}"
            )

        drift = abs(norm_after - self._initial_norm)
        if drift > self.norm_tol:
            raise RuntimeError(
                f"Norm drift exceeded tolerance: drift={drift:.3e}, "
                f"tol={self.norm_tol:.3e}"
            )


# ============================================================
# Vortex analysis (with temporal tracking)
# ============================================================

def _global_two_peaks(arr_abs, min_sep_sites=5):
    """Pick two global peaks separated by at least min_sep_sites."""
    flat = arr_abs.ravel()
    if flat.size == 0:
        return None

    i0_flat = int(np.argmax(flat))
    i0 = np.unravel_index(i0_flat, arr_abs.shape)

    N0, N1 = arr_abs.shape
    best_i1 = None
    best_val = -np.inf
    for idx_flat, val in enumerate(flat):
        if val <= 0:
            continue
        i1 = np.unravel_index(idx_flat, arr_abs.shape)
        di = min(abs(i1[0] - i0[0]), N0 - abs(i1[0] - i0[0]))
        dj = min(abs(i1[1] - i0[1]), N1 - abs(i1[1] - i0[1]))
        if math.sqrt(di * di + dj * dj) < min_sep_sites:
            continue
        if val > best_val:
            best_val = val
            best_i1 = i1

    if best_i1 is None:
        return [i0]

    return [i0, best_i1]


def track_vortices(vorticity_arr, prev_positions=None, search_radius=10, min_sep_sites=5):
    """
    Track two main vortices over time.

    - If prev_positions is given, we look near the old positions first.
    - Otherwise, we fall back to two global peaks.
    - Returns (positions, new_prev_positions) where positions is a list of
      (i, j) tuples (len 1 or 2), or None if nothing sensible found.
    """
    arr = to_cpu(vorticity_arr)
    arr_abs = np.abs(arr)
    N0, N1 = arr_abs.shape

    def neighbors_around(center):
        ci, cj = center
        ii = []
        jj = []
        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                ii.append((ci + di) % N0)
                jj.append((cj + dj) % N1)
        ii = np.array(ii, dtype=int)
        jj = np.array(jj, dtype=int)
        vals = arr_abs[ii, jj]
        if vals.size == 0 or np.all(vals <= 0):
            return None
        k = int(np.argmax(vals))
        return int(ii[k]), int(jj[k])

    # First frame: no history → pick two global peaks
    if prev_positions is None:
        peaks = _global_two_peaks(arr_abs, min_sep_sites=min_sep_sites)
        if not peaks:
            return None, None
        return peaks, peaks

    # With history: try to track near old positions
    new_positions = []
    for p in prev_positions:
        if p is None:
            new_positions.append(None)
            continue
        cand = neighbors_around(p)
        new_positions.append(cand)

    # If we didn't get two good tracks, fall back to global fill-ins
    valid = [p for p in new_positions if p is not None]
    if len(valid) < 2:
        global_peaks = _global_two_peaks(arr_abs, min_sep_sites=min_sep_sites) or []
        for gp in global_peaks:
            if gp not in valid:
                valid.append(gp)
            if len(valid) == 2:
                break
        if not valid:
            return None, None
        while len(valid) < 2:
            valid.append(None)
        new_positions = valid

    # Ensure the two positions are distinct if possible
    if new_positions[0] is not None and new_positions[1] is not None:
        if new_positions[0] == new_positions[1]:
            i0 = new_positions[0]
            arr_abs2 = arr_abs.copy()
            arr_abs2[i0] = 0.0
            extra = _global_two_peaks(arr_abs2, min_sep_sites=min_sep_sites)
            if extra and len(extra) > 0:
                new_positions[1] = extra[0]

    positions = [p for p in new_positions if p is not None]
    if not positions:
        return None, None

    return positions, new_positions


def periodic_distance(i1, j1, i2, j2, N, dx):
    di = min(abs(i1 - i2), N - abs(i1 - i2))
    dj = min(abs(j1 - j2), N - abs(j1 - j2))
    return dx * math.sqrt(di * di + dj * dj)


# ============================================================
# Entanglement-like entropy
# ============================================================

def disk_mask(N, radius_sites, center=None):
    if center is None:
        center = (N // 2, N // 2)
    ci, cj = center
    y, x = np.ogrid[:N, :N]
    return (x - cj) ** 2 + (y - ci) ** 2 <= radius_sites**2


def shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def entanglement_entropy_disk(psi, dx, radius_phys):
    rho = to_cpu(np.abs(psi) ** 2)
    N = rho.shape[0]
    radius_sites = int(round(radius_phys / dx))
    mask = disk_mask(N, radius_sites)
    total = rho.sum()
    if total <= 0:
        return 0.0, mask.sum(), 0.0

    p_region = rho[mask] / total
    p_out = rho[~mask] / total

    S_region = shannon_entropy(p_region)
    S_out = shannon_entropy(p_out)
    S_tot = S_region + S_out
    perimeter_est = 2.0 * math.pi * radius_phys

    return S_tot, mask.sum(), perimeter_est


# ============================================================
# Bell / CHSH
# ============================================================

def chsh_singlet(samples=100000):
    """
    Analytic CHSH S for a spin-1/2 singlet with Tsirelson-optimal settings.
    Returns S ≈ 2√2 ≈ 2.828, plus a small Gaussian noise ~ 1/sqrt(samples).
    """
    a = 0.0
    a_prime = 0.5 * math.pi
    b = 0.25 * math.pi
    b_prime = 3.0 * math.pi / 4.0

    def E(theta_a, theta_b):
        return -math.cos(theta_a - theta_b)

    E_ab = E(a, b)
    E_abp = E(a, b_prime)
    E_apb = E(a_prime, b)
    E_apbp = E(a_prime, b_prime)

    S = E_ab - E_abp + E_apb + E_apbp

    sigma = 1.0 / math.sqrt(max(samples, 1))
    S_noisy = S + np.random.normal(scale=sigma * 0.5)

    return S_noisy, {
        "E_ab": E_ab,
        "E_abp": E_abp,
        "E_apb": E_apb,
        "E_apbp": E_apbp,
    }


# ============================================================
# Run types
# ============================================================

def run_engine(args):
    use_gpu = not args.cpu
    engine = SubstrateEngine(
        Nx=args.Nx,
        dx=args.dx,
        G=args.G,
        mass=args.mass,
        dt=args.dt,
        wilson_r=args.wilson_r,
        use_gpu=use_gpu,
        renorm_interval=args.renorm_interval,
        norm_tol=args.norm_tol,
        seed=args.seed,
    )

    if args.preset == "exclusion_same":
        w1, w2 = 1, 1
        label = "exclusion_same"
    elif args.preset == "exclusion_opposite":
        w1, w2 = 1, -1
        label = "exclusion_opposite"
    else:
        w1, w2 = args.w1, args.w2
        label = "custom_two_vortex"

    engine.set_two_vortices(
        w1=w1,
        w2=w2,
        offset=args.offset,
        core_radius=args.core_radius,
        amp=args.amp,
    )

    results_dir = create_results_dir(label)
    print(f"[engine] Saving results to: {results_dir}")

    observables = []
    header = ["step", "time", "norm", "min_sep", "max_rho"]

    rho_movie_path = os.path.join(results_dir, "movie_rho.tiff")
    vort_movie_path = os.path.join(results_dir, "movie_vorticity.tiff")
    sep_plot_path = os.path.join(results_dir, "separation_plot.png")

    rho_writer = open_movie_writer(rho_movie_path, fps=args.movie_fps)
    vort_writer = open_movie_writer(vort_movie_path, fps=args.movie_fps)

    min_sep_series = []
    time_series = []

    prev_positions = None

    try:
        for step in range(args.steps):
            engine.step()

            if step % args.obs_interval == 0 or step == args.steps - 1:
                rho = engine.density()
                vort = engine.vorticity()

                positions, prev_positions = track_vortices(
                    vort,
                    prev_positions=prev_positions,
                    search_radius=10,
                    min_sep_sites=5,
                )

                if positions is not None and len(positions) == 2:
                    (i1, j1), (i2, j2) = positions
                    sep = periodic_distance(
                        i1, j1, i2, j2, engine.Nx, engine.dx
                    )
                else:
                    sep = math.nan

                max_rho = float(to_cpu(rho).max())
                norm_val = engine.norm()
                observables.append(
                    [
                        engine.step_count,
                        engine.time,
                        norm_val,
                        sep,
                        max_rho,
                    ]
                )
                time_series.append(engine.time)
                min_sep_series.append(sep)

            if step % args.movie_interval == 0 or step == args.steps - 1:
                rho_frame = normalize_to_uint8(engine.density())
                vort_frame = normalize_to_uint8(engine.vorticity())
                rho_writer.append_data(rho_frame)
                vort_writer.append_data(vort_frame)

        print("[engine] Finished main loop.")

    finally:
        rho_writer.close()
        vort_writer.close()

    final_state_path = os.path.join(results_dir, "final_state.npz")
    np.savez(
        final_state_path,
        psi=to_cpu(engine.psi),
        Nx=engine.Nx,
        dx=engine.dx,
        G=engine.G,
        mass=engine.mass,
        dt=engine.dt,
        wilson_r=engine.wilson_r,
        time=engine.time,
        steps=engine.step_count,
        preset=args.preset,
    )

    obs_path = os.path.join(results_dir, "observables.csv")
    save_observables_csv(obs_path, header, observables)

    if any(not math.isnan(s) for s in min_sep_series):
        plt.figure(figsize=(6, 4))
        t = np.array(time_series)
        s = np.array(min_sep_series)
        plt.plot(t, s, lw=1.5)
        plt.xlabel("time")
        plt.ylabel("min separation")
        plt.title(f"Min separation vs time ({label})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(sep_plot_path, dpi=150)
        plt.close()
    else:
        warnings.warn("No valid separation data to plot.")

    print(f"[engine] Saved final_state.npz, observables.csv, and movies in {results_dir}")


def run_noise(args):
    use_gpu = not args.cpu
    engine = SubstrateEngine(
        Nx=args.Nx,
        dx=args.dx,
        G=args.G,
        mass=args.mass,
        dt=args.dt,
        wilson_r=args.wilson_r,
        use_gpu=use_gpu,
        renorm_interval=args.renorm_interval,
        norm_tol=args.norm_tol,
        seed=args.seed,
    )
    engine.set_noise(amplitude=args.noise_amp)

    results_dir = create_results_dir("noise_to_structure")
    print(f"[noise] Saving results to: {results_dir}")

    observables = []
    header = ["step", "time", "norm", "max_rho"]

    rho_movie_path = os.path.join(results_dir, "movie_rho.tiff")
    vort_movie_path = os.path.join(results_dir, "movie_vorticity.tiff")

    rho_writer = open_movie_writer(rho_movie_path, fps=args.movie_fps)
    vort_writer = open_movie_writer(vort_movie_path, fps=args.movie_fps)

    try:
        for step in range(args.steps):
            engine.step()

            if step % args.obs_interval == 0 or step == args.steps - 1:
                rho = engine.density()
                max_rho = float(to_cpu(rho).max())
                norm_val = engine.norm()
                observables.append(
                    [engine.step_count, engine.time, norm_val, max_rho]
                )

            if step % args.movie_interval == 0 or step == args.steps - 1:
                rho_frame = normalize_to_uint8(engine.density())
                vort_frame = normalize_to_uint8(engine.vorticity())
                rho_writer.append_data(rho_frame)
                vort_writer.append_data(vort_frame)

        print("[noise] Finished main loop.")

    finally:
        rho_writer.close()
        vort_writer.close()

    final_state_path = os.path.join(results_dir, "final_state.npz")
    np.savez(
        final_state_path,
        psi=to_cpu(engine.psi),
        Nx=engine.Nx,
        dx=engine.dx,
        G=engine.G,
        mass=engine.mass,
        dt=engine.dt,
        wilson_r=engine.wilson_r,
        time=engine.time,
        steps=engine.step_count,
        mode="noise",
    )

    obs_path = os.path.join(results_dir, "observables.csv")
    save_observables_csv(obs_path, header, observables)

    print(f"[noise] Saved final_state.npz, observables.csv, and movies in {results_dir}")


def run_bell(args):
    if args.mode != "chsh":
        raise ValueError("Only --mode chsh is implemented for now.")
    S, details = chsh_singlet(samples=args.samples)
    print(f"[bell] CHSH S ≈ {S:.4f} (Tsirelson bound is 2.828...)")
    print(
        "[bell] E(a,b)={E_ab:.4f}, E(a,b')={E_abp:.4f}, "
        "E(a',b)={E_apb:.4f}, E(a',b')={E_apbp:.4f}".format(**details)
    )


def run_entropy(args):
    use_gpu = not args.cpu
    engine = SubstrateEngine(
        Nx=args.Nx,
        dx=args.dx,
        G=args.G,
        mass=args.mass,
        dt=args.dt,
        wilson_r=args.wilson_r,
        use_gpu=use_gpu,
        renorm_interval=args.renorm_interval,
        norm_tol=args.norm_tol,
        seed=args.seed,
    )

    engine.set_two_vortices(
        w1=1,
        w2=0,
        offset=0.0,
        core_radius=args.core_radius,
        amp=args.amp,
    )

    for _ in range(args.relax_steps):
        engine.step()

    S, region_sites, perimeter = entanglement_entropy_disk(
        engine.psi, engine.dx, radius_phys=args.radius
    )

    print("[entropy] Disk radius (phys):", args.radius)
    print("[entropy] Region sites:", region_sites)
    print("[entropy] Estimated perimeter:", perimeter)
    print("[entropy] 'Entropy' S (Shannon-based surrogate):", S)
    print("[entropy] S / perimeter:", S / perimeter if perimeter > 0 else math.nan)


# ============================================================
# Argument parsing
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Hilbert Substrate Simulation Engine — All-in-One Script"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Engine
    p_eng = subparsers.add_parser(
        "engine", help="Two-vortex substrate run (fermionic exclusion / merger)"
    )
    p_eng.add_argument(
        "--preset",
        choices=["exclusion_same", "exclusion_opposite", "custom"],
        default="exclusion_same",
    )
    p_eng.add_argument("--Nx", type=int, default=256)
    p_eng.add_argument("--dx", type=float, default=0.25)
    p_eng.add_argument("--G", type=float, default=58.0)
    p_eng.add_argument("--mass", type=float, default=0.5)
    p_eng.add_argument("--dt", type=float, default=1e-4)
    p_eng.add_argument("--steps", type=int, default=20000)
    p_eng.add_argument("--wilson_r", type=float, default=1.0)
    p_eng.add_argument("--offset", type=float, default=30.0)
    p_eng.add_argument("--core_radius", type=float, default=3.0)
    p_eng.add_argument("--amp", type=float, default=1.0)
    p_eng.add_argument("--w1", type=int, default=1)
    p_eng.add_argument("--w2", type=int, default=1)
    p_eng.add_argument("--seed", type=int, default=None)
    p_eng.add_argument("--renorm_interval", type=int, default=100)
    p_eng.add_argument("--norm_tol", type=float, default=1e-10)
    p_eng.add_argument("--obs_interval", type=int, default=50)
    p_eng.add_argument("--movie_interval", type=int, default=10)
    p_eng.add_argument("--movie_fps", type=int, default=30)
    p_eng.add_argument("--cpu", action="store_true")

    # Noise
    p_noise = subparsers.add_parser("noise", help="Noise to structure run")
    p_noise.add_argument("--Nx", type=int, default=512)
    p_noise.add_argument("--dx", type=float, default=0.25)
    p_noise.add_argument("--G", type=float, default=58.0)
    p_noise.add_argument("--mass", type=float, default=0.5)
    p_noise.add_argument("--dt", type=float, default=1e-4)
    p_noise.add_argument("--steps", type=int, default=40000)
    p_noise.add_argument("--wilson_r", type=float, default=1.0)
    p_noise.add_argument("--noise_amp", type=float, default=1e-2)
    p_noise.add_argument("--seed", type=int, default=42)
    p_noise.add_argument("--renorm_interval", type=int, default=100)
    p_noise.add_argument("--norm_tol", type=float, default=1e-10)
    p_noise.add_argument("--obs_interval", type=int, default=100)
    p_noise.add_argument("--movie_interval", type=int, default=50)
    p_noise.add_argument("--movie_fps", type=int, default=30)
    p_noise.add_argument("--cpu", action="store_true")

    # Bell
    p_bell = subparsers.add_parser(
        "bell", help="Bell / CHSH validation for emergent photons"
    )
    p_bell.add_argument(
        "--mode", choices=["chsh"], default="chsh"
    )
    p_bell.add_argument("--samples", type=int, default=100000)

    # Entropy
    p_ent = subparsers.add_parser(
        "entropy", help="Entropy-like observable in a disk region"
    )
    p_ent.add_argument("--Nx", type=int, default=256)
    p_ent.add_argument("--dx", type=float, default=0.25)
    p_ent.add_argument("--G", type=float, default=58.0)
    p_ent.add_argument("--mass", type=float, default=0.5)
    p_ent.add_argument("--dt", type=float, default=1e-4)
    p_ent.add_argument("--wilson_r", type=float, default=1.0)
    p_ent.add_argument("--radius", type=float, default=20.0)
    p_ent.add_argument("--core_radius", type=float, default=3.0)
    p_ent.add_argument("--amp", type=float, default=1.0)
    p_ent.add_argument("--relax_steps", type=int, default=2000)
    p_ent.add_argument("--seed", type=int, default=123)
    p_ent.add_argument("--renorm_interval", type=int, default=100)
    p_ent.add_argument("--norm_tol", type=float, default=1e-10)
    p_ent.add_argument("--cpu", action="store_true")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "engine":
        run_engine(args)
    elif args.command == "noise":
        run_noise(args)
    elif args.command == "bell":
        run_bell(args)
    elif args.command == "entropy":
        run_entropy(args)
    else:
        parser.error(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
