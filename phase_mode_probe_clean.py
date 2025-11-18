#!/usr/bin/env python3
"""
phase_mode_probe_clean.py

Phase-cleaned probe of phase / Goldstone-like excitations in the scalar+defrag
substrate.

Differences from phase_mode_probe.py:
  - At each sample time, we:
        a0(t) = (1/N) sum_x psi(x,t)
        theta0(t) = arg(a0(t))
        psi_rot(x,t) = psi(x,t) * exp(-i * theta0(t))
    and then measure the mode amplitude a_k(t) on psi_rot, not on psi.

This removes the global U(1) precession and lets us look for a genuinely
light-like dispersion of spatial phase modes.

We:
  - initialize psi ~ v * exp(i * epsilon * cos(k·x)) on a uniform background,
  - evolve with ScalarFieldDefragGPU,
  - sample a_k(t) on the de-spun field psi_rot,
  - save time series,
  - compute a power spectrum and estimate the dominant frequency omega(k).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional CuPy
try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # type: ignore

from scalar_field_defrag_gpu import ScalarFieldDefragGPU


# ---------- helpers to build initial condition and mode weight ----------

def prepare_phase_wave(
    L: int,
    dx: float,
    v: float,
    epsilon: float,
    kx_index: int,
    ky_index: int,
) -> np.ndarray:
    """
    Build an initial psi(x,y) = v * exp(i * epsilon * cos(k·x)) on a NumPy grid.

    kx_index, ky_index are integers specifying kx = 2π*kx_index/(L*dx), etc.
    """
    xs = np.arange(L) * dx
    ys = np.arange(L) * dx
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    L_phys = L * dx
    kx = 2.0 * np.pi * kx_index / L_phys
    ky = 2.0 * np.pi * ky_index / L_phys

    phase_mod = epsilon * np.cos(kx * X + ky * Y)
    psi = v * np.exp(1j * phase_mod).astype(np.complex128)
    return psi


def precompute_mode_weight(
    L: int,
    dx: float,
    kx_index: int,
    ky_index: int,
) -> np.ndarray:
    """
    Precompute W(x) = exp(-i k·x) on a NumPy grid so that

        a_k(t) = (1/N) sum_x psi_rot(x,t) * W(x)

    gives the Fourier amplitude of the selected mode on the de-spun field.
    """
    xs = np.arange(L) * dx
    ys = np.arange(L) * dx
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    L_phys = L * dx
    kx = 2.0 * np.pi * kx_index / L_phys
    ky = 2.0 * np.pi * ky_index / L_phys

    W = np.exp(-1j * (kx * X + ky * Y)).astype(np.complex128)
    return W


def global_mode_and_rotated(psi: np.ndarray) -> tuple[complex, np.ndarray]:
    """
    Compute global (k=0) mode a0 = (1/N) sum_x psi(x) and de-spun field

        psi_rot(x) = psi(x) * exp(-i * arg(a0)).

    Returns (a0, psi_rot).
    """
    a0 = psi.mean()
    theta0 = np.angle(a0)
    psi_rot = psi * np.exp(-1j * theta0)
    return a0, psi_rot


def measure_mode_amplitude(
    psi_rot: np.ndarray,
    W: np.ndarray,
) -> complex:
    """
    Compute a_k = (1/N) sum_x psi_rot(x) * W(x), where W is precomputed.
    """
    N = psi_rot.size
    return np.sum(psi_rot * W) / N


# ---------------------------- main runner ----------------------------

def run_phase_probe_clean(
    L: int,
    dx: float,
    dt: float,
    g_defrag: float,
    v: float,
    lambda_param: float,
    n_steps: int,
    sample_interval: int,
    kx_index: int,
    ky_index: int,
    epsilon: float,
    output_prefix: str,
):
    out_dir = Path(f"{output_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" phase_mode_probe_clean.py ")
    print("=" * 72)
    print(f"L={L}, dx={dx}, dt={dt}, g_defrag={g_defrag}, v={v}, lambda={lambda_param}")
    print(f"n_steps={n_steps}, sample_interval={sample_interval}")
    print(f"k-index = ({kx_index}, {ky_index}), epsilon={epsilon}")
    print(f"output_dir = {out_dir}")
    print("=" * 72)

    # Initial condition (NumPy)
    psi0_np = prepare_phase_wave(
        L=L,
        dx=dx,
        v=v,
        epsilon=epsilon,
        kx_index=kx_index,
        ky_index=ky_index,
    )

    # Simulator
    sim = ScalarFieldDefragGPU(
        L=L,
        dx=dx,
        dt=dt,
        g_defrag=g_defrag,
        v=v,
        lambda_param=lambda_param,
    )

    # Move psi to backend
    if CUPY_AVAILABLE:
        psi = cp.asarray(psi0_np)
    else:
        psi = psi0_np.copy()

    # Precompute W on CPU
    W = precompute_mode_weight(
        L=L,
        dx=dx,
        kx_index=kx_index,
        ky_index=ky_index,
    )

    times = []
    a0_re = []
    a0_im = []
    ak_re = []
    ak_im = []
    ak_abs = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % sample_interval == 0 or step == n_steps:
            # Bring psi to CPU
            if CUPY_AVAILABLE and isinstance(psi, cp.ndarray):
                psi_cpu = cp.asnumpy(psi)
            else:
                psi_cpu = psi

            # Remove global phase
            a0, psi_rot = global_mode_and_rotated(psi_cpu)
            a_k = measure_mode_amplitude(psi_rot, W)

            times.append(t)
            a0_re.append(a0.real)
            a0_im.append(a0.imag)
            ak_re.append(a_k.real)
            ak_im.append(a_k.imag)
            ak_abs.append(np.abs(a_k))

            print(
                f"[SAMPLE] step={step:5d}, t={t:8.4f}, "
                f"a0=({a0.real:+.4e},{a0.imag:+.4e}), "
                f"Re(a_k)={a_k.real:+.4e}, Im(a_k)={a_k.imag:+.4e}, "
                f"|a_k|={np.abs(a_k):.4e}"
            )

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    # Save time series
    df = pd.DataFrame(
        {
            "time": times,
            "a0_re": a0_re,
            "a0_im": a0_im,
            "ak_re": ak_re,
            "ak_im": ak_im,
            "ak_abs": ak_abs,
        }
    )
    csv_path = out_dir / f"{output_prefix}_mode_timeseries_clean.csv"
    df.to_csv(csv_path, index=False)
    print(f"[RESULT] Saved cleaned mode time series -> {csv_path}")

    # Time series plots
    plt.figure(figsize=(8, 4))
    plt.plot(times, ak_re, label="Re(a_k) (clean)")
    plt.plot(times, ak_im, label="Im(a_k) (clean)")
    plt.xlabel("time")
    plt.ylabel("a_k (clean)")
    plt.title(f"Cleaned mode amplitude a_k(t), k=({kx_index},{ky_index})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ts_png = out_dir / f"{output_prefix}_mode_time_series_clean.png"
    plt.savefig(ts_png, dpi=200)
    plt.close()
    print(f"[RESULT] Saved cleaned time-series plot -> {ts_png}")

    plt.figure(figsize=(6, 4))
    plt.plot(times, ak_abs)
    plt.xlabel("time")
    plt.ylabel("|a_k| (clean)")
    plt.title(f"|a_k(t)| cleaned, k=({kx_index},{ky_index})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    abs_png = out_dir / f"{output_prefix}_mode_abs_time_series_clean.png"
    plt.savefig(abs_png, dpi=200)
    plt.close()
    print(f"[RESULT] Saved cleaned |a_k| plot -> {abs_png}")

    # Frequency analysis of cleaned Re(a_k)
    ak_re_arr = np.array(ak_re)
    t_arr = np.array(times)
    if len(t_arr) > 1:
        dt_sample = t_arr[1] - t_arr[0]
    else:
        dt_sample = dt * sample_interval

    ak_centered = ak_re_arr - ak_re_arr.mean()
    freq = np.fft.rfftfreq(len(ak_centered), d=dt_sample)
    spectrum = np.abs(np.fft.rfft(ak_centered)) ** 2

    if len(freq) > 1:
        idx_max = np.argmax(spectrum[1:]) + 1
        f_peak = freq[idx_max]
        omega_peak = 2.0 * np.pi * f_peak
        print(f"[RESULT] Cleaned dominant frequency f ~ {f_peak:.4f}, "
              f"omega ~ {omega_peak:.4f}")
    else:
        f_peak = 0.0
        omega_peak = 0.0
        print("[WARN] Not enough samples to estimate cleaned frequency.")

    # Spectrum plot
    plt.figure(figsize=(6, 4))
    plt.plot(freq, spectrum)
    plt.xlabel("frequency f")
    plt.ylabel("|FFT(Re(a_k clean))|^2")
    plt.title(f"Cleaned mode power spectrum, k=({kx_index},{ky_index})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    spec_png = out_dir / f"{output_prefix}_mode_spectrum_clean.png"
    plt.savefig(spec_png, dpi=200)
    plt.close()
    print(f"[RESULT] Saved cleaned power spectrum plot -> {spec_png}")

    # Summary text (ASCII only for Windows)
    summary_path = out_dir / f"{output_prefix}_summary_clean.txt"
    with open(summary_path, "w") as f:
        f.write(f"k_index = ({kx_index}, {ky_index})\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"sample_interval = {sample_interval}\n")
        f.write(f"dt_sample = {dt_sample}\n")
        f.write(f"f_peak ~ {f_peak:.6f}\n")
        f.write(f"omega_peak ~ {omega_peak:.6f}\n")
    print(f"[RESULT] Saved cleaned summary -> {summary_path}")

    print("=" * 72)
    print(" Phase mode probe (cleaned) complete ")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Phase-cleaned probe of phase-mode dispersion in the "
                    "scalar+defrag substrate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--L", type=int, default=192, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--g_defrag", type=float, default=1.0, help="Defrag coupling.")
    parser.add_argument("--v", type=float, default=1.0,
                        help="VEV / background amplitude.")
    parser.add_argument("--lambda_param", type=float, default=0.5,
                        help="Self-interaction strength.")
    parser.add_argument("--n_steps", type=int, default=4000,
                        help="Number of time steps.")
    parser.add_argument("--sample_interval", type=int, default=10,
                        help="Sample mode amplitude every this many steps.")
    parser.add_argument("--kx_index", type=int, default=1,
                        help="Integer mode index in x (k_x = 2π*kx_index/L_phys).")
    parser.add_argument("--ky_index", type=int, default=0,
                        help="Integer mode index in y (k_y = 2π*ky_index/L_phys).")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Phase modulation amplitude (small).")
    parser.add_argument("--output_prefix", type=str, default="phase_clean_k1",
                        help="Prefix for output directory and files.")

    args = parser.parse_args()

    run_phase_probe_clean(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        g_defrag=args.g_defrag,
        v=args.v,
        lambda_param=args.lambda_param,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        kx_index=args.kx_index,
        ky_index=args.ky_index,
        epsilon=args.epsilon,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
