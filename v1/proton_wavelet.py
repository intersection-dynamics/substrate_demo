#!/usr/bin/env python3
"""
proton_wavelet.py

Wavelet analysis of a proton-like lump in ScalarFieldDefragGPU.

Modes:
- "single": sample one specific step (old behavior).
- "scan"  (default): scan the run every `scan_every` steps, remember the
                     heaviest lump seen, and analyze that one.

Outputs:
- A PNG with original density cutout and wavelet components of the chosen proton.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import scalar_field_defrag_gpu as sfg

try:
    import pywt
except ImportError as e:
    raise SystemExit(
        "PyWavelets is required for this script.\n"
        "Install it with:  pip install pywavelets"
    ) from e


# ---------------------------------------------------------------------------
# Lump detection
# ---------------------------------------------------------------------------

@dataclass
class ProtonLump:
    frame_index: int
    time: float
    label: int
    x_idx: float
    y_idx: float
    mass: float
    peak_rho: float
    mean_rho: float
    n_pixels: int


def find_lumps(
    rho: np.ndarray,
    dx: float,
    frame_index: int,
    time: float,
    sigma_threshold: float = 2.5,
    min_pixels: int = 12,
) -> List[ProtonLump]:
    """Detect localized density lumps via threshold + connected components."""
    assert rho.ndim == 2, "rho must be 2D"

    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh

    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, n_labels = ndi.label(mask, structure=structure)

    lumps: List[ProtonLump] = []
    voxel_area = dx * dx

    for lab in range(1, n_labels + 1):
        region = (labeled == lab)
        n_pix = int(region.sum())
        if n_pix < min_pixels:
            continue

        ys, xs = np.nonzero(region)
        x_mean_idx = xs.mean()
        y_mean_idx = ys.mean()

        rho_region = rho[region]
        mass = float(rho_region.sum() * voxel_area)
        peak = float(rho_region.max())
        mean = float(rho_region.mean())

        lumps.append(
            ProtonLump(
                frame_index=frame_index,
                time=time,
                label=lab,
                x_idx=float(x_mean_idx),
                y_idx=float(y_mean_idx),
                mass=mass,
                peak_rho=peak,
                mean_rho=mean,
                n_pixels=n_pix,
            )
        )

    return lumps


def make_cutout(
    arr: np.ndarray,
    center_x: float,
    center_y: float,
    half_width: int,
) -> Tuple[np.ndarray, slice, slice]:
    """Extract a square cutout around (center_x, center_y) in index coords."""
    Ly, Lx = arr.shape
    cx = int(round(center_x))
    cy = int(round(center_y))

    x_min = max(0, cx - half_width)
    x_max = min(Lx, cx + half_width + 1)
    y_min = max(0, cy - half_width)
    y_max = min(Ly, cy + half_width + 1)

    ys = slice(y_min, y_max)
    xs = slice(x_min, x_max)
    cut = arr[ys, xs]
    return cut, ys, xs


# ---------------------------------------------------------------------------
# Wavelet analysis
# ---------------------------------------------------------------------------

def wavelet_decompose_2d(
    rho_cut: np.ndarray,
    wavelet_name: str = "db4",
    level: int = 3,
):
    """2D wavelet decomposition of rho_cut."""
    arr = rho_cut.astype(np.float64)
    wavelet = pywt.Wavelet(wavelet_name)

    if level <= 0:
        max_level = pywt.dwt_max_level(min(arr.shape), wavelet.dec_len)
        level = max_level

    coeffs = pywt.wavedec2(arr, wavelet=wavelet, level=level)
    return coeffs, wavelet


def plot_wavelet_components(
    rho_cut: np.ndarray,
    coeffs,
    wavelet,
    dx: float,
    out_path: Path,
    meta: str = "",
):
    """Plot original cutout + approximation and detail coefficients."""
    cA = coeffs[0]
    details = coeffs[1:]
    n_levels = len(details)

    fig_rows = n_levels + 1
    fig_cols = 4

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 3 * fig_rows))

    if fig_rows == 1:
        axes = np.array([axes])
    if fig_cols == 1:
        axes = axes[:, np.newaxis]

    # Row 0: original + approximation
    ax0 = axes[0, 0]
    im0 = ax0.imshow(rho_cut, origin="lower", cmap="magma")
    ax0.set_title("Original density cutout")
    ax0.set_xticks([]); ax0.set_yticks([])
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = axes[0, 1]
    im1 = ax1.imshow(cA, origin="lower", cmap="magma")
    ax1.set_title(f"Approximation A_L (L={n_levels})")
    ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    # Detail levels: finest first
    for lvl, (cH, cV, cD) in enumerate(details[::-1], start=1):
        row = lvl

        axH = axes[row, 0]
        axV = axes[row, 1]
        axD = axes[row, 2]

        imH = axH.imshow(cH, origin="lower", cmap="bwr")
        axH.set_title(f"Level {lvl} H-detail")
        axH.set_xticks([]); axH.set_yticks([])
        fig.colorbar(imH, ax=axH, fraction=0.046)

        imV = axV.imshow(cV, origin="lower", cmap="bwr")
        axV.set_title(f"Level {lvl} V-detail")
        axV.set_xticks([]); axV.set_yticks([])
        fig.colorbar(imV, ax=axV, fraction=0.046)

        imD = axD.imshow(cD, origin="lower", cmap="bwr")
        axD.set_title(f"Level {lvl} D-detail")
        axD.set_xticks([]); axD.set_yticks([])
        fig.colorbar(imD, ax=axD, fraction=0.046)

        axes[row, 3].axis("off")

    fig.suptitle(
        f"Wavelet analysis of proton density\n"
        f"wavelet={wavelet.name}, levels={n_levels}, dx={dx}\n{meta}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main logic: single-step vs scan
# ---------------------------------------------------------------------------

def run_proton_wavelet(
    mode: str = "scan",        # "single" or "scan"
    L: int = 128,
    dx: float = 1.0,
    dt: float = 0.005,
    g_defrag: float = 1.5,
    v: float = 1.0,
    lambda_param: float = 0.0,
    n_steps: int = 5000,
    sample_step: int = 5000,
    scan_every: int = 200,
    cutout_half_width: int = 16,
    sigma_threshold: float = 1.8,
    min_pixels: int = 6,
    wavelet_name: str = "db4",
    level: int = 3,
    init_mean: float = 1.0,
    init_noise_amp: float = 0.1,
    init_seed: int = 42,
    output_dir: str = "proton_wavelet_output",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" PROTON WAVELET ANALYSIS – ScalarFieldDefragGPU ")
    print("=" * 70)
    print(f"mode={mode}")
    print(f"L={L}, dx={dx}, dt={dt}, g_defrag={g_defrag}, v={v}, lambda={lambda_param}")
    print(f"n_steps={n_steps}, sample_step={sample_step}, scan_every={scan_every}")
    print(f"cutout_half_width={cutout_half_width}")
    print(f"sigma_threshold={sigma_threshold}, min_pixels={min_pixels}")
    print(f"wavelet={wavelet_name}, level={level}")
    print("=" * 70)

    sim = sfg.ScalarFieldDefragGPU(
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

    if mode == "single":
        # Old behavior: analyze exactly sample_step
        for step in range(n_steps + 1):
            if step == sample_step or step == n_steps:
                t = step * dt

                if sfg.GPU_AVAILABLE:
                    psi_cpu = sfg.cp.asnumpy(psi)
                else:
                    psi_cpu = psi

                rho = np.abs(psi_cpu) ** 2
                lumps = find_lumps(
                    rho,
                    dx=dx,
                    frame_index=step,
                    time=t,
                    sigma_threshold=sigma_threshold,
                    min_pixels=min_pixels,
                )

                if not lumps:
                    print(f"[SAMPLE] step={step}, t={t:.3f}: no lumps found.")
                else:
                    lump = max(lumps, key=lambda L_: L_.mass)
                    print(
                        f"[SAMPLE] step={step}, t={t:.3f}: "
                        f"{len(lumps)} lumps, using label={lump.label} "
                        f"(mass={lump.mass:.3e}, peak_rho={lump.peak_rho:.3e})"
                    )
                    rho_cut, ys, xs = make_cutout(
                        rho, center_x=lump.x_idx, center_y=lump.y_idx,
                        half_width=cutout_half_width,
                    )
                    coeffs, wavelet = wavelet_decompose_2d(
                        rho_cut, wavelet_name=wavelet_name, level=level
                    )
                    meta = f"step={step}, t={t:.3f}, mass={lump.mass:.3e}"
                    out_fig = out_dir / f"proton_wavelet_step{step:05d}.png"
                    plot_wavelet_components(
                        rho_cut=rho_cut,
                        coeffs=coeffs,
                        wavelet=wavelet,
                        dx=dx,
                        out_path=out_fig,
                        meta=meta,
                    )
                    print(f"[OUT] Saved wavelet figure to {out_fig}")
                break

            if step < n_steps:
                psi, _ = sim.evolve_step_rk4(psi)

    else:
        # Scan mode: keep the best lump seen
        best_lump: Optional[ProtonLump] = None
        best_rho_cut: Optional[np.ndarray] = None

        for step in range(n_steps + 1):
            if step % scan_every == 0 or step == n_steps:
                t = step * dt

                if sfg.GPU_AVAILABLE:
                    psi_cpu = sfg.cp.asnumpy(psi)
                else:
                    psi_cpu = psi

                rho = np.abs(psi_cpu) ** 2
                lumps = find_lumps(
                    rho,
                    dx=dx,
                    frame_index=step,
                    time=t,
                    sigma_threshold=sigma_threshold,
                    min_pixels=min_pixels,
                )

                if lumps:
                    lump = max(lumps, key=lambda L_: L_.mass)
                    print(
                        f"[SCAN] step={step}, t={t:.3f}: "
                        f"{len(lumps)} lumps, best mass={lump.mass:.3e}"
                    )
                    if (best_lump is None) or (lump.mass > best_lump.mass):
                        best_lump = lump
                        best_rho_cut, ys, xs = make_cutout(
                            rho,
                            center_x=lump.x_idx,
                            center_y=lump.y_idx,
                            half_width=cutout_half_width,
                        )
                else:
                    print(f"[SCAN] step={step}, t={t:.3f}: no lumps.")

            if step < n_steps:
                psi, _ = sim.evolve_step_rk4(psi)

        if best_lump is None or best_rho_cut is None:
            print("[RESULT] No lumps found in entire scan.")
        else:
            coeffs, wavelet = wavelet_decompose_2d(
                best_rho_cut, wavelet_name=wavelet_name, level=level
            )
            meta = (
                f"best at step={best_lump.frame_index}, t={best_lump.time:.3f}, "
                f"mass={best_lump.mass:.3e}"
            )
            out_fig = out_dir / f"proton_wavelet_best.png"
            plot_wavelet_components(
                rho_cut=best_rho_cut,
                coeffs=coeffs,
                wavelet=wavelet,
                dx=dx,
                out_path=out_fig,
                meta=meta,
            )
            print(f"[RESULT] Best lump at step={best_lump.frame_index}, "
                  f"mass={best_lump.mass:.3e}")
            print(f"[OUT] Saved wavelet figure to {out_fig}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wavelet analysis of a proton-like lump in ScalarFieldDefragGPU."
    )
    parser.add_argument("--mode", type=str, default="scan",
                        choices=["scan", "single"],
                        help="Scan whole run or sample a single step.")
    parser.add_argument("--L", type=int, default=128, help="Grid size.")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--g_defrag", type=float, default=1.5, help="Defrag coupling.")
    parser.add_argument("--v", type=float, default=1.0, help="Vacuum expectation.")
    parser.add_argument("--lambda_param", type=float, default=0.0,
                        help="Self-interaction strength.")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Total evolution steps.")
    parser.add_argument("--sample_step", type=int, default=5000,
                        help="Step for 'single' mode sampling.")
    parser.add_argument("--scan_every", type=int, default=200,
                        help="Interval between scans in 'scan' mode.")
    parser.add_argument("--cutout_half_width", type=int, default=16,
                        help="Half-width of cutout around proton.")
    parser.add_argument("--sigma_threshold", type=float, default=1.8,
                        help="Density threshold parameter.")
    parser.add_argument("--min_pixels", type=int, default=6,
                        help="Minimum pixels in a lump.")
    parser.add_argument("--wavelet_name", type=str, default="db4",
                        help="PyWavelets wavelet name (db4, coif3, sym4, ...).")
    parser.add_argument("--level", type=int, default=3,
                        help="Wavelet decomposition level (<=0 → max).")
    parser.add_argument("--init_mean", type=float, default=1.0,
                        help="Initial mean amplitude.")
    parser.add_argument("--init_noise_amp", type=float, default=0.1,
                        help="Initial noise amplitude.")
    parser.add_argument("--init_seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="proton_wavelet_output",
                        help="Output directory.")

    args = parser.parse_args()

    run_proton_wavelet(
        mode=args.mode,
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        g_defrag=args.g_defrag,
        v=args.v,
        lambda_param=args.lambda_param,
        n_steps=args.n_steps,
        sample_step=args.sample_step,
        scan_every=args.scan_every,
        cutout_half_width=args.cutout_half_width,
        sigma_threshold=args.sigma_threshold,
        min_pixels=args.min_pixels,
        wavelet_name=args.wavelet_name,
        level=args.level,
        init_mean=args.init_mean,
        init_noise_amp=args.init_noise_amp,
        init_seed=args.init_seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
