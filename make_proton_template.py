#!/usr/bin/env python3
"""
make_proton_template.py

Run the scalar + defrag model once, let it form lumps, and automatically
extract the heaviest bound lump as a "proton template".

We:
  - start from uniform + noise,
  - evolve with ScalarFieldDefragGPU,
  - at analysis intervals detect density lumps in |psi|^2,
  - keep track of the heaviest lump seen so far,
  - when we see a new heaviest, we stash a copy of psi at that step
    plus the lump center + radius,
  - at the end, crop a square patch around that lump (with periodic
    wrapping) and save it as proton_template.npz with key "psi",
    plus some metadata.

No hand sculpting, no manual tuning of the lump: the template is
literally a field configuration your substrate produced on its own.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# optional CuPy
try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # type: ignore

from scalar_field_defrag_gpu import ScalarFieldDefragGPU


# ---------------- lump detection (same style as before) ----------------

@dataclass
class Lump:
    frame_index: int
    time: float
    x: float
    y: float
    radius: float
    mass: float
    peak_rho: float
    mean_rho: float
    area: float
    n_pixels: int


def detect_lumps(
    rho: np.ndarray,
    dx: float,
    frame_index: int,
    time: float,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
) -> List[Lump]:
    assert rho.ndim == 2
    ny, nx = rho.shape

    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_labels = ndi.label(mask, structure=structure)

    lumps: List[Lump] = []
    voxel_area = dx * dx

    for lab in range(1, n_labels + 1):
        region = (labeled == lab)
        n_pix = int(region.sum())
        if n_pix < min_pixels:
            continue

        ys, xs = np.nonzero(region)
        x_mean = xs.mean()
        y_mean = ys.mean()

        x_phys = float(x_mean * dx)
        y_phys = float(y_mean * dx)

        dxs = (xs - x_mean) * dx
        dys = (ys - y_mean) * dx
        r = np.sqrt(dxs**2 + dys**2)
        radius = float(np.sqrt(np.mean(r**2)))

        rho_region = rho[region]
        mass = float(rho_region.sum() * voxel_area)
        peak = float(rho_region.max())
        mean = float(rho_region.mean())
        area = float(n_pix * voxel_area)

        lumps.append(
            Lump(
                frame_index=frame_index,
                time=time,
                x=x_phys,
                y=y_phys,
                radius=radius,
                mass=mass,
                peak_rho=peak,
                mean_rho=mean,
                area=area,
                n_pixels=n_pix,
            )
        )

    return lumps


# ---------------- cropping helper ----------------

def crop_periodic_patch(
    psi: np.ndarray,
    x_center: float,
    y_center: float,
    dx: float,
    patch_radius_phys: float,
) -> np.ndarray:
    """
    Crop a square patch of side ~2*patch_radius+1 around (x_center, y_center),
    using periodic boundary conditions.

    x_center, y_center are in physical units; dx is lattice spacing.
    """
    L = psi.shape[0]
    assert psi.shape[0] == psi.shape[1], "square grid assumed"

    # convert to index coordinates
    cx = x_center / dx
    cy = y_center / dx

    patch_radius_idx = int(np.ceil(patch_radius_phys / dx))
    side = 2 * patch_radius_idx + 1

    patch = np.zeros((side, side), dtype=psi.dtype)

    for j in range(side):
        for i in range(side):
            gx = int(np.round(cx)) - patch_radius_idx + i
            gy = int(np.round(cy)) - patch_radius_idx + j
            gx %= L
            gy %= L
            patch[j, i] = psi[gy, gx]

    return patch


# ---------------- main template-extraction run ----------------

def run_make_template(
    L: int = 192,
    dx: float = 1.0,
    dt: float = 0.005,
    g_defrag: float = 1.0,
    v: float = 1.0,
    lambda_param: float = 0.5,
    n_steps: int = 2500,
    analysis_interval: int = 20,
    init_mean: float = 1.0,
    init_noise_amp: float = 0.1,
    init_seed: int = 123,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
    patch_radius_factor: float = 4.0,
    output_template: str = "proton_template.npz",
    output_log: str = "proton_template_log.csv",
):
    print("=" * 72)
    print(" make_proton_template.py ")
    print("=" * 72)
    print(f"L={L}, dx={dx}, dt={dt}, g_defrag={g_defrag}, v={v}, lambda={lambda_param}")
    print(f"n_steps={n_steps}, analysis_interval={analysis_interval}")
    print(f"init_mean={init_mean}, init_noise_amp={init_noise_amp}, seed={init_seed}")
    print("=" * 72)

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

    best_lump: Lump | None = None
    best_psi: np.ndarray | None = None

    all_lumps: list[Lump] = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % analysis_interval == 0 or step == n_steps:
            if CUPY_AVAILABLE and isinstance(psi, cp.ndarray):
                psi_cpu = cp.asnumpy(psi)
            else:
                psi_cpu = psi

            rho = np.abs(psi_cpu) ** 2
            lumps = detect_lumps(
                rho,
                dx=dx,
                frame_index=step,
                time=t,
                sigma_threshold=sigma_threshold,
                min_pixels=min_pixels,
            )
            all_lumps.extend(lumps)

            if lumps:
                # heaviest lump in this frame
                lump_max = max(lumps, key=lambda L_: L_.mass)

                if best_lump is None or lump_max.mass > best_lump.mass:
                    best_lump = lump_max
                    best_psi = psi_cpu.copy()
                    print(
                        f"[NEW BEST] step={step}, t={t:.3f}, "
                        f"mass={lump_max.mass:.3f}, radius={lump_max.radius:.3f}, "
                        f"x={lump_max.x:.2f}, y={lump_max.y:.2f}"
                    )
            else:
                print(f"[ANALYSIS] step={step:5d}, t={t:7.3f}, no lumps")

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    if best_lump is None or best_psi is None:
        print("[ERROR] No lump found; cannot make template.")
        return

    # Crop a patch around the best lump
    patch_radius_phys = best_lump.radius * patch_radius_factor
    template = crop_periodic_patch(
        best_psi,
        x_center=best_lump.x,
        y_center=best_lump.y,
        dx=dx,
        patch_radius_phys=patch_radius_phys,
    )

    # Save template + metadata
    out_path = Path(output_template)
    np.savez_compressed(
        out_path,
        psi=template,
        dx=dx,
        L=L,
        dt=dt,
        g_defrag=g_defrag,
        v=v,
        lambda_param=lambda_param,
        best_mass=best_lump.mass,
        best_radius=best_lump.radius,
        best_x=best_lump.x,
        best_y=best_lump.y,
        best_time=best_lump.time,
        best_frame_index=best_lump.frame_index,
        patch_radius_phys=patch_radius_phys,
    )
    print(f"[RESULT] Saved proton template -> {out_path}")

    # Optional: log all lumps seen for reference
    df = pd.DataFrame([asdict(l) for l in all_lumps])
    log_path = Path(output_log)
    df.to_csv(log_path, index=False)
    print(f"[RESULT] Saved lump log -> {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a proton_template.npz from scalar+defrag evolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--L", type=int, default=192)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--g_defrag", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--lambda_param", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=2500)
    parser.add_argument("--analysis_interval", type=int, default=20)
    parser.add_argument("--init_mean", type=float, default=1.0)
    parser.add_argument("--init_noise_amp", type=float, default=0.1)
    parser.add_argument("--init_seed", type=int, default=123)
    parser.add_argument("--sigma_threshold", type=float, default=2.0)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument(
        "--patch_radius_factor",
        type=float,
        default=4.0,
        help="How many radii to include in cropped patch.",
    )
    parser.add_argument(
        "--output_template",
        type=str,
        default="proton_template.npz",
    )
    parser.add_argument(
        "--output_log",
        type=str,
        default="proton_template_log.csv",
    )

    args = parser.parse_args()

    run_make_template(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        g_defrag=args.g_defrag,
        v=args.v,
        lambda_param=args.lambda_param,
        n_steps=args.n_steps,
        analysis_interval=args.analysis_interval,
        init_mean=args.init_mean,
        init_noise_amp=args.init_noise_amp,
        init_seed=args.init_seed,
        sigma_threshold=args.sigma_threshold,
        min_pixels=args.min_pixels,
        patch_radius_factor=args.patch_radius_factor,
        output_template=args.output_template,
        output_log=args.output_log,
    )


if __name__ == "__main__":
    main()
