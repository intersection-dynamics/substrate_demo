#!/usr/bin/env python3
"""
sea_of_protons_sim.py

Goal:
    Prepare an initial condition that contains a *sea* of proton-like
    lumps (copies of a previously obtained scalar/defrag bound state),
    then evolve under ScalarFieldDefragGPU and watch what they do.

Key points:
    - Uses *only* the existing scalar + defrag Hamiltonian.
    - No hand-coded forces or weak/strong behavior.
    - Initial condition is many copies of a measured lump template
      placed at random positions with minimum spacing.
    - We detect lumps over time and record their positions, masses,
      and counts. Interpretation (e.g. "strong-like interaction") is
      left to later data analysis.

Usage (example):

    python sea_of_protons_sim.py \\
        --L 256 \\
        --template proton_template.npz \\
        --template-key psi \\
        --n-protons 32 \\
        --n-steps 5000

Outputs:

    sea_of_protons_output/
        - lumps_all.csv      : detected lumps (per frame)
        - counts.csv         : number of lumps vs time
        - snapshots_stepXXXX.npz (optional, if enabled)
        - quick_counts.png   : lump count vs time

You can then:
    - visualize how many "protons" survive,
    - look at clustering, mergers, evaporation,
    - compute pair-correlation functions, etc.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# Optional CuPy
try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CUPY_AVAILABLE = False
    cp = None  # type: ignore

from scalar_field_defrag_gpu import ScalarFieldDefragGPU


# ---------------------------------------------------------------------
# Lump detection (same rules as before: data-driven threshold + labeling)
# ---------------------------------------------------------------------

@dataclass
class LumpRecord:
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


@dataclass
class CountRecord:
    frame_index: int
    time: float
    n_lumps: int


def detect_lumps(
    rho: np.ndarray,
    dx: float,
    frame_index: int,
    time: float,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
) -> Tuple[List[LumpRecord], CountRecord]:
    """
    Detect high-density lumps in |psi|^2 with a simple, data-driven rule:

      - threshold at mean + sigma_threshold * std,
      - label connected components (8-connectivity),
      - discard regions with too few pixels.

    No shape/profile is assumed.
    """
    assert rho.ndim == 2
    ny, nx = rho.shape

    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh
    structure = np.ones((3, 3), dtype=bool)
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
            LumpRecord(
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

    count_rec = CountRecord(
        frame_index=frame_index,
        time=time,
        n_lumps=len(lumps),
    )
    return lumps, count_rec


# ---------------------------------------------------------------------
# Build a "sea of protons" initial condition
# ---------------------------------------------------------------------

def load_template(path: Path, key: str | None) -> np.ndarray:
    """
    Load a complex 2D template from .npz or .npy.
    """
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            raise ValueError("For .npz templates, you must provide --template-key.")
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
        psi = data[key]
    else:
        psi = np.load(path)

    psi = np.asarray(psi)
    if psi.ndim != 2:
        raise ValueError(f"Template must be 2D; got shape {psi.shape}")
    if not np.iscomplexobj(psi):
        psi = psi.astype(np.complex128)
    return psi


def place_templates_on_grid(
    L: int,
    dx: float,
    template: np.ndarray,
    n_protons: int,
    min_spacing: float,
    max_tries: int = 1000,
    random_phase: bool = True,
    amp_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Place n_protons copies of 'template' onto an LxL lattice with periodic BC,
    enforcing a minimum center-to-center spacing.

    Args:
        L           : grid size (physical size is L*dx).
        dx          : lattice spacing.
        template    : 2D complex array (proton-like lump).
        n_protons   : how many copies to place.
        min_spacing : minimal distance between centers (physical units).
        max_tries   : max attempts to place each proton before giving up.
        random_phase: if True, multiply each copy by exp(i * theta).
        amp_jitter  : optional relative amplitude jitter (~0.1 -> ±10%).
        rng         : numpy random generator.

    Returns:
        psi_init : complex LxL array.
    """
    if rng is None:
        rng = np.random.default_rng()

    psi_init = np.zeros((L, L), dtype=np.complex128)

    Ly, Lx = template.shape
    cx_t = Ly // 2
    cy_t = Lx // 2

    centers: list[tuple[float, float]] = []
    min_spacing_idx = min_spacing / dx

    for k in range(n_protons):
        for attempt in range(max_tries):
            x_center = rng.uniform(0, L)
            y_center = rng.uniform(0, L)

            # enforce spacing in index space
            ok = True
            for (xc, yc) in centers:
                dx_ = (x_center - xc + L / 2.0) % L - L / 2.0
                dy_ = (y_center - yc + L / 2.0) % L - L / 2.0
                dist = np.sqrt(dx_**2 + dy_**2)
                if dist < min_spacing_idx:
                    ok = False
                    break

            if not ok:
                continue

            centers.append((x_center, y_center))

            # copy template onto grid with periodic wrapping
            if random_phase:
                theta = rng.uniform(0.0, 2.0 * np.pi)
                phase = np.exp(1j * theta)
            else:
                phase = 1.0

            amp = 1.0
            if amp_jitter > 0.0:
                amp *= (1.0 + amp_jitter * rng.uniform(-1.0, 1.0))

            shifted = template * (phase * amp)

            # indices in big grid
            x0 = int(np.round(x_center)) - cx_t
            y0 = int(np.round(y_center)) - cy_t

            for j in range(Ly):
                for i in range(Lx):
                    gx = (x0 + i) % L
                    gy = (y0 + j) % L
                    psi_init[gy, gx] += shifted[j, i]

            break
        else:
            print(f"[WARN] Could not place proton {k+1} after {max_tries} attempts.")

    print(f"[INIT] Placed {len(centers)}/{n_protons} protons on the grid.")
    return psi_init


# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------

def run_sea_of_protons(
    L: int,
    dx: float,
    dt: float,
    g_defrag: float,
    v: float,
    lambda_param: float,
    n_steps: int,
    analysis_interval: int,
    template_path: Path,
    template_key: str | None,
    n_protons: int,
    min_spacing: float,
    amp_jitter: float,
    random_phase: bool,
    output_dir: Path,
    save_snapshots: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template and build initial condition
    template = load_template(template_path, template_key)
    rng = np.random.default_rng()
    psi_init = place_templates_on_grid(
        L=L,
        dx=dx,
        template=template,
        n_protons=n_protons,
        min_spacing=min_spacing,
        amp_jitter=amp_jitter,
        random_phase=random_phase,
        rng=rng,
    )

    sim = ScalarFieldDefragGPU(
        L=L,
        dx=dx,
        dt=dt,
        g_defrag=g_defrag,
        v=v,
        lambda_param=lambda_param,
    )

    # Make sure psi is on the same backend as the simulator
    if CUPY_AVAILABLE:
        # sim is using CuPy; move psi_init to GPU
        psi = cp.asarray(psi_init)
    else:
        psi = psi_init.copy()

    all_lumps: List[LumpRecord] = []
    count_records: List[CountRecord] = []

    print("=" * 72)
    print(" Sea of Protons Simulation ")
    print("=" * 72)
    print(f"L={L}, dx={dx}, dt={dt}, g_defrag={g_defrag}, v={v}, lambda={lambda_param}")
    print(f"n_steps={n_steps}, analysis_interval={analysis_interval}")
    print(f"n_protons={n_protons}, min_spacing={min_spacing}")
    print(f"template={template_path}")
    print("=" * 72)

    for step in range(n_steps + 1):
        t = step * dt

        if step % analysis_interval == 0 or step == n_steps:
            # Move psi to CPU for analysis
            if CUPY_AVAILABLE and isinstance(psi, cp.ndarray):
                psi_cpu = cp.asnumpy(psi)
            else:
                psi_cpu = psi

            rho = np.abs(psi_cpu) ** 2

            lumps, count_rec = detect_lumps(
                rho,
                dx=dx,
                frame_index=step,
                time=t,
                sigma_threshold=2.0,
                min_pixels=8,
            )
            all_lumps.extend(lumps)
            count_records.append(count_rec)

            print(
                f"[ANALYSIS] step={step:5d}, t={t:8.3f}, n_lumps={count_rec.n_lumps}"
            )

            if save_snapshots:
                snap_path = output_dir / f"snap_step{step:05d}.npz"
                np.savez_compressed(snap_path, psi=psi_cpu)
                print(f"  saved snapshot -> {snap_path}")

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    # Save diagnostics
    if all_lumps:
        df_lumps = pd.DataFrame([asdict(l) for l in all_lumps])
        lumps_csv = output_dir / "lumps_all.csv"
        df_lumps.to_csv(lumps_csv, index=False)
        print(f"[RESULT] saved lump records -> {lumps_csv}")

    df_counts = pd.DataFrame([asdict(c) for c in count_records])
    counts_csv = output_dir / "counts.csv"
    df_counts.to_csv(counts_csv, index=False)
    print(f"[RESULT] saved lump counts -> {counts_csv}")

    # Quick-look plot: number of lumps vs time
    plt.figure(figsize=(6, 4))
    plt.plot(df_counts["time"], df_counts["n_lumps"], "-o", markersize=3)
    plt.xlabel("time")
    plt.ylabel("number of lumps")
    plt.title("Lump count vs time (sea of protons)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    counts_png = output_dir / "quick_counts.png"
    plt.savefig(counts_png, dpi=200)
    plt.close()
    print(f"[RESULT] saved lump-count plot -> {counts_png}")

    print("=" * 72)
    print(" Sea of protons run complete ")
    print(f" Outputs in: {output_dir}")
    print("=" * 72)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate a sea of proton-like lumps in the scalar+defrag substrate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--L", type=int, default=256, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--g_defrag", type=float, default=1.0, help="Defrag coupling.")
    parser.add_argument("--v", type=float, default=1.0, help="Vacuum expectation value.")
    parser.add_argument("--lambda_param", type=float, default=0.5,
                        help="Self-interaction strength.")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Total number of time steps.")
    parser.add_argument("--analysis_interval", type=int, default=50,
                        help="How often to detect lumps.")
    parser.add_argument("--template", type=str, required=True,
                        help="Path to proton template (.npz or .npy).")
    parser.add_argument("--template-key", type=str, default="psi",
                        help="Key for template array in .npz (ignored for .npy).")
    parser.add_argument("--n-protons", type=int, default=32,
                        help="Number of template copies to place.")
    parser.add_argument("--min-spacing", type=float, default=30.0,
                        help="Minimum separation between proton centers (physical units).")
    parser.add_argument("--amp-jitter", type=float, default=0.0,
                        help="Relative amplitude jitter for each proton (0 = none).")
    parser.add_argument("--no-random-phase", action="store_true",
                        help="Disable random global phase for each proton.")
    parser.add_argument("--output-dir", type=str, default="sea_of_protons_output",
                        help="Output directory.")
    parser.add_argument("--save-snapshots", action="store_true",
                        help="Save psi snapshots at each analysis step.")

    args = parser.parse_args()

    random_phase = not args.no_random_phase

    run_sea_of_protons(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        g_defrag=args.g_defrag,
        v=args.v,
        lambda_param=args.lambda_param,
        n_steps=args.n_steps,
        analysis_interval=args.analysis_interval,
        template_path=Path(args.template),
        template_key=args.template_key if args.template.endswith(".npz") else None,
        n_protons=args.n_protons,
        min_spacing=args.min_spacing,
        amp_jitter=args.amp_jitter,
        random_phase=random_phase,
        output_dir=Path(args.output_dir),
        save_snapshots=args.save_snapshots,
    )


if __name__ == "__main__":
    main()
