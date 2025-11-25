#!/usr/bin/env python3
"""
substrate_interactions_explorer.py

Explore whether interaction patterns (short-range repulsion / intermediate
attraction, and topological changes) emerge naturally from the scalar + defrag
substrate model.

This script:

  - Uses ScalarFieldDefragGPU from scalar_field_defrag_gpu.py.
  - Starts from uniform + small complex noise (no engineered particles).
  - Evolves the system unitarily with defrag Poisson coupling.
  - At analysis intervals it:

      * Detects density lumps in |psi|^2.
      * Records lump count vs time.
      * For the two heaviest lumps (if present), records:

          - positions (x1, y1), (x2, y2)
          - masses and radii
          - center-to-center separation r(t)

      * Estimates a phase-winding number around each lump
        (topological charge proxy) from the phase of psi.

The script does NOT hard-code any weak/strong-force behavior.
It only measures:

  - how many lumps there are,
  - how they move and interact,
  - whether their approximate winding numbers change over time.

Interpretation (weak-like transitions, strong-like repulsion/attraction)
is left to post-processing and comparison to the data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# Try to detect CuPy for GPU -> CPU transfers.
try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CUPY_AVAILABLE = False
    cp = None  # type: ignore

# Import your simulator
from scalar_field_defrag_gpu import ScalarFieldDefragGPU


# ----------------------------------------------------------------------
# Lump detection (pure measurement from |psi|^2)
# ----------------------------------------------------------------------

@dataclass
class Lump:
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


def detect_lumps(
    rho: np.ndarray,
    dx: float,
    frame_index: int,
    time: float,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
) -> List[Lump]:
    """
    Detect localized high-density lumps in a 2D density field.

    We use a simple, data-driven rule:

      - compute mean and std of rho,
      - threshold at rho > mean + sigma_threshold * std,
      - label connected components (8-connectivity),
      - discard very small regions.

    No shape or profile is assumed; we just use the field itself.

    Args:
        rho            : 2D |psi|^2 array (CPU).
        dx             : lattice spacing.
        frame_index    : integer frame index.
        time           : physical time for this frame.
        sigma_threshold: threshold above mean in std units.
        min_pixels     : minimum pixel count for a lump.

    Returns:
        List of Lump objects.
    """
    assert rho.ndim == 2
    ny, nx = rho.shape

    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
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

        # RMS radius in physical units
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
            Lump(
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


# ----------------------------------------------------------------------
# Topological proxy: winding number around a lump
# ----------------------------------------------------------------------

def bilinear_sample(field: np.ndarray, x: float, y: float) -> float:
    """
    Bilinear sample of a 2D scalar field at (x, y) in index coordinates.

    Args:
        field: 2D array, indexed as field[y, x].
        x, y : float indices (0 <= x < nx, 0 <= y < ny).

    Returns:
        Interpolated scalar value.
    """
    ny, nx = field.shape
    if x < 0 or x >= nx - 1 or y < 0 or y >= ny - 1:
        # Clamp or wrap; here we wrap periodically
        x = x % nx
        y = y % ny

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny

    dx = x - x0
    dy = y - y0

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    return (
        f00 * (1 - dx) * (1 - dy)
        + f10 * dx * (1 - dy)
        + f01 * (1 - dx) * dy
        + f11 * dx * dy
    )


def estimate_winding_number(
    phase: np.ndarray,
    x_center: float,
    y_center: float,
    radius_pixels: float,
    n_points: int = 128,
) -> float:
    """
    Estimate phase winding number around a closed loop of given radius.

    We:
      - sample the phase arg(psi) along a circle of radius_pixels
        centered at (x_center, y_center) in index coordinates,
      - unwrap the phase along the loop,
      - compute total Δθ,
      - return winding ≈ Δθ / (2π).

    This is a *measurement only*; we do not assume a vortex profile.
    """
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    phases = []
    for th in thetas:
        x = x_center + radius_pixels * np.cos(th)
        y = y_center + radius_pixels * np.sin(th)
        phi_sample = bilinear_sample(phase, x, y)
        phases.append(phi_sample)

    phases = np.unwrap(np.array(phases))
    total_delta = phases[-1] - phases[0]
    winding = total_delta / (2 * np.pi)
    return winding


@dataclass
class PairRecord:
    frame_index: int
    time: float
    n_lumps: int
    x1: float
    y1: float
    m1: float
    r1: float
    w1: float
    x2: float
    y2: float
    m2: float
    r2: float
    w2: float
    separation: float


@dataclass
class CountRecord:
    frame_index: int
    time: float
    n_lumps: int


# ----------------------------------------------------------------------
# Main exploration run
# ----------------------------------------------------------------------

def run_exploration(
    L: int = 128,
    dx: float = 1.0,
    dt: float = 0.005,
    g_defrag: float = 1.0,
    v: float = 1.0,
    lambda_param: float = 0.5,
    n_steps: int = 3000,
    analysis_interval: int = 20,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
    init_mean: float = 1.0,
    init_noise_amp: float = 0.1,
    init_seed: int = 42,
    output_dir: str = "substrate_interactions_output",
):
    """
    Run a long evolution and periodically measure:

      - number of lumps,
      - lump properties,
      - pair separation for top two lumps,
      - winding number around each of the top two lumps.

    Everything is driven by the scalar + defrag Hamiltonian.
    We only measure; we do not impose any extra structure.

    Outputs:
      - lumps_all.csv   : all detected lumps over time
      - pair_tracks.csv : separation + winding vs time for heaviest pair
      - counts.csv      : lump count vs time
      - pair_tracks.png : quick-look plot of separation vs time
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Substrate Interactions Explorer ")
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

    all_lumps: List[Lump] = []
    pair_records: List[PairRecord] = []
    count_records: List[CountRecord] = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % analysis_interval == 0 or step == n_steps:
            # Compute defrag potential (needed for energy diagnostics if you wish later)
            Phi = sim.solve_defrag_potential(psi)

            # Get CPU view of psi for analysis
            if CUPY_AVAILABLE and isinstance(psi, cp.ndarray):
                psi_cpu = cp.asnumpy(psi)
            else:
                psi_cpu = psi

            rho = np.abs(psi_cpu) ** 2
            phase = np.angle(psi_cpu)

            # Lump detection
            lumps = detect_lumps(
                rho,
                dx=dx,
                frame_index=step,
                time=t,
                sigma_threshold=sigma_threshold,
                min_pixels=min_pixels,
            )
            all_lumps.extend(lumps)

            count_records.append(
                CountRecord(frame_index=step, time=t, n_lumps=len(lumps))
            )

            print(
                f"[ANALYSIS] step={step:5d}, t={t:7.3f}, n_lumps={len(lumps)}"
            )

            # If at least two lumps exist, take the two heaviest and measure separation + winding
            if len(lumps) >= 2:
                lumps_sorted = sorted(lumps, key=lambda L_: L_.mass, reverse=True)
                L1, L2 = lumps_sorted[0], lumps_sorted[1]

                # Positions in physical units (already x,y)
                x1, y1 = L1.x, L1.y
                x2, y2 = L2.x, L2.y

                # Separation with periodic boundary conditions
                # in index space, then convert to physical units
                nx = L  # grid size
                ny = L
                # Convert back to index coordinates:
                cx1 = x1 / dx
                cy1 = y1 / dx
                cx2 = x2 / dx
                cy2 = y2 / dx

                # minimal-image separation in index units
                dx_idx = (cx2 - cx1 + nx / 2.0) % nx - nx / 2.0
                dy_idx = (cy2 - cy1 + ny / 2.0) % ny - ny / 2.0
                sep = np.sqrt((dx_idx * dx) ** 2 + (dy_idx * dx) ** 2)

                # Winding numbers: sample phase around each lump
                # Use a radius a bit larger than their RMS radius
                r1_pix = max(L1.radius / dx * 1.5, 2.0)
                r2_pix = max(L2.radius / dx * 1.5, 2.0)

                w1 = estimate_winding_number(
                    phase,
                    x_center=cx1,
                    y_center=cy1,
                    radius_pixels=r1_pix,
                    n_points=128,
                )
                w2 = estimate_winding_number(
                    phase,
                    x_center=cx2,
                    y_center=cy2,
                    radius_pixels=r2_pix,
                    n_points=128,
                )

                pair_records.append(
                    PairRecord(
                        frame_index=step,
                        time=t,
                        n_lumps=len(lumps),
                        x1=x1,
                        y1=y1,
                        m1=L1.mass,
                        r1=L1.radius,
                        w1=w1,
                        x2=x2,
                        y2=y2,
                        m2=L2.mass,
                        r2=L2.radius,
                        w2=w2,
                        separation=sep,
                    )
                )
                print(
                    f"    Heaviest pair: sep={sep:7.3f}, "
                    f"m1={L1.mass:.3e}, m2={L2.mass:.3e}, "
                    f"w1≈{w1:.2f}, w2≈{w2:.2f}"
                )

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if all_lumps:
        df_lumps = pd.DataFrame([asdict(l) for l in all_lumps])
        lumps_csv = out_dir / "lumps_all.csv"
        df_lumps.to_csv(lumps_csv, index=False)
        print(f"[RESULT] Saved all lumps to {lumps_csv}")
    else:
        print("[WARN] No lumps detected over run; lumps_all.csv not created.")

    if pair_records:
        df_pairs = pd.DataFrame([asdict(p) for p in pair_records])
        pairs_csv = out_dir / "pair_tracks.csv"
        df_pairs.to_csv(pairs_csv, index=False)
        print(f"[RESULT] Saved pair tracks to {pairs_csv}")

        # Quick-look plot: separation vs time
        plt.figure(figsize=(6, 4))
        plt.plot(df_pairs["time"], df_pairs["separation"], "-o", markersize=3)
        plt.xlabel("time")
        plt.ylabel("separation (physical units)")
        plt.title("Heaviest pair separation vs time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pairs_png = out_dir / "pair_tracks.png"
        plt.savefig(pairs_png, dpi=200)
        plt.close()
        print(f"[RESULT] Saved separation plot to {pairs_png}")
    else:
        print("[WARN] No pair data recorded; pair_tracks.csv not created.")

    df_counts = pd.DataFrame([asdict(c) for c in count_records])
    counts_csv = out_dir / "counts.csv"
    df_counts.to_csv(counts_csv, index=False)
    print(f"[RESULT] Saved lump counts to {counts_csv}")

    print("=" * 72)
    print(" Exploration complete ")
    print(f" Results in: {out_dir}")
    print("=" * 72)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Explore emergent interactions and topological changes in the "
            "scalar + defrag substrate, without hand-coding forces."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--g_defrag", type=float, default=1.0,
                        help="Defrag coupling strength.")
    parser.add_argument("--v", type=float, default=1.0,
                        help="Vacuum expectation value.")
    parser.add_argument("--lambda_param", type=float, default=0.5,
                        help="Self-interaction strength in the Mexican-hat potential.")
    parser.add_argument("--n_steps", type=int, default=3000,
                        help="Total number of time steps.")
    parser.add_argument("--analysis_interval", type=int, default=20,
                        help="How often to analyze lumps and winding.")
    parser.add_argument("--sigma_threshold", type=float, default=2.0,
                        help="Threshold above mean (in σ) for lump detection.")
    parser.add_argument("--min_pixels", type=int, default=8,
                        help="Minimum pixel count for a lump.")
    parser.add_argument("--init_mean", type=float, default=1.0,
                        help="Initial mean amplitude.")
    parser.add_argument("--init_noise_amp", type=float, default=0.1,
                        help="Initial noise amplitude.")
    parser.add_argument("--init_seed", type=int, default=42,
                        help="Random seed for initial condition.")
    parser.add_argument("--output_dir", type=str,
                        default="substrate_interactions_output",
                        help="Directory to store outputs.")

    args = parser.parse_args()

    run_exploration(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        g_defrag=args.g_defrag,
        v=args.v,
        lambda_param=args.lambda_param,
        n_steps=args.n_steps,
        analysis_interval=args.analysis_interval,
        sigma_threshold=args.sigma_threshold,
        min_pixels=args.min_pixels,
        init_mean=args.init_mean,
        init_noise_amp=args.init_noise_amp,
        init_seed=args.init_seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
