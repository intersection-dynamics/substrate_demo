#!/usr/bin/env python3
"""
substrate_interactions_explorer_ids.py

Like the earlier explorer, but with PERSISTENT lump IDs.

We:
  - evolve ScalarFieldDefragGPU,
  - detect density lumps in |psi|^2 at analysis intervals,
  - assign each lump a global_id that persists across frames
    using nearest-neighbor matching in position,
  - record lump properties to lumps_all.csv,
  - record the heaviest pair at each analysis frame to
    pair_tracks.csv, including their global IDs (id1, id2),
  - record lump counts vs time to counts.csv.

NO new forces, NO hand-coded interactions.
We only measure what the substrate is already doing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CUPY_AVAILABLE = False
    cp = None  # type: ignore

from scalar_field_defrag_gpu import ScalarFieldDefragGPU


# ----------------------------------------------------------------------
# Lump detection (same as before, but no IDs yet)
# ----------------------------------------------------------------------

@dataclass
class LumpDetection:
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
) -> List[LumpDetection]:
    """Detect high-density lumps in |psi|^2 with a simple threshold + labeling."""
    assert rho.ndim == 2
    ny, nx = rho.shape

    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_labels = ndi.label(mask, structure=structure)

    lumps: List[LumpDetection] = []
    voxel_area = dx * dx

    for lab in range(1, n_labels + 1):
        region = (labeled == lab)
        n_pix = int(region.sum())
        if n_pix < min_pixels:
            continue

        ys, xs = np.nonzero(region)
        x_mean = xs.mean()
        y_mean = ys.mean()

        x_phys = x_mean * dx
        y_phys = y_mean * dx

        # RMS radius in physical units
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
            LumpDetection(
                frame_index=frame_index,
                time=time,
                id_local=lab,
                x=float(x_phys),
                y=float(y_phys),
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
# Persistent tracking
# ----------------------------------------------------------------------

@dataclass
class TrackedLump:
    global_id: int
    x: float
    y: float
    radius: float
    mass: float


@dataclass
class LumpRecord:
    frame_index: int
    time: float
    global_id: int
    x: float
    y: float
    radius: float
    mass: float
    peak_rho: float
    mean_rho: float
    area: float
    n_pixels: int


@dataclass
class PairRecord:
    frame_index: int
    time: float
    n_lumps: int
    id1: int
    x1: float
    y1: float
    m1: float
    r1: float
    w1: float
    id2: int
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


def bilinear_sample(field: np.ndarray, x: float, y: float) -> float:
    ny, nx = field.shape
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
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    phases = []
    for th in thetas:
        x = x_center + radius_pixels * np.cos(th)
        y = y_center + radius_pixels * np.sin(th)
        phi_sample = bilinear_sample(phase, x, y)
        phases.append(phi_sample)
    phases = np.unwrap(np.array(phases))
    total_delta = phases[-1] - phases[0]
    return float(total_delta / (2 * np.pi))


def match_lumps(
    prev: Dict[int, TrackedLump],
    detections: List[LumpDetection],
    dx: float,
    max_match_distance: float = 4.0,
) -> Dict[int, TrackedLump]:
    """
    Greedy nearest-neighbor matching from previous tracked lumps to new detections.

    prev: dict global_id -> TrackedLump (from previous analysis frame)
    detections: list of LumpDetection from current frame
    max_match_distance: max distance (in physical units) to consider a match

    Returns:
        new_prev: dict global_id -> updated TrackedLump for this frame
    """
    new_prev: Dict[int, TrackedLump] = {}
    if not detections:
        return new_prev

    det_coords = np.array([[d.x, d.y] for d in detections])
    used = np.zeros(len(detections), dtype=bool)

    # Try to match old IDs
    for gid, tl in prev.items():
        # distances from old position to all detections
        dxs = det_coords[:, 0] - tl.x
        dys = det_coords[:, 1] - tl.y
        dists = np.sqrt(dxs**2 + dys**2)
        idx = np.argmin(dists)
        if dists[idx] <= max_match_distance and not used[idx]:
            d = detections[idx]
            used[idx] = True
            new_prev[gid] = TrackedLump(
                global_id=gid,
                x=d.x,
                y=d.y,
                radius=d.radius,
                mass=d.mass,
            )

    # Any remaining detections get new IDs
    max_gid = max(prev.keys(), default=-1)
    next_gid = max_gid + 1
    for i, d in enumerate(detections):
        if used[i]:
            continue
        gid = next_gid
        next_gid += 1
        new_prev[gid] = TrackedLump(
            global_id=gid,
            x=d.x,
            y=d.y,
            radius=d.radius,
            mass=d.mass,
        )

    return new_prev


# ----------------------------------------------------------------------
# Main exploration
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
    output_dir: str = "substrate_interactions_ids_output",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" Substrate Interactions Explorer (Persistent IDs) ")
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

    tracked_prev: Dict[int, TrackedLump] = {}
    all_lump_records: List[LumpRecord] = []
    pair_records: List[PairRecord] = []
    count_records: List[CountRecord] = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % analysis_interval == 0 or step == n_steps:
            Phi = sim.solve_defrag_potential(psi)

            if CUPY_AVAILABLE and isinstance(psi, cp.ndarray):
                psi_cpu = cp.asnumpy(psi)
            else:
                psi_cpu = psi

            rho = np.abs(psi_cpu) ** 2
            phase = np.angle(psi_cpu)

            detections = detect_lumps(
                rho,
                dx=dx,
                frame_index=step,
                time=t,
                sigma_threshold=sigma_threshold,
                min_pixels=min_pixels,
            )

            # Update tracking
            tracked_now = match_lumps(tracked_prev, detections, dx=dx)
            tracked_prev = tracked_now

            # Save lump records with global IDs
            n_lumps = len(tracked_now)
            count_records.append(
                CountRecord(frame_index=step, time=t, n_lumps=n_lumps)
            )

            print(
                f"[ANALYSIS] step={step:5d}, t={t:7.3f}, n_lumps={n_lumps}"
            )

            # Build map from global_id to detection for recording extra info
            det_map = {}
            for d in detections:
                # Find which global_id matched this detection (if any)
                # by nearest neighbor among tracked_now
                best_gid = None
                best_dist = np.inf
                for gid, tl in tracked_now.items():
                    if abs(tl.x - d.x) + abs(tl.y - d.y) < best_dist:
                        best_dist = abs(tl.x - d.x) + abs(tl.y - d.y)
                        best_gid = gid
                if best_gid is not None:
                    det_map[best_gid] = d

            for gid, tl in tracked_now.items():
                d = det_map.get(gid, None)
                if d is None:
                    # Should be rare; fall back to tl values only
                    peak = mean = area = 0.0
                    n_pix = 0
                else:
                    peak = d.peak_rho
                    mean = d.mean_rho
                    area = d.area
                    n_pix = d.n_pixels

                all_lump_records.append(
                    LumpRecord(
                        frame_index=step,
                        time=t,
                        global_id=gid,
                        x=tl.x,
                        y=tl.y,
                        radius=tl.radius,
                        mass=tl.mass,
                        peak_rho=peak,
                        mean_rho=mean,
                        area=area,
                        n_pixels=n_pix,
                    )
                )

            # Heaviest pair at this frame (but now with global IDs)
            if len(tracked_now) >= 2:
                # Sort by mass
                sorted_lumps = sorted(
                    tracked_now.values(),
                    key=lambda tl: tl.mass,
                    reverse=True,
                )
                tl1, tl2 = sorted_lumps[0], sorted_lumps[1]

                # Positions in physical space
                x1, y1 = tl1.x, tl1.y
                x2, y2 = tl2.x, tl2.y

                # Convert to index coords for winding + minimal-image sep
                cx1, cy1 = x1 / dx, y1 / dx
                cx2, cy2 = x2 / dx, y2 / dx
                nx = L
                ny = L

                dx_idx = (cx2 - cx1 + nx / 2.0) % nx - nx / 2.0
                dy_idx = (cy2 - cy1 + ny / 2.0) % ny - ny / 2.0
                sep = np.sqrt((dx_idx * dx) ** 2 + (dy_idx * dx) ** 2)

                # Winding numbers
                r1_pix = max(tl1.radius / dx * 1.5, 2.0)
                r2_pix = max(tl2.radius / dx * 1.5, 2.0)
                w1 = estimate_winding_number(
                    phase, cx1, cy1, radius_pixels=r1_pix, n_points=128
                )
                w2 = estimate_winding_number(
                    phase, cx2, cy2, radius_pixels=r2_pix, n_points=128
                )

                pair_records.append(
                    PairRecord(
                        frame_index=step,
                        time=t,
                        n_lumps=n_lumps,
                        id1=tl1.global_id,
                        x1=x1,
                        y1=y1,
                        m1=tl1.mass,
                        r1=tl1.radius,
                        w1=w1,
                        id2=tl2.global_id,
                        x2=x2,
                        y2=y2,
                        m2=tl2.mass,
                        r2=tl2.radius,
                        w2=w2,
                        separation=sep,
                    )
                )
                print(
                    f"    Heaviest pair: IDs=({tl1.global_id},{tl2.global_id}), "
                    f"sep={sep:7.3f}, m1={tl1.mass:.3e}, m2={tl2.mass:.3e}, "
                    f"w1≈{w1:.2f}, w2≈{w2:.2f}"
                )

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    # Save outputs
    if all_lump_records:
        df_lumps = pd.DataFrame([asdict(lr) for lr in all_lump_records])
        lumps_csv = out_dir / "lumps_all.csv"
        df_lumps.to_csv(lumps_csv, index=False)
        print(f"[RESULT] Saved all lumps to {lumps_csv}")

    if pair_records:
        df_pairs = pd.DataFrame([asdict(pr) for pr in pair_records])
        pairs_csv = out_dir / "pair_tracks.csv"
        df_pairs.to_csv(pairs_csv, index=False)
        print(f"[RESULT] Saved pair tracks to {pairs_csv}")

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

    df_counts = pd.DataFrame([asdict(cr) for cr in count_records])
    counts_csv = out_dir / "counts.csv"
    df_counts.to_csv(counts_csv, index=False)
    print(f"[RESULT] Saved lump counts to {counts_csv}")

    print("=" * 72)
    print(" Exploration complete ")
    print(f" Results in: {out_dir}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Explore emergent interactions with persistent lump IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--g_defrag", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--lambda_param", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=3000)
    parser.add_argument("--analysis_interval", type=int, default=20)
    parser.add_argument("--sigma_threshold", type=float, default=2.0)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument("--init_mean", type=float, default=1.0)
    parser.add_argument("--init_noise_amp", type=float, default=0.1)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="substrate_interactions_ids_output",
    )

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
