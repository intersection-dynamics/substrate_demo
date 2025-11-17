#!/usr/bin/env python3
"""
proton_tracker.py

Standalone "proton" (lump) finder and tracker for the ScalarFieldDefragGPU
substrate simulation.

- Uses your existing scalar_field_defrag_gpu.ScalarFieldDefragGPU
- Runs its own evolution loop (RK4) so we can snapshot psi frequently
- On each snapshot:
    * Pulls psi to CPU
    * Computes density  rho = |psi|^2
    * Detects localized high-density lumps ("protons")
    * Stores their positions & properties
- After the run:
    * Links lumps across snapshots into tracks (nearest-neighbor)
    * Saves:
        - proton_candidates.csv    (all lumps in all frames)
        - proton_tracks.csv        (worldlines)
        - proton_tracks_3d.png     (x,y vs time)
        - proton_tracks_xy.png     (x,y overlay at all times)
"""

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import ndimage as ndi

import scalar_field_defrag_gpu as sfg  # your existing module


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProtonCandidate:
    """Single proton-like density lump in one snapshot."""
    frame_index: int
    time: float
    id_local: int           # label id within this frame
    x: float                # centroid x (physical)
    y: float                # centroid y (physical)
    radius: float           # rms radius (physical)
    mass: float             # integrated density * dx^2
    peak_rho: float         # max density in lump
    mean_rho: float         # average density in lump
    area: float             # lump area in physical units
    n_pixels: int           # number of lattice sites


@dataclass
class ProtonTrack:
    """A proton tracked over multiple frames."""
    track_id: int
    frames: List[int]
    times: List[float]
    xs: List[float]
    ys: List[float]
    radii: List[float]
    masses: List[float]
    peak_rhos: List[float]


# ---------------------------------------------------------------------------
# Lump detection on a single 2D density field
# ---------------------------------------------------------------------------

def find_lumps_in_density(
    rho: np.ndarray,
    dx: float,
    frame_index: int,
    time: float,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
) -> List[ProtonCandidate]:
    """
    Detect localized high-density lumps in a 2D density field.

    Args:
        rho: 2D numpy array (|psi|^2 on CPU)
        dx:  lattice spacing (physical units)
        frame_index: integer frame id
        time: physical time for this frame
        sigma_threshold: threshold = mean + sigma_threshold*std
        min_pixels: minimum connected pixels to count as lump

    Returns:
        List of ProtonCandidate objects.
    """
    assert rho.ndim == 2, "rho must be 2D"

    mu = rho.mean()
    sigma = rho.std()
    thresh = mu + sigma_threshold * sigma

    mask = rho > thresh

    # connected components (4-connectivity or 8; we use 8 for smoother blobs)
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_labels = ndi.label(mask, structure=structure)

    candidates: List[ProtonCandidate] = []
    voxel_area = dx * dx

    for label_id in range(1, n_labels + 1):
        region_mask = (labeled == label_id)
        n_pix = int(region_mask.sum())
        if n_pix < min_pixels:
            continue

        ys_idx, xs_idx = np.nonzero(region_mask)  # note: row = y, col = x

        # centroid in index space
        x_mean_idx = xs_idx.mean()
        y_mean_idx = ys_idx.mean()

        # physical coords
        x_phys = x_mean_idx * dx
        y_phys = y_mean_idx * dx

        # rms radius
        dxs = (xs_idx - x_mean_idx) * dx
        dys = (ys_idx - y_mean_idx) * dx
        r2 = dxs * dxs + dys * dys
        r_rms = float(np.sqrt(r2.mean()))

        # density stats
        rho_region = rho[region_mask]
        mass = float(rho_region.sum() * voxel_area)
        peak = float(rho_region.max())
        mean = float(rho_region.mean())
        area = float(n_pix * voxel_area)

        cand = ProtonCandidate(
            frame_index=frame_index,
            time=time,
            id_local=label_id,
            x=float(x_phys),
            y=float(y_phys),
            radius=r_rms,
            mass=mass,
            peak_rho=peak,
            mean_rho=mean,
            area=area,
            n_pixels=n_pix,
        )
        candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# Tracking across time
# ---------------------------------------------------------------------------

def track_protons(
    all_candidates: List[List[ProtonCandidate]],
    max_link_distance: float = 3.0,
) -> List[ProtonTrack]:
    """
    Link proton candidates across frames into tracks via nearest-neighbor.

    Args:
        all_candidates: list of candidate lists, one per frame (ordered in time)
        max_link_distance: maximum allowed spatial jump between frames
                           (in same physical units as x,y)

    Returns:
        List of ProtonTrack objects.
    """
    tracks: List[ProtonTrack] = []
    next_track_id = 0

    if not all_candidates:
        return tracks

    # Initialize tracks from first frame
    for cand in all_candidates[0]:
        tracks.append(
            ProtonTrack(
                track_id=next_track_id,
                frames=[cand.frame_index],
                times=[cand.time],
                xs=[cand.x],
                ys=[cand.y],
                radii=[cand.radius],
                masses=[cand.mass],
                peak_rhos=[cand.peak_rho],
            )
        )
        next_track_id += 1

    # Process subsequent frames
    for f_idx in range(1, len(all_candidates)):
        cands = all_candidates[f_idx]
        if not cands:
            continue

        cand_positions = np.array([[c.x, c.y] for c in cands])
        used_indices = set()

        # Try to extend existing tracks
        for tr in tracks:
            last_pos = np.array([tr.xs[-1], tr.ys[-1]])

            d2 = np.sum((cand_positions - last_pos) ** 2, axis=1)
            order = np.argsort(d2)

            assigned = False
            for idx in order:
                if idx in used_indices:
                    continue
                dist = float(np.sqrt(d2[idx]))
                if dist <= max_link_distance:
                    c = cands[idx]
                    tr.frames.append(c.frame_index)
                    tr.times.append(c.time)
                    tr.xs.append(c.x)
                    tr.ys.append(c.y)
                    tr.radii.append(c.radius)
                    tr.masses.append(c.mass)
                    tr.peak_rhos.append(c.peak_rho)
                    used_indices.add(idx)
                    assigned = True
                    break

            # if not assigned, track simply doesn't continue in this frame

        # Start new tracks for unassigned candidates
        for idx, c in enumerate(cands):
            if idx in used_indices:
                continue
            tracks.append(
                ProtonTrack(
                    track_id=next_track_id,
                    frames=[c.frame_index],
                    times=[c.time],
                    xs=[c.x],
                    ys=[c.y],
                    radii=[c.radius],
                    masses=[c.mass],
                    peak_rhos=[c.peak_rho],
                )
            )
            next_track_id += 1

    return tracks


# ---------------------------------------------------------------------------
# IO helpers: saving results and plots
# ---------------------------------------------------------------------------

def save_candidates_csv(all_candidates: List[List[ProtonCandidate]], out_path: Path) -> None:
    rows: List[Dict] = []
    for frame_cands in all_candidates:
        for c in frame_cands:
            rows.append(asdict(c))

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "frame_index",
                "time",
                "id_local",
                "x",
                "y",
                "radius",
                "mass",
                "peak_rho",
                "mean_rho",
                "area",
                "n_pixels",
            ]
        )

    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(df)} proton candidates to {out_path}")


def save_tracks_csv(tracks: List[ProtonTrack], out_path: Path) -> None:
    rows: List[Dict] = []
    for tr in tracks:
        for i in range(len(tr.frames)):
            rows.append(
                {
                    "track_id": tr.track_id,
                    "frame_index": tr.frames[i],
                    "time": tr.times[i],
                    "x": tr.xs[i],
                    "y": tr.ys[i],
                    "radius": tr.radii[i],
                    "mass": tr.masses[i],
                    "peak_rho": tr.peak_rhos[i],
                }
            )

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "track_id",
                "frame_index",
                "time",
                "x",
                "y",
                "radius",
                "mass",
                "peak_rho",
            ]
        )

    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(df)} track points to {out_path}")


def plot_tracks_3d(tracks: List[ProtonTrack], out_path: Path) -> None:
    if not tracks:
        print("[WARN] No tracks to plot (3D)")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for tr in tracks:
        ax.plot(tr.xs, tr.ys, tr.times, marker="o", linestyle="-", alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("time")
    ax.set_title("Proton Tracks (x, y, t)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved 3D track plot to {out_path}")


def plot_tracks_xy(tracks: List[ProtonTrack], out_path: Path) -> None:
    if not tracks:
        print("[WARN] No tracks to plot (x-y)")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    for tr in tracks:
        ax.plot(tr.xs, tr.ys, marker="o", linestyle="-", alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Proton Tracks (x-y projection)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved x-y track plot to {out_path}")


# ---------------------------------------------------------------------------
# Main simulation + tracking loop
# ---------------------------------------------------------------------------

def run_simulation_and_track(
    L: int = 128,
    dx: float = 1.0,
    dt: float = 0.005,
    g_defrag: float = 1.0,
    v: float = 1.0,
    lambda_param: float = 0.0,
    n_steps: int = 2000,
    sample_every: int = 20,
    sigma_threshold: float = 2.0,
    min_pixels: int = 8,
    max_link_distance: float = 3.0,
    output_dir: str = "proton_output",
    init_mean: float = 1.0,
    init_noise_amp: float = 0.1,
    init_seed: int = 42,
) -> None:
    """
    Run ScalarFieldDefragGPU evolution and track proton-like lumps.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" PROTON TRACKER – ScalarFieldDefragGPU ")
    print("=" * 70)
    print(f"L = {L}, dx = {dx}, dt = {dt}")
    print(f"g_defrag = {g_defrag}, v = {v}, lambda = {lambda_param}")
    print(f"n_steps = {n_steps}, sample_every = {sample_every}")
    print(f"sigma_threshold = {sigma_threshold}, min_pixels = {min_pixels}")
    print(f"max_link_distance = {max_link_distance}")
    print("=" * 70)

    # create simulator
    sim = sfg.ScalarFieldDefragGPU(
        L=L,
        dx=dx,
        dt=dt,
        g_defrag=g_defrag,
        v=v,
        lambda_param=lambda_param,
    )

    # initial condition: uniform + noise
    psi = sim.create_uniform_noise(mean=init_mean, noise_amp=init_noise_amp, seed=init_seed)

    all_candidates: List[List[ProtonCandidate]] = []

    # evolution loop
    for step in range(n_steps + 1):
        t = step * dt

        if step % sample_every == 0 or step == n_steps:
            # Bring psi to CPU
            if sfg.GPU_AVAILABLE:
                psi_cpu = sfg.cp.asnumpy(psi)
            else:
                psi_cpu = psi

            rho = np.abs(psi_cpu) ** 2

            cands = find_lumps_in_density(
                rho,
                dx=dx,
                frame_index=step,
                time=t,
                sigma_threshold=sigma_threshold,
                min_pixels=min_pixels,
            )
            all_candidates.append(cands)

            print(
                f"[SNAP] step={step:5d}, t={t:.3f}, "
                f"num_lumps={len(cands)}, "
                f"rho_max={rho.max():.4f}"
            )

        if step < n_steps:
            psi, _ = sim.evolve_step_rk4(psi)

    # build tracks
    tracks = track_protons(all_candidates, max_link_distance=max_link_distance)
    print(f"[INFO] Built {len(tracks)} tracks.")

    # save results
    save_candidates_csv(all_candidates, out_dir / "proton_candidates.csv")
    save_tracks_csv(tracks, out_dir / "proton_tracks.csv")
    plot_tracks_3d(tracks, out_dir / "proton_tracks_3d.png")
    plot_tracks_xy(tracks, out_dir / "proton_tracks_xy.png")

    print("=" * 70)
    print(" Proton tracking complete ")
    print(f" Results in: {out_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Proton (lump) finder/tracker for ScalarFieldDefragGPU."
    )
    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--g_defrag", type=float, default=1.0, help="Defrag coupling.")
    parser.add_argument("--v", type=float, default=1.0, help="Vacuum expectation value.")
    parser.add_argument(
        "--lambda_param", type=float, default=0.0, help="Self-interaction strength."
    )
    parser.add_argument("--n_steps", type=int, default=2000, help="Number of time steps.")
    parser.add_argument(
        "--sample_every", type=int, default=20, help="Snapshot interval in steps."
    )
    parser.add_argument(
        "--sigma_threshold",
        type=float,
        default=2.0,
        help="Density threshold = mean + sigma_threshold * std.",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=8,
        help="Minimum connected pixels to count as a lump.",
    )
    parser.add_argument(
        "--max_link_distance",
        type=float,
        default=3.0,
        help="Maximum spatial jump between frames when linking tracks.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="proton_output", help="Output directory."
    )
    parser.add_argument(
        "--init_mean",
        type=float,
        default=1.0,
        help="Mean initial amplitude for uniform noise state.",
    )
    parser.add_argument(
        "--init_noise_amp",
        type=float,
        default=0.1,
        help="Noise amplitude around mean in initial state.",
    )
    parser.add_argument(
        "--init_seed", type=int, default=42, help="Random seed for initial condition."
    )

    args = parser.parse_args()

    run_simulation_and_track(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        g_defrag=args.g_defrag,
        v=args.v,
        lambda_param=args.lambda_param,
        n_steps=args.n_steps,
        sample_every=args.sample_every,
        sigma_threshold=args.sigma_threshold,
        min_pixels=args.min_pixels,
        max_link_distance=args.max_link_distance,
        output_dir=args.output_dir,
        init_mean=args.init_mean,
        init_noise_amp=args.init_noise_amp,
        init_seed=args.init_seed,
    )


if __name__ == "__main__":
    main()
