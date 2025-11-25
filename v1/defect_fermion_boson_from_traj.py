#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_frames_from_traj(traj_csv):
    """
    Load trajectories CSV and regroup as per-time 'frames'.

    Assumes columns (with header):
      traj_id,charge,time,x,y

    Returns:
      times_unique: sorted unique times
      frame_x: list of np.arrays of x positions per frame
      frame_y: list of np.arrays of y positions per frame
      frame_q: list of np.arrays of charges per frame
    """
    data = np.genfromtxt(traj_csv, delimiter=",", names=True)

    required = ["traj_id", "charge", "time", "x", "y"]
    for name in required:
        if name not in data.dtype.names:
            raise KeyError(f"CSV {traj_csv} missing required column '{name}'")

    times = data["time"]
    xs    = data["x"]
    ys    = data["y"]
    qs    = data["charge"].astype(int)

    # Group by time (treat each unique time as a frame)
    times_unique = np.unique(times)
    frame_x = []
    frame_y = []
    frame_q = []

    for t in times_unique:
        mask = (times == t)
        frame_x.append(xs[mask].copy())
        frame_y.append(ys[mask].copy())
        frame_q.append(qs[mask].copy())

    return times_unique, frame_x, frame_y, frame_q


def pair_correlation_from_frames(times_unique, frame_x, frame_y, frame_q,
                                 Lx, Ly, dx,
                                 q_select=None,
                                 r_max=None, n_bins=50):
    """
    Compute radial pair correlation g(r) from per-time frames.

    times_unique: array of times (not actually used except for counting frames)
    frame_x, frame_y, frame_q: lists per frame
    Lx, Ly: physical box size
    dx: grid spacing if x,y are indices (we multiply x,y by dx)
    q_select: None -> all charges, 1 -> only q=+1, -1 -> only q=-1
    r_max: max radius; if None, use min(Lx,Ly)/2
    n_bins: radial bins
    """
    n_frames = len(times_unique)
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly)

    dr = r_max / n_bins
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    counts = np.zeros(n_bins, dtype=float)

    V = Lx * Ly
    n_frames_used = 0
    n_per_frame_total = 0
    n_pairs_total = 0

    for k in range(n_frames):
        x = frame_x[k]
        y = frame_y[k]
        q = frame_q[k]

        if q_select is not None:
            mask = (q == q_select)
            x = x[mask]
            y = y[mask]

        if x.size < 2:
            continue

        # to physical coords if needed
        x_phys = x * dx
        y_phys = y * dx

        N = x_phys.size
        n_frames_used += 1
        n_per_frame_total += N

        # O(N^2) pair distances with periodic BCs
        for i in range(N):
            dx_ = x_phys[i] - x_phys[i+1:]
            dy_ = y_phys[i] - y_phys[i+1:]

            dx_ -= Lx * np.round(dx_ / Lx)
            dy_ -= Ly * np.round(dy_ / Ly)

            r = np.sqrt(dx_ * dx_ + dy_ * dy_)
            r = r[r < r_max]
            if r.size == 0:
                continue

            hist, _ = np.histogram(r, bins=r_edges)
            counts += hist
            n_pairs_total += r.size

    if n_frames_used == 0 or n_pairs_total == 0:
        raise RuntimeError("Not enough pairs to build g(r).")

    # Average number density over used frames
    n_avg = (n_per_frame_total / n_frames_used) / V

    shell_areas = 2.0 * np.pi * r_centers * dr
    # Normalize so that Poisson → g(r) ~ 1
    norm = n_frames_used * (n_avg ** 2) * V * shell_areas
    g_r = counts / (norm + 1e-20)

    return r_centers, g_r


def cell_occupancy_from_frames(times_unique, frame_x, frame_y, frame_q,
                               Lx, Ly, dx,
                               n_cells_x=16, n_cells_y=16,
                               q_select=None):
    """
    Compute cell occupancy distribution + Fano factor from frames.

    Returns:
      F (Fano factor), mu, sigma2, occ_all (flattened occupancies)
    """
    cell_dx = Lx / n_cells_x
    cell_dy = Ly / n_cells_y

    occ_all = []

    for k in range(len(times_unique)):
        x = frame_x[k]
        y = frame_y[k]
        q = frame_q[k]

        if q_select is not None:
            mask = (q == q_select)
            x = x[mask]
            y = y[mask]

        if x.size == 0:
            continue

        x_phys = x * dx
        y_phys = y * dx

        ix = np.floor(x_phys / cell_dx).astype(int)
        iy = np.floor(y_phys / cell_dy).astype(int)

        ix = np.clip(ix, 0, n_cells_x - 1)
        iy = np.clip(iy, 0, n_cells_y - 1)

        counts = np.zeros((n_cells_x, n_cells_y), dtype=int)
        for j in range(x_phys.size):
            counts[ix[j], iy[j]] += 1

        occ_all.append(counts.ravel())

    if not occ_all:
        raise RuntimeError("No occupancies collected.")

    occ_all = np.concatenate(occ_all)
    mu = occ_all.mean()
    sigma2 = occ_all.var()
    F = sigma2 / (mu + 1e-12)

    return F, mu, sigma2, occ_all


def main():
    parser = argparse.ArgumentParser(
        description="Fermion/boson-like test from defect trajectories "
                    "using g(r) and occupancy statistics."
    )
    parser.add_argument("--traj_csv", type=str, required=True,
                        help="Path to *_defects_trajectories.csv")
    parser.add_argument("--Lx", type=float, required=True,
                        help="Physical box size in x")
    parser.add_argument("--Ly", type=float, required=True,
                        help="Physical box size in y")
    parser.add_argument("--dx", type=float, default=1.0,
                        help="Grid spacing (if x,y are indices)")
    parser.add_argument("--n_bins_r", type=int, default=40,
                        help="Number of radial bins for g(r)")
    parser.add_argument("--r_max", type=float, default=None,
                        help="Max radius for g(r); default=min(Lx,Ly)/2")
    parser.add_argument("--n_cells_x", type=int, default=16,
                        help="Cells in x for occupancy statistics")
    parser.add_argument("--n_cells_y", type=int, default=16,
                        help="Cells in y for occupancy statistics")
    parser.add_argument("--charge", type=str, default="all",
                        choices=["all", "plus", "minus"],
                        help="Which defect charge species to analyze")
    args = parser.parse_args()

    print(f"[INFO] Loading trajectories from {args.traj_csv} ...")
    times_unique, frame_x, frame_y, frame_q = load_frames_from_traj(args.traj_csv)
    print(f"[INFO] Built {len(times_unique)} frames from trajectories.")

    if args.charge == "all":
        q_sel = None
        label = "all charges"
    elif args.charge == "plus":
        q_sel = 1
        label = "q=+1"
    else:
        q_sel = -1
        label = "q=-1"

    print(f"[INFO] Analyzing {label}")

    # --- Pair correlation g(r) ---
    r_centers, g_r = pair_correlation_from_frames(
        times_unique,
        frame_x,
        frame_y,
        frame_q,
        Lx=args.Lx,
        Ly=args.Ly,
        dx=args.dx,
        q_select=q_sel,
        r_max=args.r_max,
        n_bins=args.n_bins_r,
    )

    g0 = g_r[0]
    print(f"[RESULT] g(0) ≈ {g0:.3f} "
          "(<1: anti-bunching / fermion-like, >1: bunching / boson-like, ~1: Poisson)")

    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, g_r, "-o", markersize=3)
    plt.axhline(1.0, color="k", linestyle="--", label="Poisson")
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title(f"Pair correlation g(r) [{label}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"g_r_{args.charge}.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved g(r) plot -> g_r_{args.charge}.png")

    # --- Cell occupancy + Fano ---
    F, mu, sigma2, occ_all = cell_occupancy_from_frames(
        times_unique,
        frame_x,
        frame_y,
        frame_q,
        Lx=args.Lx,
        Ly=args.Ly,
        dx=args.dx,
        n_cells_x=args.n_cells_x,
        n_cells_y=args.n_cells_y,
        q_select=q_sel,
    )

    print(f"[RESULT] Occupancy stats [{label}]:")
    print(f"        mean μ   = {mu:.3f}")
    print(f"        var  σ²  = {sigma2:.3f}")
    print(f"        Fano F   = σ²/μ ≈ {F:.3f} "
          "(<1: sub-Poisson / exclusion, >1: bunching, ~1: Poisson)")

    plt.figure(figsize=(6, 4))
    plt.hist(occ_all, bins=np.arange(0, occ_all.max() + 1.5) - 0.5)
    plt.xlabel("defects per cell")
    plt.ylabel("count")
    plt.title(f"Cell occupancy distribution [{label}]")
    plt.tight_layout()
    plt.savefig(f"occupancy_hist_{args.charge}.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved occupancy histogram -> occupancy_hist_{args.charge}.png")

    print("[DONE] Fermion/boson-like analysis complete.")

if __name__ == "__main__":
    main()
