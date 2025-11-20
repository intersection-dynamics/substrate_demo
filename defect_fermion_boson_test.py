#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_defects(csv_path):
    """
    Load defects CSV.

    Assumes columns: frame, t, x, y, charge, traj_id
    If your format differs, tweak the column indices below.
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    frames  = data[:, 0].astype(int)
    times   = data[:, 1]
    xs      = data[:, 2]
    ys      = data[:, 3]
    charges = data[:, 4].astype(int)
    # traj_id = data[:, 5].astype(int)  # not needed here

    return {
        "frame": frames,
        "t": times,
        "x": xs,
        "y": ys,
        "q": charges,
    }

def pair_correlation(defects, Lx, Ly, dx, q_select=None,
                     r_max=None, n_bins=50):
    """
    Compute radial pair correlation g(r) for chosen charge species.

    defects: dict from load_defects
    Lx, Ly: physical box size
    dx: grid spacing (used to convert indices to physical coords if needed)
    q_select: None -> all charges, +1 or -1 for specific species
    r_max: maximum radius (if None, use min(Lx, Ly) / 2)
    n_bins: number of radial bins
    """
    frames = defects["frame"]
    xs = defects["x"]
    ys = defects["y"]
    qs = defects["q"]

    if q_select is not None:
        mask = (qs == q_select)
        frames = frames[mask]
        xs = xs[mask]
        ys = ys[mask]

    # Use physical coordinates directly. If your x,y are indices, multiply by dx.
    xs = xs * dx
    ys = ys * dx

    unique_frames = np.unique(frames)

    if r_max is None:
        r_max = 0.5 * min(Lx, Ly)

    dr = r_max / n_bins
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    counts = np.zeros(n_bins, dtype=float)
    n_pairs_total = 0
    n_per_frame_total = 0
    n_frames_used = 0

    # Box volume
    V = Lx * Ly

    for f in unique_frames:
        mask = (frames == f)
        xf = xs[mask]
        yf = ys[mask]
        N = xf.size
        if N < 2:
            continue

        n_frames_used += 1
        n_per_frame_total += N

        # Compute all pair distances with periodic BCs
        # To keep it simple, do O(N^2). For N ~ 10^3 this is heavy but doable;
        # if too slow, you can thin frames or sample subset of pairs.
        for i in range(N):
            dx_ = xf[i] - xf[i+1:]
            dy_ = yf[i] - yf[i+1:]

            # Minimal image convention
            dx_ -= Lx * np.round(dx_ / Lx)
            dy_ -= Ly * np.round(dy_ / Ly)

            r = np.sqrt(dx_ * dx_ + dy_ * dy_)
            # Only consider pairs within r_max
            r = r[r < r_max]
            if r.size == 0:
                continue

            hist, _ = np.histogram(r, bins=r_edges)
            counts += hist
            n_pairs_total += r.size

    if n_frames_used == 0 or n_pairs_total == 0:
        raise RuntimeError("Not enough pairs to build g(r).")

    # Average number density
    n_avg = (n_per_frame_total / n_frames_used) / V

    # Normalize g(r):
    # g(r) = (1 / (N * n * 2πr dr)) * ΔN(r)
    # We approximate using average density and counts.
    shell_areas = 2.0 * np.pi * r_centers * dr
    # Expected number of pairs per shell for uniform distribution:
    # ΔN_exp(r) = N_frames * N_avg * n_avg * shell_area
    # But we already used n_avg and counts over all frames; a simpler
    # practical normalization is:
    # g(r) = counts / ( (n_avg^2 * V) * shell_area * N_frames )
    # which should give g=1 for Poisson.
    norm = n_frames_used * (n_avg ** 2) * V * shell_areas
    g_r = counts / norm

    return r_centers, g_r

def cell_occupancy_stats(defects, Lx, Ly, dx, n_cells_x=16, n_cells_y=16,
                         q_select=None):
    """
    Compute cell occupancy distribution and Fano factor.

    Returns:
      F (Fano factor), mu (mean occupancy), sigma2 (variance),
      occ_all (array of all cell occupancies across frames)
    """
    frames = defects["frame"]
    xs = defects["x"]
    ys = defects["y"]
    qs = defects["q"]

    if q_select is not None:
        mask = (qs == q_select)
        frames = frames[mask]
        xs = xs[mask]
        ys = ys[mask]

    xs = xs * dx
    ys = ys * dx

    unique_frames = np.unique(frames)

    cell_dx = Lx / n_cells_x
    cell_dy = Ly / n_cells_y

    occ_all = []

    for f in unique_frames:
        mask = (frames == f)
        xf = xs[mask]
        yf = ys[mask]
        if xf.size == 0:
            continue

        # Bin into cells
        ix = np.floor(xf / cell_dx).astype(int)
        iy = np.floor(yf / cell_dy).astype(int)
        ix = np.clip(ix, 0, n_cells_x - 1)
        iy = np.clip(iy, 0, n_cells_y - 1)
        counts = np.zeros((n_cells_x, n_cells_y), dtype=int)
        for k in range(xf.size):
            counts[ix[k], iy[k]] += 1

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
        description="Test for boson-like bunching vs fermion-like exclusion "
                    "in substrate defects using g(r) and occupancy statistics."
    )
    parser.add_argument("--defects_csv", type=str, required=True,
                        help="Path to *_defects_defects.csv")
    parser.add_argument("--Lx", type=float, required=True,
                        help="Physical box size in x")
    parser.add_argument("--Ly", type=float, required=True,
                        help="Physical box size in y")
    parser.add_argument("--dx", type=float, default=1.0,
                        help="Grid spacing (if x,y are indices)")
    parser.add_argument("--n_bins_r", type=int, default=40,
                        help="Number of radial bins for g(r)")
    parser.add_argument("--r_max", type=float, default=None,
                        help="Max radius for g(r); default = min(Lx,Ly)/2")
    parser.add_argument("--n_cells_x", type=int, default=16,
                        help="Number of cells in x for occupancy statistics")
    parser.add_argument("--n_cells_y", type=int, default=16,
                        help="Number of cells in y for occupancy statistics")
    parser.add_argument("--charge", type=str, default="all",
                        choices=["all", "plus", "minus"],
                        help="Which defect charge to analyze")

    args = parser.parse_args()

    defects = load_defects(args.defects_csv)

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

    # --- Pair correlation ---
    r_centers, g_r = pair_correlation(
        defects,
        Lx=args.Lx,
        Ly=args.Ly,
        dx=args.dx,
        q_select=q_sel,
        r_max=args.r_max,
        n_bins=args.n_bins_r,
    )

    g0 = g_r[0]
    print(f"[RESULT] g(0) ≈ {g0:.3f}  "
          "(<1: anti-bunching / fermion-like, >1: bunching / boson-like, "
          "~1: Poisson)")

    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, g_r, "-o", markersize=3)
    plt.axhline(1.0, color="k", linestyle="--", label="Poisson")
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title(f"Pair correlation g(r) [{label}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("g_r_{}.png".format(args.charge))
    plt.close()

    # --- Occupancy statistics ---
    F, mu, sigma2, occ_all = cell_occupancy_stats(
        defects,
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
    print(f"        Fano F   = σ²/μ ≈ {F:.3f}  "
          "(<1: sub-Poisson / exclusion, >1: bunching, ~1: Poisson)")

    plt.figure(figsize=(6, 4))
    plt.hist(occ_all, bins=np.arange(0, occ_all.max() + 1.5) - 0.5)
    plt.xlabel("defects per cell")
    plt.ylabel("count")
    plt.title(f"Cell occupancy distribution [{label}]")
    plt.tight_layout()
    plt.savefig("occupancy_hist_{}.png".format(args.charge))
    plt.close()

    print("[INFO] Saved g(r) plot and occupancy histogram.")

if __name__ == "__main__":
    main()
