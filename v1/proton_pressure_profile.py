#!/usr/bin/env python3
"""
proton_pressure_profile.py

Compute a radial pressure profile for an emergent "proton" lump
from a scalar + defrag simulation snapshot.

Requires snapshots saved by scalar_field_defrag_gpu.py with fields:
    - psi : complex array, shape (Ny, Nx)
    - phi : real array, same shape
    - dx, dy, v, lambda_param (optional, but used if present)

Outputs:
    - <output_prefix>_pressure_profile.csv
    - <output_prefix>_pressure_profile.png
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Potential and energy densities, matching scalar_field_defrag_gpu.py
# ---------------------------------------------------------------------

def potential_energy_density(psi: np.ndarray,
                             v: float,
                             lambda_param: float) -> np.ndarray:
    """
    Mexican hat potential used in scalar_field_defrag_gpu.py:

        V(psi) = λ (|ψ|^2 - v^2)^2

    Returns V(psi) as a 2D array.
    """
    rho = np.abs(psi) ** 2
    return lambda_param * (rho - v ** 2) ** 2


def compute_gradients(psi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute gradient energy density: e_grad = 0.5 * |∇ψ|^2
    using centered finite differences via np.gradient.
    """
    dpsi_dy, dpsi_dx = np.gradient(psi, dy, dx, edge_order=2)

    if np.iscomplexobj(psi):
        grad_sq = np.abs(dpsi_dx) ** 2 + np.abs(dpsi_dy) ** 2
    else:
        grad_sq = dpsi_dx ** 2 + dpsi_dy ** 2

    e_grad = 0.5 * grad_sq
    return e_grad


def compute_defrag_energy_density(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Local defrag energy density for E_defrag ~ 0.5 ∫ rho(x) phi(x) dx:

        e_defrag(x) = 0.5 * rho(x) * phi(x)
    """
    return 0.5 * rho * phi


def find_lump_center(rho: np.ndarray) -> tuple[float, float]:
    """
    Estimate lump center from density peak and a small centroid window.

    Returns:
        (cx, cy) in lattice indices (floats).
    """
    ny, nx = rho.shape
    max_idx = np.unravel_index(np.argmax(rho), rho.shape)
    y0, x0 = max_idx

    window = 5
    half = window // 2
    x_min = max(0, x0 - half)
    x_max = min(nx, x0 + half + 1)
    y_min = max(0, y0 - half)
    y_max = min(ny, y0 + half + 1)

    sub_rho = rho[y_min:y_max, x_min:x_max]
    ys, xs = np.indices(sub_rho.shape)

    total = sub_rho.sum()
    if total <= 0:
        return float(x0), float(y0)

    cx_local = (xs * sub_rho).sum() / total
    cy_local = (ys * sub_rho).sum() / total

    cx = x_min + cx_local
    cy = y_min + cy_local
    return float(cx), float(cy)


def radial_profile(field: np.ndarray,
                   cx: float,
                   cy: float,
                   dx: float,
                   dy: float,
                   nbins: int):
    """
    Compute radial profile of `field` around center (cx, cy) in index space.

    Args:
        field : 2D array, e.g. p_local(y, x)
        cx, cy : center in lattice indices
        dx, dy : physical spacing in x and y
        nbins : number of radial bins

    Returns:
        r_centers, p_mean, p_std, n_points
    """
    ny, nx = field.shape
    y_indices, x_indices = np.indices(field.shape)

    x_phys = (x_indices - cx) * dx
    y_phys = (y_indices - cy) * dy
    r = np.sqrt(x_phys ** 2 + y_phys ** 2).ravel()

    values = field.ravel()
    r_max = r.max()
    nbins = max(1, nbins)

    bin_edges = np.linspace(0.0, r_max, nbins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    p_mean = np.zeros(nbins, dtype=float)
    p_std = np.zeros(nbins, dtype=float)
    n_points = np.zeros(nbins, dtype=int)

    bin_indices = np.digitize(r, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    for b in range(nbins):
        mask = (bin_indices == b)
        if not np.any(mask):
            p_mean[b] = np.nan
            p_std[b] = np.nan
            n_points[b] = 0
        else:
            vals = values[mask]
            p_mean[b] = np.mean(vals)
            p_std[b] = np.std(vals)
            n_points[b] = mask.sum()

    return r_centers, p_mean, p_std, n_points


def save_profile_csv(path: Path,
                     r_centers: np.ndarray,
                     p_mean: np.ndarray,
                     p_std: np.ndarray,
                     n_points: np.ndarray):
    """Save radial pressure profile to CSV."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "p_mean", "p_std", "n_points"])
        for r, pm, ps, n in zip(r_centers, p_mean, p_std, n_points):
            writer.writerow([r, pm, ps, n])


def plot_profile(path: Path,
                 r_centers: np.ndarray,
                 p_mean: np.ndarray,
                 p_std: np.ndarray,
                 title: str = "Radial Pressure Profile"):
    """Save a plot of p_mean(r) with error bars."""
    mask = np.isfinite(p_mean)
    r = r_centers[mask]
    pm = p_mean[mask]
    ps = p_std[mask]

    plt.figure()
    plt.errorbar(r, pm, yerr=ps, fmt="o-", capsize=3)
    plt.xlabel("r (physical units)")
    plt.ylabel("pressure proxy p(r)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute radial pressure profile for a proton-like lump "
                    "from a scalar+defrag snapshot (.npz)."
    )
    parser.add_argument("input",
                        help="Input .npz file containing psi and phi arrays.")
    parser.add_argument("--psi-key", default="psi",
                        help="Key name for psi array in .npz (default: psi).")
    parser.add_argument("--phi-key", default="phi",
                        help="Key name for phi array in .npz (default: phi).")
    parser.add_argument("--dx", type=float, default=None,
                        help="Grid spacing in x. If omitted, try to read dx from file.")
    parser.add_argument("--dy", type=float, default=None,
                        help="Grid spacing in y. If omitted, try to read dy from file.")
    parser.add_argument("--center-x", type=float,
                        help="Lump center x (lattice index). If omitted, "
                             "estimated from density peak.")
    parser.add_argument("--center-y", type=float,
                        help="Lump center y (lattice index). If omitted, "
                             "estimated from density peak.")
    parser.add_argument("--nbins", type=int, default=50,
                        help="Number of radial bins (default: 50).")
    parser.add_argument("--output-prefix",
                        help="Prefix for output files. Default: <input_stem>_pressure")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.output_prefix:
        out_prefix = Path(args.output_prefix)
    else:
        out_prefix = input_path.with_suffix("")

    csv_path = out_prefix.with_name(out_prefix.name + "_pressure_profile.csv")
    png_path = out_prefix.with_name(out_prefix.name + "_pressure_profile.png")

    data = np.load(input_path)

    if args.psi_key not in data or args.phi_key not in data:
        print(f"ERROR: psi-key '{args.psi_key}' or phi-key '{args.phi_key}' "
              f"not found in {input_path.name}. Keys: {list(data.keys())}",
              file=sys.stderr)
        sys.exit(1)

    psi = data[args.psi_key]
    phi = data[args.phi_key]

    if psi.shape != phi.shape:
        print("ERROR: psi and phi must have the same shape.", file=sys.stderr)
        print(f"psi.shape = {psi.shape}, phi.shape = {phi.shape}",
              file=sys.stderr)
        sys.exit(1)

    # rho = |psi|^2
    rho = np.abs(psi) ** 2

    # Use dx, dy from file if present, else from CLI or default to 1.0
    dx = args.dx if args.dx is not None else float(data.get("dx", 1.0))
    dy = args.dy if args.dy is not None else float(data.get("dy", dx))

    # Mexican hat parameters, from file if present, else defaults
    v = float(data.get("v", 1.0))
    lambda_param = float(data.get("lambda_param", 0.5))

    # Find / use lump center
    if args.center_x is not None and args.center_y is not None:
        cx = float(args.center_x)
        cy = float(args.center_y)
        print(f"Using provided center: cx={cx:.3f}, cy={cy:.3f}")
    else:
        cx, cy = find_lump_center(rho)
        print(f"Estimated center from density peak/centroid: "
              f"cx={cx:.3f}, cy={cy:.3f}")

    # Local energy densities
    e_grad = compute_gradients(psi, dx=dx, dy=dy)
    e_pot = potential_energy_density(psi, v=v, lambda_param=lambda_param)
    e_defrag = compute_defrag_energy_density(rho, phi)

    # Local pressure proxy
    p_local = e_grad - (e_pot + e_defrag)

    # Radial profile
    r_centers, p_mean, p_std, n_points = radial_profile(
        p_local, cx=cx, cy=cy, dx=dx, dy=dy, nbins=args.nbins
    )

    save_profile_csv(csv_path, r_centers, p_mean, p_std, n_points)
    plot_profile(png_path, r_centers, p_mean, p_std,
                 title="Radial Pressure Profile (scalar+defrag proton)")

    print(f"Wrote radial pressure profile CSV to: {csv_path}")
    print(f"Wrote radial pressure profile plot to: {png_path}")
    print("Note: This 'pressure' is a mechanical proxy based on your current "
          "energy functional (grad - potential - defrag).")


if __name__ == "__main__":
    main()
