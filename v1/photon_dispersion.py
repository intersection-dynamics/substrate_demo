#!/usr/bin/env python3
"""
photon_dispersion.py

Analyze a photon-test run from coupled_photon_capable.py and extract
an approximate dispersion relation ω(|k|).

Given an output directory like 'photons_free_output' that contains:

    photons_free_energies.csv
    photons_free_snap_000000.npz
    photons_free_snap_000020.npz
    ...
    photons_free_snap_004000.npz

this script:

  1. Loads all *_snap_*.npz files in time order.
  2. Extracts a chosen gauge field component (ax, ay, ex, or ey).
  3. Performs spatial FFT → field(kx, ky, t).
  4. Performs temporal FFT along t → field(kx, ky, ω).
  5. Computes power spectrum |field(k, ω)|^2.
  6. Radially bins in |k| to get power(|k|, ω).
  7. Plots a 2D dispersion image with:
         x-axis: |k|
         y-axis: ω
         color:  log10(power)
     and overlays the theoretical ω = c |k| line.

Usage example (for your photons_free run):

    python photon_dispersion.py \\
        --input_dir photons_free_output \\
        --prefix photons_free \\
        --component ex \\
        --c 1.0 \\
        --dx 1.0

This will write 'photons_free_dispersion.png' into photons_free_output/.
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_snapshots(
    input_dir: str,
    prefix: str,
    component: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all snapshot NPZ files and return:

        times: shape (Nt,)
        field: shape (Nt, L, L)

    'component' must be one of: 'ax', 'ay', 'ex', 'ey'.
    """
    pattern = os.path.join(input_dir, f"{prefix}_snap_*.npz")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found matching {pattern}")

    times: List[float] = []
    fields: List[np.ndarray] = []

    for path in files:
        data = np.load(path)
        if component not in data:
            raise KeyError(f"Component '{component}' not found in {path}")
        field = data[component]
        time = float(data["time"]) if "time" in data else None
        if time is None:
            raise KeyError(f"'time' not found in {path}")

        times.append(time)
        fields.append(field.astype(np.float64))

    times_arr = np.array(times)
    field_arr = np.stack(fields, axis=0)  # (Nt, L, L)

    return times_arr, field_arr


def compute_dispersion(
    times: np.ndarray,
    field_txy: np.ndarray,
    dx: float,
    c: float,
    n_kbins: int = 32,
    k_max_fraction: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given:

        times:      shape (Nt,)
        field_txy:  shape (Nt, L, L), real field in real space

    Return:

        k_centers:  shape (Nk,), bin centers in |k|
        omega_pos:  shape (Nw,), positive ω values
        power_kw:   shape (Nw, Nk), binned power(|k|, ω)
        theory_w:   shape (Nk,), ω_theory(k) = c * |k| (for overlay)
    """
    Nt, Lx, Ly = field_txy.shape
    assert Lx == Ly, "Only square grids supported"

    # Time step and frequency axis
    # (assumes nearly uniform spacing)
    dt_arr = np.diff(times)
    dt = float(np.mean(dt_arr))
    freq = np.fft.fftfreq(Nt, d=dt)  # cycles per unit time
    omega = 2.0 * np.pi * freq       # angular frequency

    # Spatial k-grid
    kx = 2.0 * np.pi * np.fft.fftfreq(Lx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ly, d=dx)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K_mag = np.sqrt(KX**2 + KY**2)   # (Lx, Ly)

    # 1) Spatial FFT → field(kx, ky, t)
    #    real → complex
    field_k_t = np.fft.fft2(field_txy, axes=(1, 2))  # (Nt, Lx, Ly)

    # 2) Temporal FFT → field(kx, ky, ω)
    field_k_w = np.fft.fft(field_k_t, axis=0)  # (Nt, Lx, Ly)

    # 3) Power spectrum
    power_w_k = np.abs(field_k_w) ** 2  # (Nt, Lx, Ly)

    # Only keep positive frequencies
    pos_mask = omega > 0.0
    omega_pos = omega[pos_mask]               # (Nw,)
    power_pos = power_w_k[pos_mask, :, :]     # (Nw, Lx, Ly)

    # Flatten spatial dims for binning
    K_flat = K_mag.reshape(-1)                # (Lx*Ly,)
    power_flat = power_pos.reshape(
        omega_pos.shape[0], -1
    )  # (Nw, Lx*Ly)

    # Define k magnitude bins
    k_max = k_max_fraction * float(K_flat.max())
    k_min = 0.0
    k_edges = np.linspace(k_min, k_max, n_kbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    power_kw = np.zeros((omega_pos.shape[0], n_kbins), dtype=np.float64)

    # Bin in |k|
    for b in range(n_kbins):
        mask = (K_flat >= k_edges[b]) & (K_flat < k_edges[b + 1])
        if np.any(mask):
            power_kw[:, b] = power_flat[:, mask].mean(axis=1)
        else:
            power_kw[:, b] = 0.0

    # Theoretical massless dispersion: ω = c * k
    theory_w = c * k_centers

    return k_centers, omega_pos, power_kw, theory_w


def plot_dispersion(
    k_centers: np.ndarray,
    omega_pos: np.ndarray,
    power_kw: np.ndarray,
    theory_w: np.ndarray,
    out_path: str,
    c: float,
    title: str,
):
    """
    Make a 2D dispersion plot of log10(power(|k|, ω)) and overlay ω=c|k|.
    """
    # Avoid log(0)
    eps = 1e-20
    log_power = np.log10(power_kw + eps)

    # Image extent in (x=|k|, y=ω)
    extent = [
        k_centers[0],
        k_centers[-1],
        omega_pos[0],
        omega_pos[-1],
    ]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        log_power.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    plt.colorbar(im, label="log10 power")

    # Overlay theoretical line
    plt.plot(
        k_centers,
        theory_w,
        "w--",
        linewidth=1.5,
        label=f"ω = c|k| (c={c:g})",
    )

    plt.xlabel(r"|k|")
    plt.ylabel(r"ω")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved dispersion image -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract photon dispersion ω(|k|) from photon test snapshots."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="photons_free_output",
        help="Directory containing <prefix>_snap_*.npz files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="photons_free",
        help="Prefix used when running the simulation.",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="ex",
        choices=["ax", "ay", "ex", "ey"],
        help="Field component to analyze (default: ex).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Lattice spacing (must match simulation).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Light speed parameter used in the simulation.",
    )
    parser.add_argument(
        "--n_kbins",
        type=int,
        default=32,
        help="Number of radial |k| bins.",
    )
    parser.add_argument(
        "--k_max_fraction",
        type=float,
        default=1.0,
        help="Fraction of max |k| to show (0–1).",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    prefix = args.prefix

    print(f"[INFO] Loading snapshots from {input_dir} with prefix '{prefix}'")
    times, field_txy = load_snapshots(input_dir, prefix, args.component)
    print(f"[INFO] Loaded {field_txy.shape[0]} snapshots of size "
          f"{field_txy.shape[1]}x{field_txy.shape[2]}")

    k_centers, omega_pos, power_kw, theory_w = compute_dispersion(
        times,
        field_txy,
        dx=args.dx,
        c=args.c,
        n_kbins=args.n_kbins,
        k_max_fraction=args.k_max_fraction,
    )

    out_path = os.path.join(input_dir, f"{prefix}_dispersion_{args.component}.png")
    title = f"Photon dispersion from {args.component.upper()}"

    plot_dispersion(
        k_centers,
        omega_pos,
        power_kw,
        theory_w,
        out_path,
        c=args.c,
        title=title,
    )

    print("[DONE] Dispersion analysis complete.")


if __name__ == "__main__":
    main()
