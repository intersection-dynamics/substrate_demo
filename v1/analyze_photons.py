#!/usr/bin/env python3
"""
analyze_photons.py

Post-process a photon test run from coupled_photon_capable.py.

Given an output directory like 'photons_free_output', this script:

  1. Loads the energies CSV and writes an energies.png plot.
  2. Finds a few snapshot NPZ files (early, middle, late).
  3. For each chosen snapshot, plots:
       - gauge potential magnitude |A| = sqrt(ax^2 + ay^2)
       - electric field magnitude |E| = sqrt(ex^2 + ey^2)
     and saves them as PNGs.

Usage (example):

  python analyze_photons.py --input_dir photons_free_output --out_prefix photons_free

This assumes your photon test was run with:

  python coupled_photon_capable.py --out_prefix photons_free --photon_test 1 ...
"""

import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def load_energies(csv_path):
    """Load time, energy from photons_free_energies.csv."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    E = data[:, 1]
    return t, E


def plot_energies(t, E, out_path, title="Photon test: total energy"):
    """Plot E(t) and save as PNG."""
    plt.figure(figsize=(8, 4))
    plt.plot(t, E, marker="", linewidth=1.5)
    plt.xlabel("time")
    plt.ylabel("Total energy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved energy plot -> {out_path}")


def choose_snapshots(npz_files, n_desired=3):
    """
    Choose a small subset of snapshots:
      - earliest
      - middle
      - latest
    """
    if len(npz_files) == 0:
        return []

    npz_files = sorted(npz_files)
    indices = [0]
    if len(npz_files) > 1:
        indices.append(len(npz_files) // 2)
    if len(npz_files) > 2:
        indices.append(len(npz_files) - 1)

    chosen = []
    seen = set()
    for idx in indices:
        if idx not in seen and 0 <= idx < len(npz_files):
            chosen.append(npz_files[idx])
            seen.add(idx)
    return chosen


def plot_field_magnitudes(npz_path, out_prefix):
    """
    For a single snapshot NPZ, load ax, ay, ex, ey and make two plots:

      - |A| = sqrt(ax^2 + ay^2)
      - |E| = sqrt(ex^2 + ey^2)
    """
    data = np.load(npz_path)
    ax = data["ax"]
    ay = data["ay"]
    ex = data["ex"]
    ey = data["ey"]
    t = data["time"].item() if "time" in data else None

    A_mag = np.sqrt(ax**2 + ay**2)
    E_mag = np.sqrt(ex**2 + ey**2)

    # Gauge magnitude plot
    plt.figure(figsize=(5, 4))
    plt.imshow(A_mag, origin="lower", cmap="viridis")
    plt.colorbar(label="|A|")
    if t is not None:
        plt.title(f"|A| at t={t:.3f}")
    else:
        plt.title("|A| (gauge potential magnitude)")
    plt.tight_layout()
    out_A = f"{out_prefix}_A_mag.png"
    plt.savefig(out_A, dpi=200)
    plt.close()
    print(f"[PLOT] Saved gauge magnitude plot -> {out_A}")

    # Electric magnitude plot
    plt.figure(figsize=(5, 4))
    plt.imshow(E_mag, origin="lower", cmap="magma")
    plt.colorbar(label="|E|")
    if t is not None:
        plt.title(f"|E| at t={t:.3f}")
    else:
        plt.title("|E| (electric field magnitude)")
    plt.tight_layout()
    out_E = f"{out_prefix}_E_mag.png"
    plt.savefig(out_E, dpi=200)
    plt.close()
    print(f"[PLOT] Saved electric magnitude plot -> {out_E}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze photon test output (energies + snapshots)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="photons_free_output",
        help="Directory containing *_energies.csv and *_snap_*.npz",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="photons_free",
        help="Prefix used in the simulation (e.g. photons_free).",
    )
    args = parser.parse_args()

    in_dir = args.input_dir
    prefix = args.out_prefix

    # 1) Energies
    csv_path = os.path.join(in_dir, f"{prefix}_energies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find energies CSV at {csv_path}")

    t, E = load_energies(csv_path)
    energy_png = os.path.join(in_dir, f"{prefix}_energies.png")
    plot_energies(t, E, energy_png, title="Photon test: total energy vs time")

    # 2) Snapshot NPZ files
    pattern = os.path.join(in_dir, f"{prefix}_snap_*.npz")
    npz_files = glob.glob(pattern)
    if len(npz_files) == 0:
        print(f"[WARN] No snapshot NPZ files found matching {pattern}")
        return

    chosen = choose_snapshots(npz_files, n_desired=3)
    print("[INFO] Selected snapshots:")
    for path in chosen:
        print("   ", os.path.basename(path))

    # For each chosen snapshot, plot field magnitudes
    for npz_path in chosen:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        out_pref = os.path.join(in_dir, base)
        plot_field_magnitudes(npz_path, out_pref)


if __name__ == "__main__":
    main()
