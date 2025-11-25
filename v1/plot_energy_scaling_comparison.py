#!/usr/bin/env python3
"""
plot_energy_scaling_comparison.py

Make a single figure comparing E_defrag scaling for:

- Ising + defrag   (super-extensive, ~ L^4)
- Scalar + defrag  (extensive, ~ L^2)

Inputs:
    ising_csv   : CSV with at least columns [L, E_defrag_total]
    scalar_csv  : CSV with at least columns [L, E_defrag_total]

Outputs:
    energy_scaling_comparison.png  : log–log plot of |E_defrag| vs L
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: Path):
    df = pd.read_csv(path)
    if "L" not in df.columns or "E_defrag_total" not in df.columns:
        raise ValueError(
            f"{path} must have columns 'L' and 'E_defrag_total'. "
            f"Found columns: {list(df.columns)}"
        )
    L = df["L"].to_numpy(dtype=float)
    E = df["E_defrag_total"].to_numpy(dtype=float)
    return L, E


def fit_power_law(L: np.ndarray, E: np.ndarray):
    """
    Fit |E| ~ a * L^p via linear regression on log10.

    Returns:
        p  : exponent
        a  : prefactor
    """
    L = np.asarray(L, dtype=float)
    E = np.asarray(E, dtype=float)

    mask = (L > 0) & (E != 0)
    L = L[mask]
    E = E[mask]

    logL = np.log10(L)
    logE = np.log10(np.abs(E))

    p, loga = np.polyfit(logL, logE, 1)
    a = 10 ** loga
    return p, a


def main():
    parser = argparse.ArgumentParser(
        description="Compare energy scaling: Ising vs scalar (E_defrag vs L).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ising_csv", help="CSV with Ising results.")
    parser.add_argument("scalar_csv", help="CSV with scalar-field results.")
    parser.add_argument(
        "--output",
        type=str,
        default="energy_scaling_comparison.png",
        help="Output figure filename.",
    )
    args = parser.parse_args()

    ising_path = Path(args.ising_csv)
    scalar_path = Path(args.scalar_csv)

    if not ising_path.exists():
        raise FileNotFoundError(f"Ising CSV not found: {ising_path}")
    if not scalar_path.exists():
        raise FileNotFoundError(f"Scalar CSV not found: {scalar_path}")

    # Load data
    L_ising, E_ising = load_data(ising_path)
    L_scalar, E_scalar = load_data(scalar_path)

    # Fit power laws
    p_ising, a_ising = fit_power_law(L_ising, E_ising)
    p_scalar, a_scalar = fit_power_law(L_scalar, E_scalar)

    print("\n=== ENERGY SCALING FITS ===")
    print(f"Ising + defrag : |E| ≈ {a_ising:.3e} * L^{p_ising:.3f}")
    print(f"Scalar + defrag: |E| ≈ {a_scalar:.3e} * L^{p_scalar:.3f}")

    # Log–log plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter points
    ax.loglog(L_ising, np.abs(E_ising), "o", label=f"Ising (p≈{p_ising:.2f})")
    ax.loglog(
        L_scalar,
        np.abs(E_scalar),
        "s",
        label=f"Scalar field (p≈{p_scalar:.2f})",
    )

    # Smooth L grid for fit lines
    L_fit = np.linspace(
        min(L_ising.min(), L_scalar.min()),
        max(L_ising.max(), L_scalar.max()),
        200,
    )
    ax.loglog(L_fit, a_ising * L_fit ** p_ising, "--", alpha=0.7)
    ax.loglog(L_fit, a_scalar * L_fit ** p_scalar, "--", alpha=0.7)

    ax.set_xlabel("L")
    ax.set_ylabel(r"$|E_{\mathrm{defrag}}|$")
    ax.set_title("Energy Scaling: Ising vs Scalar (defrag coupling)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = Path(args.output)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSaved figure to {out_path}\n")


if __name__ == "__main__":
    main()
