#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def fit_power_law(L, E):
    L = np.asarray(L, dtype=float)
    E = np.asarray(E, dtype=float)
    mask = (L > 0) & (E != 0)
    L = L[mask]
    E = E[mask]
    logL = np.log10(L)
    logE = np.log10(np.abs(E))
    p, loga = np.polyfit(logL, logE, 1)
    a = 10**loga
    return p, a

def main():
    csv_path = Path("energy_scaling_analysis/energy_scaling_results.csv")
    df = pd.read_csv(csv_path)

    L = df["L"].to_numpy(float)
    E = df["E_defrag_total"].to_numpy(float)

    p, a = fit_power_law(L, E)
    print(f"Ising + defrag: |E| ≈ {a:.3e} * L^{p:.3f}")

    L_fit = np.linspace(L.min(), L.max(), 200)

    plt.figure(figsize=(6,4))
    plt.loglog(L, np.abs(E), "o", label=f"Ising data (p≈{p:.2f})")
    plt.loglog(L_fit, a * L_fit**p, "--", label="power-law fit")

    plt.xlabel("L")
    plt.ylabel(r"$|E_{\mathrm{defrag}}|$")
    plt.title("Ising + defrag energy scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path("ising_energy_scaling_loglog.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved Ising-only scaling figure to {out}")

if __name__ == "__main__":
    main()
