#!/usr/bin/env python3
"""
investigate_energy_scaling_yukawa.py

Energy-scaling and stripe-width investigation for the Ising + defrag model.

Features:
- Runs IsingDefragGPU at several lattice sizes L.
- Computes defrag energy via s·Phi and via |∇Phi|^2 (consistency check).
- Estimates a dominant stripe wavelength λ from the structure factor.
- Optionally uses a Yukawa (screened) defrag potential:
    unscreened:   ∇² Φ = s
    Yukawa:  (∇² - μ²) Φ = s
  implemented here, without modifying IsingDefragGPU.

Outputs:
- CSV: energy_scaling_analysis/energy_scaling_results.csv
- PNG: energy_scaling_analysis/energy_scaling_plots.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Backend selection: GPU (CuPy) if available, else CPU (NumPy only)
# ---------------------------------------------------------------------
try:
    import cupy as cp
    from ising_defrag_gpu import IsingDefragGPU
    GPU_AVAILABLE = True
    xp = cp
    print("✓ GPU (CuPy) detected - using GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    import numpy as cp  # dummy so code paths compile
    from ising_defrag_gpu import IsingDefragGPU  # if this fails, script will crash
    xp = np
    print("WARNING: GPU not available; using CPU (NumPy) only")

# ---------------------------------------------------------------------
# Helper: Yukawa Poisson solver in Fourier space
# (∇² - μ²) Φ = s  →  Φ_k = -s_k / (k² + μ²)
# ---------------------------------------------------------------------
def solve_yukawa_potential(s_field, mu, use_gpu):
    """
    Solve (∇² - μ²) Φ = s for Φ on a periodic square grid using FFT.

    Parameters
    ----------
    s_field : array (cp.ndarray or np.ndarray)
        Source field s(x,y), assumed mean-subtracted.
    mu : float
        Screening mass. mu=0 reduces to unscreened Poisson (but here we
        usually use mu>0 when calling this).
    use_gpu : bool
        If True, treat s_field as CuPy array; otherwise NumPy.

    Returns
    -------
    Phi : array, same type/shape as s_field
    """
    if use_gpu:
        xp_local = cp
    else:
        xp_local = np

    Lx, Ly = s_field.shape
    assert Lx == Ly, "Assuming square lattice for now."
    L = Lx

    # Forward FFT
    s_fft = xp_local.fft.fft2(s_field)

    # k-grid
    kx = xp_local.fft.fftfreq(L, d=1.0) * (2 * xp_local.pi)
    ky = xp_local.fft.fftfreq(L, d=1.0) * (2 * xp_local.pi)
    KX, KY = xp_local.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    # Denominator: k^2 + mu^2
    denom = K2 + (mu**2)
    # Avoid divide-by-zero at k=0
    if mu > 0:
        denom[0, 0] = mu**2
    else:
        denom[0, 0] = 1.0

    Phi_fft = -s_fft / denom

    # Enforce zero-mean potential (just to fix gauge)
    Phi_fft[0, 0] = 0.0

    Phi = xp_local.fft.ifft2(Phi_fft).real
    return Phi

# ---------------------------------------------------------------------
# Helper: estimate stripe wavelength from structure factor
# ---------------------------------------------------------------------
def estimate_stripe_wavelength(M_field):
    """
    Estimate dominant stripe wavelength λ from the magnetization field M_field.

    Method:
        - subtract mean
        - compute structure factor S(k) = |FFT(M)|^2
        - zero out k=0
        - find k_max where S is maximum
        - λ = 2π / |k_max|

    Parameters
    ----------
    M_field : 2D numpy array

    Returns
    -------
    lambda_stripe : float
        Estimated stripe wavelength in lattice sites.
        Returns np.inf if no nonzero peak is found.
    """
    Lx, Ly = M_field.shape
    assert Lx == Ly, "Assuming square lattice."
    L = Lx

    M0 = M_field - np.mean(M_field)
    S_k = np.abs(np.fft.fft2(M0))**2

    # Remove k=0
    S_k[0, 0] = 0.0

    # Build k-grid
    kx = np.fft.fftfreq(L, d=1.0) * (2 * np.pi)
    ky = np.fft.fftfreq(L, d=1.0) * (2 * np.pi)
    KX, KY = np.meshgrid(kx, ky)

    # Flatten for argmax
    idx_max = np.argmax(S_k)
    kx_max = KX.ravel()[idx_max]
    ky_max = KY.ravel()[idx_max]
    k_mag = np.sqrt(kx_max**2 + ky_max**2)

    if k_mag == 0:
        return np.inf

    lambda_stripe = 2 * np.pi / k_mag
    return float(lambda_stripe)

# ---------------------------------------------------------------------
# Core: run one detailed simulation for a given (L, T, g_defrag, mu)
# ---------------------------------------------------------------------
def run_detailed_energy_analysis(
    L,
    T=1.0,
    g_defrag=0.5,
    n_sweeps=2000,
    mu=None,
    use_yukawa=False,
    coarse_grain_size=1,
    seed=42,
):
    """
    Run IsingDefragGPU at given parameters and extract energy & stripe data.

    If use_yukawa is False:
        - use the model's built-in defrag potential (sim.solve_defrag_potential)

    If use_yukawa is True:
        - compute defrag potential via Yukawa solver in this script with mass mu.
    """
    print("\n" + "=" * 70)
    print(f"ANALYZING L={L}, T={T}, g={g_defrag}, "
          f"{'Yukawa mu='+str(mu) if use_yukawa else 'unscreened'}")
    print("=" * 70)

    sim = IsingDefragGPU(
        L=L,
        T=T,
        g_defrag=g_defrag,
        coarse_grain_size=coarse_grain_size,
    )

    # Initial spins
    spins = sim.create_noise_spins(flip_prob=0.2, seed=seed)

    # Evolve to equilibrium
    print("Evolving to equilibrium...")
    for sweep in range(n_sweeps):
        if sweep % 500 == 0:
            print(f"  Sweep {sweep}/{n_sweeps}")

        if use_yukawa and mu is not None and mu > 0:
            # Compute coarse-grained magnetization field on GPU (or CPU)
            if GPU_AVAILABLE:
                spins_f = spins.astype(cp.float32)
                # no coarse-grain in this script; we treat coarse_grain_size=1
                s_field = spins_f - cp.mean(spins_f)
                Phi = solve_yukawa_potential(s_field, mu, use_gpu=True)
            else:
                spins_f = spins.astype(float)
                s_field = spins_f - np.mean(spins_f)
                Phi = solve_yukawa_potential(s_field, mu, use_gpu=False)
        else:
            # Use built-in Poisson solver from the model
            Phi = sim.solve_defrag_potential(spins)

        spins = sim.metropolis_sweep(spins, Phi)

    # Final potential using same rule
    if use_yukawa and mu is not None and mu > 0:
        if GPU_AVAILABLE:
            spins_f = spins.astype(cp.float32)
            s_field = spins_f - cp.mean(spins_f)
            Phi = solve_yukawa_potential(s_field, mu, use_gpu=True)
        else:
            spins_f = spins.astype(float)
            s_field = spins_f - np.mean(spins_f)
            Phi = solve_yukawa_potential(s_field, mu, use_gpu=False)
    else:
        Phi = sim.solve_defrag_potential(spins)

    # Move to CPU
    if GPU_AVAILABLE:
        spins_cpu = cp.asnumpy(spins)
        Phi_cpu = cp.asnumpy(Phi)
    else:
        spins_cpu = spins
        Phi_cpu = Phi

    # Domain wall count
    walls = 0
    for i in range(L):
        for j in range(L):
            if spins_cpu[i, j] != spins_cpu[(i + 1) % L, j]:
                walls += 1
            if spins_cpu[i, j] != spins_cpu[i, (j + 1) % L]:
                walls += 1
    walls //= 2

    # Magnetization field used for defrag (here cg=1)
    M_field = spins_cpu.astype(float)

    # Density fluctuation (zero mean)
    s = M_field - np.mean(M_field)

    # ------- Energies --------
    # Defrag energy via s·Phi
    E_defrag_total = -0.5 * g_defrag * np.sum(s * Phi_cpu)

    # Gradient-based defrag energy: E = (g/2) ∑ |∇Phi|²
    grad_Phi_x = np.gradient(Phi_cpu, axis=0)
    grad_Phi_y = np.gradient(Phi_cpu, axis=1)
    grad_Phi_mag = np.sqrt(grad_Phi_x**2 + grad_Phi_y**2)
    E_defrag_grad = 0.5 * g_defrag * np.sum(
        grad_Phi_x**2 + grad_Phi_y**2
    )
    E_grad_ratio = (
        E_defrag_grad / E_defrag_total if E_defrag_total != 0 else np.nan
    )

    # Ising energy
    E_ising = 0.0
    J = sim.J
    for i in range(L):
        for j in range(L):
            E_ising -= J * spins_cpu[i, j] * spins_cpu[(i + 1) % L, j]
            E_ising -= J * spins_cpu[i, j] * spins_cpu[i, (j + 1) % L]

    N_sites = L * L
    M_mean = np.mean(spins_cpu)

    # Stripe wavelength estimate
    lambda_stripe = estimate_stripe_wavelength(M_field)

    results = {
        "L": L,
        "N_sites": N_sites,
        "T": T,
        "g_defrag": g_defrag,
        "mu": mu if use_yukawa else 0.0,
        "use_yukawa": use_yukawa,
        "walls": walls,
        "M_mean": M_mean,
        "lambda_stripe": lambda_stripe,
        "lambda_over_L": lambda_stripe / L if np.isfinite(lambda_stripe) else np.inf,
        "E_ising": E_ising,
        "E_defrag_total": E_defrag_total,
        "E_defrag_grad": E_defrag_grad,
        "E_grad_ratio": E_grad_ratio,
        "E_total": E_ising + E_defrag_total,
        "E_defrag_per_site": E_defrag_total / N_sites,
        "E_defrag_per_L": E_defrag_total / L,
        "E_defrag_per_wall": E_defrag_total / walls if walls > 0 else 0.0,
        "Phi_mean": np.mean(Phi_cpu),
        "Phi_std": np.std(Phi_cpu),
        "Phi_max": np.max(np.abs(Phi_cpu)),
        "s_std": np.std(s),
        "grad_Phi_mean": np.mean(grad_Phi_mag),
        "grad_Phi_max": np.max(grad_Phi_mag),
    }

    print(f"\nResults for L={L}:")
    print(f"  Walls:            {walls}")
    print(f"  M_mean:           {M_mean:.3f}")
    print(f"  λ_stripe:         {lambda_stripe:.2f} (λ/L = {results['lambda_over_L']:.3f})")
    print(f"  E_defrag (s·Phi): {E_defrag_total:.2e}")
    print(f"  E_defrag (grad):  {E_defrag_grad:.2e}")
    print(f"  E_grad / E_sPhi:  {E_grad_ratio:.3e}")
    print(f"  E_defrag/site:    {results['E_defrag_per_site']:.3f}")
    print(f"  E_defrag/L:       {results['E_defrag_per_L']:.3f}")
    print(f"  E_defrag/wall:    {results['E_defrag_per_wall']:.3f}")
    print(f"  Phi_max:          {results['Phi_max']:.2e}")
    print(f"  grad_Phi_max:     {results['grad_Phi_max']:.2e}")

    return results, spins_cpu, Phi_cpu

# ---------------------------------------------------------------------
# Plotting / scaling analysis
# ---------------------------------------------------------------------
def plot_energy_scaling(df, label_prefix=""):
    """Create energy and stripe scaling plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1: E_defrag vs N
    ax = axes[0, 0]
    ax.plot(df["N_sites"], df["E_defrag_total"], "o-", markersize=8, linewidth=2)
    ax.set_xlabel("N = L²")
    ax.set_ylabel("E_defrag")
    ax.set_title(f"{label_prefix}Total Defrag Energy vs System Size")
    ax.grid(True, alpha=0.3)

    # 2: E_defrag/site vs L
    ax = axes[0, 1]
    ax.plot(df["L"], df["E_defrag_per_site"], "o-", markersize=8, linewidth=2)
    ax.set_xlabel("L")
    ax.set_ylabel("E_defrag / N")
    ax.set_title(f"{label_prefix}Energy per Site")
    ax.grid(True, alpha=0.3)

    # 3: E_defrag/L vs L
    ax = axes[0, 2]
    ax.plot(df["L"], df["E_defrag_per_L"], "o-", markersize=8, linewidth=2)
    ax.set_xlabel("L")
    ax.set_ylabel("E_defrag / L")
    ax.set_title(f"{label_prefix}Energy per Length")
    ax.grid(True, alpha=0.3)

    # 4: E_defrag/wall vs L
    ax = axes[1, 0]
    ax.plot(df["L"], df["E_defrag_per_wall"], "o-", markersize=8, linewidth=2)
    ax.set_xlabel("L")
    ax.set_ylabel("E_defrag / wall")
    ax.set_title(f"{label_prefix}Energy per Wall")
    ax.grid(True, alpha=0.3)

    # 5: Phi_max vs L
    ax = axes[1, 1]
    ax.plot(df["L"], df["Phi_max"], "o-", markersize=8, linewidth=2)
    ax.set_xlabel("L")
    ax.set_ylabel("max|Phi|")
    ax.set_title(f"{label_prefix}Max Potential")
    ax.grid(True, alpha=0.3)

    # 6: Stripe wavelength vs L
    ax = axes[1, 2]
    ax.plot(df["L"], df["lambda_stripe"], "o-", markersize=8, linewidth=2)
    ax.set_xlabel("L")
    ax.set_ylabel("λ_stripe")
    ax.set_title(f"{label_prefix}Stripe Wavelength")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

# ---------------------------------------------------------------------
def loglog_fit_E_vs_L(df, min_L_for_fit=48):
    """
    Do a log–log fit of |E_defrag| vs L for L >= min_L_for_fit.
    """
    df_valid = df[df["L"] >= min_L_for_fit].copy()
    if len(df_valid) < 3:
        df_valid = df

    L_arr = df_valid["L"].values
    E_arr = np.abs(df_valid["E_defrag_total"].values)
    log_L = np.log(L_arr)
    log_E = np.log(E_arr)

    slope, intercept = np.polyfit(log_L, log_E, 1)
    residuals = log_E - (slope * log_L + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_E - np.mean(log_E))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, intercept, r2, df_valid

# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ENERGY SCALING & STRIPE-WIDTH INVESTIGATION")
    print("=" * 70)

    # -------------------
    # Parameters to scan
    # -------------------
    L_values = [32, 48, 64, 96]
    T = 1.0
    g_defrag = 0.5
    n_sweeps = 2000

    # Two "phases" to compare: unscreened and a Yukawa example
    cases = [
        {"name": "unscreened", "use_yukawa": False, "mu": None},
        # You can add / tweak μ values here:
        # {"name": "yukawa_mu_0p05", "use_yukawa": True, "mu": 0.05},
    ]

    all_results = []

    for case in cases:
        print("\n" + "-" * 70)
        print(f"CASE: {case['name']}")
        print("-" * 70)

        case_results = []
        for L in L_values:
            res, spins, Phi = run_detailed_energy_analysis(
                L=L,
                T=T,
                g_defrag=g_defrag,
                n_sweeps=n_sweeps,
                mu=case["mu"],
                use_yukawa=case["use_yukawa"],
            )
            res["case"] = case["name"]
            case_results.append(res)
            all_results.append(res)

        df_case = pd.DataFrame(case_results)

        # Fit scaling
        slope, intercept, r2, df_valid = loglog_fit_E_vs_L(df_case)
        print("\nScaling fit for case:", case["name"])
        print(f"  log(E) ≈ {slope:.2f} log(L) + {intercept:.2f}")
        print(f"  R² = {r2:.4f}")
        print(f"  → E ~ L^{slope:.2f}")

        # Stripe scaling info
        print("\nStripe wavelength λ vs L for case:", case["name"])
        for _, row in df_case.sort_values("L").iterrows():
            print(
                f"  L={row['L']:3d} : λ={row['lambda_stripe']:7.2f}, "
                f"λ/L={row['lambda_over_L']:.3f}"
            )

        # Plot for this case (can compare by eye)
        plot_energy_scaling(df_case, label_prefix=f"{case['name']} - ")

    # Save combined results
    output_dir = Path("energy_scaling_analysis")
    output_dir.mkdir(exist_ok=True)
    df_all = pd.DataFrame(all_results)
    csv_path = output_dir / "energy_scaling_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n[PASS] Saved combined results to {csv_path}")

    # Save plots
    plots_path = output_dir / "energy_scaling_plots.png"
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PASS] Saved plots to {plots_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
