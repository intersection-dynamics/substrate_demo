#!/usr/bin/env python3
"""
info_tradeoff_sweep.py

Predictive simulation: tradeoff between Bell (CHSH) violation and
classical information extraction for a Tsirelson-level two-qubit system.

Model:
  - Start in the maximally entangled Bell state |ψ+> = (|00> + |11>)/√2
  - Use standard CHSH settings that saturate the Tsirelson bound 2√2
  - On qubit A, apply a *weak σ_z measurement* with strength η each step:
        η = 0   : no information, no disturbance (identity channel)
        η → 1   : projective σ_z, 1 bit of info, max disturbance
  - For each step k:
        * Compute mutual information I_k between an ideal σ_z(A) variable
          and the meter outcome (2-outcome POVM).
        * Apply the weak measurement channel (average over outcomes) to
          update the state.
  - After n_steps:
        * Compute total info extracted:  I_total = Σ_k I_k  (bits)
        * Compute final CHSH S_final
        * Compute Tsirelson deficit: ΔS = 2√2 - |S_final|
        * Compute von Neumann entropy increase ΔS_vN (bits)

We sweep η over [eta_min, eta_max] and record (I_total, ΔS, ΔS_vN).
In the small-information regime, the substrate-style prediction is that:

    ΔS ≈ a * I_total,     ΔS_vN ≈ b * I_total

i.e. an approximately linear tradeoff between Bell deficit and extracted
classical bits, and a Landauer-like scaling for entropy/heat per bit.

Outputs:
  - ./outputs/info_tradeoff_sweep.png : 2-panel plot
  - ./outputs/info_tradeoff_sweep_data.npz : raw arrays
  - Console: small-info linear fit slopes
"""

import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Basic linear algebra helpers
# ---------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)

sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
sy = np.array([[0, -1j],
               [1j, 0]], dtype=complex)
sz = np.array([[1, 0],
               [0, -1]], dtype=complex)


def kron(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product of an arbitrary number of operators."""
    out = np.array([[1.0]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


# ---------------------------------------------------------------------
# CHSH machinery for a 2-qubit Bell pair
# ---------------------------------------------------------------------

# A on qubit A, B on qubit B
A0 = kron(sz, I2)
A1 = kron(sx, I2)
B0 = kron(I2, (sz + sx) / math.sqrt(2.0))
B1 = kron(I2, (sz - sx) / math.sqrt(2.0))


def bell_state_rho() -> np.ndarray:
    """Return density matrix for |ψ+> = (|00> + |11>)/√2."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2.0)
    psi[3] = 1.0 / math.sqrt(2.0)
    return np.outer(psi, psi.conj())


def chsh_S(rho: np.ndarray) -> float:
    """Compute CHSH parameter S for the fixed settings (A0,A1,B0,B1)."""
    E00 = np.trace(rho @ (A0 @ B0)).real
    E01 = np.trace(rho @ (A0 @ B1)).real
    E10 = np.trace(rho @ (A1 @ B0)).real
    E11 = np.trace(rho @ (A1 @ B1)).real
    return E00 + E01 + E10 - E11


# ---------------------------------------------------------------------
# Weak measurement channel on qubit A
# ---------------------------------------------------------------------

def build_kraus(eta: float):
    """
    Build Kraus operators M0, M1 for a 2-outcome weak σ_z measurement
    on a single qubit, with information-strength parameter η in [0,1].

    On the single-qubit Hilbert space:
        p0 = (1 + η)/2
        M0 = sqrt(p0) |0><0| + sqrt(1-p0) |1><1|
        M1 = sqrt(1-p0) |0><0| + sqrt(p0) |1><1|

    These satisfy M0†M0 + M1†M1 = I.

    η = 0  ⇒  M0 = M1 = I / √2  (no info, no disturbance)
    η = 1  ⇒  projective σ_z measurement (1 bit of info)
    """
    if eta < 0.0 or eta > 1.0:
        raise ValueError("eta should be in [0, 1]")

    p0 = (1.0 + eta) / 2.0
    M0 = np.array([[math.sqrt(p0), 0.0],
                   [0.0, math.sqrt(1.0 - p0)]], dtype=complex)
    M1 = np.array([[math.sqrt(1.0 - p0), 0.0],
                   [0.0, math.sqrt(p0)]], dtype=complex)
    return M0, M1


def apply_weak_measurement_channel(rho: np.ndarray, eta: float) -> np.ndarray:
    """
    Apply the weak measurement channel (average over outcomes) on
    qubit A of a 2-qubit state rho.
    """
    M0, M1 = build_kraus(eta)
    M0A = kron(M0, I2)
    M1A = kron(M1, I2)

    rho0 = M0A @ rho @ M0A.conj().T
    rho1 = M1A @ rho @ M1A.conj().T
    return rho0 + rho1


# ---------------------------------------------------------------------
# Information-theoretic quantities
# ---------------------------------------------------------------------

# Projectors onto |0>_A and |1>_A (tensored with I on B)
Pz0 = kron(np.array([[1, 0],
                     [0, 0]], dtype=complex), I2)
Pz1 = kron(np.array([[0, 0],
                     [0, 1]], dtype=complex), I2)


def mutual_info_one_step(rho: np.ndarray, eta: float) -> float:
    """
    Compute I(Z;M) for one weak-measurement step, where:
      - Z is the outcome of an ideal σ_z measurement on qubit A
      - M is the 2-outcome weak measurement with Kraus {M0, M1}

    We form the joint distribution p(z,m) by:
      p(z,m) = Tr[ P_z * ρ_m ]
      where ρ_m = (M_m⊗I) ρ (M_m⊗I)†.

    Returns mutual information in bits.
    """
    M0, M1 = build_kraus(eta)
    Ms = [M0, M1]
    Pz = [Pz0, Pz1]

    p_zm = np.zeros((2, 2), dtype=float)

    for m, M in enumerate(Ms):
        MA = kron(M, I2)
        rho_m = MA @ rho @ MA.conj().T
        for z in (0, 1):
            p = np.trace(Pz[z] @ rho_m).real
            p_zm[z, m] = max(p, 0.0)  # clip tiny negatives

    total = p_zm.sum()
    if total <= 0:
        return 0.0
    p_zm /= total

    p_z = p_zm.sum(axis=1)
    p_m = p_zm.sum(axis=0)

    I = 0.0
    for z in (0, 1):
        for m in (0, 1):
            if p_zm[z, m] > 0 and p_z[z] > 0 and p_m[m] > 0:
                I += p_zm[z, m] * math.log2(p_zm[z, m] / (p_z[z] * p_m[m]))
    return float(I)


def von_neumann_entropy_bits(rho: np.ndarray) -> float:
    """Von Neumann entropy S(ρ) in bits."""
    vals = np.linalg.eigvalsh(rho)
    S = 0.0
    for lam in vals:
        lam = float(lam.real)
        if lam > 1e-12:
            S -= lam * math.log2(lam)
    return float(S)


# ---------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------

def run_sequence_for_eta(eta: float, n_steps: int, rho0: np.ndarray):
    """
    For a given weak-measurement strength η, run n_steps of:
      - compute I_k
      - apply weak measurement channel
    Return:
      I_total, S_final, ΔS_vN (entropy increase), final_state
    """
    rho_k = rho0.copy()
    S0_vN = von_neumann_entropy_bits(rho_k)

    I_total = 0.0
    for _ in range(n_steps):
        I_step = mutual_info_one_step(rho_k, eta)
        I_total += I_step
        rho_k = apply_weak_measurement_channel(rho_k, eta)

    S_final = chsh_S(rho_k)
    S_final_abs = abs(S_final)
    S_vN_final = von_neumann_entropy_bits(rho_k)
    delta_S_vN = S_vN_final - S0_vN

    return I_total, S_final_abs, delta_S_vN, rho_k


def info_tradeoff_sweep(
    eta_min: float,
    eta_max: float,
    n_eta: int,
    n_steps: int,
):
    """
    Sweep over η in [eta_min, eta_max], run sequences, and collect arrays.
    """
    rho0 = bell_state_rho()
    S0 = abs(chsh_S(rho0))
    tsirelson = 2.0 * math.sqrt(2.0)
    S0_vN = von_neumann_entropy_bits(rho0)

    etas = np.linspace(eta_min, eta_max, n_eta)
    I_totals = np.zeros_like(etas)
    S_final_abs = np.zeros_like(etas)
    S_deficits = np.zeros_like(etas)
    delta_S_vN = np.zeros_like(etas)

    print("=" * 70)
    print("INFO-TRADEOFF SWEEP")
    print("=" * 70)
    print(f"Initial CHSH S0       ≈ {S0:.6f}")
    print(f"Tsirelson bound       =  {tsirelson:.6f}")
    print(f"Initial entropy S_vN  ≈ {S0_vN:.6e} bits")
    print(f"n_steps per eta       =  {n_steps}")
    print(f"eta range             =  [{eta_min:.3f}, {eta_max:.3f}]")
    print(f"n_eta points          =  {n_eta}")
    print("=" * 70)

    for i, eta in enumerate(etas):
        I_total, S_abs, dS_vN, _ = run_sequence_for_eta(eta, n_steps, rho0)
        I_totals[i] = I_total
        S_final_abs[i] = S_abs
        S_deficits[i] = tsirelson - S_abs
        delta_S_vN[i] = dS_vN
        print(f"eta={eta:.3f}  I_total={I_total:.6f}  "
              f"S_final={S_abs:.6f}  ΔS={S_deficits[i]:.6f}  "
              f"ΔS_vN={delta_S_vN[i]:.6f}")

    return {
        "etas": etas,
        "I_totals": I_totals,
        "S_final_abs": S_final_abs,
        "S_deficits": S_deficits,
        "delta_S_vN": delta_S_vN,
        "S0": S0,
        "tsirelson": tsirelson,
        "S0_vN": S0_vN,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------
# Plotting and small-info linear fit
# ---------------------------------------------------------------------

def plot_results(results, out_path_png: str):
    etas = results["etas"]
    I_totals = results["I_totals"]
    S_deficits = results["S_deficits"]
    delta_S_vN = results["delta_S_vN"]
    tsirelson = results["tsirelson"]

    # Basic figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Tsirelson deficit vs I_total
    ax = axes[0]
    ax.plot(I_totals, S_deficits, "o-")
    ax.set_xlabel("Total classical information extracted I_total (bits)")
    ax.set_ylabel(r"Tsirelson deficit $2\sqrt{2} - |S|$")
    ax.set_title("Bell deficit vs extracted information")

    # Right: entropy increase vs I_total
    ax2 = axes[1]
    ax2.plot(I_totals, delta_S_vN, "o-")
    ax2.set_xlabel("Total classical information extracted I_total (bits)")
    ax2.set_ylabel(r"Entropy increase $\Delta S_{\mathrm{vN}}$ (bits)")
    ax2.set_title("Entropy/irreversibility vs extracted information")

    fig.suptitle(
        f"Information–Bell tradeoff (n_steps={results['n_steps']}, "
        f"S0≈{tsirelson:.3f})",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def small_info_fit(results, I_max_fit: float = 0.3):
    """
    Fit straight lines in the small-information regime:

        ΔS       ≈ a * I_total
        ΔS_vN    ≈ b * I_total

    using points with I_total <= I_max_fit.
    """
    I_totals = results["I_totals"]
    S_deficits = results["S_deficits"]
    delta_S_vN = results["delta_S_vN"]

    mask = I_totals <= I_max_fit
    if np.count_nonzero(mask) < 2:
        print("Not enough small-info points for a reliable fit.")
        return None

    x = I_totals[mask]
    y_def = S_deficits[mask]
    y_ent = delta_S_vN[mask]

    # Fit y = a x + b, but we expect nearly zero intercept, so we can either
    # force through origin or just report slope from polyfit.
    a_def, b_def = np.polyfit(x, y_def, 1)
    a_ent, b_ent = np.polyfit(x, y_ent, 1)

    print("\nSmall-information linear fits (using I_total <= "
          f"{I_max_fit:.3f} bits):")
    print(f"  ΔS       ≈ a * I_total + b   with a ≈ {a_def:.3f}, b ≈ {b_def:.3e}")
    print(f"  ΔS_vN    ≈ a' * I_total + b' with a'≈ {a_ent:.3f}, b'≈ {b_ent:.3e}")

    return {
        "I_max_fit": I_max_fit,
        "a_def": a_def,
        "b_def": b_def,
        "a_ent": a_ent,
        "b_ent": b_ent,
    }


# ---------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep weak-measurement strength and compute "
                    "Bell deficit vs extracted information."
    )
    parser.add_argument("--eta_min", type=float, default=0.0,
                        help="Minimum weak-measurement strength η (default 0.0)")
    parser.add_argument("--eta_max", type=float, default=0.5,
                        help="Maximum weak-measurement strength η (default 0.5)")
    parser.add_argument("--n_eta", type=int, default=11,
                        help="Number of η values to sweep (default 11)")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of weak-measurement steps per η (default 10)")
    parser.add_argument("--I_max_fit", type=float, default=0.3,
                        help="Max I_total for small-info linear fit (default 0.3 bits)")

    args = parser.parse_args()

    # Run sweep
    results = info_tradeoff_sweep(
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        n_eta=args.n_eta,
        n_steps=args.n_steps,
    )

    # Prepare outputs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Save plot
    png_path = os.path.join(out_dir, "info_tradeoff_sweep.png")
    plot_results(results, png_path)
    print(f"\n✓ Plot saved to: {png_path}")

    # Save data
    npz_path = os.path.join(out_dir, "info_tradeoff_sweep_data.npz")
    np.savez(npz_path, **results)
    print(f"✓ Data saved to: {npz_path}")

    # Small-info linear fit
    fit = small_info_fit(results, I_max_fit=args.I_max_fit)
    if fit is not None:
        print("\nFit parameters (small-info regime):")
        for k, v in fit.items():
            print(f"  {k:10s} = {v}")

    print("\n" + "=" * 70)
    print("INFO-TRADEOFF SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
