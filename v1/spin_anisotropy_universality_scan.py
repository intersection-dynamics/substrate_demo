#!/usr/bin/env python3
"""
spin_anisotropy_universality_scan.py

Universality test #1: Spin-symmetry breaking via a Zeeman-like term.

We start from the 3D two-excitation substrate model defined in
`substrate_engine_3d.py` (TwoFermion3DParams + build_twofermion3d_hamiltonian)
and then add a new term that breaks global SU(2) spin symmetry:

    H_Z = h_z * sum_r ( S1^z(r) - S2^z(r) )

where S^z has eigenvalues +1/2 (spin up) and -1/2 (spin down).
The parameter h_z is sampled randomly in a range [-h_max, +h_max].

For each sample, we:
  - build H_total = H_base + H_Z,
  - find the ground state via eigsh,
  - compute:
        antisymmetry score A (0..1)
        singlet fraction at overlap Fs (0..1)
        overlap probability P_overlap
        CHSH Bell parameter S_chsh
  - classify:
        fermionic_core = (A > 0.95 and Fs > 0.95)
        bell_violation = (|S| > 2.0)
        near_tsirelson = (| |S| - 2.828427 | < 0.05)

We then summarize what fraction of random h_z samples remain in the
"fermion-like" phase, and save full results as JSON:

    outputs_universality/spin_anisotropy_universality_results.json

This is designed to probe *universality* in the sense of robustness
to a structural spin-symmetry-breaking perturbation, not just rescaling
existing couplings.
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import substrate_engine_3d as se3d


# ---------------------------------------------------------------------
# Baseline parameters (fermion-like regime, from substrate_engine_3d)
# ---------------------------------------------------------------------


def baseline_params() -> se3d.TwoFermion3DParams:
    """
    Return the baseline TwoFermion3DParams in the fermion-like regime.
    Adjust here if your current best point changes.
    """
    return se3d.TwoFermion3DParams(
        Lx=2,
        Ly=2,
        Lz=2,
        J_hop=1.0,
        mass=0.1,
        g_defrag=1.0,
        sigma_defrag=1.0,
        lambda_G=5.0,
        lambda_S=-1.0,
        lambda_T=0.0,
        J_exch=1.0,
        max_eigsh_iter=5000,
        k_eigs=1,
    )


# ---------------------------------------------------------------------
# Zeeman-like spin anisotropy term
# ---------------------------------------------------------------------


def build_zeeman_term_diff(
    params: se3d.TwoFermion3DParams,
    h_z: float,
) -> sp.csr_matrix:
    """
    Build a diagonal sparse matrix H_Z corresponding to:

        H_Z = h_z * sum_r ( S1^z(r) - S2^z(r) )

    where S^z eigenvalues are +1/2 for spin-up (s=0) and -1/2 for spin-down (s=1).

    This is a simple spin-symmetry-breaking term: it energetically favors
    S1^z and disfavors S2^z (or vice versa, depending on sign of h_z).
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    dim = Ns * 2 * Ns * 2

    diag = np.zeros(dim, dtype=float)

    for idx in range(dim):
        r1, s1, r2, s2 = se3d.decode_basis_3d(idx, Ns)

        # S^z eigenvalues: +1/2 for up (s=0), -1/2 for down (s=1)
        S1z = 0.5 if s1 == 0 else -0.5
        S2z = 0.5 if s2 == 0 else -0.5

        # Differential Zeeman term: h_z (S1^z - S2^z)
        diag[idx] = h_z * (S1z - S2z)

    H_Z = sp.diags(diag, offsets=0, format="csr")
    return H_Z


# ---------------------------------------------------------------------
# Single-sample run and classification
# ---------------------------------------------------------------------


def run_single_sample(
    params: se3d.TwoFermion3DParams,
    h_z: float,
) -> Dict[str, Any]:
    """
    Build H_total = H_base + H_Z, find ground state, and compute diagnostics.
    """
    # Base Hamiltonian from substrate_engine_3d
    H_base = se3d.build_twofermion3d_hamiltonian(params)

    # Zeeman-like term
    H_Z = build_zeeman_term_diff(params, h_z)

    # Total Hamiltonian
    H_total = H_base + H_Z

    dim = H_total.shape[0]

    # Ground state via eigsh
    evals, evecs = eigsh(
        H_total,
        k=1,
        which="SA",
        maxiter=params.max_eigsh_iter,
    )
    E0 = float(evals[0].real)
    psi0 = evecs[:, 0]

    # Diagnostics using existing engine functions
    anti = se3d.antisymmetry_metrics(psi0, params)
    overlap = se3d.overlap_and_spin_metrics(psi0, params)
    E_gauss = se3d.gauss_energy_expectation(psi0, params)
    S_chsh = se3d.chsh_S_from_state(psi0, params)

    antisym_score = float(anti["antisym_score"])
    singlet_fraction = float(overlap["singlet_fraction"])
    overlap_prob = float(overlap["overlap_prob"])
    abs_S = abs(S_chsh)

    fermionic_core = (antisym_score > 0.95 and singlet_fraction > 0.95)
    bell_violation = (abs_S > 2.0)
    near_tsirelson = (abs(abs_S - 2.828427) < 0.05)

    return {
        "params": asdict(params),
        "h_z": h_z,
        "E0": E0,
        "E_gauss": E_gauss,
        "antisym_score": antisym_score,
        "singlet_fraction": singlet_fraction,
        "overlap_prob": overlap_prob,
        "S_chsh": S_chsh,
        "abs_S": abs_S,
        "fermionic_core": fermionic_core,
        "bell_violation": bell_violation,
        "near_tsirelson": near_tsirelson,
        "dim": dim,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Universality test: spin-symmetry breaking via Zeeman-like term "
            "H_Z = h_z (S1^z - S2^z)."
        )
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of random h_z samples (default 50)",
    )
    p.add_argument(
        "--h_max",
        type=float,
        default=0.5,
        help="Maximum |h_z| value (uniform in [-h_max, +h_max], default 0.5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility (default 2025)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base = baseline_params()
    base_dict = asdict(base)

    print("=" * 72)
    print("SPIN ANISOTROPY UNIVERSALITY TEST")
    print("=" * 72)
    print("Baseline parameters (H_base):")
    for k, v in base_dict.items():
        print(f"  {k:15s} = {v}")
    print("-" * 72)
    print("Spin-symmetry-breaking term:")
    print("  H_Z = h_z * sum_r (S1^z(r) - S2^z(r))")
    print(f"Random h_z ~ Uniform[-{args.h_max:.3f}, +{args.h_max:.3f}]")
    print(f"Number of samples : {args.n_samples}")
    print(f"Random seed       : {args.seed}")
    print("-" * 72)

    rng = np.random.default_rng(args.seed)

    records: List[Dict[str, Any]] = []
    n_core = 0
    n_bell = 0
    n_tsir = 0

    for i in range(args.n_samples):
        h_z = float(rng.uniform(-args.h_max, args.h_max))
        rec = run_single_sample(base, h_z)
        records.append(rec)

        if rec["fermionic_core"]:
            n_core += 1
        if rec["bell_violation"]:
            n_bell += 1
        if rec["near_tsirelson"]:
            n_tsir += 1

        print(
            f"[sample {i+1:3d}/{args.n_samples}] "
            f"h_z={h_z:+.3f}, "
            f"A={rec['antisym_score']:.4f}, "
            f"Fs={rec['singlet_fraction']:.4f}, "
            f"|S|={rec['abs_S']:.4f}, "
            f"core={rec['fermionic_core']}, "
            f"Bell={rec['bell_violation']}, "
            f"Tsir={rec['near_tsirelson']}"
        )

    f_core = n_core / args.n_samples
    f_bell = n_bell / args.n_samples
    f_tsir = n_tsir / args.n_samples

    print("\n" + "=" * 72)
    print("UNIVERSALITY SUMMARY (SPIN ANISOTROPY)")
    print("=" * 72)
    print(f"h_z range                     : [-{args.h_max:.3f}, +{args.h_max:.3f}]")
    print(f"n_samples                     : {args.n_samples}")
    print("-" * 72)
    print(f"Fraction with fermionic_core (A>0.95 & Fs>0.95): {f_core:6.3f}")
    print(f"Fraction with Bell violation (|S|>2.0)        : {f_bell:6.3f}")
    print(f"Fraction near Tsirelson (|S|-2.828<0.05)      : {f_tsir:6.3f}")
    print("=" * 72)

    # Save full records as JSON for further analysis
    save_dir = os.path.join(os.path.dirname(__file__), "outputs_universality")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "spin_anisotropy_universality_results.json")

    output_payload = {
        "h_max": args.h_max,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "baseline": base_dict,
        "summary": {
            "fraction_fermionic_core": f_core,
            "fraction_bell_violation": f_bell,
            "fraction_near_tsirelson": f_tsir,
        },
        "records": records,
    }

    with open(save_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Saved detailed JSON results to: {save_path}")


if __name__ == "__main__":
    main()
