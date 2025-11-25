#!/usr/bin/env python3
"""
larger_system_universality_scan.py

Universality test: scale the lattice up to Lx=Ly=Lz=4 (or other L)
and check whether the fermion-like two-excitation phase persists.

We consider two Hamiltonian "families":

  1) defrag family:
       - g_defrag > 0 provides a central Gaussian-like binding
       - lambda_T = 0 (no onsite term)
       - scan over g_defrag in [g_min, g_max]

  2) onsite family:
       - g_defrag = 0 (defrag OFF)
       - lambda_T < 0 provides onsite attraction
       - scan over lambda_T in [U_min, U_max], with U_min < U_max < 0

For each sample, we:

  - build H(params) via `substrate_engine_3d.build_twofermion3d_hamiltonian`,
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

We then summarize:

    Fraction with fermionic core
    Fraction with Bell violation
    Fraction near Tsirelson

and save full results as JSON:

    outputs_universality/larger_system_universality_results_L{L}_{family}.json
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np
from scipy.sparse.linalg import eigsh

import substrate_engine_3d as se3d


# ---------------------------------------------------------------------
# Parameter templates for the two families
# ---------------------------------------------------------------------


def defrag_baseline(L: int) -> se3d.TwoFermion3DParams:
    """
    Baseline template for the defrag family on an LxLxL lattice.
    g_defrag will be scanned; other couplings are fixed.
    """
    return se3d.TwoFermion3DParams(
        Lx=L,
        Ly=L,
        Lz=L,
        J_hop=1.0,
        mass=0.1,
        g_defrag=1.0,      # will be overridden per sample
        sigma_defrag=1.0,
        lambda_G=5.0,
        lambda_S=-1.0,
        lambda_T=0.0,
        J_exch=1.0,
        max_eigsh_iter=5000,
        k_eigs=1,
    )


def onsite_baseline(L: int) -> se3d.TwoFermion3DParams:
    """
    Baseline template for the onsite-attraction family on an LxLxL lattice.
    lambda_T will be scanned; defrag is OFF.
    """
    return se3d.TwoFermion3DParams(
        Lx=L,
        Ly=L,
        Lz=L,
        J_hop=1.0,
        mass=0.1,
        g_defrag=0.0,      # defrag OFF
        sigma_defrag=1.0,  # irrelevant when g_defrag=0
        lambda_G=5.0,
        lambda_S=-1.0,
        lambda_T=-1.0,     # will be overridden per sample
        J_exch=1.0,
        max_eigsh_iter=5000,
        k_eigs=1,
    )


# ---------------------------------------------------------------------
# Single-sample run and classification
# ---------------------------------------------------------------------


def run_single_sample(params: se3d.TwoFermion3DParams) -> Dict[str, Any]:
    """
    Build H(params), find ground state, and compute diagnostics
    for a single parameter choice.
    """
    H = se3d.build_twofermion3d_hamiltonian(params)
    dim = H.shape[0]

    # Ground state via eigsh
    evals, evecs = eigsh(
        H,
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
            "Larger-system universality test: Lx=Ly=E Lz=L (default 4), "
            "for defrag or onsite-attraction Hamiltonian families."
        )
    )
    p.add_argument(
        "--L",
        type=int,
        default=4,
        help="Lattice linear size L (Lx=Ly=Lz=L, default 4)",
    )
    p.add_argument(
        "--family",
        type=str,
        default="defrag",
        choices=["defrag", "onsite"],
        help="Hamiltonian family to scan: 'defrag' or 'onsite' (default 'defrag')",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of random parameter samples (default 20)",
    )
    # Defrag family scan range
    p.add_argument(
        "--g_min",
        type=float,
        default=0.5,
        help="Minimum g_defrag for defrag family (default 0.5)",
    )
    p.add_argument(
        "--g_max",
        type=float,
        default=1.5,
        help="Maximum g_defrag for defrag family (default 1.5)",
    )
    # Onsite family scan range
    p.add_argument(
        "--U_min",
        type=float,
        default=-2.0,
        help="Minimum lambda_T (U_min < 0) for onsite family (default -2.0)",
    )
    p.add_argument(
        "--U_max",
        type=float,
        default=-0.1,
        help="Maximum lambda_T (U_max < 0) for onsite family (default -0.1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=4040,
        help="Random seed for reproducibility (default 4040)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.family == "defrag":
        if not (args.g_min < args.g_max):
            raise ValueError("Require g_min < g_max for defrag family.")
    elif args.family == "onsite":
        if not (args.U_min < 0.0 and args.U_max < 0.0 and args.U_min < args.U_max):
            raise ValueError(
                "Require U_min < U_max < 0 for negative onsite attraction."
            )

    if args.family == "defrag":
        base = defrag_baseline(args.L)
    else:
        base = onsite_baseline(args.L)

    base_dict = asdict(base)

    print("=" * 72)
    print("LARGER-SYSTEM UNIVERSALITY TEST")
    print("=" * 72)
    print(f"Lattice size: Lx=Ly=Lz={args.L} (Ns={args.L**3} sites)")
    print(f"Hamiltonian family: {args.family}")
    print("Baseline parameter template:")
    for k, v in base_dict.items():
        print(f"  {k:15s} = {v}")
    print("-" * 72)

    if args.family == "defrag":
        print("Family details: DEFGRAG BINDING")
        print("  g_defrag is scanned in [g_min, g_max]")
        print(f"  g_min = {args.g_min:.3f}, g_max = {args.g_max:.3f}")
    else:
        print("Family details: ONSITE ATTRACTION")
        print("  defrag OFF: g_defrag = 0.0")
        print("  lambda_T is scanned in [U_min, U_max] (negative)")
        print(f"  U_min = {args.U_min:.3f}, U_max = {args.U_max:.3f}")
    print(f"Number of samples : {args.n_samples}")
    print(f"Random seed       : {args.seed}")
    print("-" * 72)

    rng = np.random.default_rng(args.seed)

    records: List[Dict[str, Any]] = []
    n_core = 0
    n_bell = 0
    n_tsir = 0

    for i in range(args.n_samples):
        d = dict(base_dict)

        if args.family == "defrag":
            g_i = float(rng.uniform(args.g_min, args.g_max))
            d["g_defrag"] = g_i
            params_i = se3d.TwoFermion3DParams(**d)
            label = f"g_defrag={g_i:+.3f}"
        else:
            lambda_T_i = float(rng.uniform(args.U_min, args.U_max))
            d["lambda_T"] = lambda_T_i
            params_i = se3d.TwoFermion3DParams(**d)
            label = f"lambda_T={lambda_T_i:+.3f}"

        rec = run_single_sample(params_i)
        records.append(rec)

        if rec["fermionic_core"]:
            n_core += 1
        if rec["bell_violation"]:
            n_bell += 1
        if rec["near_tsirelson"]:
            n_tsir += 1

        print(
            f"[sample {i+1:3d}/{args.n_samples}] "
            f"{label}, "
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
    print("UNIVERSALITY SUMMARY (LARGER SYSTEM)")
    print("=" * 72)
    print(f"Lattice size                   : Lx=Ly=Lz={args.L}")
    print(f"Family                         : {args.family}")
    print(f"n_samples                      : {args.n_samples}")
    print("-" * 72)
    print(f"Fraction with fermionic_core (A>0.95 & Fs>0.95): {f_core:6.3f}")
    print(f"Fraction with Bell violation (|S|>2.0)        : {f_bell:6.3f}")
    print(f"Fraction near Tsirelson (|S|-2.828<0.05)      : {f_tsir:6.3f}")
    print("=" * 72)

    # Save full records as JSON for further analysis
    save_dir = os.path.join(os.path.dirname(__file__), "outputs_universality")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"larger_system_universality_results_L{args.L}_{args.family}.json",
    )

    output_payload = {
        "L": args.L,
        "family": args.family,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "baseline_template": base_dict,
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
