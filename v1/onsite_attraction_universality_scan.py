#!/usr/bin/env python3
"""
onsite_attraction_universality_scan.py

Universality test #3: Replace defrag-like binding with onsite attraction.

We start from a 3D two-excitation substrate model (TwoFermion3DParams,
build_twofermion3d_hamiltonian) and construct a new "onsite-attraction"
family of Hamiltonians by:

  - turning OFF the defrag term:   g_defrag = 0.0
  - keeping lambda_G as in the baseline (it's uniform on 2x2x2 anyway)
  - using a NEGATIVE onsite interaction lambda_T < 0 as the primary
    binding mechanism, active when r1 == r2.

We then scan over lambda_T in a NEGATIVE interval:

    lambda_T ~ Uniform[U_min, U_max], with U_min < U_max < 0,

and for each sample we:

  - build H_total = H_base(lambda_T),
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

We then summarize what fraction of samples in this onsite-attraction
family show fermion-like behavior, and save full results as JSON:

    outputs_universality/onsite_attraction_universality_results.json

This probes *universality* across a structurally different binding
mechanism (onsite attraction instead of defrag/Gauss).
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
# Baseline parameters for onsite-attraction family
# ---------------------------------------------------------------------


def baseline_params() -> se3d.TwoFermion3DParams:
    """
    Return the baseline TwoFermion3DParams for the onsite-attraction family.

    Differences from the original "defrag" baseline:
      - g_defrag = 0.0 (defrag off)
      - lambda_T = -1.0 (onsite attraction; can be overridden by the scan)

    Other couplings are kept as in the earlier fermion-like regime.
    """
    return se3d.TwoFermion3DParams(
        Lx=2,
        Ly=2,
        Lz=2,
        J_hop=1.0,
        mass=0.1,
        g_defrag=0.0,      # TURN OFF defrag
        sigma_defrag=1.0,  # irrelevant when g_defrag=0
        lambda_G=5.0,      # uniform shift on 2x2x2; fine to keep
        lambda_S=-1.0,     # antiferromagnetic-like spin interaction
        lambda_T=-1.0,     # onsite attraction (U<0) - will be scanned
        J_exch=1.0,        # explicit exchange coupling
        max_eigsh_iter=5000,
        k_eigs=1,
    )


# ---------------------------------------------------------------------
# Single-sample run and classification
# ---------------------------------------------------------------------


def run_single_sample(
    params: se3d.TwoFermion3DParams,
) -> Dict[str, Any]:
    """
    Build H(lambda_T) via the standard engine, find ground state, and
    compute diagnostics for a single choice of lambda_T.
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
            "Universality test: onsite attraction binding via negative "
            "lambda_T, with defrag turned off (g_defrag=0)."
        )
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of random lambda_T samples (default 50)",
    )
    p.add_argument(
        "--U_min",
        type=float,
        default=-2.0,
        help="Minimum lambda_T (U_min < 0, default -2.0)",
    )
    p.add_argument(
        "--U_max",
        type=float,
        default=-0.1,
        help="Maximum lambda_T (U_max < 0, default -0.1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=3025,
        help="Random seed for reproducibility (default 3025)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not (args.U_min < 0.0 and args.U_max < 0.0 and args.U_min < args.U_max):
        raise ValueError(
            "Require U_min < U_max < 0 for negative onsite attraction."
        )

    base = baseline_params()
    base_dict = asdict(base)

    print("=" * 72)
    print("ONSITE ATTRACTION UNIVERSALITY TEST")
    print("=" * 72)
    print("Baseline parameters template (before setting lambda_T):")
    for k, v in base_dict.items():
        print(f"  {k:15s} = {v}")
    print("-" * 72)
    print("Binding mechanism:")
    print("  defrag term OFF: g_defrag = 0.0")
    print("  onsite attraction at r1 == r2 via lambda_T < 0")
    print(f"Random lambda_T ~ Uniform[{args.U_min:.3f}, {args.U_max:.3f}]")
    print(f"Number of samples : {args.n_samples}")
    print(f"Random seed       : {args.seed}")
    print("-" * 72)

    rng = np.random.default_rng(args.seed)

    records: List[Dict[str, Any]] = []
    n_core = 0
    n_bell = 0
    n_tsir = 0

    for i in range(args.n_samples):
        # Sample lambda_T in [U_min, U_max], both negative
        lambda_T_i = float(rng.uniform(args.U_min, args.U_max))

        # Build params_i from baseline dict, overriding lambda_T
        d = dict(base_dict)
        d["lambda_T"] = lambda_T_i
        params_i = se3d.TwoFermion3DParams(**d)

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
            f"lambda_T={lambda_T_i:+.3f}, "
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
    print("UNIVERSALITY SUMMARY (ONSITE ATTRACTION)")
    print("=" * 72)
    print(f"lambda_T range                 : [{args.U_min:.3f}, {args.U_max:.3f}]")
    print(f"n_samples                      : {args.n_samples}")
    print("-" * 72)
    print(f"Fraction with fermionic_core (A>0.95 & Fs>0.95): {f_core:6.3f}")
    print(f"Fraction with Bell violation (|S|>2.0)        : {f_bell:6.3f}")
    print(f"Fraction near Tsirelson (|S|-2.828<0.05)      : {f_tsir:6.3f}")
    print("=" * 72)

    # Save full records as JSON for further analysis
    save_dir = os.path.join(os.path.dirname(__file__), "outputs_universality")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "onsite_attraction_universality_results.json")

    output_payload = {
        "U_min": args.U_min,
        "U_max": args.U_max,
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
