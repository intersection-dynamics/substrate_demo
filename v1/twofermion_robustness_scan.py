#!/usr/bin/env python3
"""
twofermion_robustness_scan.py

Multi-parameter robustness test for the 3D two-excitation substrate model.

We:
  - Take a baseline parameter set known to produce fermion-like behavior
    (antisymmetric, singlet-dominated, Bell-violating).
  - Sample random perturbations of multiple parameters simultaneously
    by a relative factor (1 + epsilon_i), epsilon_i ∈ [-delta, +delta].
  - For each sample, run run_twofermion3d_experiment(...) and record:

        antisym_score, singlet_fraction, S_chsh, etc.

  - Count how many samples preserve:

        - high antisymmetry (A > 0.95)
        - high singlet fraction (Fs > 0.95)
        - Bell violation (|S| > 2)
        - near Tsirelson saturation (|S| ~ 2.828)

Results are printed to stdout and also saved as a JSON file:
    outputs_robustness/twofermion_robustness_results.json
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np

import substrate_engine_3d as se3d


# ---------------------------------------------------------------------
# Baseline parameter set (fermion-like regime)
# ---------------------------------------------------------------------

def baseline_params() -> se3d.TwoFermion3DParams:
    """
    Return the baseline TwoFermion3DParams in the fermion-like regime.
    Adjust if your current best point differs.
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


# Which parameters to perturb jointly
PERTURB_KEYS = [
    "g_defrag",
    "sigma_defrag",
    "lambda_G",
    "lambda_S",
    "lambda_T",
    "J_exch",
]


def make_perturbed_params(
    base: se3d.TwoFermion3DParams,
    rng: np.random.Generator,
    delta: float,
) -> se3d.TwoFermion3DParams:
    """
    Create a perturbed copy of 'base' by multiplying selected parameters
    by (1 + eps_k), eps_k ~ Uniform[-delta, +delta].

    delta ~ 0.05 corresponds to ±5% relative perturbations.
    """
    d = asdict(base)
    for key in PERTURB_KEYS:
        base_val = float(d[key])
        eps = rng.uniform(-delta, delta)
        d[key] = base_val * (1.0 + eps)
    return se3d.TwoFermion3DParams(**d)


# ---------------------------------------------------------------------
# Single-sample run and classification
# ---------------------------------------------------------------------

def run_single_sample(
    params: se3d.TwoFermion3DParams,
) -> Dict[str, Any]:
    """
    Run a single experiment and classify its fermion-like behavior.
    """
    results = se3d.run_twofermion3d_experiment(params)

    anti = results["antisymmetry"]
    overlap = results["overlap"]
    S_chsh = float(results["S_chsh"])

    antisym_score = float(anti["antisym_score"])
    singlet_fraction = float(overlap["singlet_fraction"])
    overlap_prob = float(overlap["overlap_prob"])
    abs_S = abs(S_chsh)

    fermionic_core = (antisym_score > 0.95 and singlet_fraction > 0.95)
    bell_violation = (abs_S > 2.0)
    near_tsirelson = (abs(abs_S - 2.828427) < 0.05)

    # Flatten params for JSON output
    params_dict = asdict(params)

    return {
        "params": params_dict,
        "E0": float(results["E0"]),
        "E_gauss": float(results["E_gauss"]),
        "antisym_score": antisym_score,
        "singlet_fraction": singlet_fraction,
        "overlap_prob": overlap_prob,
        "S_chsh": S_chsh,
        "abs_S": abs_S,
        "fermionic_core": fermionic_core,
        "bell_violation": bell_violation,
        "near_tsirelson": near_tsirelson,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Multi-parameter robustness scan for the 3D two-excitation "
            "substrate model (fermion-like regime)."
        )
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of random perturbation samples (default 50)",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Relative perturbation scale (delta=0.05 => ±5%% on each param)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default 1234)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base = baseline_params()
    base_dict = asdict(base)

    print("=" * 72)
    print("TWO-FERMION 3D MULTI-PARAMETER ROBUSTNESS SCAN")
    print("=" * 72)
    print("Baseline parameters:")
    for k, v in base_dict.items():
        print(f"  {k:15s} = {v}")
    print("-" * 72)
    print(f"Perturbing keys: {', '.join(PERTURB_KEYS)}")
    print(f"Relative perturbation scale delta = {args.delta:.3f}")
    print(f"Number of samples                = {args.n_samples}")
    print("-" * 72)

    rng = np.random.default_rng(args.seed)

    records: List[Dict[str, Any]] = []
    n_core = 0
    n_bell = 0
    n_tsir = 0

    for i in range(args.n_samples):
        params_i = make_perturbed_params(base, rng, args.delta)
        rec = run_single_sample(params_i)
        records.append(rec)

        if rec["fermionic_core"]:
            n_core += 1
        if rec["bell_violation"]:
            n_bell += 1
        if rec["near_tsirelson"]:
            n_tsir += 1

        print(f"[sample {i+1:3d}/{args.n_samples}] "
              f"A={rec['antisym_score']:.4f}, "
              f"Fs={rec['singlet_fraction']:.4f}, "
              f"|S|={rec['abs_S']:.4f}, "
              f"core={rec['fermionic_core']}, "
              f"Bell={rec['bell_violation']}, "
              f"Tsir={rec['near_tsirelson']}")

    f_core = n_core / args.n_samples
    f_bell = n_bell / args.n_samples
    f_tsir = n_tsir / args.n_samples

    print("\n" + "=" * 72)
    print("ROBUSTNESS SUMMARY")
    print("=" * 72)
    print(f"delta (relative perturbation): {args.delta:.3f}")
    print(f"n_samples                    : {args.n_samples}")
    print("-" * 72)
    print(f"Fraction with fermionic_core (A>0.95 & Fs>0.95): {f_core:6.3f}")
    print(f"Fraction with Bell violation (|S|>2.0)        : {f_bell:6.3f}")
    print(f"Fraction near Tsirelson (|S|-2.828<0.05)      : {f_tsir:6.3f}")
    print("=" * 72)

    # Save full records as JSON for analysis and plotting elsewhere
    save_dir = os.path.join(os.path.dirname(__file__), "outputs_robustness")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "twofermion_robustness_results.json")

    output_payload = {
        "delta": args.delta,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "baseline": base_dict,
        "perturb_keys": PERTURB_KEYS,
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
