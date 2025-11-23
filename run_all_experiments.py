#!/usr/bin/env python3
"""
run_all_experiments.py

One-button driver for the 3D substrate research bundle.

What it does:
- Reads settings from config.py
- Runs:
    * Gauss sweep        (gauss_sweep.py)
    * Exchange sweep     (exchange_sweep.py)
    * Sequential test    (sequential_perturbation_test.py)
    * Info extraction    (information_extraction_test.py)
  according to RunFlags.

- Saves:
    * Plots (.png) into outputs/
    * Raw data (.npz) into outputs/
    * A summary.json with headline metrics.

Usage (from the project folder):

    python run_all_experiments.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config import (
    experiment_config,
    lambda_G_array,
    J_exch_array,
)

# Import experiment modules, aliasing their plot functions
from gauss_sweep import (
    run_gauss_sweep,
    plot_results as plot_gauss_results,
)
from exchange_sweep import (
    run_exchange_sweep,
    plot_results as plot_exchange_results,
)
from sequential_perturbation_test import (
    run_sequential_perturbation_test,
    plot_results as plot_sequential_results,
)
from information_extraction_test import (
    run_information_extraction_test,
    plot_results as plot_info_results,
)

from substrate_engine_3d import TwoFermion3DParams


def build_params(lambda_G: float, J_exch: float) -> TwoFermion3DParams:
    """
    Build a TwoFermion3DParams instance using the shared engine/lattice config,
    with experiment-specific lambda_G and J_exch.
    """
    cfg = experiment_config
    lat = cfg.lattice
    eng = cfg.engine

    return TwoFermion3DParams(
        Lx=lat.Lx,
        Ly=lat.Ly,
        Lz=lat.Lz,
        J_hop=eng.J_hop,
        m=eng.m,
        g_defrag=eng.g_defrag,
        sigma_defrag=eng.sigma_defrag,
        lambda_G=lambda_G,
        lambda_S=eng.lambda_S,
        lambda_T=eng.lambda_T,
        J_exch=J_exch,
        max_eigsh_iter=eng.max_eigsh_iter,
        k_eigs=eng.k_eigs,
    )


def summarize_gauss(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract headline metrics from Gauss sweep results."""
    lam = np.array(results["lambda_G"])
    S_abs = np.abs(results["S_chsh"])

    max_S = float(S_abs.max())
    idx_max = int(S_abs.argmax())
    lam_at_max = float(lam[idx_max])

    mask_violate = S_abs > 2.0
    first_violate = float(lam[mask_violate][0]) if np.any(mask_violate) else None

    return {
        "max_abs_S": max_S,
        "lambda_at_max_abs_S": lam_at_max,
        "bell_violation_any": bool(max_S > 2.0),
        "first_lambda_with_violation": first_violate,
    }


def summarize_exchange(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract headline metrics from Exchange sweep results."""
    J = np.array(results["J_exch"])
    S_abs = np.abs(results["S_chsh"])

    max_S = float(S_abs.max())
    idx_max = int(S_abs.argmax())
    J_at_max = float(J[idx_max])

    mask_violate = S_abs > 2.0
    first_violate = float(J[mask_violate][0]) if np.any(mask_violate) else None

    violation_at_zero = bool(S_abs[0] > 2.0)

    return {
        "max_abs_S": max_S,
        "J_at_max_abs_S": J_at_max,
        "bell_violation_any": bool(max_S > 2.0),
        "first_J_with_violation": first_violate,
        "violation_at_J_exch_0": violation_at_zero,
        "abs_S_at_J_exch_0": float(S_abs[0]),
    }


def summarize_sequential(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract headline metrics from Sequential Perturbation test."""
    overlaps = np.array(results["overlap_with_history"][1:], dtype=float)
    dE = np.array(results["delta_energy"][1:], dtype=float)
    dS = np.array(results["delta_entropy"][1:], dtype=float)

    if len(overlaps) > 2:
        corr_overlap_abs_dE = float(np.corrcoef(overlaps, np.abs(dE))[0, 1])
        corr_overlap_abs_dS = float(np.corrcoef(overlaps, np.abs(dS))[0, 1])
    else:
        corr_overlap_abs_dE = None
        corr_overlap_abs_dS = None

    return {
        "corr_overlap_vs_abs_delta_E": corr_overlap_abs_dE,
        "corr_overlap_vs_abs_delta_S": corr_overlap_abs_dS,
        "n_steps": int(len(results["step"]) - 1),
    }


def summarize_info(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract headline metrics from Information Extraction test."""
    S_sub = np.array(results["S_substrate"], dtype=float)
    I_class = np.array(results["I_classical"], dtype=float)
    E = np.array(results["energy"], dtype=float)

    total_delta_S_sub = float(S_sub[-1] - S_sub[0])
    total_delta_I_class = float(I_class[-1] - I_class[0])
    total_delta_E = float(E[-1] - E[0])

    dI = np.array(results["delta_I_classical"][1:], dtype=float)
    dE = np.array(results["delta_energy"][1:], dtype=float)

    if len(dI) > 2:
        corr_dI_abs_dE = float(np.corrcoef(dI, np.abs(dE))[0, 1])
    else:
        corr_dI_abs_dE = None

    return {
        "total_delta_S_substrate": total_delta_S_sub,
        "total_delta_I_classical": total_delta_I_class,
        "total_delta_energy": total_delta_E,
        "corr_delta_I_vs_abs_delta_E": corr_dI_abs_dE,
        "n_steps": int(len(results["step"]) - 1),
    }


def main() -> None:
    cfg = experiment_config
    paths = cfg.paths

    print("\n" + "=" * 70)
    print("RUNNING ALL SUBSTRATE EXPERIMENTS")
    print("=" * 70)
    print(f"Base directory : {paths.base_dir}")
    print(f"Outputs        : {paths.outputs_dir}")
    print(f"RNG seed       : {cfg.rng_seed}")
    print("=" * 70)

    np.random.seed(cfg.rng_seed)
    paths.ensure_dirs()

    summary: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1) Gauss sweep
    # -------------------------------------------------------------------------
    if cfg.run_flags.run_gauss_sweep:
        print("\n[1/4] Running Gauss constraint sweep (lambda_G)...")

        lam_values = lambda_G_array(cfg)
        results_gauss = run_gauss_sweep(lam_values)
        fig_gauss = plot_gauss_results(results_gauss)

        png_path = paths.outputs_dir / "gauss_sweep_results.png"
        npz_path = paths.outputs_dir / "gauss_sweep_data.npz"

        fig_gauss.savefig(str(png_path), dpi=150, bbox_inches="tight")
        np.savez(str(npz_path), **results_gauss)

        print(f"  ✓ Gauss sweep plot saved to: {png_path}")
        print(f"  ✓ Gauss sweep data saved to: {npz_path}")

        summary["gauss_sweep"] = summarize_gauss(results_gauss)
    else:
        print("\n[1/4] Skipping Gauss sweep (disabled in RunFlags).")

    # -------------------------------------------------------------------------
    # 2) Exchange sweep
    # -------------------------------------------------------------------------
    if cfg.run_flags.run_exchange_sweep:
        print("\n[2/4] Running Exchange interaction sweep (J_exch)...")

        J_values = J_exch_array(cfg)
        results_ex = run_exchange_sweep(J_values)
        fig_ex = plot_exchange_results(results_ex)

        png_path = paths.outputs_dir / "exchange_sweep_results.png"
        npz_path = paths.outputs_dir / "exchange_sweep_data.npz"

        fig_ex.savefig(str(png_path), dpi=150, bbox_inches="tight")
        np.savez(str(npz_path), **results_ex)

        print(f"  ✓ Exchange sweep plot saved to: {png_path}")
        print(f"  ✓ Exchange sweep data saved to: {npz_path}")

        summary["exchange_sweep"] = summarize_exchange(results_ex)
    else:
        print("\n[2/4] Skipping Exchange sweep (disabled in RunFlags).")

    # -------------------------------------------------------------------------
    # 3) Sequential perturbation test
    # -------------------------------------------------------------------------
    if cfg.run_flags.run_sequential_perturbation:
        print("\n[3/4] Running Sequential Perturbation test...")

        s_cfg = cfg.sequential
        params_seq = build_params(lambda_G=s_cfg.lambda_G, J_exch=s_cfg.J_exch)

        results_seq = run_sequential_perturbation_test(
            params_seq,
            n_perturbations=s_cfg.n_perturbations,
            perturbation_strength=s_cfg.perturbation_strength,
        )
        fig_seq = plot_sequential_results(results_seq, params_seq)

        png_path = paths.outputs_dir / "sequential_perturbation_test.png"
        npz_path = paths.outputs_dir / "sequential_perturbation_data.npz"

        fig_seq.savefig(str(png_path), dpi=150, bbox_inches="tight")
        np.savez(str(npz_path), **results_seq)

        print(f"  ✓ Sequential test plot saved to: {png_path}")
        print(f"  ✓ Sequential test data saved to: {npz_path}")

        summary["sequential_perturbation_test"] = summarize_sequential(results_seq)
    else:
        print("\n[3/4] Skipping Sequential Perturbation test (disabled in RunFlags).")

    # -------------------------------------------------------------------------
    # 4) Information extraction test
    # -------------------------------------------------------------------------
    if cfg.run_flags.run_information_extraction:
        print("\n[4/4] Running Information Extraction test...")

        i_cfg = cfg.info_extraction
        params_info = build_params(lambda_G=i_cfg.lambda_G, J_exch=i_cfg.J_exch)

        results_info = run_information_extraction_test(
            params_info,
            n_perturbations=i_cfg.n_perturbations,
            strength=i_cfg.perturbation_strength,
        )
        fig_info = plot_info_results(results_info, params_info)

        png_path = paths.outputs_dir / "information_extraction_test.png"
        npz_path = paths.outputs_dir / "information_extraction_data.npz"

        fig_info.savefig(str(png_path), dpi=150, bbox_inches="tight")
        np.savez(str(npz_path), **results_info)

        print(f"  ✓ Information extraction plot saved to: {png_path}")
        print(f"  ✓ Information extraction data saved to: {npz_path}")

        summary["information_extraction_test"] = summarize_info(results_info)
    else:
        print("\n[4/4] Skipping Information Extraction test (disabled in RunFlags).")

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------
    summary_path = paths.outputs_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Summary written to: {summary_path}")
    print("Summary contents (headline metrics):")
    print(json.dumps(summary, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
