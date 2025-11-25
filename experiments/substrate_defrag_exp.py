"""
experiments/substrate_defrag_exp.py

Defrag / clumping test using engines/substrate_engine.py.

This runs the unified Hilbert-substrate engine with parameters chosen to
encourage clumping in the fermion-like sector and then prints out a short
summary (energy drift, lump counts, etc.).

Usage (from repo root):

  python experiments\\substrate_defrag_exp.py

You can also tweak CLI flags; run with -h to see options.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import sys

# ---------------------------------------------------------------------
# Make sure repo root is on sys.path so "engines" imports correctly
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]  # .../substrate_demo
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines import substrate_engine as engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a substrate defrag / clumping experiment."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=24,
        help="Lattice size N (uses NxNxN). Smaller is faster.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Number of time steps.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for outputs (JSON + NPZ).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="defrag_test",
        help="Tag for this run (goes into run_id).",
    )
    return parser.parse_args()


def make_run_dir(output_root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{tag}" if tag else ts
    base = Path(output_root) / "substrate_defrag_exp" / run_id
    base.mkdir(parents=True, exist_ok=False)
    return base


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, args.tag)

    print("==============================================")
    print("Experiment: substrate_defrag_exp")
    print(f"Run dir:    {run_dir}")
    print("==============================================")
    print(f"Using engine module: {engine.__file__}")
    print("==============================================")

    # Engine parameters chosen to encourage clumping in fermion-like sector
    engine_params = {
        "grid_size": args.grid_size,
        "steps": args.steps,
        "dt": args.dt,
        "seed": args.seed,

        # One bosonic channel, kept simple for now
        "n_boson": 1,
        "boson_mass2": (0.0,),
        "boson_lambda4": (0.0,),
        "boson_init_amp": (0.02,),

        # Fermion-like sector: 1 color, 2 spin → 2 complex components
        "n_color": 1,
        "n_spin": 2,

        # Tachyonic-ish mass^2 to encourage structure, with quartic stabilizer
        "fermion_mass2": (-1.0,),
        "fermion_lambda4": (2.0,),
        "fermion_init_amp": (0.05,),

        # No boson–fermion cross-coupling yet
        "g_bf": 0.0,

        # Lump detection (on fermion density)
        "lump_sigma_threshold": 2.0,
        "lump_min_voxels": 4,
    }

    print("Engine parameters:")
    for k, v in engine_params.items():
        print(f"  {k}: {v}")

    print("----------------------------------------------")
    print("Running engine...")
    results = engine.run_experiment(engine_params)
    print("Engine run completed.")
    print("----------------------------------------------")

    # Save outputs
    # - params + metrics + diagnostics as JSON
    # - timeseries as NPZ
    params_path = run_dir / "params.json"
    summary_path = run_dir / "summary.json"
    timeseries_path = run_dir / "timeseries.npz"

    with params_path.open("w", encoding="utf-8") as f:
        json.dump(results["params"], f, indent=2, sort_keys=True)

    summary_obj = {
        "metrics": results["metrics"],
        "diagnostics": results["diagnostics"],
        "verdicts": results["verdicts"],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2, sort_keys=True)

    import numpy as np

    ts = results["timeseries"]
    np.savez_compressed(
        timeseries_path,
        step=np.array(ts["step"], dtype=np.int32),
        energy=np.array(ts["energy"], dtype=np.float64),
        norm=np.array(ts["norm"], dtype=np.float64),
        phi_rms=np.array(ts["phi_rms"], dtype=np.float64),
        phi_max=np.array(ts["phi_max"], dtype=np.float64),
        fermion_lump_count=np.array(ts["fermion_lump_count"], dtype=np.int32),
        fermion_com=np.array(ts["fermion_com"], dtype=np.float64),
    )

    # Print a short human-readable summary
    m = results["metrics"]
    d = results["diagnostics"]
    v = results["verdicts"]

    print("==== Run Summary ====")
    print(f"E_initial: {m['E_initial']:.6e}")
    print(f"E_final:   {m['E_final']:.6e}")
    print(f"E_min:     {m['E_min']:.6e}")
    print(f"E_max:     {m['E_max']:.6e}")
    print(f"N_initial: {m['N_initial']:.6e}")
    print(f"N_final:   {m['N_final']:.6e}")
    print(f"phi_rms_final: {m['phi_rms_final']:.6e}")
    print(f"phi_max_final: {m['phi_max_final']:.6e}")
    print(f"fermion_lump_count_final: {m['fermion_lump_count_final']}")
    print(f"fermion_com_final: {m['fermion_com_final']}")

    print("---- Diagnostics ----")
    print(f"abs_energy_drift: {d['abs_energy_drift']:.6e}")
    print(f"rel_energy_drift: {d['rel_energy_drift']:.3f}")
    print(f"abs_norm_drift:   {d['abs_norm_drift']:.6e}")
    print(f"rel_norm_drift:   {d['rel_norm_drift']:.3f}")
    print(f"stable:           {d['stable']}")
    print(f"notes:            {d['notes']}")

    print("---- Verdicts ----")
    print(f"has_fermion_lumps: {v['has_fermion_lumps']}")
    print(f"energy_reasonable: {v['energy_reasonable']}")
    print(f"norm_reasonable:   {v['norm_reasonable']}")
    print("====================")
    print(f"Saved params to:     {params_path}")
    print(f"Saved summary to:    {summary_path}")
    print(f"Saved timeseries to: {timeseries_path}")


if __name__ == "__main__":
    main()
