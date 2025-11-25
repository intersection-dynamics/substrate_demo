"""
analysis/aggregate_substrate_runs.py

Aggregate results from multiple substrate_pointer_exp runs into a single CSV file.

For each run directory under:
    <input_root>/<run_id>/

it expects at least:
    summary.json

and (optionally, but usually present):
    metadata.json
    params.json

It flattens the nested JSON structures into a single-row dict per run
and writes them all to a CSV.

Usage (from repo root):

    python analysis\\aggregate_substrate_runs.py

With options:

    python analysis\\aggregate_substrate_runs.py ^
        --input-root outputs\\substrate_pointer_exp ^
        --output-csv analysis\\substrate_pointer_aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate substrate_pointer_exp results into a single CSV."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="outputs/substrate_pointer_exp",
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="analysis/substrate_pointer_aggregate.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include diagnostics.* fields from summary.json.",
    )
    parser.add_argument(
        "--include-verdicts",
        action="store_true",
        help="Include verdicts.* fields from summary.json.",
    )
    return parser.parse_args()


def flatten_dict(
    d: Dict[str, Any],
    prefix: str = "",
    sep: str = ".",
    max_list_len: int = 8,
) -> Dict[str, Any]:
    """
    Flatten a nested dict into a single-level dict with dotted keys.

    Lists are serialized to JSON strings if they exceed max_list_len
    or contain non-scalar items.
    """
    flat: Dict[str, Any] = {}

    def _is_scalar(x: Any) -> bool:
        return isinstance(x, (str, int, float, bool)) or x is None

    def _flatten(obj: Any, current_prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{current_prefix}{sep}{k}" if current_prefix else str(k)
                _flatten(v, new_prefix)
        elif isinstance(obj, list):
            # If small list of scalars, join into a simple string
            if len(obj) <= max_list_len and all(_is_scalar(x) for x in obj):
                key = current_prefix
                flat[key] = ",".join("" if x is None else str(x) for x in obj)
            else:
                # Otherwise store as JSON string
                key = current_prefix
                flat[key] = json.dumps(obj)
        else:
            key = current_prefix
            flat[key] = obj

    _flatten(d, prefix)
    return flat


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def collect_run_rows(
    input_root: Path,
    include_diagnostics: bool,
    include_verdicts: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not input_root.exists():
        print(f"[WARN] Input root does not exist: {input_root}")
        return rows

    for run_dir in sorted(input_root.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            # Might be a partial or non-experiment directory
            continue

        metadata_path = run_dir / "metadata.json"
        params_path = run_dir / "params.json"

        summary = load_json_if_exists(summary_path)
        metadata = load_json_if_exists(metadata_path)
        params = load_json_if_exists(params_path)

        # Build a combined dict for flattening
        combined: Dict[str, Any] = {}

        # Basic identifiers
        run_id = summary.get("run_id") or metadata.get("run_id") or run_dir.name
        combined["run_id"] = run_id
        combined["run_dir"] = str(run_dir)

        # Timestamp
        combined["summary.timestamp"] = summary.get("timestamp")
        combined["metadata.timestamp"] = metadata.get("timestamp")

        # Top-level in summary
        combined["framework_version"] = summary.get("framework_version")
        combined["script"] = summary.get("script")

        # Params: prefer summary.params, then params.json, then metadata.engine_params
        params_src = summary.get("params") or params or metadata.get("engine_params") or {}
        combined["params_source"] = (
            "summary.params"
            if "params" in summary
            else "params.json"
            if params
            else "metadata.engine_params"
            if "engine_params" in metadata
            else "none"
        )

        # Metrics from summary
        metrics = summary.get("metrics", {})

        # Diagnostics and verdicts (optional)
        diagnostics = summary.get("diagnostics", {}) if include_diagnostics else {}
        verdicts = summary.get("verdicts", {}) if include_verdicts else {}

        # Flatten everything under namespaces
        combined_flat = {}
        combined_flat.update(flatten_dict(params_src, prefix="params"))
        combined_flat.update(flatten_dict(metrics, prefix="metrics"))
        if include_diagnostics:
            combined_flat.update(flatten_dict(diagnostics, prefix="diagnostics"))
        if include_verdicts:
            combined_flat.update(flatten_dict(verdicts, prefix="verdicts"))

        # Add the simple fields last so they don't get overwritten
        combined_flat["run_id"] = run_id
        combined_flat["run_dir"] = str(run_dir)
        combined_flat["framework_version"] = summary.get("framework_version")
        combined_flat["script"] = summary.get("script")
        combined_flat["summary.timestamp"] = summary.get("timestamp")
        combined_flat["metadata.timestamp"] = metadata.get("timestamp")
        combined_flat["params_source"] = combined["params_source"]

        rows.append(combined_flat)

    return rows


def write_csv(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("[INFO] No runs found to aggregate; CSV not written.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Collect all keys across rows
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Wrote {len(rows)} rows to {output_csv}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_csv = Path(args.output_csv).resolve()

    print(f"[INFO] Aggregating runs from: {input_root}")
    rows = collect_run_rows(
        input_root=input_root,
        include_diagnostics=args.include_diagnostics,
        include_verdicts=args.include_verdicts,
    )
    print(f"[INFO] Found {len(rows)} completed runs.")
    write_csv(output_csv, rows)


if __name__ == "__main__":
    main()
