#!/usr/bin/env python3
"""Summarize lightweight FR3 ROS run_summary.json files into CSV/JSON tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


ORDER = ["pd_only", "davide_online", "nn", "nn_g12_fallback"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    postprocess = run_dir / "postprocess"
    postprocess.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted((run_dir / "raw").glob("*/*/run_summary.json")):
        try:
            records.append(json.loads(path.read_text()))
        except Exception as exc:
            print(f"Could not parse {path}: {exc}")

    if not records:
        raise SystemExit(f"No run_summary.json files found under {run_dir / 'raw'}")

    df = pd.DataFrame(records)
    if "mode" in df.columns:
        df["mode"] = pd.Categorical(df["mode"], categories=ORDER, ordered=True)
    if "scenario" in df.columns:
        df = df.sort_values(["scenario", "mode"], kind="stable")

    csv_path = postprocess / f"{args.prefix}_summary.csv"
    json_path = postprocess / f"{args.prefix}_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2))

    agg_rows = []
    if "mode" in df.columns:
        for mode, g in df.groupby("mode", observed=True):
            success_bool = (
                g["success"].fillna(False).astype(bool)
                if "success" in g
                else pd.Series(False, index=g.index)
            )
            successes = g[success_bool]
            agg_rows.append({
                "mode": str(mode),
                "n": int(len(g)),
                "success_count": int(success_bool.sum()),
                "timeout_or_fail_count": int((~success_bool).sum()),
                "safe_h_count": int((pd.to_numeric(g.get("min_h"), errors="coerce") >= 0.0).sum()) if "min_h" in g else 0,
                "mean_success_time_s": float(pd.to_numeric(successes.get("task_completion_time_s"), errors="coerce").mean()) if len(successes) else None,
                "median_success_time_s": float(pd.to_numeric(successes.get("task_completion_time_s"), errors="coerce").median()) if len(successes) else None,
                "mean_min_h": float(pd.to_numeric(g.get("min_h"), errors="coerce").mean()) if "min_h" in g else None,
                "mean_min_psi": float(pd.to_numeric(g.get("min_psi"), errors="coerce").mean()) if "min_psi" in g else None,
                "mean_qp_fail_count": float(pd.to_numeric(g.get("qp_fail_count"), errors="coerce").mean()) if "qp_fail_count" in g else None,
                "mean_fallback_count": float(pd.to_numeric(g.get("fallback_count"), errors="coerce").mean()) if "fallback_count" in g else None,
            })

    agg = pd.DataFrame(agg_rows)
    agg_csv = postprocess / f"{args.prefix}_aggregate.csv"
    agg_json = postprocess / f"{args.prefix}_aggregate.json"
    agg.to_csv(agg_csv, index=False)
    agg_json.write_text(agg.to_json(orient="records", indent=2))

    print(f"Wrote per-run summary CSV: {csv_path}")
    print(f"Wrote per-run summary JSON: {json_path}")
    print(f"Wrote aggregate CSV: {agg_csv}")
    print(f"Wrote aggregate JSON: {agg_json}")
    print()
    if not agg.empty:
        print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
