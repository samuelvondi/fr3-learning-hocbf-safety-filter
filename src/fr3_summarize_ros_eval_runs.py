#!/usr/bin/env python3
"""
Summarize final FR3 ROS benchmark runs stored under:

  <run-dir>/raw/<mode>/<scenario>/run_data.csv
  <run-dir>/raw/<mode>/<scenario>/run_summary.json

Outputs are written only under:

  <run-dir>/postprocess/

By default this script writes a scenario-level summary CSV/JSON. Use
--write-steps only when a combined step-level CSV is explicitly needed; it can
be very large for all-1000, four-mode sweeps.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NUM_ARM_JOINTS = 7


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def series_min(df: pd.DataFrame, col: str) -> float:
    x = finite_series(df, col)
    return float(x.min()) if len(x) else float("nan")


def series_max(df: pd.DataFrame, col: str) -> float:
    x = finite_series(df, col)
    return float(x.max()) if len(x) else float("nan")


def series_mean(df: pd.DataFrame, col: str) -> float:
    x = finite_series(df, col)
    return float(x.mean()) if len(x) else float("nan")


def series_p95(df: pd.DataFrame, col: str) -> float:
    x = finite_series(df, col)
    return float(x.quantile(0.95)) if len(x) else float("nan")


def series_first(df: pd.DataFrame, col: str) -> float:
    x = finite_series(df, col)
    return float(x.iloc[0]) if len(x) else float("nan")


def series_last(df: pd.DataFrame, col: str) -> float:
    x = finite_series(df, col)
    return float(x.iloc[-1]) if len(x) else float("nan")


def parse_vec(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
    elif isinstance(value, str) and value.strip():
        try:
            arr = np.asarray(ast.literal_eval(value), dtype=float)
        except Exception:
            arr = np.full(NUM_ARM_JOINTS, np.nan, dtype=float)
    else:
        arr = np.full(NUM_ARM_JOINTS, np.nan, dtype=float)
    try:
        return arr.reshape(NUM_ARM_JOINTS)
    except Exception:
        return np.full(NUM_ARM_JOINTS, np.nan, dtype=float)


def path_metrics(df: pd.DataFrame, dt: float) -> dict[str, float]:
    q_rows = [parse_vec(v) for v in df.get("joint_q", pd.Series([], dtype=object)).tolist()]
    q_rows = [q for q in q_rows if np.all(np.isfinite(q))]
    if len(q_rows) >= 2:
        path_length_q = float(sum(np.linalg.norm(q_rows[i] - q_rows[i - 1]) for i in range(1, len(q_rows))))
        direct_dist = float(np.linalg.norm(q_rows[-1] - q_rows[0]))
        path_ineff = float(path_length_q / max(direct_dist, 1e-12))
    else:
        path_length_q = 0.0
        path_ineff = float("nan")

    ddq_safe = [parse_vec(v) for v in df.get("joint_ddq", pd.Series([], dtype=object)).tolist()]
    jerk_vals: list[float] = []
    for prev, curr in zip(ddq_safe[:-1], ddq_safe[1:]):
        if np.all(np.isfinite(prev)) and np.all(np.isfinite(curr)):
            jerk_vals.append(float(np.mean(((curr - prev) / float(dt)) ** 2)))
    mean_sq_jerk = float(np.mean(jerk_vals)) if jerk_vals else 0.0

    ddq_nom = [parse_vec(v) for v in df.get("joint_ddq_nominal", pd.Series([], dtype=object)).tolist()]
    qpf = [boolish(v) for v in df.get("qp_infeasible", pd.Series([False] * len(df))).tolist()]
    interventions: list[float] = []
    for failed, safe, nom in zip(qpf, ddq_safe, ddq_nom):
        if failed:
            continue
        if np.all(np.isfinite(safe)) and np.all(np.isfinite(nom)):
            interventions.append(float(np.linalg.norm(safe - nom)))
    mean_intervention = float(np.mean(interventions)) if interventions else 0.0

    return {
        "path_length_q": path_length_q,
        "path_inefficiency": path_ineff,
        "mean_squared_jerk": mean_sq_jerk,
        "mean_intervention_l2": mean_intervention,
    }


def enrich_summary(summary: dict[str, Any], df: pd.DataFrame, mode: str, scenario: str, dt: float) -> dict[str, Any]:
    status = str(summary.get("status", summary.get("final_run_status", "UNKNOWN")))
    success = bool(summary.get("success", status == "SUCCESS"))
    steps = int(summary.get("steps", len(df)))
    qp_fail_count = int(sum(boolish(v) for v in df.get("qp_infeasible", pd.Series([False] * len(df))).tolist()))
    fallback_count = int(sum(boolish(v) for v in df.get("fallback_used", pd.Series([False] * len(df))).tolist()))

    initial_dist_to_goal = finite_or_nan(summary.get("initial_dist_to_goal", series_first(df, "distance_to_goal")))
    final_dist_to_goal = finite_or_nan(summary.get("final_dist_to_goal", series_last(df, "distance_to_goal")))
    summary_progress = finite_or_nan(summary.get("progress_score", float("nan")))
    if np.isfinite(summary_progress):
        progress_score = float(summary_progress)
    elif np.isfinite(initial_dist_to_goal) and abs(initial_dist_to_goal) > 1e-12 and np.isfinite(final_dist_to_goal):
        progress_score = float(1.0 - final_dist_to_goal / initial_dist_to_goal)
    else:
        progress_score = float("nan")

    out = dict(summary)
    out.update({
        "scenario": scenario,
        "mode": mode,
        "status": status,
        "final_run_status": status,
        "success": success,
        "steps": steps,
        "last_logged_time_s": finite_or_nan(summary.get("last_logged_time_s", series_last(df, "time"))),
        "task_completion_time_s": finite_or_nan(summary.get("task_completion_time_s", series_last(df, "time") if success else float("nan"))),
        "min_h": finite_or_nan(summary.get("min_h", series_min(df, "min_h"))),
        "min_dist": finite_or_nan(summary.get("min_dist", series_min(df, "min_dist"))),
        "min_psi": finite_or_nan(summary.get("min_psi", series_min(df, "min_psi"))),
        "min_g1_selected": finite_or_nan(summary.get("min_g1_selected", series_min(df, "min_g1_selected"))),
        "min_g2_selected": finite_or_nan(summary.get("min_g2_selected", series_min(df, "min_g2_selected"))),
        "qp_fail_count": qp_fail_count,
        "qp_fail_rate": float(qp_fail_count / max(1, steps)),
        "fallback_count": fallback_count,
        "fallback_rate": float(fallback_count / max(1, steps)),
        "p1_mean": finite_or_nan(summary.get("p1_mean", series_mean(df, "p1"))),
        "p1_max": finite_or_nan(summary.get("p1_max", series_max(df, "p1"))),
        "p2_mean": finite_or_nan(summary.get("p2_mean", series_mean(df, "p2"))),
        "p2_max": finite_or_nan(summary.get("p2_max", series_max(df, "p2"))),
        "initial_dist_to_goal": initial_dist_to_goal,
        "final_dist_to_goal": final_dist_to_goal,
        "progress_score": progress_score,
        "solve_time_mean_s": finite_or_nan(summary.get("solve_time_mean_s", series_mean(df, "solve_time"))),
        "solve_time_p95_s": finite_or_nan(summary.get("solve_time_p95_s", series_p95(df, "solve_time"))),
        "solve_time_max_s": finite_or_nan(summary.get("solve_time_max_s", series_max(df, "solve_time"))),
        "final_qp_solve_time_mean_s": finite_or_nan(summary.get("final_qp_solve_time_mean_s", series_mean(df, "final_qp_solve_time_s"))),
        "final_qp_solve_time_p95_s": finite_or_nan(summary.get("final_qp_solve_time_p95_s", series_p95(df, "final_qp_solve_time_s"))),
        "final_qp_solve_time_max_s": finite_or_nan(summary.get("final_qp_solve_time_max_s", series_max(df, "final_qp_solve_time_s"))),
        "nn_inference_time_mean_s": finite_or_nan(summary.get("nn_inference_time_mean_s", series_mean(df, "nn_inference_time_s"))),
        "nn_inference_time_p95_s": finite_or_nan(summary.get("nn_inference_time_p95_s", series_p95(df, "nn_inference_time_s"))),
        "nn_inference_time_max_s": finite_or_nan(summary.get("nn_inference_time_max_s", series_max(df, "nn_inference_time_s"))),
    })
    out.update(path_metrics(df, dt=dt))
    return out


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def standardize_steps(df: pd.DataFrame, mode: str, scenario: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "mode", mode)
    out.insert(0, "scenario", scenario)
    out.insert(0, "step", np.arange(len(out), dtype=int))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to <results-root>/<run-name>.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--modes", default="", help="Optional comma-separated filter.")
    parser.add_argument("--prefix", default="", help="Output prefix; defaults to the run folder name.")
    parser.add_argument("--write-steps", action="store_true", help="Write a combined step-level CSV. This may be very large.")
    args = parser.parse_args()

    run_dir = expand(args.run_dir)
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    post_dir = run_dir / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip() or run_dir.name
    mode_filter = {m.strip() for m in args.modes.split(",") if m.strip()} if args.modes.strip() else None

    summary_rows: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, str]] = []
    runtime_samples: dict[str, dict[str, list[float]]] = {}

    run_data_paths = sorted(raw_dir.glob("*/*/run_data.csv"))
    if not run_data_paths:
        raise RuntimeError(f"No run_data.csv files found under {raw_dir}")

    for run_data_path in run_data_paths:
        mode = run_data_path.parent.parent.name
        scenario = run_data_path.parent.name
        if mode_filter is not None and mode not in mode_filter:
            continue
        df = pd.read_csv(run_data_path)
        summary_json = run_data_path.parent / "run_summary.json"
        if summary_json.exists():
            with open(summary_json, "r") as f:
                summary = json.load(f)
        else:
            summary = {}
        summary_rows.append(enrich_summary(summary, df, mode=mode, scenario=scenario, dt=args.dt))

        mode_runtime = runtime_samples.setdefault(
            mode,
            {"filter_step": [], "final_qp": [], "nn_inference": []},
        )
        for runtime_key, col in [
            ("filter_step", "solve_time"),
            ("final_qp", "final_qp_solve_time_s"),
            ("nn_inference", "nn_inference_time_s"),
        ]:
            vals = finite_series(df, col).to_numpy(dtype=float)
            mode_runtime[runtime_key].extend(float(v) for v in vals if np.isfinite(v))

        if args.write_steps:
            step_frames.append(standardize_steps(df, mode=mode, scenario=scenario))
        source_rows.append({
            "mode": mode,
            "scenario": scenario,
            "run_data": str(run_data_path),
            "run_summary": str(summary_json),
        })

    if not summary_rows:
        raise RuntimeError("No summary rows remained after filtering.")

    summary_rows = sorted(summary_rows, key=lambda r: (str(r.get("mode", "")), str(r.get("scenario", ""))))
    summary_csv = post_dir / f"{prefix}_summary.csv"
    summary_json_out = post_dir / f"{prefix}_summary.json"
    sources_csv = post_dir / f"{prefix}_sources.csv"
    csv_write(summary_csv, summary_rows)
    summary_json_out.write_text(json.dumps({"summaries": summary_rows}, indent=2, allow_nan=True))
    csv_write(sources_csv, source_rows)

    runtime_rows: list[dict[str, Any]] = []
    for mode in sorted(runtime_samples):
        row: dict[str, Any] = {"mode": mode}
        for runtime_key in ["filter_step", "final_qp", "nn_inference"]:
            arr = np.asarray(runtime_samples[mode].get(runtime_key, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{runtime_key}_mean_s"] = float(np.nanmean(arr)) if arr.size else float("nan")
            row[f"{runtime_key}_p95_s"] = float(np.nanpercentile(arr, 95)) if arr.size else float("nan")
            row[f"{runtime_key}_max_s"] = float(np.nanmax(arr)) if arr.size else float("nan")
            row[f"{runtime_key}_n_samples"] = int(arr.size)
        runtime_rows.append(row)

    runtime_csv = post_dir / f"{prefix}_runtime_aggregates.csv"
    runtime_json = post_dir / f"{prefix}_runtime_aggregates.json"
    csv_write(runtime_csv, runtime_rows)
    runtime_json.write_text(json.dumps({"runtime_aggregates": runtime_rows}, indent=2, allow_nan=True))

    print(f"Wrote summary CSV:  {summary_csv}")
    print(f"Wrote summary JSON: {summary_json_out}")
    print(f"Wrote sources CSV:  {sources_csv}")
    print(f"Wrote runtime CSV:  {runtime_csv}")
    print(f"Wrote runtime JSON: {runtime_json}")

    if args.write_steps:
        steps_df = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
        steps_csv = post_dir / f"{prefix}_steps.csv"
        steps_df.to_csv(steps_csv, index=False)
        print(f"Wrote steps CSV:    {steps_csv}")


if __name__ == "__main__":
    main()
