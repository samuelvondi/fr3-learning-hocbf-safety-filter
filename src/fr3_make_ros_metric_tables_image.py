#!/usr/bin/env python3
"""
Create paper-style metric tables for the final FR3 ROS/Gazebo evaluation.

Input:
  <run-dir>/postprocess/<prefix>_summary.csv

Output folder:
  <run-dir>/postprocess/tables/

Outputs:
  <table-prefix>_metric_tables.png
  <table-prefix>_metric_tables.pdf
  <table-prefix>_metric_tables_values.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAFETY_TOL = 1e-4
METHOD_ORDER = ["pd_only", "davide_online", "nn", "nn_g12_fallback"]
METHOD_LABELS = {
    "pd_only": "PD only",
    "davide_online": "Davide online",
    "nn": "NN",
    "nn_g12_fallback": "NN + fallback",
}


def expand(path: str | Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def count_pct(count: int, total: int) -> str:
    if total <= 0:
        return "--"
    return f"{count}/{total} ({100.0 * count / total:.1f}%)"


def med_q1_q3(series: pd.Series, decimals: int = 3) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return "--"
    med = float(np.nanmedian(x))
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def med_q1_q3_sci(series: pd.Series, decimals: int = 2) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return "--"
    med = float(np.nanmedian(x))
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    return f"{med:.{decimals}e} [{q1:.{decimals}e}, {q3:.{decimals}e}]"


def worst_sci(series: pd.Series, decimals: int = 2) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return "--"
    return f"{float(np.nanmin(x)):.{decimals}e}"


def mean_value(series: pd.Series, decimals: int = 3) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return "--"
    return f"{float(np.nanmean(x)):.{decimals}f}"


def ms_mean_value(series_s: pd.Series, decimals: int = 3) -> str:
    x = pd.to_numeric(series_s, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return "--"
    return f"{1000.0 * float(np.nanmean(x)):.{decimals}f}"


def runtime_triplet_ms(row: dict[str, Any] | None, prefix: str, decimals: int = 3) -> str:
    if row is None:
        return "--"
    vals = []
    for suffix in ["mean_s", "p95_s", "max_s"]:
        try:
            x = float(row.get(f"{prefix}_{suffix}", np.nan))
        except Exception:
            x = np.nan
        vals.append(x)
    if not all(np.isfinite(v) for v in vals):
        return "--"
    return " / ".join(f"{1000.0 * v:.{decimals}f}" for v in vals)


def runtime_value_ms(row: dict[str, Any] | None, prefix: str, suffix: str) -> float:
    if row is None:
        return float("nan")
    try:
        x = float(row.get(f"{prefix}_{suffix}_s", np.nan))
    except Exception:
        x = np.nan
    return 1000.0 * x if np.isfinite(x) else float("nan")


def draw_table(ax, title: str, columns: list[str], rows: list[list[str]], font_size: float, col_widths: list[float]):
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=8)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.88],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.30)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left", fontweight="bold")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument(
        "--runtime-aggregates-csv",
        default="",
        help="Optional runtime aggregate CSV from fr3_summarize_ros_eval_runs.py. "
             "Defaults to <prefix>_runtime_aggregates.csv next to the summary CSV when present.",
    )
    parser.add_argument("--out-dir", default="", help="Defaults to <summary CSV folder>/tables")
    parser.add_argument("--table-prefix", default="fr3_ros_final")
    args = parser.parse_args()

    summary_csv = expand(args.summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    out_dir = expand(args.out_dir) if args.out_dir.strip() else summary_csv.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    if args.runtime_aggregates_csv.strip():
        runtime_csv = expand(args.runtime_aggregates_csv)
    else:
        inferred = summary_csv.with_name(summary_csv.name.replace("_summary.csv", "_runtime_aggregates.csv"))
        runtime_csv = inferred if inferred.exists() else None

    runtime_by_mode: dict[str, dict[str, Any]] = {}
    if runtime_csv is not None and Path(runtime_csv).exists():
        runtime_df = pd.read_csv(runtime_csv)
        for _, row in runtime_df.iterrows():
            runtime_by_mode[str(row.get("mode", ""))] = row.to_dict()

    initial_goal = numeric(df, "initial_dist_to_goal").replace(0, np.nan)
    final_goal = numeric(df, "final_dist_to_goal")
    computed_progress = 1.0 - final_goal / initial_goal
    if "progress_score" in df.columns:
        reported_progress = numeric(df, "progress_score")
        df["progress_score"] = reported_progress.where(reported_progress.notna(), computed_progress)
    else:
        df["progress_score"] = computed_progress

    df["success_bool"] = as_bool(df["success"]) if "success" in df.columns else False
    df["h_violation"] = numeric(df, "min_h") < -SAFETY_TOL
    df["g1_violation"] = numeric(df, "min_g1_selected") < -SAFETY_TOL
    df["g2_violation"] = numeric(df, "min_g2_selected") < -SAFETY_TOL
    df["safe_success"] = df["success_bool"] & (~df["h_violation"])
    df["qp_failed_run"] = numeric(df, "qp_fail_count").fillna(0.0) > 0.0
    df["fallback_used_run"] = numeric(df, "fallback_count").fillna(0.0) > 0.0

    task_rows: list[list[str]] = []
    safety_h_rows: list[list[str]] = []
    safety_g_rows: list[list[str]] = []
    efficiency_rows: list[list[str]] = []
    robustness_rows: list[list[str]] = []
    runtime_rows: list[list[str]] = []
    flat_records: list[dict[str, Any]] = []

    for mode in METHOD_ORDER:
        sdf = df[df["mode"] == mode].copy()
        if sdf.empty:
            continue
        n = int(len(sdf))
        label = METHOD_LABELS[mode]
        successes = int(sdf["success_bool"].sum())
        safe_successes = int(sdf["safe_success"].sum())
        h_viol = int(sdf["h_violation"].sum())
        g1_viol = int(sdf["g1_violation"].sum())
        g2_viol = int(sdf["g2_violation"].sum())
        qp_fail_runs = int(sdf["qp_failed_run"].sum())
        total_qp_fail_steps = int(np.nansum(numeric(sdf, "qp_fail_count").to_numpy(dtype=float)))
        fallback_runs = int(sdf["fallback_used_run"].sum())
        total_fallback_steps = float(np.nansum(numeric(sdf, "fallback_count").to_numpy(dtype=float)))
        total_steps = float(np.nansum(numeric(sdf, "steps").to_numpy(dtype=float)))
        fallback_activation = total_fallback_steps / total_steps if total_steps > 0 else np.nan

        task_rows.append([
            label,
            count_pct(successes, n),
            count_pct(safe_successes, n),
            med_q1_q3(sdf.loc[sdf["success_bool"], "task_completion_time_s"], decimals=2),
            med_q1_q3(numeric(sdf, "progress_score"), decimals=3),
            med_q1_q3(numeric(sdf, "final_dist_to_goal"), decimals=3),
        ])

        safety_h_rows.append([
            label,
            count_pct(h_viol, n),
            med_q1_q3_sci(numeric(sdf, "min_h"), decimals=2),
            worst_sci(numeric(sdf, "min_h"), decimals=2),
        ])

        safety_g_rows.append([
            label,
            count_pct(g1_viol, n) if mode != "pd_only" else "--",
            med_q1_q3_sci(numeric(sdf, "min_g1_selected"), decimals=2) if mode != "pd_only" else "--",
            worst_sci(numeric(sdf, "min_g1_selected"), decimals=2) if mode != "pd_only" else "--",
            count_pct(g2_viol, n) if mode != "pd_only" else "--",
            med_q1_q3_sci(numeric(sdf, "min_g2_selected"), decimals=2) if mode != "pd_only" else "--",
            worst_sci(numeric(sdf, "min_g2_selected"), decimals=2) if mode != "pd_only" else "--",
        ])

        efficiency_rows.append([
            label,
            med_q1_q3(numeric(sdf.loc[sdf["success_bool"]], "path_inefficiency"), decimals=3),
            med_q1_q3(numeric(sdf, "mean_squared_jerk"), decimals=1),
            med_q1_q3(numeric(sdf, "mean_intervention_l2"), decimals=3),
        ])

        robustness_rows.append([
            label,
            count_pct(qp_fail_runs, n) if mode != "pd_only" else "--",
            str(total_qp_fail_steps) if mode != "pd_only" else "--",
            count_pct(fallback_runs, n) if mode == "nn_g12_fallback" else "--",
            f"{100.0 * fallback_activation:.1f}%" if mode == "nn_g12_fallback" and np.isfinite(fallback_activation) else "--",
        ])

        runtime_agg_row = runtime_by_mode.get(mode)
        runtime_rows.append([
            label,
            runtime_triplet_ms(runtime_agg_row, "filter_step"),
            runtime_triplet_ms(runtime_agg_row, "final_qp") if mode != "pd_only" else "--",
            runtime_triplet_ms(runtime_agg_row, "nn_inference") if mode in {"nn", "nn_g12_fallback"} else "--",
            mean_value(numeric(sdf, "p1_mean"), decimals=3) if mode != "pd_only" else "--",
            mean_value(numeric(sdf, "p2_mean"), decimals=3) if mode != "pd_only" else "--",
        ])

        flat_records.append({
            "mode": mode,
            "method": label,
            "n_scenarios": n,
            "success_count": successes,
            "success_rate_pct": 100.0 * successes / n,
            "safe_success_count": safe_successes,
            "safe_success_rate_pct": 100.0 * safe_successes / n,
            "h_violation_count": h_viol,
            "g1_violation_count": g1_viol if mode != "pd_only" else np.nan,
            "g2_violation_count": g2_viol if mode != "pd_only" else np.nan,
            "qp_failed_run_count": qp_fail_runs if mode != "pd_only" else np.nan,
            "total_qp_failed_steps": total_qp_fail_steps if mode != "pd_only" else np.nan,
            "fallback_used_run_count": fallback_runs if mode == "nn_g12_fallback" else np.nan,
            "fallback_activation_rate_pct": 100.0 * fallback_activation if mode == "nn_g12_fallback" and np.isfinite(fallback_activation) else np.nan,
            "completion_time_success_median_s": float(pd.to_numeric(sdf.loc[sdf["success_bool"], "task_completion_time_s"], errors="coerce").median()),
            "progress_score_median": float(pd.to_numeric(sdf["progress_score"], errors="coerce").median()),
            "final_goal_distance_median_m": float(pd.to_numeric(sdf["final_dist_to_goal"], errors="coerce").median()),
            "min_h_median": float(pd.to_numeric(sdf["min_h"], errors="coerce").median()),
            "min_h_worst": float(pd.to_numeric(sdf["min_h"], errors="coerce").min()),
            "min_g1_median": float(pd.to_numeric(sdf["min_g1_selected"], errors="coerce").median()) if mode != "pd_only" else np.nan,
            "min_g1_worst": float(pd.to_numeric(sdf["min_g1_selected"], errors="coerce").min()) if mode != "pd_only" else np.nan,
            "min_g2_median": float(pd.to_numeric(sdf["min_g2_selected"], errors="coerce").median()) if mode != "pd_only" else np.nan,
            "min_g2_worst": float(pd.to_numeric(sdf["min_g2_selected"], errors="coerce").min()) if mode != "pd_only" else np.nan,
            "path_inefficiency_success_median": float(pd.to_numeric(sdf.loc[sdf["success_bool"], "path_inefficiency"], errors="coerce").median()),
            "mean_squared_jerk_median": float(pd.to_numeric(sdf["mean_squared_jerk"], errors="coerce").median()),
            "intervention_median": float(pd.to_numeric(sdf["mean_intervention_l2"], errors="coerce").median()),
            "filter_step_time_mean_ms": runtime_value_ms(runtime_agg_row, "filter_step", "mean"),
            "filter_step_time_p95_ms": runtime_value_ms(runtime_agg_row, "filter_step", "p95"),
            "filter_step_time_max_ms": runtime_value_ms(runtime_agg_row, "filter_step", "max"),
            "final_qp_time_mean_ms": runtime_value_ms(runtime_agg_row, "final_qp", "mean") if mode != "pd_only" else np.nan,
            "final_qp_time_p95_ms": runtime_value_ms(runtime_agg_row, "final_qp", "p95") if mode != "pd_only" else np.nan,
            "final_qp_time_max_ms": runtime_value_ms(runtime_agg_row, "final_qp", "max") if mode != "pd_only" else np.nan,
            "nn_inference_time_mean_ms": runtime_value_ms(runtime_agg_row, "nn_inference", "mean") if mode in {"nn", "nn_g12_fallback"} else np.nan,
            "nn_inference_time_p95_ms": runtime_value_ms(runtime_agg_row, "nn_inference", "p95") if mode in {"nn", "nn_g12_fallback"} else np.nan,
            "nn_inference_time_max_ms": runtime_value_ms(runtime_agg_row, "nn_inference", "max") if mode in {"nn", "nn_g12_fallback"} else np.nan,
            "mean_p1_across_scenarios": float(pd.to_numeric(sdf["p1_mean"], errors="coerce").mean()) if mode != "pd_only" else np.nan,
            "mean_p2_across_scenarios": float(pd.to_numeric(sdf["p2_mean"], errors="coerce").mean()) if mode != "pd_only" else np.nan,
        })

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(40, 25))
    gs = fig.add_gridspec(
        nrows=7,
        ncols=1,
        height_ratios=[0.72, 1.14, 0.92, 1.24, 1.10, 1.10, 1.18],
        hspace=0.26,
    )

    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")
    ax_title.text(0.0, 0.98, "Final FR3 ROS/Gazebo benchmark — paper metric tables", fontsize=27, fontweight="bold", va="top")
    ax_title.text(0.0, 0.60, "Safe success = task success with min h ≥ −1e−4. Safety violation counts use min h/G1/G2 < −1e−4.", fontsize=13.5, va="top")
    ax_title.text(0.0, 0.22, "Continuous metrics are median [Q1, Q3]. Runtime values are mean / p95 / max in milliseconds.", fontsize=12.8, va="top")

    draw_table(fig.add_subplot(gs[1, 0]), "1. Task", ["Method", "Success rate", "Safe success rate", "Completion time [s]", "Progress score", "Final goal distance [m]"], task_rows, 10.2, [0.13, 0.16, 0.16, 0.19, 0.17, 0.19])
    draw_table(fig.add_subplot(gs[2, 0]), "2a. Safety — h", ["Method", "h violation count\nmin h < -1e-4", "min h\nmedian [Q1, Q3]", "min h\nworst"], safety_h_rows, 10.2, [0.18, 0.24, 0.38, 0.20])
    draw_table(fig.add_subplot(gs[3, 0]), "2b. Safety — selected G1 and G2", ["Method", "G1 violation count", "min G1\nmedian [Q1, Q3]", "min G1\nworst", "G2 violation count", "min G2\nmedian [Q1, Q3]", "min G2\nworst"], safety_g_rows, 9.4, [0.13, 0.16, 0.19, 0.10, 0.16, 0.19, 0.10])
    draw_table(fig.add_subplot(gs[4, 0]), "3. Efficiency / smoothness", ["Method", "Path inefficiency ratio\n(successes)", "Mean squared jerk", "Intervention magnitude"], efficiency_rows, 10.2, [0.18, 0.30, 0.26, 0.26])
    draw_table(fig.add_subplot(gs[5, 0]), "4. Robustness", ["Method", "Runs with any\nQP infeasibility", "Total QP-failed steps", "Fallback-used runs", "Fallback activation rate"], robustness_rows, 10.2, [0.18, 0.24, 0.20, 0.20, 0.18])
    draw_table(fig.add_subplot(gs[6, 0]), "5. Runtime / p12", ["Method", "Filter runtime [ms]\nmean / p95 / max", "Final QP solve [ms]\nmean / p95 / max", "NN inference [ms]\nmean / p95 / max", "Mean p1", "Mean p2"], runtime_rows, 9.8, [0.14, 0.25, 0.25, 0.20, 0.08, 0.08])

    out_png = out_dir / f"{args.table_prefix}_metric_tables.png"
    out_pdf = out_dir / f"{args.table_prefix}_metric_tables.pdf"
    out_csv = out_dir / f"{args.table_prefix}_metric_tables_values.csv"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    pd.DataFrame(flat_records).to_csv(out_csv, index=False)
    print(f"Wrote image: {out_png}")
    print(f"Wrote PDF:   {out_pdf}")
    print(f"Wrote CSV:   {out_csv}")


if __name__ == "__main__":
    main()
