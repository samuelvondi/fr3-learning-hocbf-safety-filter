#!/usr/bin/env python3
from __future__ import annotations

"""
Generic FR3 final metric-table generator.

Use this single script for any offline comparison result produced by
fr3_compare_rollout_policies.py, as long as you have matching:

    <prefix>_summary.csv
    <prefix>_steps.csv

It can combine multiple result files into one table, for example:
  - one run containing Davide + ep22 NN + ep22 NN+FB
  - a second run containing ep34 NN + ep34 NN+FB

Main metrics:
  1. Task
     - success rate
     - safe success rate: success AND min_h >= -safety_tol
     - completion time on successful scenarios
     - progress score
     - final goal distance

  2. Safety
     - h / G1 / G2 violation counts using min < -safety_tol
     - median [Q1, Q3] and worst value

  3. Efficiency / smoothness
     - path inefficiency ratio on successes
     - mean squared jerk
     - intervention magnitude

  4. Robustness
     - runs with any QP infeasibility
     - total QP-failed steps
     - fallback-used runs
     - fallback activation rate

  5. Runtime / p12
     - QP solve time: mean / p95 / max in ms
     - total step time: mean / p95 / max in ms
     - mean p1 / p2 across scenarios

Progress-score logic:
  1. use summary column progress_score/progress if present
  2. else compute 1 - final_dist_to_goal / initial_dist_to_goal if present
  3. else compute 1 - goal_final / goal_init if present
  4. else compute from first/last dist_to_goal in steps.csv

Example: all-1000 ep22 + ep34 table

python fr3_make_final_metric_tables_generic.py \
  --title "Final offline all-1000 benchmark — complete paper metric tables" \
  --out-prefix fr3_final_all1000_v10ep22_ep34_metric_tables \
  --run fr3_all1000_v10ep22_vs_davide_terminal_qpfail_summary.csv \
        fr3_all1000_v10ep22_vs_davide_terminal_qpfail_steps.csv \
        "davide_online=Davide online" "nn=ep22 NN" "nn_g12_fallback=ep22 NN + fallback" \
  --run fr3_all1000_v10ep34_nn_and_fallback_terminal_qpfail_summary.csv \
        fr3_all1000_v10ep34_nn_and_fallback_terminal_qpfail_steps.csv \
        "nn=ep34 NN" "nn_g12_fallback=ep34 NN + fallback" \
  --order "Davide online" "ep22 NN" "ep22 NN + fallback" "ep34 NN" "ep34 NN + fallback" \
  --short "Davide online=Davide" "ep22 NN=ep22 NN" "ep22 NN + fallback=ep22 NN+FB" \
          "ep34 NN=ep34 NN" "ep34 NN + fallback=ep34 NN+FB"
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
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


def runtime_triplet_ms(series_s: pd.Series, decimals: int = 3) -> str:
    x = pd.to_numeric(series_s, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return "--"
    x_ms = 1000.0 * x
    mean = float(np.nanmean(x_ms))
    p95 = float(np.nanpercentile(x_ms, 95))
    vmax = float(np.nanmax(x_ms))
    return f"{mean:.{decimals}f} / {p95:.{decimals}f} / {vmax:.{decimals}f}"


def fmt_rate(rate: float, decimals: int = 1) -> str:
    if not np.isfinite(rate):
        return "--"
    return f"{100.0 * rate:.{decimals}f}%"


def parse_key_values(items: list[str] | None, arg_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"{arg_name} entries must have form key=value, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid {arg_name} entry: {item!r}")
        out[k] = v
    return out


def parse_runs(run_args: list[list[str]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for raw in run_args:
        if len(raw) < 3:
            raise ValueError(
                "Each --run needs at least: SUMMARY.csv STEPS.csv mode=MethodName"
            )
        summary = Path(raw[0])
        steps = Path(raw[1])
        method_names = parse_key_values(raw[2:], "--run mode map")
        if not summary.exists():
            raise FileNotFoundError(f"Missing summary file: {summary}")
        if not steps.exists():
            raise FileNotFoundError(f"Missing steps file: {steps}")
        runs.append({"summary": summary, "steps": steps, "method_names": method_names})
    return runs


def draw_table(
    ax,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    font_size: float,
    col_widths: list[float],
):
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


# -----------------------------------------------------------------------------
# Progress / final goal helpers
# -----------------------------------------------------------------------------

def compute_step_distance_features(steps_df: pd.DataFrame) -> pd.DataFrame:
    """Return first/last distance-to-goal per method/scenario if available."""
    if "dist_to_goal" not in steps_df.columns:
        return pd.DataFrame(columns=["method", "scenario", "step_initial_dist_to_goal", "step_final_dist_to_goal"])

    tmp = steps_df.copy()
    tmp["dist_to_goal_num"] = pd.to_numeric(tmp["dist_to_goal"], errors="coerce")
    tmp = tmp.dropna(subset=["dist_to_goal_num"])
    if tmp.empty:
        return pd.DataFrame(columns=["method", "scenario", "step_initial_dist_to_goal", "step_final_dist_to_goal"])

    tmp = tmp.sort_values(["method", "scenario", "step"])
    g = tmp.groupby(["method", "scenario"], observed=True)["dist_to_goal_num"]
    out = g.agg(step_initial_dist_to_goal="first", step_final_dist_to_goal="last").reset_index()
    return out


def add_progress_and_goal(summary_df: pd.DataFrame, steps_df: pd.DataFrame) -> pd.DataFrame:
    """Add progress_score_final and final_goal_distance_final robustly."""
    df = summary_df.copy()

    step_feat = compute_step_distance_features(steps_df)
    if not step_feat.empty:
        df = df.merge(step_feat, on=["method", "scenario"], how="left")
    else:
        df["step_initial_dist_to_goal"] = np.nan
        df["step_final_dist_to_goal"] = np.nan

    # Final goal distance, preferring summary column but falling back to steps.
    if "final_dist_to_goal" in df.columns:
        final_goal = safe_numeric(df, "final_dist_to_goal")
    elif "goal_final" in df.columns:
        final_goal = safe_numeric(df, "goal_final")
    else:
        final_goal = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    final_goal = final_goal.fillna(pd.to_numeric(df["step_final_dist_to_goal"], errors="coerce"))
    df["final_goal_distance_final"] = final_goal

    # Progress score priority.
    progress = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    for c in ["progress_score", "progress"]:
        if c in df.columns:
            progress = progress.fillna(pd.to_numeric(df[c], errors="coerce"))

    if "initial_dist_to_goal" in df.columns:
        init = safe_numeric(df, "initial_dist_to_goal").replace(0.0, np.nan)
        fin = safe_numeric(df, "final_dist_to_goal") if "final_dist_to_goal" in df.columns else final_goal
        progress = progress.fillna(1.0 - fin / init)

    if "goal_init" in df.columns:
        init = safe_numeric(df, "goal_init").replace(0.0, np.nan)
        fin = safe_numeric(df, "goal_final") if "goal_final" in df.columns else final_goal
        progress = progress.fillna(1.0 - fin / init)

    init_step = pd.to_numeric(df["step_initial_dist_to_goal"], errors="coerce").replace(0.0, np.nan)
    fin_step = pd.to_numeric(df["step_final_dist_to_goal"], errors="coerce")
    progress = progress.fillna(1.0 - fin_step / init_step)

    df["progress_score_final"] = progress
    return df


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_combined_data(runs: list[dict[str, Any]], method_order: list[str] | None):
    all_summary = []
    all_steps = []
    encountered_methods: list[str] = []

    for run in runs:
        summary = pd.read_csv(run["summary"])
        steps = pd.read_csv(run["steps"])

        method_names = run["method_names"]
        summary["method"] = summary["mode"].map(method_names)
        steps["method"] = steps["mode"].map(method_names)

        summary = summary[summary["method"].notna()].copy()
        steps = steps[steps["method"].notna()].copy()

        for name in method_names.values():
            if name not in encountered_methods:
                encountered_methods.append(name)

        all_summary.append(summary)
        all_steps.append(steps)

    summary_df = pd.concat(all_summary, ignore_index=True)
    steps_df = pd.concat(all_steps, ignore_index=True)

    order = method_order or encountered_methods
    summary_df["method"] = pd.Categorical(summary_df["method"], categories=order, ordered=True)
    steps_df["method"] = pd.Categorical(steps_df["method"], categories=order, ordered=True)

    summary_df = summary_df.sort_values(["method", "scenario"]).reset_index(drop=True)
    steps_df = steps_df.sort_values(["method", "scenario", "step"]).reset_index(drop=True)

    return summary_df, steps_df, order


# -----------------------------------------------------------------------------
# Main table generation
# -----------------------------------------------------------------------------

def build_tables(
    summary_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    method_order: list[str],
    short_methods: dict[str, str],
    safety_tol: float,
):
    summary_df = summary_df.copy()
    steps_df = steps_df.copy()

    summary_df["success_bool"] = as_bool(summary_df["success"]) if "success" in summary_df.columns else False
    steps_df["fallback_bool"] = as_bool(steps_df["fallback_used"]) if "fallback_used" in steps_df.columns else False
    steps_df["qp_solved_bool"] = as_bool(steps_df["qp_solved"]) if "qp_solved" in steps_df.columns else True

    # Scenario-level min G2 from the step-level metric.
    if "min_g2" in steps_df.columns:
        min_g2_df = (
            steps_df.groupby(["method", "scenario"], observed=True)["min_g2"]
            .min()
            .reset_index()
            .rename(columns={"min_g2": "min_g2_summary"})
        )
        summary_df = summary_df.merge(min_g2_df, on=["method", "scenario"], how="left")
    else:
        summary_df["min_g2_summary"] = np.nan

    # Fallback usage per scenario.
    fallback_group = (
        steps_df.groupby(["method", "scenario"], observed=True)
        .agg(
            fallback_used_any=("fallback_bool", "any"),
            fallback_steps=("fallback_bool", "sum"),
            total_logged_steps=("step", "count"),
        )
        .reset_index()
    )
    summary_df = summary_df.merge(fallback_group, on=["method", "scenario"], how="left")

    # Progress and goal distance.
    summary_df = add_progress_and_goal(summary_df, steps_df)

    # QP and safety flags.
    summary_df["qp_fail_count_num"] = safe_numeric(summary_df, "qp_fail_count").fillna(0.0)
    summary_df["qp_failed_run"] = summary_df["qp_fail_count_num"] > 0.0

    summary_df["h_violation_run"] = safe_numeric(summary_df, "min_h") < -float(safety_tol)
    summary_df["g1_violation_run"] = safe_numeric(summary_df, "min_psi") < -float(safety_tol)
    summary_df["g2_violation_run"] = safe_numeric(summary_df, "min_g2_summary") < -float(safety_tol)
    summary_df["safe_success_bool"] = summary_df["success_bool"] & (~summary_df["h_violation_run"])

    runtime_steps = steps_df[steps_df["qp_solved_bool"]].copy()

    task_rows: list[list[str]] = []
    safety_h_rows: list[list[str]] = []
    safety_g_rows: list[list[str]] = []
    eff_rows: list[list[str]] = []
    robust_rows: list[list[str]] = []
    runtime_p12_rows: list[list[str]] = []
    flat_records: list[dict[str, Any]] = []

    for method in method_order:
        sdf = summary_df[summary_df["method"].astype(str) == method].copy()
        stp_runtime = runtime_steps[runtime_steps["method"].astype(str) == method].copy()
        short = short_methods.get(method, method)

        n = len(sdf)
        if n == 0:
            continue

        success_count = int(sdf["success_bool"].sum())
        safe_success_count = int(sdf["safe_success_bool"].sum())

        completion_col = "task_completion_time_s" if "task_completion_time_s" in sdf.columns else None
        if completion_col is None and "success_time_s" in sdf.columns:
            completion_col = "success_time_s"
        if completion_col is None and "sim_time_s" in sdf.columns:
            completion_col = "sim_time_s"
        completion_success = (
            med_q1_q3(sdf.loc[sdf["success_bool"], completion_col], decimals=2)
            if completion_col is not None
            else "--"
        )

        progress_score = (
            med_q1_q3(sdf["progress_score_final"], decimals=3)
            if sdf["progress_score_final"].notna().any()
            else "--"
        )
        final_goal = med_q1_q3(sdf["final_goal_distance_final"], decimals=3)

        task_rows.append([
            short,
            count_pct(success_count, n),
            count_pct(safe_success_count, n),
            completion_success,
            progress_score,
            final_goal,
        ])

        h_viol_count = int(sdf["h_violation_run"].sum())
        g1_viol_count = int(sdf["g1_violation_run"].sum())
        g2_viol_count = int(sdf["g2_violation_run"].sum())

        safety_h_rows.append([
            short,
            count_pct(h_viol_count, n),
            med_q1_q3_sci(sdf["min_h"], decimals=2),
            worst_sci(sdf["min_h"], decimals=2),
        ])

        safety_g_rows.append([
            short,
            count_pct(g1_viol_count, n),
            med_q1_q3_sci(sdf["min_psi"], decimals=2),
            worst_sci(sdf["min_psi"], decimals=2),
            count_pct(g2_viol_count, n),
            med_q1_q3_sci(sdf["min_g2_summary"], decimals=2),
            worst_sci(sdf["min_g2_summary"], decimals=2),
        ])

        path_ineff = (
            med_q1_q3(sdf.loc[sdf["success_bool"], "path_inefficiency"], decimals=3)
            if "path_inefficiency" in sdf.columns
            else "--"
        )
        jerk = med_q1_q3(safe_numeric(sdf, "mean_squared_jerk"), decimals=1)
        intervention = med_q1_q3(safe_numeric(sdf, "mean_intervention_l2"), decimals=3)

        eff_rows.append([
            short,
            path_ineff,
            jerk,
            intervention,
        ])

        qp_fail_runs = count_pct(int(sdf["qp_failed_run"].sum()), n)
        total_qp_failed_steps = str(int(np.nansum(sdf["qp_fail_count_num"].to_numpy(dtype=float))))

        if "fallback" in method.lower() or "fb" in short.lower():
            fallback_used_runs = count_pct(int(sdf["fallback_used_any"].fillna(False).sum()), n)
            total_fb_steps = float(sdf["fallback_steps"].fillna(0).sum())
            total_logged = float(sdf["total_logged_steps"].fillna(0).sum())
            fallback_activation_rate = fmt_rate(total_fb_steps / total_logged) if total_logged > 0 else "--"
        else:
            fallback_used_runs = "--"
            fallback_activation_rate = "--"
            total_fb_steps = np.nan
            total_logged = np.nan

        robust_rows.append([
            short,
            qp_fail_runs,
            total_qp_failed_steps,
            fallback_used_runs,
            fallback_activation_rate,
        ])

        qp_runtime = (
            runtime_triplet_ms(stp_runtime["solve_time_s"], decimals=3)
            if "solve_time_s" in stp_runtime.columns
            else "--"
        )
        total_step_runtime = (
            runtime_triplet_ms(stp_runtime["step_elapsed_s"], decimals=3)
            if "step_elapsed_s" in stp_runtime.columns
            else "--"
        )
        mean_p1 = mean_value(safe_numeric(sdf, "p1_mean"), decimals=3)
        mean_p2 = mean_value(safe_numeric(sdf, "p2_mean"), decimals=3)

        runtime_p12_rows.append([
            short,
            qp_runtime,
            total_step_runtime,
            mean_p1,
            mean_p2,
        ])

        # Numeric values CSV.
        flat_records.append({
            "method": method,
            "short_method": short,
            "n_scenarios": n,
            "success_count": success_count,
            "success_rate_pct": 100.0 * success_count / n,
            "safe_success_count_min_h_ge_minus_tol": safe_success_count,
            "safe_success_rate_pct_min_h_ge_minus_tol": 100.0 * safe_success_count / n,
            "progress_score_median": float(pd.to_numeric(sdf["progress_score_final"], errors="coerce").median()),
            "final_goal_distance_median": float(pd.to_numeric(sdf["final_goal_distance_final"], errors="coerce").median()),
            "h_violation_count_min_h_lt_minus_tol": h_viol_count,
            "h_violation_rate_pct_min_h_lt_minus_tol": 100.0 * h_viol_count / n,
            "g1_violation_count_min_g1_lt_minus_tol": g1_viol_count,
            "g1_violation_rate_pct_min_g1_lt_minus_tol": 100.0 * g1_viol_count / n,
            "g2_violation_count_min_g2_lt_minus_tol": g2_viol_count,
            "g2_violation_rate_pct_min_g2_lt_minus_tol": 100.0 * g2_viol_count / n,
            "min_h_median": float(pd.to_numeric(sdf["min_h"], errors="coerce").median()),
            "min_h_q1": float(pd.to_numeric(sdf["min_h"], errors="coerce").quantile(0.25)),
            "min_h_q3": float(pd.to_numeric(sdf["min_h"], errors="coerce").quantile(0.75)),
            "min_h_worst": float(pd.to_numeric(sdf["min_h"], errors="coerce").min()),
            "min_g1_median": float(pd.to_numeric(sdf["min_psi"], errors="coerce").median()),
            "min_g1_q1": float(pd.to_numeric(sdf["min_psi"], errors="coerce").quantile(0.25)),
            "min_g1_q3": float(pd.to_numeric(sdf["min_psi"], errors="coerce").quantile(0.75)),
            "min_g1_worst": float(pd.to_numeric(sdf["min_psi"], errors="coerce").min()),
            "min_g2_median": float(pd.to_numeric(sdf["min_g2_summary"], errors="coerce").median()),
            "min_g2_q1": float(pd.to_numeric(sdf["min_g2_summary"], errors="coerce").quantile(0.25)),
            "min_g2_q3": float(pd.to_numeric(sdf["min_g2_summary"], errors="coerce").quantile(0.75)),
            "min_g2_worst": float(pd.to_numeric(sdf["min_g2_summary"], errors="coerce").min()),
            "path_inefficiency_success_median": float(pd.to_numeric(
                sdf.loc[sdf["success_bool"], "path_inefficiency"] if "path_inefficiency" in sdf.columns else pd.Series(dtype=float),
                errors="coerce",
            ).median()),
            "mean_squared_jerk_median": float(safe_numeric(sdf, "mean_squared_jerk").median()),
            "intervention_median": float(safe_numeric(sdf, "mean_intervention_l2").median()),
            "qp_fail_run_count": int(sdf["qp_failed_run"].sum()),
            "qp_fail_run_rate_pct": 100.0 * int(sdf["qp_failed_run"].sum()) / n,
            "total_qp_failed_steps": int(np.nansum(sdf["qp_fail_count_num"].to_numpy(dtype=float))),
            "fallback_used_run_count": int(sdf["fallback_used_any"].fillna(False).sum())
                if ("fallback" in method.lower() or "fb" in short.lower()) else np.nan,
            "fallback_used_run_rate_pct": 100.0 * int(sdf["fallback_used_any"].fillna(False).sum()) / n
                if ("fallback" in method.lower() or "fb" in short.lower()) else np.nan,
            "fallback_activation_rate_pct": 100.0 * total_fb_steps / total_logged
                if ("fallback" in method.lower() or "fb" in short.lower()) and np.isfinite(total_logged) and total_logged > 0 else np.nan,
            "qp_solve_mean_ms": 1000.0 * float(pd.to_numeric(
                stp_runtime.get("solve_time_s", pd.Series(dtype=float)), errors="coerce"
            ).mean()),
            "qp_solve_p95_ms": 1000.0 * float(pd.to_numeric(
                stp_runtime.get("solve_time_s", pd.Series(dtype=float)), errors="coerce"
            ).quantile(0.95)),
            "qp_solve_max_ms": 1000.0 * float(pd.to_numeric(
                stp_runtime.get("solve_time_s", pd.Series(dtype=float)), errors="coerce"
            ).max()),
            "offline_total_step_mean_ms": 1000.0 * float(pd.to_numeric(
                stp_runtime.get("step_elapsed_s", pd.Series(dtype=float)), errors="coerce"
            ).mean()),
            "offline_total_step_p95_ms": 1000.0 * float(pd.to_numeric(
                stp_runtime.get("step_elapsed_s", pd.Series(dtype=float)), errors="coerce"
            ).quantile(0.95)),
            "offline_total_step_max_ms": 1000.0 * float(pd.to_numeric(
                stp_runtime.get("step_elapsed_s", pd.Series(dtype=float)), errors="coerce"
            ).max()),
            "mean_p1_across_scenarios": float(safe_numeric(sdf, "p1_mean").mean()),
            "mean_p2_across_scenarios": float(safe_numeric(sdf, "p2_mean").mean()),
        })

    rows = {
        "task": task_rows,
        "safety_h": safety_h_rows,
        "safety_g": safety_g_rows,
        "eff": eff_rows,
        "robust": robust_rows,
        "runtime": runtime_p12_rows,
    }
    flat = pd.DataFrame(flat_records)
    return rows, flat


def render_tables(
    rows: dict[str, list[list[str]]],
    flat: pd.DataFrame,
    title: str,
    out_prefix: str,
    safety_tol: float,
    figsize: tuple[float, float],
):
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=7,
        ncols=1,
        height_ratios=[0.72, 1.18, 0.95, 1.25, 1.12, 1.12, 1.18],
        hspace=0.26,
    )

    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")
    ax_title.text(0.0, 0.98, title, fontsize=27, fontweight="bold", va="top")
    ax_title.text(
        0.0,
        0.60,
        f"Safe success = task success with min h ≥ −{safety_tol:g}. "
        f"Safety violation counts use min h/G1/G2 < −{safety_tol:g}.",
        fontsize=13.5,
        va="top",
    )
    ax_title.text(
        0.0,
        0.22,
        "Continuous rollout metrics are median [Q1, Q3]. Safety also reports worst min h / G1 / G2. "
        "Runtime is mean / p95 / max in milliseconds. Offline rollout metrics stop at the first hard-QP failure.",
        fontsize=12.8,
        va="top",
    )

    draw_table(
        fig.add_subplot(gs[1, 0]),
        "1. Task",
        ["Method", "Success rate", "Safe success rate", "Completion time [s]", "Progress score", "Final goal distance [m]"],
        rows["task"],
        font_size=10.2,
        col_widths=[0.13, 0.16, 0.16, 0.20, 0.16, 0.19],
    )

    draw_table(
        fig.add_subplot(gs[2, 0]),
        "2a. Safety — h",
        ["Method", f"h violation count\nmin h < -{safety_tol:g}", "min h\nmedian [Q1, Q3]", "min h\nworst"],
        rows["safety_h"],
        font_size=10.2,
        col_widths=[0.18, 0.24, 0.38, 0.20],
    )

    draw_table(
        fig.add_subplot(gs[3, 0]),
        "2b. Safety — G1 and G2",
        [
            "Method",
            f"G1 violation count\nmin G1 < -{safety_tol:g}",
            "min G1\nmedian [Q1, Q3]",
            "min G1\nworst",
            f"G2 violation count\nmin G2 < -{safety_tol:g}",
            "min G2\nmedian [Q1, Q3]",
            "min G2\nworst",
        ],
        rows["safety_g"],
        font_size=9.4,
        col_widths=[0.13, 0.16, 0.19, 0.10, 0.16, 0.19, 0.10],
    )

    draw_table(
        fig.add_subplot(gs[4, 0]),
        "3. Efficiency / smoothness",
        ["Method", "Path inefficiency ratio\n(successes)", "Mean squared jerk", "Intervention magnitude"],
        rows["eff"],
        font_size=10.2,
        col_widths=[0.18, 0.30, 0.26, 0.26],
    )

    draw_table(
        fig.add_subplot(gs[5, 0]),
        "4. Robustness",
        ["Method", "Runs with any\nQP infeasibility", "Total QP-failed steps", "Fallback-used runs", "Fallback activation rate"],
        rows["robust"],
        font_size=10.2,
        col_widths=[0.18, 0.24, 0.20, 0.20, 0.18],
    )

    draw_table(
        fig.add_subplot(gs[6, 0]),
        "5. Runtime / p12",
        ["Method", "QP solve time [ms]\nmean / p95 / max", "Offline total step time [ms]\nmean / p95 / max", "Mean p1", "Mean p2"],
        rows["runtime"],
        font_size=10.0,
        col_widths=[0.16, 0.31, 0.31, 0.11, 0.11],
    )

    out = Path(out_prefix)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    flat.to_csv(out.with_name(out.name + "_values.csv"), index=False)

    print(f"Wrote image: {out.with_suffix('.png').resolve()}")
    print(f"Wrote PDF:   {out.with_suffix('.pdf').resolve()}")
    print(f"Wrote CSV:   {out.with_name(out.name + '_values.csv').resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic final metric-table generator for FR3 offline comparison CSVs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs="+",
        required=True,
        metavar="ARG",
        help=(
            "One result group: SUMMARY.csv STEPS.csv mode=MethodName [mode=MethodName ...]. "
            "Repeat --run to combine several comparisons. Quote names with spaces."
        ),
    )
    parser.add_argument("--order", nargs="*", default=None, help="Method display order. Quote names with spaces.")
    parser.add_argument("--short", nargs="*", default=None, help="Optional Full method name=Short label entries.")
    parser.add_argument("--title", default="Final offline benchmark — complete paper metric tables")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--safety-tol", type=float, default=1e-4)
    parser.add_argument("--fig-width", type=float, default=40.0)
    parser.add_argument("--fig-height", type=float, default=25.0)
    args = parser.parse_args()

    runs = parse_runs(args.run)
    summary_df, steps_df, method_order = load_combined_data(runs, args.order)

    short_methods = {m: m for m in method_order}
    short_methods.update(parse_key_values(args.short, "--short"))

    rows, flat = build_tables(
        summary_df=summary_df,
        steps_df=steps_df,
        method_order=method_order,
        short_methods=short_methods,
        safety_tol=args.safety_tol,
    )

    render_tables(
        rows=rows,
        flat=flat,
        title=args.title,
        out_prefix=args.out_prefix,
        safety_tol=args.safety_tol,
        figsize=(args.fig_width, args.fig_height),
    )


if __name__ == "__main__":
    main()
