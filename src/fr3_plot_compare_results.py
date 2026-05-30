#!/usr/bin/env python3
from __future__ import annotations

"""
Clean plotting helper for FR3 rollout comparison results.

Input files are produced by compare_rollout_policies.py:
    <out-prefix>_steps.csv
    <out-prefix>_summary.csv

This script plots whichever of these policies are present in the CSV:
    original
    davide_online
    nn
    nn_g12_fallback

Main things it shows:
    - goal distance over time
    - geometric safety margin min_h over time
    - G1 / psi margin over time
    - G2 / HOCBF-QP margin over time
    - p1/p2 over time
    - intervention ||ddq_safe - ddq_nom|| over time
    - first safety-violation line per policy
    - aggregate summary plots across scenarios

Example:
    python fr3_plot_compare_results.py \
      --steps cmp_best_track_smoke_steps.csv \
      --summary cmp_best_track_smoke_summary.csv \
      --out-dir plots_cmp_best_track \
      --all-scenarios

Single scenario:
    python fr3_plot_compare_results.py \
      --steps cmp_best_track_smoke_steps.csv \
      --summary cmp_best_track_smoke_summary.csv \
      --scenario scenario_0000 \
      --out-dir plots_cmp_best_track \
      --show
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT_MODES = ["original", "davide_online", "nn", "nn_g12_fallback"]
MODE_LABEL = {
    "original": "Original",
    "davide_online": "Davide online",
    "nn": "NN",
    "nn_g12_fallback": "NN + G12 fallback",
}

# Keep line styles simple and readable. Matplotlib default colors are used.
MODE_STYLE = {
    "original": {"linestyle": "-", "linewidth": 2.0},
    "davide_online": {"linestyle": "-", "linewidth": 2.0},
    "nn": {"linestyle": "-", "linewidth": 2.2},
    "nn_g12_fallback": {"linestyle": "-", "linewidth": 2.4},
}


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def load_results(steps_csv: str, summary_csv: Optional[str]) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    steps = pd.read_csv(steps_csv)
    _require_cols(steps, ["scenario", "mode", "step", "time"], "steps CSV")

    steps = steps[steps["mode"].isin(PLOT_MODES)].copy()
    steps["mode"] = pd.Categorical(steps["mode"], categories=PLOT_MODES, ordered=True)
    steps = steps.sort_values(["scenario", "mode", "step"]).reset_index(drop=True)

    summary = None
    if summary_csv is not None and os.path.exists(summary_csv):
        summary = pd.read_csv(summary_csv)
        _require_cols(summary, ["scenario", "mode"], "summary CSV")
        summary = summary[summary["mode"].isin(PLOT_MODES)].copy()
        summary["mode"] = pd.Categorical(summary["mode"], categories=PLOT_MODES, ordered=True)
        summary = summary.sort_values(["scenario", "mode"]).reset_index(drop=True)

    return steps, summary


def find_first_safety_violation(
    df_mode: pd.DataFrame,
    h_tol: float = 0.0,
    g2_tol: Optional[float] = None,
    include_qp_fail: bool = True,
) -> tuple[Optional[float], Optional[int], Optional[str]]:
    """
    Return first safety-related violation for one mode/scenario.

    Priority:
      1. geometric safety violation: min_h < -h_tol
      2. HOCBF/QP pair-row violation: min_g2 < -g2_tol, if available
      3. QP failure, if available and include_qp_fail=True

    For the thesis plots, min_h is the most important one because it is the
    actual collision/barrier safety margin. min_g2 is useful as a diagnostic.
    """
    candidates: list[tuple[int, float, str]] = []

    if "min_h" in df_mode.columns:
        m = pd.to_numeric(df_mode["min_h"], errors="coerce") < -float(h_tol)
        if bool(m.any()):
            idx = int(np.flatnonzero(m.to_numpy())[0])
            row = df_mode.iloc[idx]
            candidates.append((idx, float(row["time"]), "min_h < 0"))

    if g2_tol is not None and "min_g2" in df_mode.columns:
        g = pd.to_numeric(df_mode["min_g2"], errors="coerce") < -float(g2_tol)
        if bool(g.any()):
            idx = int(np.flatnonzero(g.to_numpy())[0])
            row = df_mode.iloc[idx]
            candidates.append((idx, float(row["time"]), "min_g2 < 0"))

    if include_qp_fail and "qp_solved" in df_mode.columns:
        solved = _bool_series(df_mode["qp_solved"])
        m = ~solved
        if bool(m.any()):
            idx = int(np.flatnonzero(m.to_numpy())[0])
            row = df_mode.iloc[idx]
            candidates.append((idx, float(row["time"]), "QP failed"))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0])
    idx, t, reason = candidates[0]
    return t, idx, reason




def _first_below(
    df_mode: pd.DataFrame,
    col: str,
    tol: float = 0.0,
) -> tuple[Optional[float], Optional[float]]:
    """Return first time/value where a numeric column is below -tol."""
    if col not in df_mode.columns:
        return None, None
    y = pd.to_numeric(df_mode[col], errors="coerce")
    m = y < -float(tol)
    if not bool(m.any()):
        return None, None
    idx = int(np.flatnonzero(m.to_numpy())[0])
    row = df_mode.iloc[idx]
    return float(row["time"]), float(y.iloc[idx])


def _mark_first_h_violation_line(
    ax,
    df_mode: pd.DataFrame,
    tol: float = 0.0,
) -> None:
    """Mark the first geometric h violation with a vertical line instead of an x marker."""
    t, _ = _first_below(df_mode, "min_h", tol=tol)
    if t is None:
        return
    ax.axvline(t, linestyle="-", linewidth=1.6, alpha=0.75, zorder=5)


def _any_first_below(
    df_s: pd.DataFrame,
    col: str,
    modes: list[str],
    tol: float = 0.0,
) -> bool:
    """True if any plotted mode has a first below-threshold marker for this column."""
    if col not in df_s.columns:
        return False
    for mode in modes:
        df_m = df_s[df_s["mode"].astype(str) == mode]
        if df_m.empty:
            continue
        t, _ = _first_below(df_m, col, tol=tol)
        if t is not None:
            return True
    return False


def _goal_tolerance_for_scenario(summary: Optional[pd.DataFrame], scenario: str) -> Optional[float]:
    if summary is None or "goal_tolerance" not in summary.columns:
        return None
    rows = summary[summary["scenario"] == scenario]
    if rows.empty:
        return None
    val = pd.to_numeric(rows["goal_tolerance"], errors="coerce").dropna()
    if val.empty:
        return None
    return float(val.iloc[0])


def _task_completion_times_for_scenario(
    summary: Optional[pd.DataFrame],
    scenario: str,
) -> dict[str, Optional[float]]:
    """Return per-mode task completion time in simulated seconds.

    Prefer task_completion_time_s from the summary CSV. If older summaries do
    not have that column, fall back to sim_time_s for successful rollouts.
    """
    out: dict[str, Optional[float]] = {}
    if summary is None or summary.empty:
        return out

    rows = summary[summary["scenario"] == scenario].copy()
    if rows.empty:
        return out

    for mode in PLOT_MODES:
        r = rows[rows["mode"].astype(str) == mode]
        if r.empty:
            continue
        row = r.iloc[0]

        success = False
        if "success" in row.index:
            success = str(row["success"]).lower() in ["true", "1", "yes"]

        t_done = None
        if "task_completion_time_s" in row.index:
            val = pd.to_numeric(pd.Series([row["task_completion_time_s"]]), errors="coerce").iloc[0]
            if pd.notna(val):
                t_done = float(val)
        elif success and "sim_time_s" in row.index:
            val = pd.to_numeric(pd.Series([row["sim_time_s"]]), errors="coerce").iloc[0]
            if pd.notna(val):
                t_done = float(val)

        out[mode] = t_done

    return out



def _h_violation_stats_for_mode(
    df_mode: pd.DataFrame,
    h_tol: float = 0.0,
) -> dict[str, float | int | None]:
    """Compute geometric h-constraint violation statistics for one mode/scenario.

    The actual constraint is h >= 0. The h_tol only controls which samples are
    counted as violated in the plot, matching the dashboard threshold.
    """
    if "min_h" not in df_mode.columns or df_mode.empty:
        return {
            "num_violated_steps": 0,
            "num_steps": int(len(df_mode)),
            "max_h_violation": 0.0,
            "worst_h": None,
            "worst_time": None,
        }

    y = pd.to_numeric(df_mode["min_h"], errors="coerce")
    t = pd.to_numeric(df_mode["time"], errors="coerce") if "time" in df_mode.columns else None

    valid = y.notna()
    if not bool(valid.any()):
        return {
            "num_violated_steps": 0,
            "num_steps": int(len(df_mode)),
            "max_h_violation": 0.0,
            "worst_h": None,
            "worst_time": None,
        }

    y_valid = y[valid]
    num_steps = int(len(y_valid))
    violated = y_valid < -float(h_tol)
    num_violated = int(violated.sum())

    worst_idx = y_valid.idxmin()
    worst_h = float(y.loc[worst_idx])
    max_violation = max(0.0, -worst_h)

    worst_time = None
    if t is not None and worst_idx in t.index and pd.notna(t.loc[worst_idx]):
        worst_time = float(t.loc[worst_idx])

    return {
        "num_violated_steps": num_violated,
        "num_steps": num_steps,
        "max_h_violation": float(max_violation),
        "worst_h": worst_h,
        "worst_time": worst_time,
    }


def _add_h_violation_stats_box(
    ax,
    df_s: pd.DataFrame,
    modes: list[str],
    h_tol: float = 0.0,
) -> None:
    """Add max h-violation magnitude and number of violated steps to h plot."""
    lines = []
    for mode in modes:
        df_m = df_s[df_s["mode"].astype(str) == mode].sort_values("step")
        if df_m.empty:
            continue
        stats = _h_violation_stats_for_mode(df_m, h_tol=h_tol)
        n_bad = int(stats["num_violated_steps"] or 0)
        n_total = int(stats["num_steps"] or 0)
        max_v = float(stats["max_h_violation"] or 0.0)

        if n_bad == 0:
            lines.append(f"{MODE_LABEL.get(mode, mode)}: max viol=0, steps=0/{n_total}")
        else:
            lines.append(
                f"{MODE_LABEL.get(mode, mode)}: max viol={max_v:.2e}, steps={n_bad}/{n_total}"
            )

    if not lines:
        return

    ax.text(
        0.985,
        0.045,
        "h violation summary\n" + "\n".join(lines),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.88,
        },
        zorder=10,
    )



def _margin_violation_stats_for_mode(
    df_mode: pd.DataFrame,
    col: str,
    tol: float = 0.0,
) -> dict[str, float | int | None]:
    """Compute generic margin violation stats for a column with constraint value >= 0."""
    if col not in df_mode.columns or df_mode.empty:
        return {
            "num_violated_steps": 0,
            "num_steps": int(len(df_mode)),
            "max_violation": 0.0,
            "worst_value": None,
            "worst_time": None,
        }

    y = pd.to_numeric(df_mode[col], errors="coerce")
    t = pd.to_numeric(df_mode["time"], errors="coerce") if "time" in df_mode.columns else None

    valid = y.notna()
    if not bool(valid.any()):
        return {
            "num_violated_steps": 0,
            "num_steps": int(len(df_mode)),
            "max_violation": 0.0,
            "worst_value": None,
            "worst_time": None,
        }

    y_valid = y[valid]
    num_steps = int(len(y_valid))
    violated = y_valid < -float(tol)
    num_violated = int(violated.sum())

    worst_idx = y_valid.idxmin()
    worst_value = float(y.loc[worst_idx])
    max_violation = max(0.0, -worst_value)

    worst_time = None
    if t is not None and worst_idx in t.index and pd.notna(t.loc[worst_idx]):
        worst_time = float(t.loc[worst_idx])

    return {
        "num_violated_steps": num_violated,
        "num_steps": num_steps,
        "max_violation": float(max_violation),
        "worst_value": worst_value,
        "worst_time": worst_time,
    }


def _add_margin_violation_stats_box(
    ax,
    df_s: pd.DataFrame,
    modes: list[str],
    col: str,
    title: str,
    tol: float = 0.0,
    loc: str = "lower right",
) -> None:
    """Add max violation magnitude and violated-step count for G1/G2 style margins."""
    lines = []
    for mode in modes:
        df_m = df_s[df_s["mode"].astype(str) == mode].sort_values("step")
        if df_m.empty:
            continue
        stats = _margin_violation_stats_for_mode(df_m, col=col, tol=tol)
        n_bad = int(stats["num_violated_steps"] or 0)
        n_total = int(stats["num_steps"] or 0)
        max_v = float(stats["max_violation"] or 0.0)

        if n_bad == 0:
            lines.append(f"{MODE_LABEL.get(mode, mode)}: max viol=0, steps=0/{n_total}")
        else:
            lines.append(
                f"{MODE_LABEL.get(mode, mode)}: max viol={max_v:.2e}, steps={n_bad}/{n_total}"
            )

    if not lines:
        return

    if loc == "upper right":
        xy = (0.985, 0.955)
        va = "top"
    else:
        xy = (0.985, 0.045)
        va = "bottom"

    ax.text(
        xy[0],
        xy[1],
        title + "\n" + "\n".join(lines),
        transform=ax.transAxes,
        va=va,
        ha="right",
        fontsize=7.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.88,
        },
        zorder=10,
    )


def _mark_first_below_line(
    ax,
    df_mode: pd.DataFrame,
    col: str,
    tol: float = 0.0,
) -> None:
    """Mark first negative margin crossing with a solid vertical line."""
    t, _ = _first_below(df_mode, col, tol=tol)
    if t is None:
        return
    ax.axvline(t, linestyle="-", linewidth=1.6, alpha=0.75, zorder=5)

def _add_task_time_box(ax, completion_times: dict[str, Optional[float]]) -> None:
    """Show task completion time in the distance-to-goal plot."""
    lines = []
    for mode in PLOT_MODES:
        if mode not in completion_times:
            continue
        t_done = completion_times[mode]
        if t_done is None or not np.isfinite(t_done):
            lines.append(f"{MODE_LABEL.get(mode, mode)}: not reached")
        else:
            lines.append(f"{MODE_LABEL.get(mode, mode)}: {t_done:.2f}s")

    if not lines:
        return

    ax.text(
        0.985,
        0.92,
        "Task time\n" + "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.88,
        },
        zorder=10,
    )


def _plot_first_violation_lines(
    axes,
    per_mode_violation: dict[str, tuple[Optional[float], Optional[str]]],
) -> None:
    """Draw thin vertical guide lines at first safety events.

    Intentionally no inline text here: vertical labels overlap badly when
    several policies violate close together. A compact summary box is added
    separately on the safety axis.
    """
    for _, (t_violate, _) in per_mode_violation.items():
        if t_violate is None:
            continue
        for ax in axes:
            ax.axvline(t_violate, linestyle="-", linewidth=1.0, alpha=0.35)


def _add_violation_summary_box(
    ax,
    per_mode_violation: dict[str, tuple[Optional[float], Optional[str]]],
) -> None:
    """Add one readable legend-like box instead of many overlapping labels."""
    lines = []
    for mode in PLOT_MODES:
        t_violate, reason = per_mode_violation.get(mode, (None, None))
        if t_violate is None:
            lines.append(f"{MODE_LABEL.get(mode, mode)}: no violation")
        else:
            lines.append(f"{MODE_LABEL.get(mode, mode)}: t={t_violate:.2f}s, {reason}")

    if not lines:
        return

    ax.text(
        0.985,
        0.965,
        "First safety event\n" + "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.88,
        },
        zorder=10,
    )


def plot_scenario_dashboard(
    steps: pd.DataFrame,
    summary: Optional[pd.DataFrame],
    scenario: str,
    out_dir: str,
    h_tol: float = 0.0,
    g2_tol: Optional[float] = None,
    show: bool = False,
) -> Optional[str]:
    df_s = steps[steps["scenario"] == scenario].copy()
    if df_s.empty:
        print(f"[skip] no step rows for scenario {scenario}")
        return None

    present_modes = [m for m in PLOT_MODES if m in set(df_s["mode"].astype(str))]
    if not present_modes:
        print(f"[skip] no requested modes for scenario {scenario}")
        return None

    goal_tol = _goal_tolerance_for_scenario(summary, scenario)
    completion_times = _task_completion_times_for_scenario(summary, scenario)

    # One-page dashboard. G1 and G2 are separated because they tell different
    # stories: G1/psi is the first-order barrier margin, while G2 is the actual
    # second-order HOCBF-QP row margin under the chosen acceleration.
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(11.5, 15.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.05, 1.0, 1.0, 1.0, 1.0]},
    )
    fig.suptitle(f"FR3 rollout comparison — {scenario}", fontsize=14, fontweight="bold")

    violation_info: dict[str, tuple[Optional[float], Optional[str]]] = {}

    for mode in present_modes:
        df_m = df_s[df_s["mode"].astype(str) == mode].sort_values("step")
        t = pd.to_numeric(df_m["time"], errors="coerce").to_numpy()
        label = MODE_LABEL.get(mode, mode)
        style = MODE_STYLE.get(mode, {})

        if "dist_to_goal" in df_m.columns:
            y = pd.to_numeric(df_m["dist_to_goal"], errors="coerce").to_numpy()
            axes[0].plot(t, y, label=label, **style)

        if "min_h" in df_m.columns:
            y = pd.to_numeric(df_m["min_h"], errors="coerce").to_numpy()
            axes[1].plot(t, y, label=label, **style)
            _mark_first_h_violation_line(axes[1], df_m, tol=h_tol)

        # G1 is psi = Lf_h + p1*h. In the CSV it is stored as min_psi.
        if "min_psi" in df_m.columns:
            y = pd.to_numeric(df_m["min_psi"], errors="coerce").to_numpy()
            axes[2].plot(t, y, label=label, **style)
            _mark_first_below_line(axes[2], df_m, "min_psi", tol=0.0)

        # G2 is the active collision-row QP margin. In the CSV it is stored as min_g2.
        if "min_g2" in df_m.columns:
            y = pd.to_numeric(df_m["min_g2"], errors="coerce").to_numpy()
            axes[3].plot(t, y, label=label, **style)
            _mark_first_below_line(axes[3], df_m, "min_g2", tol=(0.0 if g2_tol is None else g2_tol))

        if "p1" in df_m.columns:
            y = pd.to_numeric(df_m["p1"], errors="coerce").to_numpy()
            axes[4].plot(t, y, label=f"{label} p1", **style)
        if "p2" in df_m.columns:
            y = pd.to_numeric(df_m["p2"], errors="coerce").to_numpy()
            p2_style = dict(style)
            p2_style["linewidth"] = max(1.3, float(p2_style.get("linewidth", 2.0)) - 0.4)
            p2_style["linestyle"] = "-"
            axes[4].plot(t, y, label=f"{label} p2", **p2_style)

        if "intervention_l2" in df_m.columns:
            y = pd.to_numeric(df_m["intervention_l2"], errors="coerce").to_numpy()
            axes[5].plot(t, y, label=label, **style)

        t_violate, _, reason = find_first_safety_violation(
            df_m,
            h_tol=h_tol,
            g2_tol=g2_tol,
            include_qp_fail=True,
        )
        violation_info[mode] = (t_violate, reason)

    if goal_tol is not None:
        axes[0].axhline(goal_tol, linestyle="--", linewidth=1.2, alpha=0.7, label="goal tolerance")

    for mode, t_done in completion_times.items():
        if t_done is None or not np.isfinite(t_done):
            continue
        axes[0].axvline(t_done, linestyle="-", linewidth=1.2, alpha=0.45)
    _add_task_time_box(axes[0], completion_times)

    axes[1].axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.8, label="safety boundary h=0")
    axes[2].axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.8, label="G1 boundary")
    axes[3].axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.8, label="G2 boundary")

    _plot_first_violation_lines(axes, violation_info)
    _add_violation_summary_box(axes[1], violation_info)
    _add_h_violation_stats_box(axes[1], df_s, present_modes, h_tol=h_tol)
    _add_margin_violation_stats_box(
        axes[2],
        df_s,
        present_modes,
        col="min_psi",
        title="G1 violation summary",
        tol=0.0,
        loc="lower right",
    )
    _add_margin_violation_stats_box(
        axes[3],
        df_s,
        present_modes,
        col="min_g2",
        title="G2 violation summary",
        tol=(0.0 if g2_tol is None else g2_tol),
        loc="lower right",
    )

    axes[0].set_ylabel("dist to goal [m]")
    axes[1].set_ylabel("min h [m²]")
    axes[2].set_ylabel("min G1 / psi")
    axes[3].set_ylabel("min G2 margin")
    axes[4].set_ylabel("p1, p2")
    axes[5].set_ylabel("intervention L2")
    axes[5].set_xlabel("time [s]")

    axes[0].set_title("Task tracking and completion time")
    axes[1].set_title("Geometric safety margin")
    axes[2].set_title("G1 / psi margin")
    axes[3].set_title("G2 / HOCBF-QP margin")
    axes[4].set_title("HOCBF parameters")
    axes[5].set_title("QP intervention: ||ddq_safe - ddq_nom||")

    for i, ax in enumerate(axes):
        ax.grid(True, alpha=0.3)
        if i == 4:
            ax.legend(loc="best", fontsize=8, ncol=2)
        else:
            ax.legend(loc="best", fontsize=8.5)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_safe_name(scenario)}_dashboard.png")
    fig.savefig(out_path, dpi=180)
    print(f"saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path

def plot_scenario_joints(
    steps: pd.DataFrame,
    scenario: str,
    out_dir: str,
    show: bool = False,
) -> Optional[str]:
    """Optional clean 7-joint position plot. Useful when a rollout behaves strangely."""
    df_s = steps[steps["scenario"] == scenario].copy()
    if df_s.empty:
        return None

    q_cols = [f"q_post{i}" for i in range(7)]
    if not all(c in df_s.columns for c in q_cols):
        print(f"[skip] no q_post columns for joint plot of {scenario}")
        return None

    present_modes = [m for m in PLOT_MODES if m in set(df_s["mode"].astype(str))]
    fig, axes = plt.subplots(7, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(f"Joint positions — {scenario}", fontsize=14, fontweight="bold")

    for j, ax in enumerate(axes):
        for mode in present_modes:
            df_m = df_s[df_s["mode"].astype(str) == mode].sort_values("step")
            t = pd.to_numeric(df_m["time"], errors="coerce").to_numpy()
            q = pd.to_numeric(df_m[f"q_post{j}"], errors="coerce").to_numpy()
            ax.plot(t, q, label=MODE_LABEL.get(mode, mode), **MODE_STYLE.get(mode, {}))
        ax.set_ylabel(f"q{j+1} [rad]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_safe_name(scenario)}_joints.png")
    fig.savefig(out_path, dpi=180)
    print(f"saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_aggregate_summary(
    summary: pd.DataFrame,
    out_dir: str,
    h_tol: float = 0.0,
    show: bool = False,
) -> Optional[str]:
    if summary is None or summary.empty:
        return None

    df = summary[summary["mode"].isin(PLOT_MODES)].copy()
    if df.empty:
        return None

    for c in [
        "final_dist_to_goal",
        "min_h",
        "min_psi",
        "mean_intervention_l2",
        "qp_fail_count",
        "p1_mean",
        "p2_mean",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "success" in df.columns:
        df["success_bool"] = _bool_series(df["success"])
    else:
        df["success_bool"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("FR3 rollout aggregate comparison", fontsize=14, fontweight="bold")

    labels = [MODE_LABEL[m] for m in PLOT_MODES if m in set(df["mode"].astype(str))]
    modes = [m for m in PLOT_MODES if m in set(df["mode"].astype(str))]
    x = np.arange(len(modes))

    # Final distance to goal distribution.
    if "final_dist_to_goal" in df.columns:
        data = [df[df["mode"].astype(str) == m]["final_dist_to_goal"].dropna().to_numpy() for m in modes]
        axes[0, 0].boxplot(data, labels=labels, showmeans=True)
        axes[0, 0].set_ylabel("final dist to goal [m]")
        axes[0, 0].set_title("Tracking result")
        axes[0, 0].grid(True, alpha=0.3)

    # Minimum h distribution.
    if "min_h" in df.columns:
        data = [df[df["mode"].astype(str) == m]["min_h"].dropna().to_numpy() for m in modes]
        axes[0, 1].boxplot(data, labels=labels, showmeans=True)
        axes[0, 1].axhline(-h_tol, linestyle="--", linewidth=1.2, alpha=0.8, label="safety boundary")
        axes[0, 1].set_ylabel("min h [m²]")
        axes[0, 1].set_title("Geometric safety margin")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc="best", fontsize=9)

    # Success rate and QP failures.
    success_rate = [float(df[df["mode"].astype(str) == m]["success_bool"].mean()) for m in modes]
    axes[1, 0].bar(x, success_rate)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_ylabel("success rate")
    axes[1, 0].set_title("Goal success")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # Mean intervention.
    if "mean_intervention_l2" in df.columns:
        means = [float(df[df["mode"].astype(str) == m]["mean_intervention_l2"].mean()) for m in modes]
        axes[1, 1].bar(x, means)
        axes[1, 1].set_xticks(x, labels)
        axes[1, 1].set_ylabel("mean intervention L2")
        axes[1, 1].set_title("Conservatism / intervention")
        axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "aggregate_summary.png")
    fig.savefig(out_path, dpi=180)
    print(f"saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_goal_vs_safety_scatter(
    summary: pd.DataFrame,
    out_dir: str,
    h_tol: float = 0.0,
    show: bool = False,
) -> Optional[str]:
    if summary is None or summary.empty:
        return None

    df = summary[summary["mode"].isin(PLOT_MODES)].copy()
    if df.empty or "final_dist_to_goal" not in df.columns or "min_h" not in df.columns:
        return None

    df["final_dist_to_goal"] = pd.to_numeric(df["final_dist_to_goal"], errors="coerce")
    df["min_h"] = pd.to_numeric(df["min_h"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in PLOT_MODES:
        df_m = df[df["mode"].astype(str) == mode]
        if df_m.empty:
            continue
        ax.scatter(
            df_m["final_dist_to_goal"],
            df_m["min_h"],
            label=MODE_LABEL.get(mode, mode),
            alpha=0.8,
        )

    ax.axhline(-h_tol, linestyle="--", linewidth=1.2, alpha=0.8, label="safety boundary")
    ax.set_xlabel("final dist to goal [m]")
    ax.set_ylabel("minimum h [m²]")
    ax.set_title("Tracking vs safety")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "goal_vs_safety_scatter.png")
    fig.savefig(out_path, dpi=180)
    print(f"saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def pick_default_scenario(steps: pd.DataFrame, summary: Optional[pd.DataFrame]) -> str:
    if summary is not None and not summary.empty and "final_dist_to_goal" in summary.columns:
        nn = summary[summary["mode"].astype(str) == "nn"].copy()
        if not nn.empty:
            nn["final_dist_to_goal"] = pd.to_numeric(nn["final_dist_to_goal"], errors="coerce")
            nn = nn.sort_values("final_dist_to_goal", ascending=False)
            return str(nn.iloc[0]["scenario"])
    return str(steps["scenario"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FR3 comparison rollouts: original vs Davide vs NN.")
    parser.add_argument("--steps", required=True, help="Path to <out-prefix>_steps.csv from compare_rollout_policies.py")
    parser.add_argument("--summary", default=None, help="Path to <out-prefix>_summary.csv from compare_rollout_policies.py")
    parser.add_argument("--out-dir", default="fr3_compare_plots", help="Directory for PNG plots")
    parser.add_argument("--scenario", default=None, help="Scenario stem to plot, e.g. scenario_0000")
    parser.add_argument("--all-scenarios", action="store_true", help="Create one dashboard per scenario")
    parser.add_argument("--plot-joints", action="store_true", help="Also save 7-joint position plots per scenario")
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving PNGs")
    parser.add_argument("--h-tol", type=float, default=0.0, help="Geometric safety tolerance. Violation if min_h < -h_tol")
    parser.add_argument("--g2-tol", type=float, default=None, help="Optional HOCBF/QP row tolerance. Violation if min_g2 < -g2_tol")
    args = parser.parse_args()

    steps, summary = load_results(args.steps, args.summary)

    if steps.empty:
        raise RuntimeError("No step rows for the requested plot modes were found.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if summary is not None:
        plot_aggregate_summary(summary, args.out_dir, h_tol=args.h_tol, show=args.show)
        plot_goal_vs_safety_scatter(summary, args.out_dir, h_tol=args.h_tol, show=args.show)

    if args.all_scenarios:
        scenarios = sorted(steps["scenario"].unique())
    elif args.scenario is not None:
        scenarios = [args.scenario]
    else:
        scenarios = [pick_default_scenario(steps, summary)]
        print(f"No --scenario given. Plotting default scenario: {scenarios[0]}")

    for scenario in scenarios:
        plot_scenario_dashboard(
            steps=steps,
            summary=summary,
            scenario=scenario,
            out_dir=args.out_dir,
            h_tol=args.h_tol,
            g2_tol=args.g2_tol,
            show=args.show,
        )
        if args.plot_joints:
            plot_scenario_joints(steps, scenario, args.out_dir, show=args.show)

    print("done")


if __name__ == "__main__":
    main()