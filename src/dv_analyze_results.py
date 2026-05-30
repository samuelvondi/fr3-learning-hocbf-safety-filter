#!/usr/bin/env python3

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_BASE_DIR = os.path.join(PACKAGE_ROOT, "plots", "sim")
DEFAULT_CONFIG_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(PACKAGE_ROOT)),
    "share",
    "cbf_safety_filter",
    "config",
    "generated_scenarios",
)


def parse_array_from_string(s):
    """Safely convert a CSV-stored Python/JSON list into a numpy array."""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=float)
    if pd.isna(s):
        return np.array([])
    text = str(s).strip()
    try:
        return np.asarray(json.loads(text), dtype=float)
    except Exception:
        # Pandas may preserve Python-list-like strings. Fall back carefully.
        try:
            import ast
            return np.asarray(ast.literal_eval(text), dtype=float)
        except Exception:
            return np.array([])


def scalar_or_none(x, digits=None):
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if not np.isfinite(xf):
        return None
    return round(xf, digits) if digits is not None else xf


def get_metric_dataframe(df: pd.DataFrame):
    """
    Return a trajectory-quality dataframe truncated before the first QP failure.

    This avoids contaminating h/G1/jerk/path metrics with the post-failure
    emergency-hold tail, while QP failure itself is still reported separately.
    """
    if "qp_infeasible" not in df.columns:
        return df.copy(), None

    qp_mask = df["qp_infeasible"].astype(bool).to_numpy()
    fail_indices = np.flatnonzero(qp_mask)
    if len(fail_indices) == 0:
        return df.copy(), None

    first_fail_pos = int(fail_indices[0])
    metrics_df = df.iloc[:first_fail_pos].copy()
    fail_row = df.iloc[first_fail_pos].copy()
    return metrics_df, fail_row


def compute_mean_squared_jerk(df_metrics: pd.DataFrame):
    if df_metrics.empty:
        return np.nan
    if "joint_ddq" not in df_metrics.columns or df_metrics["joint_ddq"].isnull().all():
        return np.nan

    accelerations = np.array(df_metrics["joint_ddq"].apply(parse_array_from_string).tolist())
    if accelerations.ndim != 2 or accelerations.shape[0] <= 1:
        return np.nan

    time_vals = df_metrics["time"].to_numpy(dtype=float)
    time_diffs = np.diff(time_vals)
    time_diffs[np.abs(time_diffs) < 1e-12] = 1e-6
    jerk = np.diff(accelerations, axis=0) / time_diffs[:, np.newaxis]
    return float(np.mean(np.sum(np.square(jerk), axis=1)))


def compute_path_inefficiency(df_metrics: pd.DataFrame):
    if df_metrics.empty:
        return np.nan, np.nan
    if "joint_q" not in df_metrics.columns or df_metrics["joint_q"].isnull().all():
        return np.nan, np.nan

    positions = np.array(df_metrics["joint_q"].apply(parse_array_from_string).tolist())
    if positions.ndim != 2 or positions.shape[0] <= 1:
        return np.nan, np.nan

    step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_path_length = float(np.sum(step_distances))

    q_initial = positions[0]
    q_final = positions[-1]
    direct_distance = float(np.linalg.norm(q_final - q_initial))

    if direct_distance < 1e-6:
        path_ratio = 1.0
    else:
        path_ratio = total_path_length / direct_distance

    return total_path_length, float(path_ratio)


def series_stat(df: pd.DataFrame, column: str, fn: str):
    if df.empty or column not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan
    if fn == "mean":
        return float(np.mean(vals))
    if fn == "max":
        return float(np.max(vals))
    if fn == "min":
        return float(np.min(vals))
    if fn == "p95":
        return float(np.percentile(vals, 95))
    raise ValueError(fn)


def infer_method_tag(df: pd.DataFrame, csv_file_path: str, results_base_dir: str):
    if "p12_mode" in df.columns and df["p12_mode"].notna().any():
        return str(df["p12_mode"].dropna().iloc[0])

    # Fallback: infer from result subfolder if files are stored under plots/sim/<tag>/<scenario>/run_data.csv
    parent = Path(csv_file_path).parent
    base = Path(results_base_dir)
    try:
        rel = parent.relative_to(base)
        if len(rel.parts) >= 2:
            return rel.parts[-2]
    except Exception:
        pass
    return "unknown"


def load_scenario_metadata(config_base_dir: str, scenario_name: str):
    scenario_yaml_path = os.path.join(config_base_dir, f"{scenario_name}.yaml")
    gamma = None
    beta = None
    obstacles_json = None

    if os.path.exists(scenario_yaml_path):
        with open(scenario_yaml_path, "r") as f:
            scenario_config = yaml.safe_load(f)
        hocbf_params = scenario_config.get("hocbf_controller", {}).get("ros__parameters", {})
        gamma = hocbf_params.get("gamma_js")
        beta = hocbf_params.get("beta_js")
        obstacles_json = json.dumps(scenario_config.get("obstacles", []))
    return gamma, beta, obstacles_json


def analyze_all_scenarios(results_base_dir: str, config_base_dir: str, summary_filename: str):
    if not os.path.isdir(results_base_dir):
        print(f"Error: Results directory not found at '{results_base_dir}'")
        return

    csv_files = glob.glob(os.path.join(results_base_dir, "**", "run_data.csv"), recursive=True)
    if not csv_files:
        print(f"No 'run_data.csv' files found in '{results_base_dir}'.")
        return

    print(f"Found {len(csv_files)} result files. Analyzing...")

    all_summaries = []

    for csv_file_path in sorted(csv_files):
        try:
            scenario_name = os.path.basename(os.path.dirname(csv_file_path))
            gamma, beta, obstacles_json = load_scenario_metadata(config_base_dir, scenario_name)
            df = pd.read_csv(csv_file_path)

            if df.empty:
                all_summaries.append(
                    {
                        "scenario": scenario_name,
                        "method": "unknown",
                        "status": "EMPTY_DATA",
                        "duration_s": 0.0,
                        "metrics_duration_s": 0.0,
                        "qp_infeasible_count": 0,
                        "qp_failed": False,
                        "first_qp_fail_time_s": None,
                        "min_h_val": None,
                        "min_dist": None,
                        "min_psi_val": None,
                        "mean_squared_jerk": None,
                        "path_length_q": None,
                        "path_inefficiency_ratio": None,
                        "gamma": gamma,
                        "beta": beta,
                        "obstacles_json": obstacles_json,
                    }
                )
                continue

            method = infer_method_tag(df, csv_file_path, results_base_dir)
            final_status = str(df["final_run_status"].iloc[0]) if "final_run_status" in df.columns else "UNKNOWN"
            total_duration = float(pd.to_numeric(df["time"], errors="coerce").dropna().iloc[-1]) if "time" in df.columns else np.nan

            metrics_df, fail_row = get_metric_dataframe(df)
            qp_infeasible_count = int(df["qp_infeasible"].astype(bool).sum()) if "qp_infeasible" in df.columns else 0
            qp_failed = bool(qp_infeasible_count > 0)
            first_qp_fail_time_s = scalar_or_none(fail_row["time"], 6) if fail_row is not None and "time" in fail_row else None
            metrics_duration_s = (
                float(pd.to_numeric(metrics_df["time"], errors="coerce").dropna().iloc[-1])
                if not metrics_df.empty and "time" in metrics_df.columns
                else 0.0
            )

            min_h = series_stat(metrics_df, "min_h", "min")
            min_dist = series_stat(metrics_df, "min_dist", "min")
            min_psi = series_stat(metrics_df, "min_psi", "min")

            min_h_at_qp_fail = scalar_or_none(fail_row["min_h"], 8) if fail_row is not None and "min_h" in fail_row else None
            min_dist_at_qp_fail = scalar_or_none(fail_row["min_dist"], 8) if fail_row is not None and "min_dist" in fail_row else None
            min_psi_at_qp_fail = scalar_or_none(fail_row["min_psi"], 8) if fail_row is not None and "min_psi" in fail_row else None

            mean_squared_jerk = compute_mean_squared_jerk(metrics_df)
            path_length_q, path_inefficiency_ratio = compute_path_inefficiency(metrics_df)

            fallback_step_count = int(df["fallback_used"].astype(bool).sum()) if "fallback_used" in df.columns else 0
            fallback_used = bool(fallback_step_count > 0)

            summary = {
                "scenario": scenario_name,
                "method": method,
                "status": final_status,
                "duration_s": scalar_or_none(total_duration, 4),
                "metrics_duration_s": scalar_or_none(metrics_duration_s, 4),
                "qp_infeasible_count": qp_infeasible_count,
                "qp_failed": qp_failed,
                "first_qp_fail_time_s": first_qp_fail_time_s,
                "min_h_val": scalar_or_none(min_h, 8),
                "min_dist": scalar_or_none(min_dist, 8),
                "min_psi_val": scalar_or_none(min_psi, 8),
                "min_h_at_qp_fail": min_h_at_qp_fail,
                "min_dist_at_qp_fail": min_dist_at_qp_fail,
                "min_psi_at_qp_fail": min_psi_at_qp_fail,
                "avg_final_qp_solve_time_s": scalar_or_none(series_stat(df, "final_qp_solve_time_s", "mean"), 8),
                "avg_total_filter_time_s": scalar_or_none(series_stat(df, "total_filter_time_s", "mean"), 8),
                "avg_nn_inference_time_s": scalar_or_none(series_stat(df, "nn_inference_time_s", "mean"), 8),
                "fallback_used": fallback_used,
                "fallback_step_count": fallback_step_count,
                "p1_mean": scalar_or_none(series_stat(metrics_df, "p1", "mean"), 8),
                "p1_p95": scalar_or_none(series_stat(metrics_df, "p1", "p95"), 8),
                "p1_max": scalar_or_none(series_stat(metrics_df, "p1", "max"), 8),
                "p2_mean": scalar_or_none(series_stat(metrics_df, "p2", "mean"), 8),
                "p2_p95": scalar_or_none(series_stat(metrics_df, "p2", "p95"), 8),
                "p2_max": scalar_or_none(series_stat(metrics_df, "p2", "max"), 8),
                "mean_squared_jerk": scalar_or_none(mean_squared_jerk, 8),
                "path_length_q": scalar_or_none(path_length_q, 8),
                "path_inefficiency_ratio": scalar_or_none(path_inefficiency_ratio, 8),
                "gamma": gamma,
                "beta": beta,
                "obstacles_json": obstacles_json,
            }
            all_summaries.append(summary)

        except Exception as exc:
            print(f"Error processing file {csv_file_path}: {exc}")

    summary_df = pd.DataFrame(all_summaries)
    if summary_df.empty:
        print("No valid runs to summarize.")
        return

    summary_df = summary_df.sort_values(by=["method", "scenario"])
    summary_file_path = os.path.join(results_base_dir, summary_filename)
    summary_df.to_csv(summary_file_path, index=False)

    print("\n" + "=" * 80)
    print("Batch Analysis Complete")
    print("=" * 80)
    print(f"Per-run summary saved to: {summary_file_path}")

    for method, group in summary_df.groupby("method", dropna=False):
        total = len(group)
        success_count = int((group["status"] == "SUCCESS").sum())
        timeout_count = int(group["status"].isin(["TIMEOUT", "NONE"]).sum())
        qp_fail_runs = int(group["qp_failed"].fillna(False).astype(bool).sum())
        h_viol_runs = int((pd.to_numeric(group["min_h_val"], errors="coerce") < 0).sum())
        collision_runs = int((pd.to_numeric(group["min_dist"], errors="coerce") < 0).sum())
        psi_viol_runs = int((pd.to_numeric(group["min_psi_val"], errors="coerce") < 0).sum())
        fallback_runs = int(group["fallback_used"].fillna(False).astype(bool).sum())

        row = {
            "method": method,
            "runs": total,
            "successes": success_count,
            "timeouts": timeout_count,
            "qp_fail_runs": qp_fail_runs,
            "h_violation_runs_pre_qp_fail": h_viol_runs,
            "collision_runs_pre_qp_fail": collision_runs,
            "psi_violation_runs_pre_qp_fail": psi_viol_runs,
            "fallback_runs": fallback_runs,
            "avg_duration_s": pd.to_numeric(group["duration_s"], errors="coerce").mean(),
            "avg_final_qp_solve_time_s": pd.to_numeric(group["avg_final_qp_solve_time_s"], errors="coerce").mean(),
            "avg_total_filter_time_s": pd.to_numeric(group["avg_total_filter_time_s"], errors="coerce").mean(),
            "avg_nn_inference_time_s": pd.to_numeric(group["avg_nn_inference_time_s"], errors="coerce").mean(),
            "avg_mean_squared_jerk": pd.to_numeric(group["mean_squared_jerk"], errors="coerce").mean(),
            "avg_path_inefficiency_ratio": pd.to_numeric(group["path_inefficiency_ratio"], errors="coerce").mean(),
        }
        print("\n---", method, "---")
        for key, value in row.items():
            if isinstance(value, float):
                print(f"{key:40s}: {value:.8g}")
            else:
                print(f"{key:40s}: {value}")

    print("\nAnalysis finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base-dir", default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--config-base-dir", default=DEFAULT_CONFIG_BASE_DIR)
    parser.add_argument("--summary-filename", default="batch_summary.csv")
    args = parser.parse_args()

    analyze_all_scenarios(
        results_base_dir=args.results_base_dir,
        config_base_dir=args.config_base_dir,
        summary_filename=args.summary_filename,
    )


if __name__ == "__main__":
    main()
