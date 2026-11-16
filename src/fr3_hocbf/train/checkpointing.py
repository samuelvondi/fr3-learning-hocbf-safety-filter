"""Epoch statistics, checkpoints and the safe-candidate checkpoint set of the rollout training."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from fr3_hocbf.policy.features import INPUT_DIM
from fr3_hocbf.policy.p12_net import P12ParamNet
from fr3_hocbf.sim.qp_data import NUM_BOUND_ROWS, NUM_PAIR_ROWS, NUM_TOTAL_ROWS
from fr3_hocbf.train.config import RolloutTrainCfg


def aggregate_stats(stats: list[dict]) -> dict:
    valid = [s for s in stats if not s.get("solver_failed", False)]
    if not valid:
        return {}

    keys_mean = [
        "loss",
        "safety",
        "h_safety",
        "h_viol_freq",
        "g1_safety",
        "slack",
        "goal",
        "goal_initial",
        "goal_final",
        "goal_dist_mean",
        "goal_progress_ratio",
        "normalized_mean_goal_distance",
        "goal_delta",
        "preg",
        "h_mean",
        "g1_mean",
        "g2_mean",
        "p1_mean",
        "p2_mean",
        "mean_segment_time_s",
    ]
    out = {}
    for key in keys_mean:
        out[key] = float(np.mean([s[key] for s in valid]))

    out["hard_viol_max"] = float(np.max([s["hard_viol_max"] for s in valid]))
    out["pair_viol_max"] = float(np.max([s["pair_viol_max"] for s in valid]))
    out["bound_viol_max"] = float(np.max([s["bound_viol_max"] for s in valid]))
    out["viol_max"] = out["hard_viol_max"]
    out["slack_max"] = float(np.max([s["slack_max"] for s in valid]))
    out["num_valid_segments"] = int(len(valid))
    out["num_failed_segments"] = int(len(stats) - len(valid))

    finite_h = [s["min_h"] for s in valid if np.isfinite(s["min_h"])]
    finite_psi = [s["min_psi"] for s in valid if np.isfinite(s["min_psi"])]
    finite_g2 = [s["min_g2"] for s in valid if np.isfinite(s.get("min_g2", float("nan")))]
    out["min_h"] = float(np.min(finite_h)) if finite_h else float("nan")
    out["min_psi"] = float(np.min(finite_psi)) if finite_psi else float("nan")
    out["min_g2"] = float(np.min(finite_g2)) if finite_g2 else float("nan")
    return out


def maybe_load_model_weights(model: P12ParamNet, path: str | None) -> None:
    if path is None:
        return
    ckpt = torch.load(os.path.expanduser(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)


def save_checkpoint(
    path: str,
    model: P12ParamNet,
    cfg: RolloutTrainCfg,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    epoch: int,
    metric_name: str,
    metric_value: float,
    extra: dict | None = None,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "feature_version": "q_dq_ddqnom_ddqnomnorm_h_Lfh",
        "hidden_dim": cfg.hidden_dim,
        "p1_floor": cfg.p1_floor,
        "p2_floor": cfg.p2_floor,
        "p1_max": cfg.p1_max,
        "p2_max": cfg.p2_max,
        "init_p1": cfg.init_p1,
        "init_p2": cfg.init_p2,
        "feat_mean": feat_mean.detach().cpu(),
        "feat_std": feat_std.detach().cpu(),
        "dt": cfg.dt,
        "num_pair_rows": NUM_PAIR_ROWS,
        "num_bound_rows": NUM_BOUND_ROWS,
        "num_total_rows": NUM_TOTAL_ROWS,
        "rho": cfg.rho,
        "dqp_row_norm_eps": cfg.dqp_row_norm_eps,
        "train_layer_type": "rollout_bound_slack_cvxpy",
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "cfg": asdict(cfg),
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def safety_score_from_epoch(ep: dict) -> float:
    """
    Lower is safer. Checkpoint safety now uses only failed segments and
    post-step geometric h violation. Pair-row QP residuals are diagnostics only.
    """
    if not ep:
        return float("inf")
    failed = float(ep.get("num_failed_segments", 0))
    min_h = float(ep.get("min_h", float("nan")))
    h_violation = max(0.0, -min_h) if np.isfinite(min_h) else 0.0
    return failed * 1e6 + h_violation


def track_metric_from_epoch(ep: dict, cfg: RolloutTrainCfg) -> float:
    if not ep:
        return float("inf")
    value = ep.get(cfg.track_metric, float("inf"))
    try:
        value_f = float(value)
    except Exception:
        return float("inf")
    return value_f if np.isfinite(value_f) else float("inf")


def is_safe_enough(ep: dict, cfg: RolloutTrainCfg) -> bool:
    if not ep:
        return False
    if int(ep.get("num_failed_segments", 0)) != 0:
        return False
    min_h = float(ep.get("min_h", float("nan")))
    if np.isfinite(min_h) and min_h < -cfg.safe_min_h_tol:
        return False
    return True


def _finite_or_nan(value) -> float:
    try:
        value_f = float(value)
    except Exception:
        return float("nan")
    return value_f if np.isfinite(value_f) else float("nan")


def _violation_from_min(value) -> float:
    value_f = _finite_or_nan(value)
    return max(0.0, -value_f) if np.isfinite(value_f) else float("nan")


def _float_for_filename(value: float) -> str:
    value_f = _finite_or_nan(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.3e}".replace("+", "").replace("-", "m").replace(".", "p")


def safe_candidate_checkpoint_path(cfg: RolloutTrainCfg, epoch: int, split: str, track_value: float, ep: dict) -> str:
    safe_dir = Path(cfg.safe_candidate_dir).expanduser()
    name = (
        f"epoch_{epoch:04d}_{split}_"
        f"{cfg.track_metric}_{_float_for_filename(track_value)}_"
        f"minh_{_float_for_filename(ep.get('min_h', float('nan')))}_"
        f"ming1_{_float_for_filename(ep.get('min_psi', float('nan')))}_"
        f"ming2_{_float_for_filename(ep.get('min_g2', float('nan')))}.pt"
    )
    return str(safe_dir / name)


def make_candidate_summary_row(epoch: int, split: str, ckpt_path: str, cfg: RolloutTrainCfg, ep: dict) -> dict:
    track_value = track_metric_from_epoch(ep, cfg)
    row = {
        "epoch": int(epoch),
        "split": split,
        "checkpoint": ckpt_path,
        "track_metric_name": cfg.track_metric,
        "track_metric_value": track_value,
        "loss_mean": _finite_or_nan(ep.get("loss")),
        "safety_loss_mean": _finite_or_nan(ep.get("safety")),
        "h_safety_loss_mean": _finite_or_nan(ep.get("h_safety")),
        "h_viol_freq_loss_mean": _finite_or_nan(ep.get("h_viol_freq")),
        "g1_safety_loss_mean": _finite_or_nan(ep.get("g1_safety")),
        "slack_loss_mean": _finite_or_nan(ep.get("slack")),
        "goal_loss_mean": _finite_or_nan(ep.get("goal")),
        "goal_initial_mean": _finite_or_nan(ep.get("goal_initial")),
        "goal_final_mean": _finite_or_nan(ep.get("goal_final")),
        "goal_dist_mean": _finite_or_nan(ep.get("goal_dist_mean")),
        "goal_progress_ratio_mean": _finite_or_nan(ep.get("goal_progress_ratio")),
        "normalized_mean_goal_distance": _finite_or_nan(ep.get("normalized_mean_goal_distance")),
        "goal_delta_mean": _finite_or_nan(ep.get("goal_delta")),
        "h_mean": _finite_or_nan(ep.get("h_mean")),
        "min_h": _finite_or_nan(ep.get("min_h")),
        "h_max_violation": _violation_from_min(ep.get("min_h")),
        "g1_mean": _finite_or_nan(ep.get("g1_mean")),
        "min_g1_or_psi": _finite_or_nan(ep.get("min_psi")),
        "g1_max_violation": _violation_from_min(ep.get("min_psi")),
        "g2_mean": _finite_or_nan(ep.get("g2_mean")),
        "min_g2": _finite_or_nan(ep.get("min_g2")),
        "g2_max_violation": _violation_from_min(ep.get("min_g2")),
        "hard_viol_max": _finite_or_nan(ep.get("hard_viol_max")),
        "pair_viol_max": _finite_or_nan(ep.get("pair_viol_max")),
        "bound_viol_max": _finite_or_nan(ep.get("bound_viol_max")),
        "slack_max": _finite_or_nan(ep.get("slack_max")),
        "p1_mean": _finite_or_nan(ep.get("p1_mean")),
        "p2_mean": _finite_or_nan(ep.get("p2_mean")),
        "mean_segment_time_s": _finite_or_nan(ep.get("mean_segment_time_s")),
        "epoch_elapsed_s": _finite_or_nan(ep.get("elapsed_s")),
        "num_valid_segments": int(ep.get("num_valid_segments", 0)),
        "num_failed_segments": int(ep.get("num_failed_segments", 0)),
    }
    return row


def write_candidate_summaries(rows: list[dict], cfg: RolloutTrainCfg) -> None:
    if not rows:
        return

    csv_path = Path(cfg.safe_candidate_csv).expanduser()
    txt_path = Path(cfg.safe_candidate_txt).expanduser()
    if str(csv_path.parent) not in ("", "."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if str(txt_path.parent) not in ("", "."):
        txt_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Human-readable side-by-side summary sorted by the selected tracking metric.
    rows_sorted = sorted(rows, key=lambda r: _finite_or_nan(r.get("track_metric_value")))
    cols = [
        "epoch",
        "split",
        "track_metric_value",
        "h_safety_loss_mean",
        "h_viol_freq_loss_mean",
        "g1_safety_loss_mean",
        "h_mean",
        "min_h",
        "h_max_violation",
        "g1_mean",
        "min_g1_or_psi",
        "g1_max_violation",
        "g2_mean",
        "min_g2",
        "g2_max_violation",
        "p1_mean",
        "p2_mean",
        "mean_segment_time_s",
        "num_valid_segments",
        "num_failed_segments",
        "checkpoint",
    ]

    def fmt(v):
        if isinstance(v, int):
            return str(v)
        vf = _finite_or_nan(v)
        if np.isfinite(vf):
            return f"{vf:.6e}"
        return str(v)

    table_rows = [[fmt(r.get(c, "")) for c in cols] for r in rows_sorted]
    widths = [len(c) for c in cols]
    for row in table_rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    lines = []
    lines.append("Safe candidate checkpoints")
    lines.append(f"safe gate: num_failed_segments == 0 and min_h >= -{cfg.safe_min_h_tol:g}")
    lines.append(f"sorted by: {cfg.track_metric} (lower is better)")
    lines.append("")
    lines.append(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in table_rows:
        lines.append(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))

    txt_path.write_text("\n".join(lines) + "\n")


def guard_against_accidental_overwrite(output_paths: Iterable[str], overwrite: bool = False) -> None:
    """Refuse to start training if output files already exist, unless --overwrite is set.

    This protects long training runs from accidentally replacing previous checkpoints/logs.
    Relative paths are resolved against the current working directory, exactly like torch.save/open.
    """
    unique_paths: list[Path] = []
    seen: set[Path] = set()

    for raw_path in output_paths:
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser()
        resolved_key = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        unique_paths.append(path)

    existing = [path for path in unique_paths if path.exists()]
    if existing and not overwrite:
        msg_lines = [
            "Refusing to start because these output files already exist:",
            *[f"  - {path}" for path in existing],
            "",
            "Use different output names, move/delete the old files, or rerun with --overwrite.",
        ]
        raise FileExistsError("\n".join(msg_lines))

    if existing and overwrite:
        print("WARNING: --overwrite enabled. Existing output files may be replaced:")
        for path in existing:
            print("  -", path)

    # Create parent folders if the user passed paths inside a new run directory.
    for path in unique_paths:
        parent = path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
