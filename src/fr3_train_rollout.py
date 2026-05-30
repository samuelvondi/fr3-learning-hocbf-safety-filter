#!/usr/bin/env python3
from __future__ import annotations

"""
Short-horizon closed-loop rollout training for FR3 p12 / BarrierNet.

This is the bound/input-slack training version.

Training-time differentiable QP:
    - HOCBF / geometric pair rows stay hard.
    - Joint acceleration / velocity / position bound rows are softened with slack.
    - Bound slack measures how much extra actuator/joint-limit authority would be required.
    - Geometric safety cannot be bought with slack.

Runtime/final evaluation should still use the original hard QP + fallback.

Training-time DQP:
    min_{u,s_b} 0.5 ||u-u_nom||^2 + 0.5 rho ||s_b||^2
    s.t.        A_pair  u       >= b_pair
                A_bound u + s_b >= b_bound
                s_b >= 0

Main changes versus the first prototype:
    1) Supports Cartesian goal tracking loss, default disabled for safety-first training.
    2) Uses a deeper P12ParamNet.
    3) Uses bound/input slack only, not all-row slack.
    4) Uses h_all and Lf_h_all directly as features.
    5) Uses post-step geometric h violation as the safety loss.
    6) Saves every checkpoint that passes the safe-given-track gate for later offline comparison.

Example:
    python fr3_train_rollout.py \
      ~/davide_fr3_ws/src/cbf_safety_filter/config/generated_scenarios \
      --rollout-dir ~/davide_fr3_ws/src/cbf_safety_filter/data/rollouts/good_200 \
      --sample-start-from-rollout \
      --stats-in fr3_feature_stats_h_lfh_good200.npz \
      --epochs 3 \
      --segments-per-epoch 3 \
      --horizon 25 \
      --start-step-min 0 \
      --start-step-max 2500 \
      --rho 1e5 \
      --w-safety 100.0 \
      --w-g1-safety 10.0 \
      --safety-topk-frac 0.05 \
      --w-slack 1.0 \
      --w-goal 0.0 \
      --w-preg 0.0 \
      --p1-max 200.0 \
      --p2-max 250.0 \
      --model-out fr3_p12_rollout_latest.pt \
      --log-jsonl fr3_train_rollout_log.jsonl
"""

import argparse
import csv
import glob
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin
import torch
import torch.nn as nn

from fr3_rollout import (
    NUM_ARM_JOINTS,
    get_default_initial_state,
    load_scenario_yaml,
    make_rollout_context,
)
from fr3_torch_kinematics import EE_FRAME_NAME, load_fr3_torch_kinematics
from fr3_torch_geometry import compute_all_pair_terms_torch
from fr3_nominal_controller import nominal_controller_js_standalone
from fr3_qp_data import (
    NUM_BOUND_ROWS,
    NUM_PAIR_ROWS,
    NUM_TOTAL_ROWS,
    get_joint_limits,
)
from fr3_cvx_layer_bound_slack import (
    bound_violation_batch,
    hard_violation_batch,
    make_bound_slack_qp_layer,
    mean_sq_bound_slack,
    pair_violation_batch,
    solve_bound_slack_qp_batch,
)


DT_DEFAULT = 1.0 / 50.0
INPUT_DIM = 7 + 7 + 7 + 7 + NUM_PAIR_ROWS + NUM_PAIR_ROWS  # q, dq, ddq_nominal, ddq_nominal/ddq_scale, h_all, Lf_h_all = 114


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------


def violation_frequency_loss_from_sq(
    viol_sq: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft count of post-step h violations: 0 when safe, approximately 1 when violated."""
    if viol_sq.numel() == 0:
        return torch.zeros((), device=viol_sq.device, dtype=viol_sq.dtype)
    return (2.0 * torch.sigmoid(viol_sq / eps) - 1.0).mean()

@dataclass
class RolloutTrainCfg:
    seed: int = 0
    epochs: int = 3
    segments_per_epoch: int = 1
    horizon: int = 10

    # If enabled, each segment starts from q,dq sampled from a generated good rollout NPZ.
    # The NPZ is used only for initial state/window sampling, not as labels.
    sample_from_rollouts: bool = False
    start_step_min: int = 0
    start_step_max: int | None = None

    lr: float = 1e-3
    weight_decay: float = 0.0

    hidden_dim: int = 256
    p1_floor: float = 1e-3
    p2_floor: float = 1e-3
    p1_max: float = 200.0
    p2_max: float = 250.0

    # Initial NN output at zero normalized input.
    # This makes an untrained network behave like the original/Davide baseline
    # instead of starting near half of the p caps.
    init_p1: float = 10.0
    init_p2: float = 15.0

    dt: float = DT_DEFAULT

    # Outer losses.
    # Safety is post-step geometric h plus post-step G1/psi violation.
    # Both safety terms use a top-k loss over actual positive violations so rare
    # dangerous pair/timestep violations are not diluted by many safe rows.
    # Goal is disabled by default for safety-first training.
    w_safety: float = 100.0          # post-step h top-k loss weight
    w_g1_safety: float = 10.0       # post-step G1/psi top-k loss weight
    w_viol_freq: float = 0.0        # post-step h violation-frequency loss weight
    viol_freq_eps: float = 1e-6     # sharpness scale for violation-frequency loss
    w_slack: float = 1.0
    w_goal: float = 0.0
    w_preg: float = 0.0

    # Post-step safety margins.
    # h loss:  topk(ReLU(h_safety_margin  - h_next)^2)
    # G1 loss: topk(ReLU(g1_safety_margin - psi_next)^2)
    h_safety_margin: float = 0.0
    g1_safety_margin: float = 0.0
    safety_topk_frac: float = 0.05
    slack_huber_delta: float = 0.0

    # Training-time DQP.
    # HOCBF/pair rows are hard; bound/input rows are softened with slack.
    rho: float = 1e5
    dqp_row_norm_eps: float = 1e-6

    # Rollout behavior.
    move_obstacles: bool = True
    detach_nominal: bool = True
    grad_clip: float = 5.0

    # Validation / checkpointing.
    val_every: int = 1
    val_segments: int = 5
    val_rollout_dir: str | None = None
    # safe_pair_viol_tol is kept for backwards-compatible CLIs/logs,
    # but checkpointing no longer uses pair-row violation.
    safe_pair_viol_tol: float = 1e-6
    safe_min_h_tol: float = 1e-6
    checkpoint_tol: float = 1e-9
    track_metric: str = "normalized_mean_goal_distance"
    goal_progress_eps: float = 1e-3

    # Logging / checkpoints.
    model_out: str = "fr3_p12_rollout_latest.pt"
    safe_candidate_dir: str = "safe_candidates"
    safe_candidate_csv: str = "safe_candidates_summary.csv"
    safe_candidate_txt: str = "safe_candidates_summary.txt"
    log_jsonl: str = "fr3_train_rollout_log.jsonl"


# ---------------------------------------------------------------------
# feature utilities kept here so singlestep_train.py can be deleted
# ---------------------------------------------------------------------
def nominal_ddq_scale_numpy() -> np.ndarray:
    """
    Per-joint nominal-acceleration normalization scale.

    This is the 3D analogue of the 2D u_nom / u_max feature.  Because the FR3
    acceleration limits may be asymmetric, we use the larger absolute limit per
    joint:
        scale_j = max(|ddq_min_j|, |ddq_max_j|)
    """
    limits = get_joint_limits()
    scale = np.maximum(
        np.abs(np.asarray(limits["ddq_min_arm"], dtype=float)),
        np.abs(np.asarray(limits["ddq_max_arm"], dtype=float)),
    )
    return np.clip(scale.reshape(NUM_ARM_JOINTS), 1e-6, None)


def build_raw_feature_matrix(data):
    """
    Build the 114D feature matrix from saved rollout NPZ data.

    Feature order:
        q_arm,
        dq_arm,
        ddq_nominal_arm,
        ddq_nominal_arm / ddq_scale,
        h_all,
        Lf_h_all

    The normalized nominal acceleration gives the NN direct information about
    how large the nominal input request is relative to the available joint
    acceleration authority, analogous to the 2D u_nom/u_max feature.

    Important: this intentionally uses h_all and Lf_h_all directly. It does not
    use the old d_all = h_all + d_margin feature, because h_all is a squared
    distance barrier and d_margin is a distance.
    """
    q_arm = data["q_arm"]
    dq_arm = data["dq_arm"]
    ddq_nominal_arm = data["ddq_nominal_arm"]
    ddq_nominal_norm = ddq_nominal_arm / nominal_ddq_scale_numpy()[None, :]
    h_all = data["h_all"]
    Lf_h_all = data["Lf_h_all"]

    z_raw = np.concatenate(
        [
            q_arm,
            dq_arm,
            ddq_nominal_arm,
            ddq_nominal_norm,
            h_all,
            Lf_h_all,
        ],
        axis=1,
    )
    return z_raw


def compute_feature_stats(z_raw):
    feat_mean = z_raw.mean(axis=0)
    feat_std = z_raw.std(axis=0)
    feat_std = np.clip(feat_std, 1e-6, None)
    return feat_mean, feat_std


def normalize_features_numpy(z_raw, mean, std):
    std_safe = np.clip(std, 1e-6, None)
    return (z_raw - mean) / std_safe


def load_feature_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    stats = np.load(os.path.expanduser(path))
    if "feat_mean" in stats.files:
        mean = stats["feat_mean"]
        std = stats["feat_std"]
    elif "mean" in stats.files:
        mean = stats["mean"]
        std = stats["std"]
    else:
        raise KeyError(f"Could not find feat_mean/feat_std in {path}")

    mean_t = torch.tensor(mean, dtype=torch.double)
    std_t = torch.tensor(std, dtype=torch.double)
    std_t = torch.clamp(std_t, min=1e-6)

    if mean_t.numel() != INPUT_DIM:
        raise ValueError(f"Expected feature dim {INPUT_DIM}, got {mean_t.numel()} from {path}")

    return mean_t, std_t


def normalize_features_torch(z_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (z_raw - mean) / torch.clamp(std, min=1e-6)


def topk_positive_violation_sq_loss(violation: torch.Tensor, frac: float) -> torch.Tensor:
    """
    Average the largest fraction of positive violations squared.

    This sits between a diluted mean-over-all loss and a noisy pure max loss:
    it ignores already-safe zero rows and focuses training on the worst unsafe
    pair/timestep constraints. If no row violates, the loss is exactly zero.
    """
    v = violation.reshape(-1)
    v = v[v > 0.0]
    if v.numel() == 0:
        return torch.zeros((), dtype=violation.dtype, device=violation.device)

    frac_f = float(frac)
    if frac_f <= 0.0:
        k = 1
    elif frac_f >= 1.0:
        k = int(v.numel())
    else:
        k = max(1, int(np.ceil(frac_f * int(v.numel()))))

    return torch.topk(v ** 2, k=k).values.mean()


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
class P12ParamNet(nn.Module):
    """
    Deeper NN for FR3 p12/gamma-beta prediction.

    Input is already normalized outside the model.
    Output is bounded positive:
        p1 in [p1_floor, p1_max]
        p2 in [p2_floor, p2_max]

    The final layer is initialized so that a zero normalized input predicts
    approximately the original/Davide baseline gains, e.g. p1=10, p2=15.
    This avoids starting at the middle of the output caps, which would be
    around p1=100, p2=125 for caps 200/250.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        p1_floor: float = 1e-3,
        p2_floor: float = 1e-3,
        p1_max: float = 200.0,
        p2_max: float = 250.0,
        init_p1: float = 10.0,
        init_p2: float = 15.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.p1_floor = float(p1_floor)
        self.p2_floor = float(p2_floor)
        self.p1_max = float(p1_max)
        self.p2_max = float(p2_max)
        self.init_p1 = float(init_p1)
        self.init_p2 = float(init_p2)

        h1 = int(hidden_dim)
        h2 = max(64, h1)
        h3 = max(64, h1 // 2)
        h4 = max(32, h1 // 4)

        self.output_layer = nn.Linear(h4, 2)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, h3),
            nn.LayerNorm(h3),
            nn.SiLU(),
            nn.Linear(h3, h4),
            nn.SiLU(),
            self.output_layer,
        )

        self.initialize_output_to_nominal(self.init_p1, self.init_p2)

    @staticmethod
    def _target_to_logit(target: float, floor: float, cap: float) -> float:
        """Convert desired bounded output value into the required sigmoid logit."""
        if cap <= floor:
            raise ValueError(f"cap must be larger than floor, got floor={floor}, cap={cap}")
        y = (float(target) - float(floor)) / (float(cap) - float(floor))
        y = min(max(y, 1e-6), 1.0 - 1e-6)
        return float(np.log(y / (1.0 - y)))

    def initialize_output_to_nominal(self, init_p1: float = 10.0, init_p2: float = 15.0):
        """
        Shift the final-layer bias so z_norm=0 gives the desired nominal p1,p2.

        We do not zero the final-layer weights, because keeping random weights
        gives the hidden layers gradients immediately. Instead, we compensate the
        current raw output at z_norm=0 by shifting only the final bias.
        """
        target_logits = torch.tensor(
            [
                self._target_to_logit(init_p1, self.p1_floor, self.p1_max),
                self._target_to_logit(init_p2, self.p2_floor, self.p2_max),
            ],
            dtype=self.output_layer.bias.dtype,
            device=self.output_layer.bias.device,
        )

        with torch.no_grad():
            z0 = torch.zeros(1, self.input_dim, dtype=self.output_layer.bias.dtype, device=self.output_layer.bias.device)
            raw0 = self.net(z0).view(-1)
            self.output_layer.bias.add_(target_logits - raw0)

    def forward(self, z_norm: torch.Tensor) -> torch.Tensor:
        raw_out = self.net(z_norm)

        p1 = self.p1_floor + (self.p1_max - self.p1_floor) * torch.sigmoid(raw_out[:, 0:1])
        p2 = self.p2_floor + (self.p2_max - self.p2_floor) * torch.sigmoid(raw_out[:, 1:2])

        return torch.cat([p1, p2], dim=1)


# ---------------------------------------------------------------------
# NPZ helper kept here so eval_hard_qp_saved_states.py can import from this file
# ---------------------------------------------------------------------
REQUIRED_KEYS = [
    "q_arm",
    "dq_arm",
    "ddq_nominal_arm",
    "ddq_safe_arm",
    "gamma",
    "beta",
    "n_crit",
    "d_crit",
    "d_dot_crit",
    "tau_crit",
    "h_all",
    "Lf_h_all",
    "vrel_sq2_all",
    "Lg_psi_all",
    "pair_mask",
]


def load_npz_checked(path):
    data = np.load(path)
    for key in REQUIRED_KEYS:
        if key not in data.files:
            raise KeyError(f"Missing key '{key}' in {path}")
    return data


# ---------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------
def read_list_file(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def expand_scenario_inputs(inputs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in inputs:
        p = os.path.expanduser(raw)

        if os.path.isfile(p) and p.endswith(".txt"):
            out.extend(read_list_file(p))
            continue

        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.yaml"))))
            out.extend(sorted(glob.glob(os.path.join(p, "*.yml"))))
            continue

        matches = sorted(glob.glob(p))
        if matches:
            out.extend(matches)
        else:
            out.append(p)

    out = [os.path.abspath(os.path.expanduser(p)) for p in out if p.endswith((".yaml", ".yml"))]
    out = sorted(list(dict.fromkeys(out)))
    return out


def expand_rollout_inputs(inputs: Iterable[str]) -> list[str]:
    """Accept rollout NPZ files, directories, globs, or txt files containing NPZ paths."""
    out: list[str] = []
    for raw in inputs:
        p = os.path.expanduser(raw)

        if os.path.isfile(p) and p.endswith(".txt"):
            out.extend(read_list_file(p))
            continue

        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.npz"))))
            continue

        matches = sorted(glob.glob(p))
        if matches:
            out.extend(matches)
        else:
            out.append(p)

    out = [os.path.abspath(os.path.expanduser(p)) for p in out if p.endswith(".npz")]
    out = sorted(list(dict.fromkeys(out)))
    return out


def scenario_name_from_rollout_path(path: str) -> str:
    """scenario_0081_offline_rollout.npz -> scenario_0081"""
    stem = Path(path).stem
    if stem.endswith("_offline_rollout"):
        stem = stem[: -len("_offline_rollout")]
    return stem


def build_scenario_map(scenario_paths: list[str]) -> dict[str, str]:
    return {Path(p).stem: p for p in scenario_paths}




def read_jsonl_file(path: str) -> list[dict]:
    """Read a JSONL file into a list of dict rows."""
    rows: list[dict] = []
    with open(os.path.expanduser(path), "r") as f:
        for line_i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_i} is not valid JSONL: {exc}") from exc
            rows.append(row)
    return rows


def normalize_rollout_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def load_critical_window_map(path: str | None) -> dict[str, dict]:
    """
    Load critical-window metadata keyed by absolute rollout_path.

    Rows are expected to contain:
        rollout_path, critical_start_min, critical_start_max,
        found_unsafe, critical_window_valid
    """
    if path is None:
        return {}

    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"--critical-windows file not found: {path}")

    out: dict[str, dict] = {}
    for row in read_jsonl_file(p):
        rollout_path = row.get("rollout_path") or row.get("output_path") or row.get("path")
        if rollout_path is None:
            continue
        key = normalize_rollout_path(str(rollout_path))
        out[key] = row
    return out


def rollout_record_from_path(path: str, category: str = "unknown", meta: dict | None = None) -> dict:
    rec = {
        "rollout_path": normalize_rollout_path(path),
        "category": category,
    }
    if meta:
        rec.update(meta)
    return rec


def expand_labeled_rollout_inputs(inputs: Iterable[str], critical_windows_path: str | None = None) -> list[dict]:
    """
    Accept labeled JSONL files, plain txt files, NPZs, directories, or globs.

    JSONL rows may contain rollout_path or output_path plus category.
    For bad_safety / bad_goal rows, critical_start_min/max are attached from
    --critical-windows when available.
    """
    critical_map = load_critical_window_map(critical_windows_path)
    records: list[dict] = []

    for raw in inputs:
        p = os.path.expanduser(raw)

        # Labeled JSONL file.
        if os.path.isfile(p) and p.endswith(".jsonl"):
            for row in read_jsonl_file(p):
                rollout_path = row.get("rollout_path") or row.get("output_path") or row.get("path")
                if rollout_path is None:
                    raise KeyError(f"{p} row has no rollout_path/output_path/path field: {row}")
                cat = str(row.get("category", "unknown"))
                rec = rollout_record_from_path(str(rollout_path), category=cat, meta=row)
                records.append(rec)
            continue

        # Plain text list of rollout paths.
        if os.path.isfile(p) and p.endswith(".txt"):
            for path in read_list_file(p):
                records.append(rollout_record_from_path(path, category="unknown"))
            continue

        # Directory or glob / npz path.
        for path in expand_rollout_inputs([p]):
            records.append(rollout_record_from_path(path, category="unknown"))

    # Attach critical windows where available.
    out: list[dict] = []
    seen: set[str] = set()
    for rec in records:
        rp = normalize_rollout_path(rec["rollout_path"])
        if rp in seen:
            continue
        seen.add(rp)
        rec["rollout_path"] = rp
        cat = str(rec.get("category", "unknown"))

        cw = critical_map.get(rp)
        if cw is not None:
            rec.setdefault("critical_window_valid", cw.get("critical_window_valid"))
            rec.setdefault("found_unsafe", cw.get("found_unsafe"))
            rec.setdefault("first_unsafe_step", cw.get("first_unsafe_step"))
            rec.setdefault("unsafe_reason", cw.get("unsafe_reason"))
            rec.setdefault("critical_start_min", cw.get("critical_start_min"))
            rec.setdefault("critical_start_max", cw.get("critical_start_max"))

        # Only bad_safety / bad_goal are allowed to override the global start range.
        if cat not in ("bad_safety", "bad_qp", "bad_goal"):
            rec.pop("critical_start_min", None)
            rec.pop("critical_start_max", None)
            rec.pop("critical_window_valid", None)

        out.append(rec)

    return out


def summarize_rollout_records(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        cat = str(rec.get("category", "unknown"))
        counts[cat] = counts.get(cat, 0) + 1
    return counts

def sample_start_state_from_rollout(
    rollout_files: list,
    scenario_map: dict[str, str],
    horizon: int,
    start_step_min: int,
    start_step_max: int | None,
    rng: random.Random,
) -> tuple[str, dict]:
    """
    Sample one rollout file/record and one start index.

    rollout_files may be either:
      - list[str] of NPZ paths, or
      - list[dict] with rollout_path, category, and optionally
        critical_start_min/critical_start_max.

    For records with category bad_safety / bad_goal and a valid critical window,
    the per-rollout critical range overrides global start_step_min/max.
    """
    if not rollout_files:
        raise RuntimeError("sample_from_rollouts=True but no rollout files were provided.")

    entry = rng.choice(rollout_files)
    if isinstance(entry, dict):
        rollout_path = normalize_rollout_path(entry["rollout_path"])
        category = str(entry.get("category", "unknown"))
        use_critical = (
            category in ("bad_safety", "bad_qp", "bad_goal")
            and entry.get("critical_start_min") is not None
            and entry.get("critical_start_max") is not None
            and bool(entry.get("critical_window_valid", True))
        )
        if use_critical:
            local_start_step_min = int(entry["critical_start_min"])
            local_start_step_max = int(entry["critical_start_max"])
            start_source = "critical_window"
        else:
            local_start_step_min = int(start_step_min)
            local_start_step_max = start_step_max
            start_source = "global_window"
    else:
        rollout_path = normalize_rollout_path(str(entry))
        category = "unknown"
        local_start_step_min = int(start_step_min)
        local_start_step_max = start_step_max
        start_source = "global_window"

    scenario_name = scenario_name_from_rollout_path(rollout_path)
    if scenario_name not in scenario_map:
        raise KeyError(
            f"Could not find matching scenario YAML for rollout {rollout_path}. "
            f"Expected scenario named {scenario_name}."
        )

    z = np.load(rollout_path)
    n = int(z["q_arm"].shape[0])
    if n <= 0:
        raise RuntimeError(f"Rollout has no samples: {rollout_path}")

    lo = max(0, int(local_start_step_min))
    hi_from_horizon = max(0, n - int(horizon) - 1)
    hi = hi_from_horizon if local_start_step_max is None else min(int(local_start_step_max), hi_from_horizon)
    if hi < lo:
        lo = 0
        hi = max(0, n - int(horizon) - 1)
        start_source = start_source + "_fallback_full_rollout"

    k0 = rng.randint(lo, hi) if hi > lo else lo

    start_state = {
        "rollout_path": rollout_path,
        "scenario_name": scenario_name,
        "start_step": int(k0),
        "start_step_min_used": int(lo),
        "start_step_max_used": int(hi),
        "start_source": start_source,
        "category": category,
        "q_arm": np.asarray(z["q_arm"][k0], dtype=float).copy(),
        "dq_arm": np.asarray(z["dq_arm"][k0], dtype=float).copy(),
    }
    return scenario_map[scenario_name], start_state


# ---------------------------------------------------------------------
# batched full-QP construction in torch
# ---------------------------------------------------------------------
def build_collision_rows_torch(
    p12: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
    vrel_sq2_all: torch.Tensor,
    Lg_psi_all: torch.Tensor,
    pair_mask: torch.Tensor,
):
    """
    Build A_coll u >= b_coll for a whole batch.

    Shapes:
      p12         : (B, 2)
      h_all       : (B, 43)
      Lf_h_all    : (B, 43)
      vrel_sq2_all: (B, 43)
      Lg_psi_all  : (B, 43, 7)
      pair_mask   : (B, 43)
    """
    p1 = p12[:, 0:1]
    p2 = p12[:, 1:2]

    psi_all = Lf_h_all + p1 * h_all
    Lf_psi_all = vrel_sq2_all + p1 * Lf_h_all

    A_coll = Lg_psi_all
    b_coll = -Lf_psi_all - p2 * psi_all
    mask_coll = pair_mask

    return A_coll, b_coll, mask_coll


def build_bound_rows_torch(
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    dt: float,
    ddq_min_arm: torch.Tensor,
    ddq_max_arm: torch.Tensor,
    dq_min_arm: torch.Tensor,
    dq_max_arm: torch.Tensor,
    q_min_arm: torch.Tensor,
    q_max_arm: torch.Tensor,
):
    """
    Build fixed 42 joint-bound rows for the whole batch.

    Returns:
      A_bounds: (B, 42, 7)
      b_bounds: (B, 42)
      mask_bounds: (B, 42)
    """
    B = q_arm.shape[0]
    device = q_arm.device
    dtype = q_arm.dtype

    I = torch.eye(NUM_ARM_JOINTS, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    b_acc_low = ddq_min_arm.unsqueeze(0).expand(B, -1)
    b_acc_high = -ddq_max_arm.unsqueeze(0).expand(B, -1)

    b_vel_low = (dq_min_arm.unsqueeze(0) - dq_arm) / dt
    b_vel_high = -(dq_max_arm.unsqueeze(0) - dq_arm) / dt

    b_pos_low = 2.0 * (q_min_arm.unsqueeze(0) - q_arm - dq_arm * dt) / (dt ** 2)
    b_pos_high = -2.0 * (q_max_arm.unsqueeze(0) - q_arm - dq_arm * dt) / (dt ** 2)

    A_bounds = torch.cat(
        [I, -I, I, -I, I, -I],
        dim=1,
    )
    b_bounds = torch.cat(
        [b_acc_low, b_acc_high, b_vel_low, b_vel_high, b_pos_low, b_pos_high],
        dim=1,
    )
    mask_bounds = torch.ones(B, NUM_BOUND_ROWS, device=device, dtype=dtype)

    return A_bounds, b_bounds, mask_bounds


def build_full_qp_torch_batch(
    p12: torch.Tensor,
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
    vrel_sq2_all: torch.Tensor,
    Lg_psi_all: torch.Tensor,
    pair_mask: torch.Tensor,
    dt: float,
    ddq_min_arm: torch.Tensor,
    ddq_max_arm: torch.Tensor,
    dq_min_arm: torch.Tensor,
    dq_max_arm: torch.Tensor,
    q_min_arm: torch.Tensor,
    q_max_arm: torch.Tensor,
):
    A_coll, b_coll, mask_coll = build_collision_rows_torch(
        p12=p12,
        h_all=h_all,
        Lf_h_all=Lf_h_all,
        vrel_sq2_all=vrel_sq2_all,
        Lg_psi_all=Lg_psi_all,
        pair_mask=pair_mask,
    )

    A_bounds, b_bounds, mask_bounds = build_bound_rows_torch(
        q_arm=q_arm,
        dq_arm=dq_arm,
        dt=dt,
        ddq_min_arm=ddq_min_arm,
        ddq_max_arm=ddq_max_arm,
        dq_min_arm=dq_min_arm,
        dq_max_arm=dq_max_arm,
        q_min_arm=q_min_arm,
        q_max_arm=q_max_arm,
    )

    A_full = torch.cat([A_coll, A_bounds], dim=1)
    b_full = torch.cat([b_coll, b_bounds], dim=1)
    mask_full = torch.cat([mask_coll, mask_bounds], dim=1)

    # zero inactive rows so they become 0 >= 0
    A_full = A_full * mask_full.unsqueeze(-1)
    b_full = b_full * mask_full

    return A_full, b_full, mask_full


# ---------------------------------------------------------------------
# Pinocchio helper
# ---------------------------------------------------------------------
def ee_position(model, data, ee_frame_id: int, q_full: np.ndarray, dq_full: np.ndarray | None = None) -> np.ndarray:
    if dq_full is None:
        pin.forwardKinematics(model, data, q_full)
    else:
        pin.forwardKinematics(model, data, q_full, dq_full, np.zeros(model.nv))
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_frame_id].translation.copy()


# ---------------------------------------------------------------------
# obstacle helpers in torch
# ---------------------------------------------------------------------
def obstacles_numpy_to_torch(obstacles_np: list[dict], dtype=torch.double, device="cpu") -> list[dict]:
    out = []
    for obs in obstacles_np:
        out.append(
            {
                "pose_start": torch.tensor(obs["pose_start"], dtype=dtype, device=device),
                "pose_end": torch.tensor(obs["pose_end"], dtype=dtype, device=device),
                "radius": float(obs["radius"]),
                "velocity": torch.tensor(obs["velocity"], dtype=dtype, device=device),
                "name": obs.get("name", ""),
            }
        )
    return out


def clone_obstacles_torch(obstacles: list[dict]) -> list[dict]:
    out = []
    for obs in obstacles:
        out.append(
            {
                "pose_start": obs["pose_start"].clone(),
                "pose_end": obs["pose_end"].clone(),
                "radius": float(obs["radius"]),
                "velocity": obs["velocity"].clone(),
                "name": obs.get("name", ""),
            }
        )
    return out


def update_obstacles_torch(obstacles: list[dict], dt: float) -> list[dict]:
    updated = clone_obstacles_torch(obstacles)
    for obs in updated:
        obs["pose_start"] = obs["pose_start"] + obs["velocity"] * dt
        obs["pose_end"] = obs["pose_end"] + obs["velocity"] * dt
    return updated


# ---------------------------------------------------------------------
# nominal controller wrapper
# ---------------------------------------------------------------------
def compute_nominal_detached(
    pin_model,
    pin_data,
    ee_frame_id: int,
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    goal_np: np.ndarray,
    target_prev_np: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Uses the existing Pinocchio/numpy nominal controller.

    For this rollout-training file, nominal ddq is treated as an external
    detached command. Gradients still flow through the NN, DQP, geometry, and
    integrated q,dq trajectory.
    """
    q_np = q_arm.detach().cpu().numpy().astype(float)
    dq_np = dq_arm.detach().cpu().numpy().astype(float)

    ddq_nom_np, target_next_np = nominal_controller_js_standalone(
        model=pin_model,
        data=pin_data,
        ee_frame_id=ee_frame_id,
        q_arm_curr=q_np,
        dq_arm_curr=dq_np,
        target_ee_pos_cartesian=goal_np,
        target=target_prev_np,
    )
    ddq_nom = torch.tensor(ddq_nom_np, dtype=q_arm.dtype, device=q_arm.device)
    return ddq_nom, np.asarray(target_next_np, dtype=float)


# ---------------------------------------------------------------------
# rollout segment training
# ---------------------------------------------------------------------
def build_feature_vector(
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    ddq_nominal: torch.Tensor,
    ddq_nominal_scale: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
) -> torch.Tensor:
    """
    Build the 114D NN feature vector.

    Feature order:
        q_arm,
        dq_arm,
        ddq_nominal,
        ddq_nominal / ddq_scale,
        h_all,
        Lf_h_all
    """
    ddq_nominal_norm = ddq_nominal / torch.clamp(ddq_nominal_scale, min=1e-6)
    z = torch.cat([q_arm, dq_arm, ddq_nominal, ddq_nominal_norm, h_all, Lf_h_all], dim=0)
    if tuple(z.shape) != (INPUT_DIM,):
        raise RuntimeError(f"Feature shape mismatch: {tuple(z.shape)}")
    return z.unsqueeze(0)


def train_one_segment(
    scenario_path: str,
    pin_context: dict,
    torch_kin,
    model: P12ParamNet,
    bound_slack_layer,
    optimizer,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    limits_t: dict[str, torch.Tensor],
    cfg: RolloutTrainCfg,
    train: bool = True,
    start_state: dict | None = None,
) -> dict:
    """
    Train/evaluate one short rollout segment.
    Batch size is intentionally 1 in this first version.
    """
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    pin_model = pin_context["model"]
    pin_data = pin_model.createData()
    ee_frame_id = pin_context["ee_frame_id"]

    scenario = load_scenario_yaml(scenario_path)
    q_full_np, dq_full_np = get_default_initial_state(pin_model)

    start_step = 0
    start_rollout_path = ""
    if start_state is not None:
        start_step = int(start_state["start_step"])
        start_rollout_path = str(start_state.get("rollout_path", ""))
        q_full_np[:NUM_ARM_JOINTS] = np.asarray(start_state["q_arm"], dtype=float)
        dq_full_np[:NUM_ARM_JOINTS] = np.asarray(start_state["dq_arm"], dtype=float)

    q = torch.tensor(q_full_np[:NUM_ARM_JOINTS], dtype=torch.double)
    dq = torch.tensor(dq_full_np[:NUM_ARM_JOINTS], dtype=torch.double)

    obstacles = obstacles_numpy_to_torch(scenario["obstacles"], dtype=torch.double, device="cpu")

    # fr3_rollout.py updates obstacles before computing the first geometry row of a step.
    # If we start from saved step k, advance the stored obstacle state to time k*dt;
    # the loop update then moves it to (k+1)*dt before the first geometry evaluation.
    if start_step > 0:
        for obs in obstacles:
            obs["pose_start"] = obs["pose_start"] + obs["velocity"] * (float(start_step) * cfg.dt)
            obs["pose_end"] = obs["pose_end"] + obs["velocity"] * (float(start_step) * cfg.dt)

    goal_np = np.asarray(scenario["goal_ee_pos"], dtype=float)
    d_margin = float(scenario["d_margin"])

    # Match fr3_rollout convention: target smoothing starts from current EE pose.
    target_prev_np = ee_position(pin_model, pin_data, ee_frame_id, q_full_np, dq_full_np)

    loss_total = torch.zeros((), dtype=torch.double)
    loss_safety_total = torch.zeros((), dtype=torch.double)
    loss_h_safety_total = torch.zeros((), dtype=torch.double)
    loss_g1_safety_total = torch.zeros((), dtype=torch.double)
    loss_h_viol_freq_total = torch.zeros((), dtype=torch.double)
    loss_slack_total = torch.zeros((), dtype=torch.double)
    loss_goal_total = torch.zeros((), dtype=torch.double)
    loss_preg_total = torch.zeros((), dtype=torch.double)
    goal_t = torch.tensor(goal_np, dtype=torch.double)
    with torch.no_grad():
        ee_pos_initial = torch_kin.get_frame_position_torch(q, EE_FRAME_NAME)
        goal_initial_dist = float(torch.linalg.norm(ee_pos_initial - goal_t).detach().cpu().item())

    hard_viol_max_seen = 0.0
    pair_viol_max_seen = 0.0
    bound_viol_max_seen = 0.0
    slack_max_seen = 0.0
    goal_dist_sum = 0.0
    goal_final_dist = float("nan")
    min_h_seen = float("inf")
    min_psi_seen = float("inf")
    min_g2_seen = float("inf")
    h_sum = 0.0
    psi_sum = 0.0
    g2_sum = 0.0
    h_count = 0
    psi_count = 0
    g2_count = 0
    p1_sum = 0.0
    p2_sum = 0.0

    # Stop at scenario duration if horizon is too long.
    max_steps_from_scenario = int(float(scenario["max_sim_duration_s"]) / cfg.dt)
    H = min(int(cfg.horizon), max_steps_from_scenario)

    # Per-joint normalization scale for the new ddq_nominal / ddq_scale NN feature.
    ddq_nominal_scale = torch.maximum(
        torch.abs(limits_t["ddq_min_arm"]),
        torch.abs(limits_t["ddq_max_arm"]),
    )

    for k in range(H):
        if cfg.move_obstacles:
            obstacles = update_obstacles_torch(obstacles, cfg.dt)
        else:
            obstacles = clone_obstacles_torch(obstacles)

        # Recompute differentiable geometry from current q,dq.
        robot_links = torch_kin.get_active_link_endpoint_data_torch(q, create_graph=train)
        pair_terms = compute_all_pair_terms_torch(
            robot_links=robot_links,
            obstacles=obstacles,
            dq=dq,
            d_margin=d_margin,
        )

        # Detached nominal command for this rollout-training version.
        ddq_nominal, target_prev_np = compute_nominal_detached(
            pin_model=pin_model,
            pin_data=pin_data,
            ee_frame_id=ee_frame_id,
            q_arm=q,
            dq_arm=dq,
            goal_np=goal_np,
            target_prev_np=target_prev_np,
        )

        # NN features and p12.
        z_raw = build_feature_vector(
            q_arm=q,
            dq_arm=dq,
            ddq_nominal=ddq_nominal,
            ddq_nominal_scale=ddq_nominal_scale,
            h_all=pair_terms["h_all"],
            Lf_h_all=pair_terms["Lf_h_all"],
        )
        z_norm = normalize_features_torch(z_raw, feat_mean, feat_std)
        p12 = model(z_norm)

        A_full, b_full, _ = build_full_qp_torch_batch(
            p12=p12,
            q_arm=q.unsqueeze(0),
            dq_arm=dq.unsqueeze(0),
            h_all=pair_terms["h_all"].unsqueeze(0),
            Lf_h_all=pair_terms["Lf_h_all"].unsqueeze(0),
            vrel_sq2_all=pair_terms["vrel_sq2_all"].unsqueeze(0),
            Lg_psi_all=pair_terms["Lg_psi_all"].unsqueeze(0),
            pair_mask=pair_terms["pair_mask"].unsqueeze(0),
            dt=cfg.dt,
            ddq_min_arm=limits_t["ddq_min_arm"],
            ddq_max_arm=limits_t["ddq_max_arm"],
            dq_min_arm=limits_t["dq_min_arm"],
            dq_max_arm=limits_t["dq_max_arm"],
            q_min_arm=limits_t["q_min_arm"],
            q_max_arm=limits_t["q_max_arm"],
        )

        # Bound/input-slack DQP:
        #   - pair/HOCBF rows hard
        #   - bound/input rows soft
        try:
            u_safe, s_bound, _, _ = solve_bound_slack_qp_batch(
                layer=bound_slack_layer,
                u_nom=ddq_nominal.unsqueeze(0),
                A_full=A_full,
                b_full=b_full,
                num_pair_rows=NUM_PAIR_ROWS,
                eps=cfg.dqp_row_norm_eps,
            )
        except Exception as exc:
            active = pair_terms["pair_mask"] > 0.5
            min_h_dbg = (
                float(pair_terms["h_all"][active].min().detach().cpu().item())
                if bool(torch.any(active).detach().cpu().item())
                else float("nan")
            )
            psi_dbg = pair_terms["Lf_h_all"] + p12[0, 0] * pair_terms["h_all"]
            min_psi_dbg = (
                float(psi_dbg[active].min().detach().cpu().item())
                if bool(torch.any(active).detach().cpu().item())
                else float("nan")
            )
            return {
                "solver_failed": True,
                "fail_reason": repr(exc),
                "fail_step": int(k),
                "loss": float("nan"),
                "safety": float("nan"),
                "h_safety": float("nan"),
                "g1_safety": float("nan"),
                "slack": float("nan"),
                "goal": float("nan"),
                "goal_initial": float("nan"),
                "goal_final": float("nan"),
                "goal_dist_mean": float("nan"),
                "goal_progress_ratio": float("nan"),
                "normalized_mean_goal_distance": float("nan"),
                "goal_delta": float("nan"),
                "preg": float("nan"),
                "hard_viol_max": float("nan"),
                "pair_viol_max": float("nan"),
                "bound_viol_max": float("nan"),
                "viol_max": float("nan"),
                "slack_max": float("nan"),
                "min_h": min_h_dbg,
                "min_psi": min_psi_dbg,
                "min_g2": float("nan"),
                "h_mean": float("nan"),
                "g1_mean": float("nan"),
                "g2_mean": float("nan"),
                "p1_mean": float(p12[0, 0].detach().cpu().item()),
                "p2_mean": float(p12[0, 1].detach().cpu().item()),
                "horizon": H,
                "scenario": Path(scenario_path).stem,
                "start_step": start_step,
                "start_rollout_path": start_rollout_path,
            }

        hard_viol = hard_violation_batch(
            u=u_safe,
            A_full_raw=A_full,
            b_full_raw=b_full,
        )
        pair_viol = pair_violation_batch(
            u=u_safe,
            A_full_raw=A_full,
            b_full_raw=b_full,
            num_pair_rows=NUM_PAIR_ROWS,
        )
        bound_viol = bound_violation_batch(
            u=u_safe,
            A_full_raw=A_full,
            b_full_raw=b_full,
            num_pair_rows=NUM_PAIR_ROWS,
        )

        # Pre-step G2/HOCBF margin for the actually applied acceleration.
        # Pair rows are A_pair u >= b_pair, so the G2 margin is A_pair u - b_pair.
        # This is a diagnostic/checkpoint statistic; h and psi below are post-step.
        active_pre = pair_terms["pair_mask"] > 0.5
        if bool(torch.any(active_pre).detach().cpu().item()):
            A_pair = A_full[0, :NUM_PAIR_ROWS, :]
            b_pair = b_full[0, :NUM_PAIR_ROWS]
            g2_margin_all = torch.mv(A_pair, u_safe[0]) - b_pair
            g2_active = g2_margin_all[active_pre]
            min_g2_seen = min(min_g2_seen, float(g2_active.min().detach().cpu().item()))
            g2_sum += float(g2_active.sum().detach().cpu().item())
            g2_count += int(g2_active.numel())

        # Pair-row violation is only a diagnostic now. Since pair/HOCBF rows
        # are hard in the DQP, it is not a useful training safety loss.
        # Bound/input violations are represented through bound slack.
        if cfg.slack_huber_delta > 0.0:
            abs_s = s_bound.abs()
            # Huber-like slack penalty:
            # small slack -> quadratic, large slack -> linear gradient.
            # This keeps slack important but prevents rare huge slack values
            # from dominating the whole update and destroying p1/p2.
            loss_slack = torch.where(
                abs_s <= cfg.slack_huber_delta,
                s_bound.pow(2),
                2.0 * cfg.slack_huber_delta * abs_s - cfg.slack_huber_delta**2,
            ).mean()
        else:
            loss_slack = mean_sq_bound_slack(s_bound)
        loss_preg = (p12 ** 2).mean()

        # Differentiable dynamics integration.
        # For training we integrate the bound-slack solution directly.
        # Runtime/final evaluation still uses hard QP + fallback.
        u = u_safe[0]
        q_next = q + dq * cfg.dt + 0.5 * u * (cfg.dt ** 2)
        dq_next = dq + u * cfg.dt

        # Post-step h and G1/psi top-k safety losses.
        # This is the actual safety signal used for training: after applying
        # the learned p1,p2 through the DQP and integrating one step, h and
        # psi = Lf_h + p1*h should remain above their chosen margins.
        post_robot_links = torch_kin.get_active_link_endpoint_data_torch(q_next, create_graph=train)
        post_pair_terms = compute_all_pair_terms_torch(
            robot_links=post_robot_links,
            obstacles=obstacles,
            dq=dq_next,
            d_margin=d_margin,
        )
        post_active = post_pair_terms["pair_mask"] > 0.5
        psi_post_all = post_pair_terms["Lf_h_all"] + p12[0, 0] * post_pair_terms["h_all"]
        if bool(torch.any(post_active).detach().cpu().item()):
            h_post_active = post_pair_terms["h_all"][post_active]
            psi_post_active = psi_post_all[post_active]

            h_post_violation = torch.relu(cfg.h_safety_margin - h_post_active)
            g1_post_violation = torch.relu(cfg.g1_safety_margin - psi_post_active)

            loss_h_safety = topk_positive_violation_sq_loss(
                h_post_violation, frac=cfg.safety_topk_frac
            )
            loss_g1_safety = topk_positive_violation_sq_loss(
                g1_post_violation, frac=cfg.safety_topk_frac
            )
            loss_h_viol_freq = violation_frequency_loss_from_sq(
                h_post_violation ** 2,
                eps=cfg.viol_freq_eps,
            )
        else:
            loss_h_safety = torch.zeros((), dtype=torch.double, device=q.device)
            loss_g1_safety = torch.zeros((), dtype=torch.double, device=q.device)
            loss_h_viol_freq = torch.zeros((), dtype=torch.double, device=q.device)

        loss_safety = loss_h_safety + loss_g1_safety

        # Cartesian task loss on the post-step end-effector position.
        ee_pos_next = torch_kin.get_frame_position_torch(q_next, EE_FRAME_NAME)
        goal_err = ee_pos_next - goal_t
        loss_goal = torch.sum(goal_err ** 2)
        goal_dist = torch.linalg.norm(goal_err)

        step_loss = (
            cfg.w_safety * loss_h_safety
            + cfg.w_g1_safety * loss_g1_safety
            + cfg.w_viol_freq * loss_h_viol_freq
            + cfg.w_slack * loss_slack
            + cfg.w_goal * loss_goal
            + cfg.w_preg * loss_preg
        )

        loss_total = loss_total + step_loss
        loss_safety_total = loss_safety_total + loss_safety
        loss_h_safety_total = loss_h_safety_total + loss_h_safety
        loss_g1_safety_total = loss_g1_safety_total + loss_g1_safety
        loss_h_viol_freq_total = loss_h_viol_freq_total + loss_h_viol_freq
        loss_slack_total = loss_slack_total + loss_slack
        loss_goal_total = loss_goal_total + loss_goal
        loss_preg_total = loss_preg_total + loss_preg

        hard_viol_max_seen = max(hard_viol_max_seen, float(hard_viol.max().detach().cpu().item()))
        pair_viol_max_seen = max(pair_viol_max_seen, float(pair_viol.max().detach().cpu().item()))
        bound_viol_max_seen = max(bound_viol_max_seen, float(bound_viol.max().detach().cpu().item()))
        slack_max_seen = max(slack_max_seen, float(s_bound.max().detach().cpu().item()))
        goal_dist_value = float(goal_dist.detach().cpu().item())
        goal_dist_sum += goal_dist_value
        goal_final_dist = goal_dist_value
        p1_sum += float(p12[0, 0].detach().cpu().item())
        p2_sum += float(p12[0, 1].detach().cpu().item())

        # Diagnostics/checkpoint statistics use post-step h and psi, not pair-row
        # QP residuals. This matches the new safety loss.
        if bool(torch.any(post_active).detach().cpu().item()):
            h_post_active = post_pair_terms["h_all"][post_active]
            psi_post_active = psi_post_all[post_active]
            min_h_seen = min(min_h_seen, float(h_post_active.min().detach().cpu().item()))
            min_psi_seen = min(min_psi_seen, float(psi_post_active.min().detach().cpu().item()))
            h_sum += float(h_post_active.sum().detach().cpu().item())
            psi_sum += float(psi_post_active.sum().detach().cpu().item())
            h_count += int(h_post_active.numel())
            psi_count += int(psi_post_active.numel())

        q, dq = q_next, dq_next

    loss_total = loss_total / float(max(1, H))
    loss_safety_total = loss_safety_total / float(max(1, H))
    loss_h_safety_total = loss_h_safety_total / float(max(1, H))
    loss_g1_safety_total = loss_g1_safety_total / float(max(1, H))
    loss_h_viol_freq_total = loss_h_viol_freq_total / float(max(1, H))
    loss_slack_total = loss_slack_total / float(max(1, H))
    loss_goal_total = loss_goal_total / float(max(1, H))
    loss_preg_total = loss_preg_total / float(max(1, H))

    if train:
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

    goal_mean_dist = float(goal_dist_sum / float(max(1, H)))
    goal_progress_ratio = float(goal_final_dist / max(goal_initial_dist, cfg.goal_progress_eps))
    normalized_mean_goal_distance = float(goal_mean_dist / max(goal_initial_dist, cfg.goal_progress_eps))
    goal_delta = float(goal_final_dist - goal_initial_dist)

    return {
        "solver_failed": False,
        "fail_reason": "",
        "fail_step": -1,
        "loss": float(loss_total.detach().cpu().item()),
        "safety": float(loss_safety_total.detach().cpu().item()),
        "h_safety": float(loss_h_safety_total.detach().cpu().item()),
        "h_viol_freq": float(loss_h_viol_freq_total.detach().cpu().item()),
        "g1_safety": float(loss_g1_safety_total.detach().cpu().item()),
        "slack": float(loss_slack_total.detach().cpu().item()),
        "goal": float(loss_goal_total.detach().cpu().item()),
        "goal_initial": float(goal_initial_dist),
        "goal_final": float(goal_final_dist),
        "goal_dist_mean": goal_mean_dist,
        "goal_progress_ratio": goal_progress_ratio,
        "normalized_mean_goal_distance": normalized_mean_goal_distance,
        "goal_delta": goal_delta,
        "preg": float(loss_preg_total.detach().cpu().item()),
        "hard_viol_max": hard_viol_max_seen,
        "pair_viol_max": pair_viol_max_seen,
        "bound_viol_max": bound_viol_max_seen,
        "viol_max": hard_viol_max_seen,  # backwards-compatible printed metric
        "slack_max": slack_max_seen,
        "min_h": min_h_seen,
        "min_psi": min_psi_seen,
        "min_g2": min_g2_seen,
        "h_mean": float(h_sum / float(max(1, h_count))),
        "g1_mean": float(psi_sum / float(max(1, psi_count))),
        "g2_mean": float(g2_sum / float(max(1, g2_count))),
        "p1_mean": p1_sum / float(max(1, H)),
        "p2_mean": p2_sum / float(max(1, H)),
        "mean_segment_time_s": float(H) * float(cfg.dt),
        "horizon": H,
        "scenario": Path(scenario_path).stem,
        "start_step": start_step,
        "start_rollout_path": start_rollout_path,
    }


# ---------------------------------------------------------------------
# epoch aggregation
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="Scenario YAML(s), directories, globs, or txt files")
    parser.add_argument("--stats-in", type=str, required=True, help="Feature stats .npz for the 114D input")
    parser.add_argument("--init-model", type=str, default=None, help="Optional checkpoint to warm-start from")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--segments-per-epoch", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)

    parser.add_argument("--rollout-dir", type=str, default=None, help="Optional directory/list/glob of good rollout NPZs for random start windows")
    parser.add_argument("--val-rollout-dir", type=str, default=None, help="Optional held-out directory/list/glob of rollout NPZs for validation start windows")
    parser.add_argument("--rollout-labeled", type=str, default=None, help="Optional labeled JSONL for training rollout records. Overrides --rollout-dir when set.")
    parser.add_argument("--val-rollout-labeled", type=str, default=None, help="Optional labeled JSONL for validation rollout records. Overrides --val-rollout-dir when set.")
    parser.add_argument("--critical-windows", type=str, default=None, help="Optional JSONL with critical_start_min/max for bad_safety and bad_goal rollouts.")
    parser.add_argument("--sample-start-from-rollout", action="store_true", help="Start each training segment from a random q,dq sampled from generated good rollouts")
    parser.add_argument("--start-step-min", type=int, default=0)
    parser.add_argument("--start-step-max", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--p1-floor", type=float, default=1e-3)
    parser.add_argument("--p2-floor", type=float, default=1e-3)
    parser.add_argument("--p1-max", type=float, default=200.0)
    parser.add_argument("--p2-max", type=float, default=250.0)
    parser.add_argument("--init-p1", type=float, default=10.0, help="Initial p1/gamma output at zero normalized input")
    parser.add_argument("--init-p2", type=float, default=15.0, help="Initial p2/beta output at zero normalized input")
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)

    parser.add_argument("--rho", type=float, default=1e5)
    parser.add_argument("--dqp-row-norm-eps", type=float, default=1e-6)
    parser.add_argument("--w-safety", type=float, default=100.0, help="Weight for post-step h top-k safety loss")
    parser.add_argument("--w-g1-safety", type=float, default=10.0, help="Weight for post-step G1/psi top-k safety loss")
    parser.add_argument("--w-viol-freq", type=float, default=0.0, help="Weight for post-step h violation-frequency loss")
    parser.add_argument("--viol-freq-eps", type=float, default=1e-6, help="Epsilon/sharpness scale for h violation-frequency loss")
    parser.add_argument("--w-slack", type=float, default=1.0)
    parser.add_argument(
        "--slack-huber-delta",
        type=float,
        default=0.0,
        help="Huber delta for slack loss. 0 disables Huber and uses squared slack.",
    )
    parser.add_argument("--w-goal", type=float, default=0.0)
    parser.add_argument("--w-preg", type=float, default=0.0)
    parser.add_argument("--h-safety-margin", type=float, default=0.0, help="Post-step h margin used in top-k ReLU(margin - h_next)^2")
    parser.add_argument("--g1-safety-margin", type=float, default=0.0, help="Post-step G1/psi margin used in top-k ReLU(margin - psi_next)^2")
    parser.add_argument("--safety-topk-frac", type=float, default=0.05, help="Fraction of positive safety violations used by the top-k h/G1 losses")
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-segments", type=int, default=5)
    parser.add_argument("--safe-pair-viol-tol", type=float, default=1e-6)
    parser.add_argument("--safe-min-h-tol", type=float, default=1e-6)
    parser.add_argument("--checkpoint-tol", type=float, default=1e-9)
    parser.add_argument(
        "--track-metric",
        type=str,
        default="normalized_mean_goal_distance",
        choices=["goal_progress_ratio", "normalized_mean_goal_distance", "goal_delta", "goal_final", "goal_dist_mean"],
        help="Metric used for best_track_given_safe; lower is better.",
    )
    parser.add_argument("--goal-progress-eps", type=float, default=1e-3)
    parser.add_argument("--safe-candidate-dir", type=str, default="safe_candidates")
    parser.add_argument("--safe-candidate-csv", type=str, default="safe_candidates_summary.csv")
    parser.add_argument("--safe-candidate-txt", type=str, default="safe_candidates_summary.txt")
    parser.add_argument("--best-safe-out", type=str, default=None, help="Deprecated/ignored: safe candidates are saved instead of best_safe")
    parser.add_argument("--best-track-out", type=str, default=None, help="Deprecated/ignored: safe candidates are saved instead of one best_track checkpoint")

    parser.add_argument("--move-obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-scenarios", type=int, default=None)

    parser.add_argument("--model-out", type=str, default="fr3_p12_rollout_latest.pt")
    parser.add_argument("--log-jsonl", type=str, default="fr3_train_rollout_log.jsonl")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output checkpoint/log files to be overwritten. Default: refuse to start if outputs exist.",
    )

    args = parser.parse_args()

    cfg = RolloutTrainCfg(
        seed=args.seed,
        epochs=args.epochs,
        segments_per_epoch=args.segments_per_epoch,
        horizon=args.horizon,
        sample_from_rollouts=bool(args.sample_start_from_rollout),
        start_step_min=args.start_step_min,
        start_step_max=args.start_step_max,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        p1_floor=args.p1_floor,
        p2_floor=args.p2_floor,
        p1_max=args.p1_max,
        p2_max=args.p2_max,
        init_p1=args.init_p1,
        init_p2=args.init_p2,
        dt=args.dt,
        w_safety=args.w_safety,
        w_g1_safety=args.w_g1_safety,
        w_viol_freq=args.w_viol_freq,
        viol_freq_eps=args.viol_freq_eps,
        w_slack=args.w_slack,
        w_goal=args.w_goal,
        w_preg=args.w_preg,
        h_safety_margin=args.h_safety_margin,
        g1_safety_margin=args.g1_safety_margin,
        safety_topk_frac=args.safety_topk_frac,
        slack_huber_delta=args.slack_huber_delta,
        rho=args.rho,
        dqp_row_norm_eps=args.dqp_row_norm_eps,
        move_obstacles=args.move_obstacles,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        val_segments=args.val_segments,
        val_rollout_dir=args.val_rollout_dir,
        safe_pair_viol_tol=args.safe_pair_viol_tol,
        safe_min_h_tol=args.safe_min_h_tol,
        checkpoint_tol=args.checkpoint_tol,
        track_metric=args.track_metric,
        goal_progress_eps=args.goal_progress_eps,
        model_out=args.model_out,
        safe_candidate_dir=args.safe_candidate_dir,
        safe_candidate_csv=args.safe_candidate_csv,
        safe_candidate_txt=args.safe_candidate_txt,
        log_jsonl=args.log_jsonl,
    )

    guard_against_accidental_overwrite(
        output_paths=[cfg.model_out, cfg.safe_candidate_csv, cfg.safe_candidate_txt, cfg.log_jsonl],
        overwrite=bool(args.overwrite),
    )
    Path(cfg.safe_candidate_dir).expanduser().mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_default_dtype(torch.float64)

    scenario_paths = expand_scenario_inputs(args.scenarios)
    if args.limit_scenarios is not None:
        scenario_paths = scenario_paths[: int(args.limit_scenarios)]
    if not scenario_paths:
        raise RuntimeError("No scenario YAML files found.")

    print("num_scenarios =", len(scenario_paths))
    for p in scenario_paths[:10]:
        print("  ", p)
    if len(scenario_paths) > 10:
        print("  ...")

    scenario_map = build_scenario_map(scenario_paths)
    rollout_files: list = []
    val_rollout_files: list = []
    if args.sample_start_from_rollout:
        if args.rollout_labeled is None and args.rollout_dir is None:
            raise ValueError("--sample-start-from-rollout requires --rollout-dir or --rollout-labeled")

        if args.rollout_labeled is not None:
            rollout_files = expand_labeled_rollout_inputs([args.rollout_labeled], critical_windows_path=args.critical_windows)
            train_source = args.rollout_labeled
        else:
            rollout_files = expand_labeled_rollout_inputs([args.rollout_dir], critical_windows_path=args.critical_windows)
            train_source = args.rollout_dir

        if not rollout_files:
            raise RuntimeError(f"No rollout records found in training source {train_source}")

        print("num_train_rollout_records for start sampling =", len(rollout_files))
        print("train rollout categories =", summarize_rollout_records(rollout_files))
        for rec in rollout_files[:10]:
            print(
                "  train rollout",
                rec["category"],
                Path(rec["rollout_path"]).name,
                "critical=" + str(rec.get("critical_start_min")) + ":" + str(rec.get("critical_start_max")),
            )
        if len(rollout_files) > 10:
            print("  ...")

        if args.val_rollout_labeled is not None:
            val_rollout_files = expand_labeled_rollout_inputs([args.val_rollout_labeled], critical_windows_path=args.critical_windows)
            val_source = args.val_rollout_labeled
        elif args.val_rollout_dir is not None:
            val_rollout_files = expand_labeled_rollout_inputs([args.val_rollout_dir], critical_windows_path=args.critical_windows)
            val_source = args.val_rollout_dir
        else:
            val_rollout_files = rollout_files
            val_source = train_source
            print("WARNING: no validation rollout source set; validation samples from training rollout records.")

        if not val_rollout_files:
            raise RuntimeError(f"No rollout records found in validation source {val_source}")

        print("num_val_rollout_records for held-out validation =", len(val_rollout_files))
        print("val rollout categories =", summarize_rollout_records(val_rollout_files))
        for rec in val_rollout_files[:10]:
            print(
                "  val rollout",
                rec["category"],
                Path(rec["rollout_path"]).name,
                "critical=" + str(rec.get("critical_start_min")) + ":" + str(rec.get("critical_start_max")),
            )
        if len(val_rollout_files) > 10:
            print("  ...")

    feat_mean, feat_std = load_feature_stats(args.stats_in)
    print("stats_in =", args.stats_in)
    print("feature dim =", feat_mean.numel())

    # Pinocchio context is used for nominal controller and initial state.
    pin_context = make_rollout_context()

    # Torch kinematics is used for differentiable geometry.
    torch_kin = load_fr3_torch_kinematics()
    print("torch URDF =", torch_kin.urdf_path)

    model = P12ParamNet(
        input_dim=INPUT_DIM,
        hidden_dim=cfg.hidden_dim,
        p1_floor=cfg.p1_floor,
        p2_floor=cfg.p2_floor,
        p1_max=cfg.p1_max,
        p2_max=cfg.p2_max,
        init_p1=cfg.init_p1,
        init_p2=cfg.init_p2,
    ).double()
    maybe_load_model_weights(model, args.init_model)
    if args.init_model is not None:
        print("warm-started from", args.init_model)

    with torch.no_grad():
        z0 = torch.zeros(1, INPUT_DIM, dtype=torch.double)
        p12_0 = model(z0)
        print("initial p12 at zero normalized input:", p12_0.detach().cpu().numpy())
        print("p1 cap =", cfg.p1_max, "p2 cap =", cfg.p2_max)

    bound_slack_layer = make_bound_slack_qp_layer(
        num_u=NUM_ARM_JOINTS,
        num_pair_rows=NUM_PAIR_ROWS,
        num_bound_rows=NUM_BOUND_ROWS,
        rho=cfg.rho,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    limits_np = get_joint_limits()
    limits_t = {key: torch.tensor(value, dtype=torch.double) for key, value in limits_np.items()}

    print("cfg =", json.dumps(asdict(cfg), indent=2))

    # Reset log file.
    with open(cfg.log_jsonl, "w") as f:
        f.write("")

    safe_candidate_rows: list[dict] = []
    rng = random.Random(cfg.seed)
    total_t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        seg_stats: list[dict] = []

        for seg_idx in range(cfg.segments_per_epoch):
            start_state = None
            if cfg.sample_from_rollouts:
                scenario_path, start_state = sample_start_state_from_rollout(
                    rollout_files=rollout_files,
                    scenario_map=scenario_map,
                    horizon=cfg.horizon,
                    start_step_min=cfg.start_step_min,
                    start_step_max=cfg.start_step_max,
                    rng=rng,
                )
                print(
                    f"epoch {epoch}/{cfg.epochs} segment {seg_idx + 1}/{cfg.segments_per_epoch} | "
                    f"{Path(scenario_path).name} start_step={start_state['start_step']} "
                    f"from={Path(start_state['rollout_path']).name}"
                )
            else:
                scenario_path = rng.choice(scenario_paths)
                print(
                    f"epoch {epoch}/{cfg.epochs} segment {seg_idx + 1}/{cfg.segments_per_epoch} | "
                    f"{Path(scenario_path).name}"
                )

            stats = train_one_segment(
                scenario_path=scenario_path,
                pin_context=pin_context,
                torch_kin=torch_kin,
                model=model,
                bound_slack_layer=bound_slack_layer,
                optimizer=optimizer,
                feat_mean=feat_mean,
                feat_std=feat_std,
                limits_t=limits_t,
                cfg=cfg,
                train=True,
                start_state=start_state,
            )

            if stats.get("solver_failed", False):
                print(
                    "  skipped segment due to DQP failure | "
                    f"scenario={stats.get('scenario')} start_step={stats.get('start_step')} "
                    f"fail_step={stats.get('fail_step')} min_h={stats.get('min_h'):.3e} "
                    f"min_psi={stats.get('min_psi'):.3e} p1={stats.get('p1_mean'):.3e} "
                    f"p2={stats.get('p2_mean'):.3e} reason={stats.get('fail_reason')}"
                )
                seg_stats.append(stats)
                continue

            seg_stats.append(stats)

        ep = aggregate_stats(seg_stats)
        if not ep:
            print(f"[Epoch {epoch:03d}] no valid segments; all sampled windows failed DQP solve")
            continue

        ep["epoch"] = epoch
        ep["elapsed_s"] = time.time() - epoch_t0

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={ep['loss']:.6e} "
            f"safety={ep['safety']:.3e} "
            f"h_safety={ep.get('h_safety', float('nan')):.3e} "
            f"h_freq={ep.get('h_viol_freq', float('nan')):.3e} "
            f"g1_safety={ep.get('g1_safety', float('nan')):.3e} "
            f"slack={ep['slack']:.3e} "
            f"goal={ep['goal']:.3e} "
            f"goal_init={ep['goal_initial']:.3e} "
            f"goal_final={ep['goal_final']:.3e} "
            f"progress={ep['goal_progress_ratio']:.3e} "
            f"norm_mean_goal={ep['normalized_mean_goal_distance']:.3e} "
            f"preg={ep['preg']:.3e} "
            f"hard_viol_max={ep['hard_viol_max']:.3e} "
            f"pair_viol_max={ep['pair_viol_max']:.3e} "
            f"bound_viol_max={ep['bound_viol_max']:.3e} "
            f"slack_max={ep['slack_max']:.3e} "
            f"min_h={ep['min_h']:.3e} "
            f"min_psi={ep['min_psi']:.3e} "
            f"min_g2={ep.get('min_g2', float('nan')):.3e} "
            f"p1_mean={ep['p1_mean']:.3e} "
            f"p2_mean={ep['p2_mean']:.3e} "
            f"valid={ep['num_valid_segments']} "
            f"failed={ep['num_failed_segments']}"
        )

        # Optional validation segments. Checkpointing uses validation when enabled,
        # otherwise it falls back to the training epoch aggregate.
        val_ep = None
        if cfg.val_every > 0 and cfg.val_segments > 0 and (epoch % cfg.val_every == 0):
            val_stats: list[dict] = []
            for val_idx in range(cfg.val_segments):
                start_state = None
                if cfg.sample_from_rollouts:
                    val_scenario_path, start_state = sample_start_state_from_rollout(
                        rollout_files=val_rollout_files,
                        scenario_map=scenario_map,
                        horizon=cfg.horizon,
                        start_step_min=cfg.start_step_min,
                        start_step_max=cfg.start_step_max,
                        rng=rng,
                    )
                else:
                    val_scenario_path = rng.choice(scenario_paths)

                stats_val = train_one_segment(
                    scenario_path=val_scenario_path,
                    pin_context=pin_context,
                    torch_kin=torch_kin,
                    model=model,
                    bound_slack_layer=bound_slack_layer,
                    optimizer=optimizer,
                    feat_mean=feat_mean,
                    feat_std=feat_std,
                    limits_t=limits_t,
                    cfg=cfg,
                    train=False,
                    start_state=start_state,
                )
                val_stats.append(stats_val)

            val_ep = aggregate_stats(val_stats)
            if val_ep:
                val_ep["epoch"] = epoch
                val_ep["split"] = "val"
                print(
                    f"  [val] "
                    f"loss={val_ep['loss']:.6e} "
                    f"safety={val_ep['safety']:.3e} "
                    f"h_safety={val_ep.get('h_safety', float('nan')):.3e} "
                    f"h_freq={val_ep.get('h_viol_freq', float('nan')):.3e} "
                    f"g1_safety={val_ep.get('g1_safety', float('nan')):.3e} "
                    f"slack={val_ep['slack']:.3e} "
                    f"goal={val_ep['goal']:.3e} "
                    f"goal_init={val_ep['goal_initial']:.3e} "
                    f"goal_final={val_ep['goal_final']:.3e} "
                    f"progress={val_ep['goal_progress_ratio']:.3e} "
                    f"norm_mean_goal={val_ep['normalized_mean_goal_distance']:.3e} "
                    f"pair_viol_max={val_ep['pair_viol_max']:.3e} "
                    f"min_h={val_ep['min_h']:.3e} "
                    f"min_psi={val_ep['min_psi']:.3e} "
                    f"min_g2={val_ep.get('min_g2', float('nan')):.3e} "
                    f"valid={val_ep['num_valid_segments']} "
                    f"failed={val_ep['num_failed_segments']}"
                )

        log_obj = {"train": ep}
        if val_ep is not None:
            log_obj["val"] = val_ep
        with open(cfg.log_jsonl, "a") as f:
            f.write(json.dumps(log_obj) + "\n")

        ckpt_ep = val_ep if val_ep else ep
        ckpt_split = "val" if val_ep else "train"

        # Always save the latest valid epoch to model_out.
        save_checkpoint(
            path=cfg.model_out,
            model=model,
            cfg=cfg,
            feat_mean=feat_mean,
            feat_std=feat_std,
            epoch=epoch,
            metric_name=f"{ckpt_split}_latest_loss",
            metric_value=float(ckpt_ep["loss"]),
            extra={ckpt_split: ckpt_ep},
        )

        # Safe candidate checkpoints:
        # Save every epoch that passes the same safety gate previously used for
        # best_track_given_safe. This creates a candidate set for offline comparison.
        current_track = track_metric_from_epoch(ckpt_ep, cfg)
        if is_safe_enough(ckpt_ep, cfg):
            candidate_path = safe_candidate_checkpoint_path(
                cfg=cfg,
                epoch=epoch,
                split=ckpt_split,
                track_value=current_track,
                ep=ckpt_ep,
            )
            save_checkpoint(
                path=candidate_path,
                model=model,
                cfg=cfg,
                feat_mean=feat_mean,
                feat_std=feat_std,
                epoch=epoch,
                metric_name=f"{ckpt_split}_{cfg.track_metric}_safe_candidate",
                metric_value=current_track,
                extra={ckpt_split: ckpt_ep},
            )
            row = make_candidate_summary_row(
                epoch=epoch,
                split=ckpt_split,
                ckpt_path=candidate_path,
                cfg=cfg,
                ep=ckpt_ep,
            )
            safe_candidate_rows.append(row)
            write_candidate_summaries(safe_candidate_rows, cfg)
            print(
                f"  saved safe candidate: {candidate_path} "
                f"{cfg.track_metric}={current_track:.3e} "
                f"min_h={ckpt_ep.get('min_h', float('nan')):.3e} "
                f"min_psi={ckpt_ep.get('min_psi', float('nan')):.3e} "
                f"min_g2={ckpt_ep.get('min_g2', float('nan')):.3e}"
            )

    print("\n=== ROLLOUT TRAINING DONE ===")
    print("elapsed_min =", (time.time() - total_t0) / 60.0)
    print("num_safe_candidates =", len(safe_candidate_rows))
    print("latest_model_out =", cfg.model_out)
    print("safe_candidate_dir =", cfg.safe_candidate_dir)
    print("safe_candidate_csv =", cfg.safe_candidate_csv)
    print("safe_candidate_txt =", cfg.safe_candidate_txt)
    print("log_jsonl =", cfg.log_jsonl)


if __name__ == "__main__":
    main()