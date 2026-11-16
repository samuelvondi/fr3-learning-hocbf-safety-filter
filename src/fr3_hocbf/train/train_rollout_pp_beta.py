#!/usr/bin/env python3
from __future__ import annotations

"""
Rollout training for the per-pair beta policy (PerPairBetaNet, fr3_hocbf/policy/pp_beta.py).

Same differentiable bound-slack QP rollout, scenarios, losses, hyperparameters
and checkpoint gate as fr3_hocbf/train/train_rollout.py. The only differences:
  - the network predicts shared gamma (B, 1) and per-pair beta (B, 43)
  - HOCBF row j uses beta_j
  - p-regularization is 0.5*(gamma^2 + mean over active pairs of beta_j^2),
    which matches the old mean([gamma, beta]^2) scale
  - checkpoints carry arch="pp_beta_v1" and the per-pair feature stats

fr3_hocbf/train/train_rollout.py itself is not modified.

Example (published configuration, README section 5.5):
    python -m fr3_hocbf.train.train_rollout_pp_beta \
      ../config/generated_scenarios_fullpool_candidates \
      --rollout-labeled ../data/fr3_fullpool_train_weighted_local.jsonl \
      --val-rollout-labeled ../data/fr3_fullpool_val_unique_local.jsonl \
      --stats-in ../artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt \
      --pp-stats-in fr3_pp_feature_stats_v10.npz \
      --sample-start-from-rollout \
      --epochs 50 --horizon 50 --segments-per-epoch 80 \
      --start-step-min 0 --start-step-max 700 \
      --val-every 2 --val-segments 20 \
      --w-safety 100.0 --w-g1-safety 20.0 --w-goal 0.01 \
      --w-slack 0.01 --w-preg 5e-7 --safety-topk-frac 1.0 --slack-huber-delta 1.0 \
      --no-move-obstacles \
      --model-out fr3_v10_pp_beta_latest.pt \
      --log-jsonl fr3_v10_pp_beta_log.jsonl \
      --safe-candidate-dir safe_candidates_v10_pp_beta \
      --safe-candidate-csv safe_candidates_v10_pp_beta_summary.csv \
      --safe-candidate-txt safe_candidates_v10_pp_beta_summary.txt
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fr3_hocbf.common_io import expand_scenario_inputs
from fr3_hocbf.dataset.rollout_records import build_scenario_map, expand_labeled_rollout_inputs, sample_start_state_from_rollout, summarize_rollout_records
from fr3_hocbf.sim.rollout import NUM_ARM_JOINTS, ee_position, get_default_initial_state, load_scenario_yaml, make_rollout_context
from fr3_hocbf.differentiable.kinematics import EE_FRAME_NAME, load_fr3_torch_kinematics
from fr3_hocbf.differentiable.kinematics_fast import load_fr3_torch_kinematics_fast
from fr3_hocbf.differentiable.geometry import compute_all_pair_terms_torch
from fr3_hocbf.differentiable.qp_rows import build_full_qp_torch_batch_pp
from fr3_hocbf.sim.qp_data import NUM_BOUND_ROWS, NUM_PAIR_ROWS, NUM_TOTAL_ROWS, get_joint_limits
from fr3_hocbf.differentiable.qp_layer import (
    bound_violation_batch,
    hard_violation_batch,
    make_bound_slack_qp_layer,
    mean_sq_bound_slack,
    pair_violation_batch,
    solve_bound_slack_qp_batch,
)
from fr3_hocbf.train.train_rollout import (
    DT_DEFAULT,
    RolloutTrainCfg,
    aggregate_stats,
    clone_obstacles_torch,
    compute_nominal_detached,
    guard_against_accidental_overwrite,
    is_safe_enough,
    make_candidate_summary_row,
    obstacles_numpy_to_torch,
    safe_candidate_checkpoint_path,
    topk_positive_violation_sq_loss,
    track_metric_from_epoch,
    update_obstacles_torch,
    violation_frequency_loss_from_sq,
    write_candidate_summaries,
)
from fr3_hocbf.policy.features import INPUT_DIM, build_feature_vector, load_feature_stats, normalize_features_torch
from fr3_hocbf.policy.pp_beta import (
    PP_ARCH,
    PP_FEATURE_NAMES,
    PerPairBetaNet,
    kinematic_ddq_bounds_torch,
)


def masked_beta_mean(beta: torch.Tensor, pair_mask: torch.Tensor, power: int = 1) -> torch.Tensor:
    """Mean of beta_j**power over active pairs; over all pairs if none is active."""
    active = pair_mask > 0.5
    vals = beta[active] if bool(torch.any(active)) else beta.reshape(-1)
    return (vals ** power).mean()


def train_one_segment_pp(
    scenario_path: str,
    pin_context: dict,
    torch_kin,
    model: PerPairBetaNet,
    bound_slack_layer,
    optimizer,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    limits_t: dict[str, torch.Tensor],
    cfg: RolloutTrainCfg,
    train: bool = True,
    start_state: dict | None = None,
) -> dict:
    """train_one_segment from fr3_hocbf/train/train_rollout.py with per-pair beta. Comments there apply here."""
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
    if start_step > 0:
        for obs in obstacles:
            obs["pose_start"] = obs["pose_start"] + obs["velocity"] * (float(start_step) * cfg.dt)
            obs["pose_end"] = obs["pose_end"] + obs["velocity"] * (float(start_step) * cfg.dt)

    goal_np = np.asarray(scenario["goal_ee_pos"], dtype=float)
    d_margin = float(scenario["d_margin"])
    target_prev_np = ee_position(pin_model, pin_data, ee_frame_id, q_full_np, dq_full_np)

    zero = lambda: torch.zeros((), dtype=torch.double)
    loss_total, loss_safety_total, loss_h_safety_total, loss_g1_safety_total = zero(), zero(), zero(), zero()
    loss_h_viol_freq_total, loss_slack_total, loss_goal_total, loss_preg_total = zero(), zero(), zero(), zero()
    goal_t = torch.tensor(goal_np, dtype=torch.double)
    with torch.no_grad():
        ee_pos_initial = torch_kin.get_frame_position_torch(q, EE_FRAME_NAME)
        goal_initial_dist = float(torch.linalg.norm(ee_pos_initial - goal_t).item())

    hard_viol_max_seen = pair_viol_max_seen = bound_viol_max_seen = slack_max_seen = 0.0
    goal_dist_sum = 0.0
    goal_final_dist = float("nan")
    min_h_seen = min_psi_seen = min_g2_seen = float("inf")
    h_sum = psi_sum = g2_sum = 0.0
    h_count = psi_count = g2_count = 0
    p1_sum = p2_sum = 0.0

    max_steps_from_scenario = int(float(scenario["max_sim_duration_s"]) / cfg.dt)
    H = min(int(cfg.horizon), max_steps_from_scenario)
    ddq_nominal_scale = torch.maximum(torch.abs(limits_t["ddq_min_arm"]), torch.abs(limits_t["ddq_max_arm"]))

    for k in range(H):
        obstacles = update_obstacles_torch(obstacles, cfg.dt) if cfg.move_obstacles else clone_obstacles_torch(obstacles)

        robot_links = torch_kin.get_active_link_endpoint_data_torch(q, create_graph=train)
        pair_terms = compute_all_pair_terms_torch(robot_links=robot_links, obstacles=obstacles, dq=dq, d_margin=d_margin)

        ddq_nominal, target_prev_np = compute_nominal_detached(
            pin_model=pin_model,
            pin_data=pin_data,
            ee_frame_id=ee_frame_id,
            q_arm=q,
            dq_arm=dq,
            goal_np=goal_np,
            target_prev_np=target_prev_np,
        )

        z_raw = build_feature_vector(
            q_arm=q,
            dq_arm=dq,
            ddq_nominal=ddq_nominal,
            ddq_nominal_scale=ddq_nominal_scale,
            h_all=pair_terms["h_all"],
            Lf_h_all=pair_terms["Lf_h_all"],
        )
        z_norm = normalize_features_torch(z_raw, feat_mean, feat_std)

        q_b, dq_b = q.unsqueeze(0), dq.unsqueeze(0)
        ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds_torch(q_b, dq_b, limits_t, cfg.dt)
        pt_b = {key: pair_terms[key].unsqueeze(0) for key in ("h_all", "Lf_h_all", "vrel_sq2_all", "Lg_psi_all", "pair_mask")}
        gamma, beta = model(
            z_norm,
            {**pt_b, "u_nom": ddq_nominal.unsqueeze(0), "ddq_min_kin": ddq_min_kin, "ddq_max_kin": ddq_max_kin},
        )

        A_full, b_full, _ = build_full_qp_torch_batch_pp(gamma, beta, q_b, dq_b, pt_b, cfg.dt, limits_t)

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
            has_active = bool(torch.any(active))
            psi_dbg = pair_terms["Lf_h_all"] + gamma[0, 0] * pair_terms["h_all"]
            nan_keys = [
                "loss", "safety", "h_safety", "g1_safety", "slack", "goal", "goal_initial", "goal_final",
                "goal_dist_mean", "goal_progress_ratio", "normalized_mean_goal_distance", "goal_delta", "preg",
                "hard_viol_max", "pair_viol_max", "bound_viol_max", "viol_max", "slack_max", "min_g2",
                "h_mean", "g1_mean", "g2_mean",
            ]
            out = {key: float("nan") for key in nan_keys}
            out.update(
                {
                    "solver_failed": True,
                    "fail_reason": repr(exc),
                    "fail_step": int(k),
                    "min_h": float(pair_terms["h_all"][active].min().item()) if has_active else float("nan"),
                    "min_psi": float(psi_dbg[active].min().item()) if has_active else float("nan"),
                    "p1_mean": float(gamma[0, 0].item()),
                    "p2_mean": float(masked_beta_mean(beta[0], pair_terms["pair_mask"]).item()),
                    "horizon": H,
                    "scenario": Path(scenario_path).stem,
                    "start_step": start_step,
                    "start_rollout_path": start_rollout_path,
                }
            )
            return out

        hard_viol = hard_violation_batch(u=u_safe, A_full_raw=A_full, b_full_raw=b_full)
        pair_viol = pair_violation_batch(u=u_safe, A_full_raw=A_full, b_full_raw=b_full, num_pair_rows=NUM_PAIR_ROWS)
        bound_viol = bound_violation_batch(u=u_safe, A_full_raw=A_full, b_full_raw=b_full, num_pair_rows=NUM_PAIR_ROWS)

        active_pre = pair_terms["pair_mask"] > 0.5
        if bool(torch.any(active_pre)):
            g2_margin_all = torch.mv(A_full[0, :NUM_PAIR_ROWS, :], u_safe[0]) - b_full[0, :NUM_PAIR_ROWS]
            g2_active = g2_margin_all[active_pre]
            min_g2_seen = min(min_g2_seen, float(g2_active.min().item()))
            g2_sum += float(g2_active.sum().item())
            g2_count += int(g2_active.numel())

        if cfg.slack_huber_delta > 0.0:
            abs_s = s_bound.abs()
            loss_slack = torch.where(
                abs_s <= cfg.slack_huber_delta,
                s_bound.pow(2),
                2.0 * cfg.slack_huber_delta * abs_s - cfg.slack_huber_delta**2,
            ).mean()
        else:
            loss_slack = mean_sq_bound_slack(s_bound)
        # Old: mean([gamma, beta]^2) = 0.5*(gamma^2 + beta^2). Per-pair: beta^2 -> mean over active beta_j^2.
        loss_preg = 0.5 * (gamma[0, 0] ** 2 + masked_beta_mean(beta[0], pair_terms["pair_mask"], power=2))

        u = u_safe[0]
        q_next = q + dq * cfg.dt + 0.5 * u * (cfg.dt ** 2)
        dq_next = dq + u * cfg.dt

        post_robot_links = torch_kin.get_active_link_endpoint_data_torch(q_next, create_graph=train)
        post_pair_terms = compute_all_pair_terms_torch(
            robot_links=post_robot_links, obstacles=obstacles, dq=dq_next, d_margin=d_margin
        )
        post_active = post_pair_terms["pair_mask"] > 0.5
        psi_post_all = post_pair_terms["Lf_h_all"] + gamma[0, 0] * post_pair_terms["h_all"]
        if bool(torch.any(post_active)):
            h_post_active = post_pair_terms["h_all"][post_active]
            psi_post_active = psi_post_all[post_active]
            h_post_violation = torch.relu(cfg.h_safety_margin - h_post_active)
            g1_post_violation = torch.relu(cfg.g1_safety_margin - psi_post_active)
            loss_h_safety = topk_positive_violation_sq_loss(h_post_violation, frac=cfg.safety_topk_frac)
            loss_g1_safety = topk_positive_violation_sq_loss(g1_post_violation, frac=cfg.safety_topk_frac)
            loss_h_viol_freq = violation_frequency_loss_from_sq(h_post_violation ** 2, eps=cfg.viol_freq_eps)
        else:
            loss_h_safety, loss_g1_safety, loss_h_viol_freq = zero(), zero(), zero()

        loss_safety = loss_h_safety + loss_g1_safety

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

        hard_viol_max_seen = max(hard_viol_max_seen, float(hard_viol.max().item()))
        pair_viol_max_seen = max(pair_viol_max_seen, float(pair_viol.max().item()))
        bound_viol_max_seen = max(bound_viol_max_seen, float(bound_viol.max().item()))
        slack_max_seen = max(slack_max_seen, float(s_bound.max().item()))
        goal_dist_value = float(goal_dist.item())
        goal_dist_sum += goal_dist_value
        goal_final_dist = goal_dist_value
        p1_sum += float(gamma[0, 0].item())
        p2_sum += float(masked_beta_mean(beta[0], pair_terms["pair_mask"]).item())

        if bool(torch.any(post_active)):
            h_post_active = post_pair_terms["h_all"][post_active]
            psi_post_active = psi_post_all[post_active]
            min_h_seen = min(min_h_seen, float(h_post_active.min().item()))
            min_psi_seen = min(min_psi_seen, float(psi_post_active.min().item()))
            h_sum += float(h_post_active.sum().item())
            psi_sum += float(psi_post_active.sum().item())
            h_count += int(h_post_active.numel())
            psi_count += int(psi_post_active.numel())

        q, dq = q_next, dq_next

    norm = float(max(1, H))
    loss_total = loss_total / norm

    if train:
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

    goal_mean_dist = float(goal_dist_sum / norm)
    return {
        "solver_failed": False,
        "fail_reason": "",
        "fail_step": -1,
        "loss": float(loss_total.item()),
        "safety": float(loss_safety_total.item()) / norm,
        "h_safety": float(loss_h_safety_total.item()) / norm,
        "h_viol_freq": float(loss_h_viol_freq_total.item()) / norm,
        "g1_safety": float(loss_g1_safety_total.item()) / norm,
        "slack": float(loss_slack_total.item()) / norm,
        "goal": float(loss_goal_total.item()) / norm,
        "goal_initial": float(goal_initial_dist),
        "goal_final": float(goal_final_dist),
        "goal_dist_mean": goal_mean_dist,
        "goal_progress_ratio": float(goal_final_dist / max(goal_initial_dist, cfg.goal_progress_eps)),
        "normalized_mean_goal_distance": float(goal_mean_dist / max(goal_initial_dist, cfg.goal_progress_eps)),
        "goal_delta": float(goal_final_dist - goal_initial_dist),
        "preg": float(loss_preg_total.item()) / norm,
        "hard_viol_max": hard_viol_max_seen,
        "pair_viol_max": pair_viol_max_seen,
        "bound_viol_max": bound_viol_max_seen,
        "viol_max": hard_viol_max_seen,
        "slack_max": slack_max_seen,
        "min_h": min_h_seen,
        "min_psi": min_psi_seen,
        "min_g2": min_g2_seen,
        "h_mean": float(h_sum / float(max(1, h_count))),
        "g1_mean": float(psi_sum / float(max(1, psi_count))),
        "g2_mean": float(g2_sum / float(max(1, g2_count))),
        "p1_mean": p1_sum / norm,
        "p2_mean": p2_sum / norm,  # mean over steps of the active-pair mean beta_j
        "mean_segment_time_s": float(H) * float(cfg.dt),
        "horizon": H,
        "scenario": Path(scenario_path).stem,
        "start_step": start_step,
        "start_rollout_path": start_rollout_path,
    }


def load_global_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """114-D stats from a .npz (as fr3_hocbf/train/train_rollout.py) or embedded in an existing checkpoint."""
    if path.endswith(".pt"):
        ckpt = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
        mean = torch.as_tensor(ckpt["feat_mean"], dtype=torch.double)
        std = torch.clamp(torch.as_tensor(ckpt["feat_std"], dtype=torch.double), min=1e-6)
        if mean.numel() != INPUT_DIM:
            raise ValueError(f"Expected feature dim {INPUT_DIM}, got {mean.numel()} from {path}")
        return mean, std
    return load_feature_stats(path)


def save_checkpoint_pp(path, model, cfg, feat_mean, feat_std, epoch, metric_name, metric_value, args, extra=None):
    payload = {
        "arch": PP_ARCH,
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "feature_version": "q_dq_ddqnom_ddqnomnorm_h_Lfh",
        "pp_feature_names": list(PP_FEATURE_NAMES),
        "pair_embed_dim": int(args.pair_embed_dim),
        "pp_head_hidden": int(args.pp_head_hidden),
        "pp_stats_in": args.pp_stats_in,
        "fast_kinematics": bool(args.fast_kinematics),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="Scenario YAML(s), directories, globs, or txt files")
    parser.add_argument("--stats-in", type=str, required=True, help="114D feature stats .npz, or a checkpoint .pt with embedded feat_mean/feat_std")
    parser.add_argument("--pp-stats-in", type=str, required=True, help="Per-pair feature stats .npz from fr3_hocbf/dataset/compute_pp_feature_stats.py")
    parser.add_argument("--pair-embed-dim", type=int, default=8)
    parser.add_argument("--pp-head-hidden", type=int, default=64)
    parser.add_argument(
        "--fast-kinematics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Analytic endpoint Jacobians from one FK pass (fr3_hocbf/differentiable/kinematics_fast.py), ~4.7x faster per segment. "
        "Same math; --no-fast-kinematics uses the original per-endpoint autograd Jacobians.",
    )

    # Everything below mirrors fr3_hocbf/train/train_rollout.py (same names and defaults).
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--segments-per-epoch", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--rollout-labeled", type=str, default=None)
    parser.add_argument("--val-rollout-labeled", type=str, default=None)
    parser.add_argument("--critical-windows", type=str, default=None)
    parser.add_argument("--sample-start-from-rollout", action="store_true")
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
    parser.add_argument("--init-p1", type=float, default=10.0)
    parser.add_argument("--init-p2", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--rho", type=float, default=1e5)
    parser.add_argument("--dqp-row-norm-eps", type=float, default=1e-6)
    parser.add_argument("--w-safety", type=float, default=100.0)
    parser.add_argument("--w-g1-safety", type=float, default=10.0)
    parser.add_argument("--w-viol-freq", type=float, default=0.0)
    parser.add_argument("--viol-freq-eps", type=float, default=1e-6)
    parser.add_argument("--w-slack", type=float, default=1.0)
    parser.add_argument("--slack-huber-delta", type=float, default=0.0)
    parser.add_argument("--w-goal", type=float, default=0.0)
    parser.add_argument("--w-preg", type=float, default=0.0)
    parser.add_argument("--h-safety-margin", type=float, default=0.0)
    parser.add_argument("--g1-safety-margin", type=float, default=0.0)
    parser.add_argument("--safety-topk-frac", type=float, default=0.05)
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
    )
    parser.add_argument("--goal-progress-eps", type=float, default=1e-3)
    parser.add_argument("--safe-candidate-dir", type=str, default="safe_candidates_pp_beta")
    parser.add_argument("--safe-candidate-csv", type=str, default="safe_candidates_pp_beta_summary.csv")
    parser.add_argument("--safe-candidate-txt", type=str, default="safe_candidates_pp_beta_summary.txt")
    parser.add_argument("--move-obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--model-out", type=str, default="fr3_pp_beta_rollout_latest.pt")
    parser.add_argument("--log-jsonl", type=str, default="fr3_train_rollout_pp_beta_log.jsonl")
    parser.add_argument("--overwrite", action="store_true")
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
    scenario_map = build_scenario_map(scenario_paths)

    rollout_files: list = []
    val_rollout_files: list = []
    if args.sample_start_from_rollout:
        if args.rollout_labeled is None:
            raise ValueError("--sample-start-from-rollout requires --rollout-labeled")
        rollout_files = expand_labeled_rollout_inputs([args.rollout_labeled], critical_windows_path=args.critical_windows)
        print("train rollout categories =", summarize_rollout_records(rollout_files))
        if args.val_rollout_labeled is not None:
            val_rollout_files = expand_labeled_rollout_inputs([args.val_rollout_labeled], critical_windows_path=args.critical_windows)
        else:
            val_rollout_files = rollout_files
            print("WARNING: no validation rollout source set; validation samples from training rollout records.")
        print("val rollout categories =", summarize_rollout_records(val_rollout_files))

    feat_mean, feat_std = load_global_stats(args.stats_in)
    pp_stats = np.load(os.path.expanduser(args.pp_stats_in))
    print("stats_in =", args.stats_in, "| pp_stats_in =", args.pp_stats_in)

    pin_context = make_rollout_context()
    torch_kin = load_fr3_torch_kinematics_fast() if args.fast_kinematics else load_fr3_torch_kinematics()
    print("torch kinematics =", type(torch_kin).__name__)

    model = PerPairBetaNet(
        input_dim=INPUT_DIM,
        hidden_dim=cfg.hidden_dim,
        pair_embed_dim=args.pair_embed_dim,
        head_hidden=args.pp_head_hidden,
        p1_floor=cfg.p1_floor,
        p2_floor=cfg.p2_floor,
        p1_max=cfg.p1_max,
        p2_max=cfg.p2_max,
        init_p1=cfg.init_p1,
        init_p2=cfg.init_p2,
    )
    model.set_pp_stats(pp_stats["pp_mean"], pp_stats["pp_std"])
    print("num_params =", sum(p.numel() for p in model.parameters()))

    bound_slack_layer = make_bound_slack_qp_layer(
        num_u=NUM_ARM_JOINTS, num_pair_rows=NUM_PAIR_ROWS, num_bound_rows=NUM_BOUND_ROWS, rho=cfg.rho
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    limits_t = {key: torch.tensor(value, dtype=torch.double) for key, value in get_joint_limits().items()}

    print("cfg =", json.dumps(asdict(cfg), indent=2))
    with open(cfg.log_jsonl, "w") as f:
        f.write("")

    seg_kwargs = dict(
        pin_context=pin_context,
        torch_kin=torch_kin,
        model=model,
        bound_slack_layer=bound_slack_layer,
        optimizer=optimizer,
        feat_mean=feat_mean,
        feat_std=feat_std,
        limits_t=limits_t,
        cfg=cfg,
    )

    def sample(records, rng):
        if cfg.sample_from_rollouts:
            return sample_start_state_from_rollout(
                rollout_files=records,
                scenario_map=scenario_map,
                horizon=cfg.horizon,
                start_step_min=cfg.start_step_min,
                start_step_max=cfg.start_step_max,
                rng=rng,
            )
        return rng.choice(scenario_paths), None

    safe_candidate_rows: list[dict] = []
    rng = random.Random(cfg.seed)
    total_t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        seg_stats = []
        for seg_idx in range(cfg.segments_per_epoch):
            scenario_path, start_state = sample(rollout_files, rng)
            print(f"epoch {epoch}/{cfg.epochs} segment {seg_idx + 1}/{cfg.segments_per_epoch} | {Path(scenario_path).name}"
                  + (f" start_step={start_state['start_step']}" if start_state else ""))
            stats = train_one_segment_pp(scenario_path=scenario_path, train=True, start_state=start_state, **seg_kwargs)
            if stats.get("solver_failed", False):
                print(f"  skipped segment due to DQP failure | fail_step={stats.get('fail_step')} reason={stats.get('fail_reason')}")
            seg_stats.append(stats)

        ep = aggregate_stats(seg_stats)
        if not ep:
            print(f"[Epoch {epoch:03d}] no valid segments; all sampled windows failed DQP solve")
            continue
        ep["epoch"] = epoch
        ep["elapsed_s"] = time.time() - epoch_t0
        print(
            f"[Epoch {epoch:03d}] loss={ep['loss']:.6e} safety={ep['safety']:.3e} slack={ep['slack']:.3e} "
            f"goal={ep['goal']:.3e} norm_mean_goal={ep['normalized_mean_goal_distance']:.3e} preg={ep['preg']:.3e} "
            f"min_h={ep['min_h']:.3e} min_psi={ep['min_psi']:.3e} min_g2={ep.get('min_g2', float('nan')):.3e} "
            f"gamma_mean={ep['p1_mean']:.3e} beta_active_mean={ep['p2_mean']:.3e} "
            f"valid={ep['num_valid_segments']} failed={ep['num_failed_segments']} elapsed={ep['elapsed_s']:.0f}s"
        )

        val_ep = None
        if cfg.val_every > 0 and cfg.val_segments > 0 and (epoch % cfg.val_every == 0):
            val_stats = []
            for _ in range(cfg.val_segments):
                val_scenario_path, start_state = sample(val_rollout_files, rng)
                val_stats.append(train_one_segment_pp(scenario_path=val_scenario_path, train=False, start_state=start_state, **seg_kwargs))
            val_ep = aggregate_stats(val_stats)
            if val_ep:
                val_ep["epoch"] = epoch
                val_ep["split"] = "val"
                print(
                    f"  [val] loss={val_ep['loss']:.6e} norm_mean_goal={val_ep['normalized_mean_goal_distance']:.3e} "
                    f"min_h={val_ep['min_h']:.3e} min_psi={val_ep['min_psi']:.3e} "
                    f"valid={val_ep['num_valid_segments']} failed={val_ep['num_failed_segments']}"
                )

        log_obj = {"train": ep}
        if val_ep is not None:
            log_obj["val"] = val_ep
        with open(cfg.log_jsonl, "a") as f:
            f.write(json.dumps(log_obj) + "\n")

        ckpt_ep = val_ep if val_ep else ep
        ckpt_split = "val" if val_ep else "train"
        save_checkpoint_pp(cfg.model_out, model, cfg, feat_mean, feat_std, epoch,
                           f"{ckpt_split}_latest_loss", float(ckpt_ep["loss"]), args, extra={ckpt_split: ckpt_ep})

        current_track = track_metric_from_epoch(ckpt_ep, cfg)
        if is_safe_enough(ckpt_ep, cfg):
            candidate_path = safe_candidate_checkpoint_path(cfg=cfg, epoch=epoch, split=ckpt_split, track_value=current_track, ep=ckpt_ep)
            save_checkpoint_pp(candidate_path, model, cfg, feat_mean, feat_std, epoch,
                               f"{ckpt_split}_{cfg.track_metric}_safe_candidate", current_track, args, extra={ckpt_split: ckpt_ep})
            safe_candidate_rows.append(make_candidate_summary_row(epoch=epoch, split=ckpt_split, ckpt_path=candidate_path, cfg=cfg, ep=ckpt_ep))
            write_candidate_summaries(safe_candidate_rows, cfg)
            print(f"  saved safe candidate: {candidate_path} {cfg.track_metric}={current_track:.3e}")

    print("\n=== PER-PAIR BETA ROLLOUT TRAINING DONE ===")
    print("elapsed_min =", (time.time() - total_t0) / 60.0)
    print("num_safe_candidates =", len(safe_candidate_rows))
    print("latest_model_out =", cfg.model_out)
    print("safe_candidate_dir =", cfg.safe_candidate_dir)


if __name__ == "__main__":
    main()
