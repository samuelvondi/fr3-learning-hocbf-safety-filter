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
    python -m fr3_hocbf.train.train_rollout \
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
from fr3_hocbf.sim.rollout import (
    ee_position,
    NUM_ARM_JOINTS,
    get_default_initial_state,
    load_scenario_yaml,
    make_rollout_context,
)
from fr3_hocbf.differentiable.kinematics import EE_FRAME_NAME, load_fr3_torch_kinematics
from fr3_hocbf.differentiable.kinematics_fast import load_fr3_torch_kinematics_fast
from fr3_hocbf.differentiable.geometry import compute_all_pair_terms_torch
from fr3_hocbf.differentiable.qp_rows import build_full_qp_torch_batch, build_full_qp_torch_batch_pp
from fr3_hocbf.policy.p12_net import P12ParamNet
from fr3_hocbf.policy.pp_beta import PP_ARCH, PP_FEATURE_NAMES, PerPairBetaNet, kinematic_ddq_bounds_torch
from fr3_hocbf.train.config import DT_DEFAULT, RolloutTrainCfg
from fr3_hocbf.train.checkpointing import (
    aggregate_stats,
    guard_against_accidental_overwrite,
    is_safe_enough,
    make_candidate_summary_row,
    maybe_load_model_weights,
    safe_candidate_checkpoint_path,
    save_checkpoint,
    track_metric_from_epoch,
    write_candidate_summaries,
)
from fr3_hocbf.policy.features import INPUT_DIM, build_feature_vector, load_feature_stats, normalize_features_torch
from fr3_hocbf.sim.nominal_controller import nominal_controller_js_standalone
from fr3_hocbf.sim.qp_data import (
    NUM_BOUND_ROWS,
    NUM_PAIR_ROWS,
    get_joint_limits,
)
from fr3_hocbf.differentiable.qp_layer import (
    bound_violation_batch,
    hard_violation_batch,
    make_bound_slack_qp_layer,
    mean_sq_bound_slack,
    pair_violation_batch,
    solve_bound_slack_qp_batch,
)


# ---------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------


def violation_frequency_loss_from_sq(
    viol_sq: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft count of post-step h violations: 0 when safe, approximately 1 when violated."""
    if viol_sq.numel() == 0:
        return torch.zeros((), device=viol_sq.device, dtype=viol_sq.dtype)
    return (2.0 * torch.sigmoid(viol_sq / eps) - 1.0).mean()


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


def masked_beta_mean(beta: torch.Tensor, pair_mask: torch.Tensor, power: int = 1) -> torch.Tensor:
    """Mean of beta_j**power over active pairs; over all pairs if none is active."""
    active = pair_mask > 0.5
    vals = beta[active] if bool(torch.any(active)) else beta.reshape(-1)
    return (vals ** power).mean()


def train_one_segment(
    scenario_path: str,
    pin_context: dict,
    torch_kin,
    model: P12ParamNet | PerPairBetaNet,
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

    P12ParamNet: shared gamma, beta. PerPairBetaNet: shared gamma and beta_j per pair row;
    HOCBF row j uses beta_j and the p-regularisation uses the mean of beta_j^2 over active pairs.
    """
    per_pair = isinstance(model, PerPairBetaNet)
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

    # fr3_hocbf/sim/rollout.py updates obstacles before computing the first geometry row of a step.
    # If we start from saved step k, advance the stored obstacle state to time k*dt;
    # the loop update then moves it to (k+1)*dt before the first geometry evaluation.
    if start_step > 0:
        for obs in obstacles:
            obs["pose_start"] = obs["pose_start"] + obs["velocity"] * (float(start_step) * cfg.dt)
            obs["pose_end"] = obs["pose_end"] + obs["velocity"] * (float(start_step) * cfg.dt)

    goal_np = np.asarray(scenario["goal_ee_pos"], dtype=float)
    d_margin = float(scenario["d_margin"])

    # Match fr3_hocbf.sim.rollout convention: target smoothing starts from current EE pose.
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
        if per_pair:
            q_b, dq_b = q.unsqueeze(0), dq.unsqueeze(0)
            ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds_torch(q_b, dq_b, limits_t, cfg.dt)
            pt_b = {key: pair_terms[key].unsqueeze(0) for key in ("h_all", "Lf_h_all", "vrel_sq2_all", "Lg_psi_all", "pair_mask")}
            gamma, beta = model(
                z_norm,
                {**pt_b, "u_nom": ddq_nominal.unsqueeze(0), "ddq_min_kin": ddq_min_kin, "ddq_max_kin": ddq_max_kin},
            )
            A_full, b_full, _ = build_full_qp_torch_batch_pp(gamma, beta, q_b, dq_b, pt_b, cfg.dt, limits_t)
        else:
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

        # gamma (p1) is indexed afresh at every use, exactly as before the per-pair
        # option existed, so the autograd graph and the results stay bit-identical.
        def p1_now():
            return gamma[0, 0] if per_pair else p12[0, 0]

        # Logged beta: p2, or the mean of beta_j over the active pairs.
        def p2_logged():
            return masked_beta_mean(beta[0], pair_terms["pair_mask"]) if per_pair else p12[0, 1]

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
            psi_dbg = pair_terms["Lf_h_all"] + p1_now() * pair_terms["h_all"]
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
                "p1_mean": float(p1_now().detach().cpu().item()),
                "p2_mean": float(p2_logged().detach().cpu().item()),
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
        if per_pair:
            # Old: mean([gamma, beta]^2) = 0.5*(gamma^2 + beta^2). Per-pair: beta^2 -> mean over active beta_j^2.
            loss_preg = 0.5 * (gamma[0, 0] ** 2 + masked_beta_mean(beta[0], pair_terms["pair_mask"], power=2))
        else:
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
        psi_post_all = post_pair_terms["Lf_h_all"] + p1_now() * post_pair_terms["h_all"]
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
        p1_sum += float(p1_now().detach().cpu().item())
        p2_sum += float(p2_logged().detach().cpu().item())

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
        "p2_mean": p2_sum / float(max(1, H)),  # per-pair: mean over steps of the active-pair mean beta_j
        "mean_segment_time_s": float(H) * float(cfg.dt),
        "horizon": H,
        "scenario": Path(scenario_path).stem,
        "start_step": start_step,
        "start_rollout_path": start_rollout_path,
    }


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="Scenario YAML(s), directories, globs, or txt files")
    parser.add_argument("--stats-in", type=str, required=True, help="Feature stats .npz for the 114D input, or a checkpoint .pt with embedded feat_mean/feat_std")
    parser.add_argument("--init-model", type=str, default=None, help="Optional checkpoint to warm-start from")
    parser.add_argument(
        "--policy",
        choices=["shared", "per_pair"],
        default="shared",
        help="shared: P12ParamNet (one gamma, one beta). per_pair: PerPairBetaNet (one gamma, beta_j per pair row).",
    )
    parser.add_argument("--pp-stats-in", type=str, default=None, help="per_pair: per-pair feature stats .npz (dataset.compute_feature_stats --kind per_pair)")
    parser.add_argument("--pair-embed-dim", type=int, default=8, help="per_pair: size of the learned pair embedding")
    parser.add_argument("--pp-head-hidden", type=int, default=64, help="per_pair: hidden width of the shared beta head")
    parser.add_argument(
        "--fast-kinematics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Analytic endpoint Jacobians from one FK pass (differentiable/kinematics_fast.py), ~4.7x faster per "
        "segment. Same math, but not bit-identical to the default per-endpoint autograd Jacobians.",
    )

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
    per_pair = args.policy == "per_pair"
    if per_pair and args.pp_stats_in is None:
        parser.error("--policy per_pair requires --pp-stats-in")

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
    torch_kin = load_fr3_torch_kinematics_fast() if args.fast_kinematics else load_fr3_torch_kinematics()
    print("torch URDF =", torch_kin.urdf_path, "|", type(torch_kin).__name__)

    policy_meta = None
    if per_pair:
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
        pp_stats = np.load(os.path.expanduser(args.pp_stats_in))
        model.set_pp_stats(pp_stats["pp_mean"], pp_stats["pp_std"])
        print("pp_stats_in =", args.pp_stats_in, "| num_params =", sum(p.numel() for p in model.parameters()))
        # Extra checkpoint fields read by policy/pp_beta.load_pp_policy.
        policy_meta = {
            "arch": PP_ARCH,
            "pp_feature_names": list(PP_FEATURE_NAMES),
            "pair_embed_dim": int(args.pair_embed_dim),
            "pp_head_hidden": int(args.pp_head_hidden),
            "pp_stats_in": args.pp_stats_in,
            "fast_kinematics": bool(args.fast_kinematics),
        }
    else:
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

    if not per_pair:
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
            policy_meta=policy_meta,
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
                policy_meta=policy_meta,
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