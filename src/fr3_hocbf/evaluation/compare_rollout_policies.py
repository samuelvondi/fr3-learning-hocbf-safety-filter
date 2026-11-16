#!/usr/bin/env python3
from __future__ import annotations

"""
Closed-loop rollout comparison for FR3 p12 / BarrierNet policies.

This is the first real closed-loop evaluator:
  - loads scenario YAML(s)
  - simulates q,dq forward from the same initial state
  - recomputes geometry every step
  - computes nominal ddq every step
  - chooses p1,p2 from a policy
  - can compare a pure NN policy against an NN + G12 fallback runtime stack
  - solves the ORIGINAL HARD QP
  - integrates the resulting safe acceleration
  - logs rollout-level metrics, including task completion time

No differentiable QP is used here.
No training slack is used here.

Supported modes:
  original      fixed scenario gamma,beta from YAML
  fixed         fixed --p1-fixed,--p2-fixed
  davide_online Davide-style dynamic gamma,beta update
  nn              trained P12ParamNet, pure prediction without fallback
  nn_g12_fallback trained P12ParamNet protected by an independent minimal G12 fallback
  nn_pp_beta          trained PerPairBetaNet (shared gamma, per-pair beta_j), no fallback
  nn_pp_beta_fallback PerPairBetaNet with the G12 fallback applied per pair (--pp-model)

Recommended first smoke test:
  python compare_rollout_policies.py \
    ~/davide_fr3_ws/src/cbf_safety_filter/config/generated_scenarios/scenario_0081.yaml \
    --modes original,davide_online,fixed,nn,nn_g12_fallback \
    --p1-fixed 2.0 --p2-fixed 3.0 \
    --model fr3_p12_good200_pipeline_proof.pt \
    --stats fr3_feature_stats_h_lfh_good200.npz \
    --max-steps 300 \
    --out-prefix cmp_smoke_0081
"""

import argparse
import os
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch

from fr3_hocbf.common_io import expand_scenario_inputs, write_csv, write_json
from fr3_hocbf.sim.rollout import (
    ee_position,
    NUM_ARM_JOINTS,
    MAX_OBSTACLES,
    make_rollout_context,
    load_scenario_yaml,
    get_default_initial_state,
    clone_obstacles,
    update_obstacles,
    compute_all_pair_terms,
    compute_barrier_metrics,
)
from fr3_hocbf.sim.nominal_controller import nominal_controller_js_standalone
from fr3_hocbf.sim.qp_data import (
    NUM_PAIR_ROWS,
    NUM_BOUND_ROWS,
    NUM_TOTAL_ROWS,
    build_full_qp_numpy,
)
from fr3_hocbf.policy.p12_net import P12ParamNet
from fr3_hocbf.policy.features import normalize_features_numpy
from fr3_hocbf.policy.pp_beta import PP_ARCH, load_pp_policy


OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

# Davide dynamic CBF parameter constants.
EPSILON_DENOMINATOR = 1e-6
GAMMA_ADJUST_BUFFER = 0.5
GAMMA_MAX_LIMIT = 200.0
BETA_ADJUST_BUFFER = 0.5
BETA_MAX_LIMIT = 250.0
NUM_ROBOT_OBSTACLE_ROWS = 8 * MAX_OBSTACLES  # 8 active links * 5 obstacle slots = 40


# ---------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# hard QP solve
# ---------------------------------------------------------------------
def solve_hard_qp_osqp(
    u_nom: np.ndarray,
    A_full: np.ndarray,
    b_full: np.ndarray,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    max_iter: int = 25000,
) -> tuple[np.ndarray, bool, str, float]:
    """
    Solve:
        min 0.5 ||u - u_nom||^2
        s.t. A_full u >= b_full
    """
    u_nom = np.asarray(u_nom, dtype=float).reshape(NUM_ARM_JOINTS)
    A_full = np.asarray(A_full, dtype=float).reshape(NUM_TOTAL_ROWS, NUM_ARM_JOINTS)
    b_full = np.asarray(b_full, dtype=float).reshape(NUM_TOTAL_ROWS)

    u = cp.Variable(NUM_ARM_JOINTS)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(u - u_nom)),
        [A_full @ u >= b_full],
    )

    t0 = time.time()
    status = "not_solved"
    solved = False
    u_star = np.full(NUM_ARM_JOINTS, np.nan, dtype=float)

    try:
        problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
        )
        status = str(problem.status)
    except cp.error.SolverError:
        status = "osqp_solver_error"
        try:
            problem.solve(verbose=False)
            status = str(problem.status)
        except Exception as exc:
            status = f"fallback_solver_error:{repr(exc)}"

    solve_time_s = time.time() - t0

    if problem.status in OPTIMAL_STATUSES and u.value is not None:
        u_star = np.asarray(u.value, dtype=float).reshape(NUM_ARM_JOINTS)
        solved = True

    return u_star, solved, status, solve_time_s


def hard_violation_numpy(u: np.ndarray, A_full: np.ndarray, b_full: np.ndarray) -> tuple[float, float]:
    lhs = A_full @ u
    margin = lhs - b_full
    violation = np.maximum(b_full - lhs, 0.0)
    return float(np.max(violation)), float(np.min(margin))


# ---------------------------------------------------------------------
# NN policy helpers
# ---------------------------------------------------------------------
def load_nn_policy(model_path: str, stats_path: str | None, hidden_dim: int, p1_floor: float, p2_floor: float):
    ckpt = torch.load(os.path.expanduser(model_path), map_location="cpu")

    if stats_path is not None:
        stats = np.load(os.path.expanduser(stats_path))
        if "feat_mean" in stats.files:
            feat_mean = stats["feat_mean"].astype(float)
            feat_std = stats["feat_std"].astype(float)
        elif "mean" in stats.files:
            feat_mean = stats["mean"].astype(float)
            feat_std = stats["std"].astype(float)
        else:
            raise KeyError(f"Could not find feat_mean/feat_std in {stats_path}")
    elif isinstance(ckpt, dict) and "feat_mean" in ckpt and "feat_std" in ckpt:
        feat_mean = np.asarray(ckpt["feat_mean"], dtype=float)
        feat_std = np.asarray(ckpt["feat_std"], dtype=float)
    else:
        raise ValueError("NN mode requires --stats unless the checkpoint contains feat_mean/feat_std")

    input_dim = int(feat_mean.shape[0])

    if isinstance(ckpt, dict):
        hidden_dim = int(ckpt.get("hidden_dim", hidden_dim))
        p1_floor = float(ckpt.get("p1_floor", p1_floor))
        p2_floor = float(ckpt.get("p2_floor", p2_floor))
        p1_max = float(ckpt.get("p1_max", 200.0))
        p2_max = float(ckpt.get("p2_max", 250.0))
    else:
        p1_max = 200.0
        p2_max = 250.0

    model = P12ParamNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        p1_floor=p1_floor,
        p2_floor=p2_floor,
        p1_max=p1_max,
        p2_max=p2_max,
    ).double()

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model, feat_mean, feat_std


@torch.no_grad()
def predict_p12_nn_current(
    model: P12ParamNet,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    ddq_nominal_arm: np.ndarray,
    joint_limits: dict,
    h_all: np.ndarray,
    Lf_h_all: np.ndarray,
) -> tuple[float, float]:
    ddq_nominal_arm = np.asarray(ddq_nominal_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    ddq_scale = np.maximum(
        np.abs(np.asarray(joint_limits["ddq_min_arm"], dtype=float).reshape(NUM_ARM_JOINTS)),
        np.abs(np.asarray(joint_limits["ddq_max_arm"], dtype=float).reshape(NUM_ARM_JOINTS)),
    )
    ddq_scale = np.clip(ddq_scale, 1e-6, None)
    ddq_nominal_norm = ddq_nominal_arm / ddq_scale

    feature_dim = int(np.asarray(feat_mean).reshape(-1).shape[0])
    old_dim = 7 + 7 + 7 + NUM_PAIR_ROWS + NUM_PAIR_ROWS
    new_dim = 7 + 7 + 7 + 7 + NUM_PAIR_ROWS + NUM_PAIR_ROWS

    if feature_dim == old_dim:
        # Backwards compatibility for existing 107D checkpoints/stats.
        z_parts = [
            np.asarray(q_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            np.asarray(dq_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            ddq_nominal_arm,
            np.asarray(h_all, dtype=float).reshape(NUM_PAIR_ROWS),
            np.asarray(Lf_h_all, dtype=float).reshape(NUM_PAIR_ROWS),
        ]
    elif feature_dim == new_dim:
        # New 114D feature vector with 3D analogue of u_nom/u_max.
        z_parts = [
            np.asarray(q_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            np.asarray(dq_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            ddq_nominal_arm,
            ddq_nominal_norm,
            np.asarray(h_all, dtype=float).reshape(NUM_PAIR_ROWS),
            np.asarray(Lf_h_all, dtype=float).reshape(NUM_PAIR_ROWS),
        ]
    else:
        raise ValueError(
            f"Unsupported feature dimension {feature_dim}; expected old {old_dim}D or new {new_dim}D stats."
        )

    z_raw = np.concatenate(z_parts, axis=0)[None, :]

    z_norm = normalize_features_numpy(z_raw, feat_mean, feat_std)
    z_t = torch.tensor(z_norm, dtype=torch.double)
    p12 = model(z_t).detach().cpu().numpy()[0]
    return float(p12[0]), float(p12[1])


# ---------------------------------------------------------------------
# Davide online parameter update
# ---------------------------------------------------------------------
def kinematic_ddq_bounds(q_arm: np.ndarray, dq_arm: np.ndarray, limits: dict, dt: float) -> tuple[np.ndarray, np.ndarray]:
    q_arm = np.asarray(q_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    dq_arm = np.asarray(dq_arm, dtype=float).reshape(NUM_ARM_JOINTS)

    ddq_max_arm = limits["ddq_max_arm"]
    ddq_min_arm = limits["ddq_min_arm"]
    dq_max_arm = limits["dq_max_arm"]
    dq_min_arm = limits["dq_min_arm"]
    q_max_arm = limits["q_max_arm"]
    q_min_arm = limits["q_min_arm"]

    ddq_max_from_vel = (dq_max_arm - dq_arm) / dt
    ddq_min_from_vel = (dq_min_arm - dq_arm) / dt
    ddq_max_from_pos = 2.0 * (q_max_arm - q_arm - dq_arm * dt) / (dt ** 2)
    ddq_min_from_pos = 2.0 * (q_min_arm - q_arm - dq_arm * dt) / (dt ** 2)

    ddq_max_kinematic = np.minimum(ddq_max_arm, np.minimum(ddq_max_from_vel, ddq_max_from_pos))
    ddq_min_kinematic = np.maximum(ddq_min_arm, np.maximum(ddq_min_from_vel, ddq_min_from_pos))
    return ddq_min_kinematic, ddq_max_kinematic


def davide_online_p12_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    initial_gamma: float,
    initial_beta: float,
    pair_scope: str = "all",
) -> tuple[float, float, float, float]:
    """
    Single-step Davide-style dynamic gamma/beta calculation.

    This mirrors the two-pass logic:
      gamma from -Lf_h/h, then beta from -S_sup/psi.
    Only active pairs (below the activation threshold) are considered.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)

    h = h_all[active]
    Lf_h = Lf_h_all[active]
    vrel_sq2 = vrel_sq2_all[active]
    Lg = Lg_psi_all[active, :]

    max_gamma_required = -float("inf")
    for hi, lfi in zip(h, Lf_h):
        if hi > EPSILON_DENOMINATOR:
            gamma_needed = -float(lfi) / float(hi)
            max_gamma_required = max(max_gamma_required, gamma_needed)

    gamma = float(initial_gamma)
    if max_gamma_required > -float("inf"):
        adjusted_gamma = max(0.0, max_gamma_required + GAMMA_ADJUST_BUFFER)
        gamma = max(float(initial_gamma), min(adjusted_gamma, GAMMA_MAX_LIMIT))

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)

    psi = Lf_h + gamma * h
    Lf_psi = vrel_sq2 + gamma * Lf_h

    max_beta_required = -float("inf")
    for psii, lfpsii, lgi in zip(psi, Lf_psi, Lg):
        sup_lg_u = float(np.sum(np.where(lgi >= 0.0, lgi * ddq_max_kin, lgi * ddq_min_kin)))
        S_sup = float(lfpsii) + sup_lg_u

        if psii > EPSILON_DENOMINATOR:
            beta_needed = -S_sup / float(psii)
            max_beta_required = max(max_beta_required, beta_needed)

    beta = float(initial_beta)
    if max_beta_required > -float("inf"):
        adjusted_beta = max(0.0, max_beta_required + BETA_ADJUST_BUFFER)
        beta = max(float(initial_beta), min(adjusted_beta, BETA_MAX_LIMIT))

    return gamma, beta, float(max_gamma_required), float(max_beta_required)


# ---------------------------------------------------------------------
# Independent G12 fallback helpers for NN runtime policy
# ---------------------------------------------------------------------
def pair_scope_mask(pair_mask: np.ndarray, pair_scope: str) -> np.ndarray:
    """Return the active pair mask restricted to obstacle rows or all pair rows."""
    pair_mask = np.asarray(pair_mask, dtype=float).reshape(NUM_PAIR_ROWS) > 0.5
    if pair_scope == "obstacle":
        scoped = np.zeros(NUM_PAIR_ROWS, dtype=bool)
        scoped[:NUM_ROBOT_OBSTACLE_ROWS] = pair_mask[:NUM_ROBOT_OBSTACLE_ROWS]
        return scoped
    if pair_scope == "all":
        return pair_mask
    raise ValueError(f"Unknown pair_scope: {pair_scope}")


def min_g1_current(all_pair_terms: dict, p1: float, pair_scope: str) -> float:
    """
    Current-state G1/psi diagnostic:
        G1 = Lf_h + p1*h

    This is computed before integration, on the same state used to build the QP.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)
    if not np.any(active):
        return float("nan")
    g1_all = Lf_h_all + float(p1) * h_all
    return float(np.min(g1_all[active]))


def g12_check_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    p1: float,
    p2: float,
    pair_scope: str,
) -> tuple[float, float]:
    """
    Pre-QP analytic feasibility check of candidate p1,p2 (paper Sec. II-D).

    G1 over active pairs with h > 0:
        G1 = Lf_h + p1*h
    G2 over active pairs with psi > 0, using the largest contribution the
    bounded input can make (S_sup = sup_{u in U} Lg_psi u):
        G2 = Lf_psi + S_sup + p2*psi

    Returns (min G1, min G2); nan if no pair qualifies. A negative value means
    p1 resp. p2 is too small and the fallback has to repair it.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)

    psi_all = Lf_h_all + float(p1) * h_all
    g1_rows = active & (h_all > EPSILON_DENOMINATOR)
    min_g1 = float(np.min(psi_all[g1_rows])) if np.any(g1_rows) else float("nan")

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)
    Lf_psi_all = vrel_sq2_all + float(p1) * Lf_h_all
    S_sup_all = np.sum(np.where(Lg_psi_all >= 0.0, Lg_psi_all * ddq_max_kin, Lg_psi_all * ddq_min_kin), axis=1)
    g2_all = Lf_psi_all + S_sup_all + float(p2) * psi_all
    g2_rows = active & (psi_all > EPSILON_DENOMINATOR)
    min_g2 = float(np.min(g2_all[g2_rows])) if np.any(g2_rows) else float("nan")

    return min_g1, min_g2


def g12_values_hold(min_g1: float, min_g2: float, tol: float) -> bool:
    """Accept nan margins when there are no active rows, otherwise require >= -tol."""
    g1_ok = (not np.isfinite(min_g1)) or (min_g1 >= -float(tol))
    g2_ok = (not np.isfinite(min_g2)) or (min_g2 >= -float(tol))
    return bool(g1_ok and g2_ok)


def minimal_g12_p12_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    p1_floor: float = 1e-3,
    p2_floor: float = 1e-3,
    pair_scope: str = "all",
) -> tuple[float, float, float, float]:
    """
    Compute an independent fallback p1,p2 from the current G1/G2 conditions.

    This deliberately does NOT use the scenario YAML gamma/beta values as a
    baseline/floor. It computes the smallest current-state gains suggested by
    the G1/G2 inequalities, then clips them to the same caps used elsewhere.
    The NN fallback passes gamma_NN as p1_floor, which gives
    gamma_final = max(gamma_NN, gamma_req), and a small constant p2_floor, so
    beta_final is the minimal (conservative) required beta. psi is evaluated
    with gamma_final, so beta is solved against gamma_final.

    G1 condition:
        G1 = Lf_h + p1*h >= 0
        p1 >= -Lf_h/h, if h > 0

    G2 condition after p1 is selected:
        G2 = Lf_psi + Lg_psi*u + p2*psi >= 0

    For G2, we use the same acceleration-bound support estimate as the Davide
    online calculation. This is not a QP over p1,p2; it is the analytic/current
    state gain calculation analogous to the 2D G12 fallback.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)

    if not np.any(active):
        return float(p1_floor), float(p2_floor), float("nan"), float("nan")

    h = h_all[active]
    Lf_h = Lf_h_all[active]
    vrel_sq2 = vrel_sq2_all[active]
    Lg = Lg_psi_all[active, :]

    max_gamma_required = -float("inf")
    for hi, lfi in zip(h, Lf_h):
        if hi > EPSILON_DENOMINATOR:
            gamma_needed = -float(lfi) / float(hi)
            max_gamma_required = max(max_gamma_required, gamma_needed)
        # h <= 0: already in collision. Raising p1 only makes psi more
        # negative and the QP row harder, so such pairs are skipped (eq. 21).

    if max_gamma_required > -float("inf"):
        gamma = max(float(p1_floor), max(0.0, max_gamma_required + GAMMA_ADJUST_BUFFER))
        gamma = min(gamma, GAMMA_MAX_LIMIT)
    else:
        gamma = float(p1_floor)

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)

    psi = Lf_h + gamma * h
    Lf_psi = vrel_sq2 + gamma * Lf_h

    max_beta_required = -float("inf")
    for psii, lfpsii, lgi in zip(psi, Lf_psi, Lg):
        sup_lg_u = float(np.sum(np.where(lgi >= 0.0, lgi * ddq_max_kin, lgi * ddq_min_kin)))
        S_sup = float(lfpsii) + sup_lg_u

        if psii > EPSILON_DENOMINATOR:
            beta_needed = -S_sup / float(psii)
            max_beta_required = max(max_beta_required, beta_needed)
        # psi <= 0: raising p2 only tightens the QP row, so such pairs are
        # skipped (eq. 23).

    if max_beta_required > -float("inf"):
        beta = max(float(p2_floor), max(0.0, max_beta_required + BETA_ADJUST_BUFFER))
        beta = min(beta, BETA_MAX_LIMIT)
    else:
        beta = float(p2_floor)

    return float(gamma), float(beta), float(max_gamma_required), float(max_beta_required)


def build_and_solve_hard_qp_for_p12(
    *,
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    ddq_nominal: np.ndarray,
    all_pair_terms: dict,
    joint_limits: dict,
    p1: float,
    p2: float,
    dt: float,
    eps_abs: float,
    eps_rel: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, str, float]:
    """Build the hard FR3 QP for a candidate p1,p2 and solve it once."""
    u_nom, A_full, b_full, mask_full = build_full_qp_numpy(
        q_arm=q_arm,
        dq_arm=dq_arm,
        u_nom=ddq_nominal,
        h_all=all_pair_terms["h_all"],
        Lf_h_all=all_pair_terms["Lf_h_all"],
        vrel_sq2_all=all_pair_terms["vrel_sq2_all"],
        Lg_psi_all=all_pair_terms["Lg_psi_all"],
        pair_mask=all_pair_terms["pair_mask"],
        p1=p1,
        p2=p2,
        dt=dt,
        ddq_min_arm=joint_limits["ddq_min_arm"],
        ddq_max_arm=joint_limits["ddq_max_arm"],
        dq_min_arm=joint_limits["dq_min_arm"],
        dq_max_arm=joint_limits["dq_max_arm"],
        q_min_arm=joint_limits["q_min_arm"],
        q_max_arm=joint_limits["q_max_arm"],
    )

    ddq_safe, qp_solved, qp_status, solve_time_s = solve_hard_qp_osqp(
        u_nom=u_nom,
        A_full=A_full,
        b_full=b_full,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
    )
    return u_nom, A_full, b_full, mask_full, ddq_safe, qp_solved, qp_status, solve_time_s


# ---------------------------------------------------------------------
# Per-pair beta policy (modes nn_pp_beta, nn_pp_beta_fallback)
# ---------------------------------------------------------------------
def load_pp_pack(model_path: str, stats_path: str | None, args):
    """
    PerPairBetaNet checkpoint -> ("pp", model, feat_mean, feat_std).

    An old P12ParamNet checkpoint is accepted as well -> ("broadcast", nn_pack):
    beta_j = shared beta_NN for all pairs. Used for the sanity check that the
    per-pair path reproduces nn / nn_g12_fallback exactly.
    """
    ckpt = torch.load(os.path.expanduser(model_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("arch") == PP_ARCH:
        model, feat_mean, feat_std = load_pp_policy(os.path.expanduser(model_path))
        return ("pp", model, feat_mean, feat_std)
    return ("broadcast", load_nn_policy(model_path, stats_path, args.hidden_dim, args.p1_floor, args.p2_floor))


@torch.no_grad()
def predict_pp_current(
    pp_pack,
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    ddq_nominal_arm: np.ndarray,
    joint_limits: dict,
    all_pair_terms: dict,
    dt: float,
) -> tuple[float, np.ndarray]:
    """Return gamma (scalar) and beta (43,) for the current state."""
    if pp_pack[0] == "broadcast":
        nn_model, feat_mean, feat_std = pp_pack[1]
        p1, p2 = predict_p12_nn_current(
            model=nn_model,
            feat_mean=feat_mean,
            feat_std=feat_std,
            q_arm=q_arm,
            dq_arm=dq_arm,
            ddq_nominal_arm=ddq_nominal_arm,
            joint_limits=joint_limits,
            h_all=all_pair_terms["h_all"].copy(),
            Lf_h_all=all_pair_terms["Lf_h_all"].copy(),
        )
        return p1, np.full(NUM_PAIR_ROWS, p2, dtype=float)

    _, model, feat_mean, feat_std = pp_pack
    ddq_nominal_arm = np.asarray(ddq_nominal_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    ddq_scale = np.clip(
        np.maximum(np.abs(np.asarray(joint_limits["ddq_min_arm"], dtype=float)), np.abs(np.asarray(joint_limits["ddq_max_arm"], dtype=float))),
        1e-6,
        None,
    )
    z_raw = np.concatenate(
        [q_arm, dq_arm, ddq_nominal_arm, ddq_nominal_arm / ddq_scale, all_pair_terms["h_all"], all_pair_terms["Lf_h_all"]]
    )[None, :]
    z_t = torch.tensor(normalize_features_numpy(z_raw, feat_mean, feat_std), dtype=torch.double)
    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, joint_limits, dt)

    def t(x):
        return torch.tensor(np.asarray(x, dtype=float), dtype=torch.double).unsqueeze(0)

    gamma, beta = model(
        z_t,
        {
            "h_all": t(all_pair_terms["h_all"]),
            "Lf_h_all": t(all_pair_terms["Lf_h_all"]),
            "vrel_sq2_all": t(all_pair_terms["vrel_sq2_all"]),
            "Lg_psi_all": t(all_pair_terms["Lg_psi_all"]),
            "pair_mask": t(all_pair_terms["pair_mask"]),
            "u_nom": t(ddq_nominal_arm),
            "ddq_min_kin": t(ddq_min_kin),
            "ddq_max_kin": t(ddq_max_kin),
        },
    )
    return float(gamma[0, 0]), beta[0].numpy().astype(float)


def pp_g12_fallback_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    gamma_nn: float,
    beta_nn: np.ndarray,
    pair_scope: str,
    tol: float,
    p1_floor: float,
    p2_floor: float,
    reduce: str = "per_pair",
    unflagged: str = "keep",
) -> dict:
    """
    G12 fallback for a per-pair beta vector.

    Pre-QP check: same G1/G2 rows as g12_check_current, but G2_j uses beta_NN_j.
    If it holds, the NN gains are kept. Otherwise:
      gamma: unchanged rule, gamma_final = max(gamma_NN, gamma_req + 0.5) (capped).
      reduce="scalar": exactly the existing fallback (beta_req reduced by max over
          pairs, applied to every pair). Sanity-check path.
      reduce="per_pair": G2_j is re-checked with gamma_final and beta_NN_j; pairs that
          fail get beta_j = max(p2_floor, beta_req_j + 0.5) (capped). Unflagged pairs:
            unflagged="keep":       beta_NN_j
            unflagged="lower_pair": min(beta_NN_j, max(p2_floor, beta_req_j + 0.5)) for psi_j > 0
            unflagged="cap_old":    min(beta_NN_j, B), B = the existing scalar fallback beta
                                    (max over pairs of beta_req + 0.5). Equals the existing
                                    fallback when all beta_NN_j are equal and B <= beta_NN.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)
    beta_nn = np.asarray(beta_nn, dtype=float).reshape(NUM_PAIR_ROWS)

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)
    S_sup_all = np.sum(np.where(Lg_psi_all >= 0.0, Lg_psi_all * ddq_max_kin, Lg_psi_all * ddq_min_kin), axis=1)

    psi_nn = Lf_h_all + float(gamma_nn) * h_all
    g1_rows = active & (h_all > EPSILON_DENOMINATOR)
    g2_rows = active & (psi_nn > EPSILON_DENOMINATOR)
    g2_nn = vrel_sq2_all + float(gamma_nn) * Lf_h_all + S_sup_all + beta_nn * psi_nn
    min_g1 = float(np.min(psi_nn[g1_rows])) if np.any(g1_rows) else float("nan")
    min_g2 = float(np.min(g2_nn[g2_rows])) if np.any(g2_rows) else float("nan")

    no_pairs = np.zeros(NUM_PAIR_ROWS, dtype=bool)
    out = {
        "gamma": float(gamma_nn),
        "beta": beta_nn.copy(),
        "fallback_used": False,
        "fallback_reason": "",
        "min_g1_candidate": min_g1,
        "min_g2_candidate": min_g2,
        "max_gamma_required": float("nan"),
        "max_beta_required": float("nan"),
        "fb_pairs": no_pairs.copy(),
        "lowered_pairs": no_pairs.copy(),
        "beta_cap_pairs": no_pairs.copy(),
        "gamma_cap_hit": False,
    }
    if g12_values_hold(min_g1=min_g1, min_g2=min_g2, tol=tol):
        return out

    out["fallback_used"] = True
    out["fallback_reason"] = "G1_negative" if (np.isfinite(min_g1) and min_g1 < -float(tol)) else "G2_sup_negative"

    gamma, beta_scalar, max_gamma_required, max_beta_required = minimal_g12_p12_current(
        q_arm=q_arm,
        dq_arm=dq_arm,
        all_pair_terms=all_pair_terms,
        limits=limits,
        dt=dt,
        p1_floor=max(float(p1_floor), float(gamma_nn)),
        p2_floor=float(p2_floor),
        pair_scope=pair_scope,
    )
    out["gamma"] = float(gamma)
    out["max_gamma_required"] = float(max_gamma_required)
    out["gamma_cap_hit"] = bool(np.isfinite(max_gamma_required) and max_gamma_required + GAMMA_ADJUST_BUFFER > GAMMA_MAX_LIMIT)

    if reduce == "scalar":
        out["beta"] = np.full(NUM_PAIR_ROWS, float(beta_scalar), dtype=float)
        out["max_beta_required"] = float(max_beta_required)
        out["fb_pairs"] = active.copy()
        cap = bool(np.isfinite(max_beta_required) and max_beta_required + BETA_ADJUST_BUFFER > BETA_MAX_LIMIT)
        out["beta_cap_pairs"] = active & cap
        return out
    if reduce != "per_pair":
        raise ValueError(f"Unknown pp fallback reduce: {reduce}")

    # Per pair, against gamma_final. psi_j <= 0 rows are skipped as in eq. 23.
    psi = Lf_h_all + gamma * h_all
    Lf_psi = vrel_sq2_all + gamma * Lf_h_all
    rows = active & (psi > EPSILON_DENOMINATOR)
    beta_req = np.full(NUM_PAIR_ROWS, -np.inf)
    beta_req[rows] = -(Lf_psi[rows] + S_sup_all[rows]) / psi[rows]
    g2 = Lf_psi + S_sup_all + beta_nn * psi
    flagged = rows & (g2 < -float(tol))
    target = np.minimum(np.maximum(float(p2_floor), np.maximum(0.0, beta_req + BETA_ADJUST_BUFFER)), BETA_MAX_LIMIT)

    beta = beta_nn.copy()
    beta[flagged] = target[flagged]
    lowered = no_pairs.copy()
    if unflagged == "lower_pair":
        lowered = rows & ~flagged & (target < beta_nn)
        beta[lowered] = target[lowered]
    elif unflagged == "cap_old":
        lowered = active & ~flagged & (float(beta_scalar) < beta_nn)
        beta[lowered] = float(beta_scalar)
    elif unflagged != "keep":
        raise ValueError(f"Unknown pp unflagged rule: {unflagged}")

    out["beta"] = beta
    out["fb_pairs"] = flagged
    out["lowered_pairs"] = lowered
    out["beta_cap_pairs"] = flagged & (beta_req + BETA_ADJUST_BUFFER > BETA_MAX_LIMIT)
    out["max_beta_required"] = float(np.max(beta_req[rows])) if np.any(rows) else float("nan")
    return out


def _pairs_str(mask: np.ndarray) -> str:
    return ";".join(str(int(j)) for j in np.flatnonzero(mask))


def _pair_values_str(values: np.ndarray, mask: np.ndarray) -> str:
    return ";".join(f"{int(j)}:{float(values[j]):.6g}" for j in np.flatnonzero(mask))


# ---------------------------------------------------------------------
# rollout simulation
# ---------------------------------------------------------------------


def rollout_one_policy(
    context: dict,
    scenario_path: str,
    mode: str,
    args,
    nn_pack=None,
    pp_pack=None,
    pp_log: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    model = context["model"]
    data = model.createData()
    ee_frame_id = context["ee_frame_id"]
    active_links = context["active_links"]
    self_collision_link_pair = context["self_collision_link_pair"]
    joint_limits = context["joint_limits"]

    scenario = load_scenario_yaml(scenario_path)
    dt = float(args.dt)

    q_full, dq_full = get_default_initial_state(model)
    q0_arm = q_full[:NUM_ARM_JOINTS].copy()

    obstacles = clone_obstacles(scenario["obstacles"])
    goal = np.asarray(scenario["goal_ee_pos"], dtype=float)
    d_margin = float(scenario["d_margin"])
    initial_gamma = float(scenario["gamma"])
    initial_beta = float(scenario["beta"])

    max_steps_from_scenario = int(float(scenario["max_sim_duration_s"]) / dt)
    max_steps = args.max_steps if args.max_steps is not None else max_steps_from_scenario
    max_steps = min(int(max_steps), max_steps_from_scenario)

    # target smoothing starts from current EE position, matching fr3_hocbf/sim/rollout.py.
    target_prev = ee_position(model, data, ee_frame_id, q_full, dq_full)

    at_goal_timer = 0.0
    success = False
    final_status = "MAX_STEPS"

    rows: list[dict] = []
    path_length_q = 0.0
    qp_fail_count = 0
    fallback_count = 0
    hard_viol_max_overall = 0.0
    min_h_overall = float("inf")
    min_psi_overall = float("inf")
    min_dist_overall = float("inf")
    sum_intervention = 0.0
    sum_solve_time = 0.0
    jerk_sq_values: list[float] = []
    prev_ddq_safe = None

    t_rollout_start = time.time()

    for k in range(max_steps):
        step_t0 = time.time()

        q_arm = q_full[:NUM_ARM_JOINTS].copy()
        dq_arm = dq_full[:NUM_ARM_JOINTS].copy()

        # Match fr3_hocbf/sim/rollout.py convention: move obstacles before geometry/QP.
        next_obstacles = update_obstacles(obstacles, dt) if args.move_obstacles else clone_obstacles(obstacles)

        all_pair_terms = compute_all_pair_terms(
            model=model,
            data=data,
            q_full=q_full,
            dq_full=dq_full,
            obstacles=next_obstacles,
            active_links=active_links,
            self_collision_link_pair=self_collision_link_pair,
            d_margin=d_margin,
        )

        ddq_nominal, target_next = nominal_controller_js_standalone(
            model=model,
            data=data,
            ee_frame_id=ee_frame_id,
            q_arm_curr=q_arm,
            dq_arm_curr=dq_arm,
            target_ee_pos_cartesian=goal,
            target=target_prev,
        )

        h_feat_all = all_pair_terms["h_all"].copy()
        Lf_h_feat_all = all_pair_terms["Lf_h_all"].copy()

        max_gamma_required = float("nan")
        max_beta_required = float("nan")
        fallback_used = False
        fallback_reason = ""
        p1_nn = float("nan")
        p2_nn = float("nan")
        p1_fallback = float("nan")
        p2_fallback = float("nan")
        min_g1_candidate = float("nan")
        min_g2_candidate = float("nan")
        pp_step = None

        if mode == "original":
            p1 = initial_gamma
            p2 = initial_beta
        elif mode == "fixed":
            p1 = float(args.p1_fixed)
            p2 = float(args.p2_fixed)
        elif mode == "davide_online":
            p1, p2, max_gamma_required, max_beta_required = davide_online_p12_current(
                q_arm=q_arm,
                dq_arm=dq_arm,
                all_pair_terms=all_pair_terms,
                limits=joint_limits,
                dt=dt,
                initial_gamma=initial_gamma,
                initial_beta=initial_beta,
                pair_scope=args.online_pair_scope,
            )
            if p1 > initial_gamma or p2 > initial_beta:
                fallback_used = True
                fallback_count += 1
                fallback_reason = "G1_raised" if p1 > initial_gamma else "G2_sup_raised"
        elif mode in ("nn", "nn_g12_fallback"):
            if nn_pack is None:
                raise RuntimeError("NN modes require --model")
            nn_model, feat_mean, feat_std = nn_pack
            p1, p2 = predict_p12_nn_current(
                model=nn_model,
                feat_mean=feat_mean,
                feat_std=feat_std,
                q_arm=q_arm,
                dq_arm=dq_arm,
                ddq_nominal_arm=ddq_nominal,
                joint_limits=joint_limits,
                h_all=h_feat_all,
                Lf_h_all=Lf_h_feat_all,
            )
            p1_nn = float(p1)
            p2_nn = float(p2)

            if mode == "nn_g12_fallback" and args.nn_g12_fallback:
                # Analytic pre-QP check of the NN gains (paper Sec. II-D).
                min_g1_candidate, min_g2_candidate = g12_check_current(
                    q_arm=q_arm,
                    dq_arm=dq_arm,
                    all_pair_terms=all_pair_terms,
                    limits=joint_limits,
                    dt=dt,
                    p1=p1,
                    p2=p2,
                    pair_scope=args.g12_fallback_pair_scope,
                )
                nn_holds = g12_values_hold(
                    min_g1=min_g1_candidate,
                    min_g2=min_g2_candidate,
                    tol=args.g12_fallback_tol,
                )

                if not nn_holds:
                    fallback_used = True
                    fallback_count += 1
                    if np.isfinite(min_g1_candidate) and min_g1_candidate < -float(args.g12_fallback_tol):
                        fallback_reason = "G1_negative"
                    else:
                        fallback_reason = "G2_sup_negative"

                    # Independent fallback: compute minimal current-state p1,p2
                    # from G1/G2, without using scenario gamma/beta as floors.
                    # gamma_final = max(gamma_NN, gamma_req). beta is conservative:
                    # beta_final = beta_req (+ buffer), which may lie below beta_NN,
                    # so psi may only decrease slowly and the robot brakes earlier.
                    p1_floor = max(float(args.g12_fallback_p1_floor), p1_nn)
                    p2_floor = float(args.g12_fallback_p2_floor)
                    p1, p2, max_gamma_required, max_beta_required = minimal_g12_p12_current(
                        q_arm=q_arm,
                        dq_arm=dq_arm,
                        all_pair_terms=all_pair_terms,
                        limits=joint_limits,
                        dt=dt,
                        p1_floor=p1_floor,
                        p2_floor=p2_floor,
                        pair_scope=args.g12_fallback_pair_scope,
                    )
                    p1_fallback = float(p1)
                    p2_fallback = float(p2)
        elif mode in ("nn_pp_beta", "nn_pp_beta_fallback"):
            if pp_pack is None:
                raise RuntimeError("Per-pair modes require --pp-model")
            gamma_nn, beta_nn = predict_pp_current(
                pp_pack=pp_pack,
                q_arm=q_arm,
                dq_arm=dq_arm,
                ddq_nominal_arm=ddq_nominal,
                joint_limits=joint_limits,
                all_pair_terms=all_pair_terms,
                dt=dt,
            )
            pp_step = {
                "gamma": gamma_nn,
                "beta": beta_nn,
                "fb_pairs": np.zeros(NUM_PAIR_ROWS, dtype=bool),
                "lowered_pairs": np.zeros(NUM_PAIR_ROWS, dtype=bool),
                "beta_cap_pairs": np.zeros(NUM_PAIR_ROWS, dtype=bool),
                "gamma_cap_hit": False,
            }
            if mode == "nn_pp_beta_fallback":
                pp_step = pp_g12_fallback_current(
                    q_arm=q_arm,
                    dq_arm=dq_arm,
                    all_pair_terms=all_pair_terms,
                    limits=joint_limits,
                    dt=dt,
                    gamma_nn=gamma_nn,
                    beta_nn=beta_nn,
                    pair_scope=args.g12_fallback_pair_scope,
                    tol=args.g12_fallback_tol,
                    p1_floor=args.g12_fallback_p1_floor,
                    p2_floor=args.g12_fallback_p2_floor,
                    reduce=args.pp_fallback_reduce,
                    unflagged=args.pp_unflagged,
                )
                fallback_used = pp_step["fallback_used"]
                fallback_reason = pp_step["fallback_reason"]
                min_g1_candidate = pp_step["min_g1_candidate"]
                min_g2_candidate = pp_step["min_g2_candidate"]
                max_gamma_required = pp_step["max_gamma_required"]
                max_beta_required = pp_step["max_beta_required"]
                if fallback_used:
                    fallback_count += 1
            p1 = float(pp_step["gamma"])
            p2 = pp_step["beta"]  # (43,): HOCBF row j uses beta_j
            p1_nn = float(gamma_nn)
            if fallback_used:
                p1_fallback = p1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Scalar beta for the step log; per-pair modes log the mean over active pairs
        # (all pairs if none is active) and write beta_j to the separate pp step log.
        if np.ndim(p2) == 0:
            p2_log = float(p2)
        else:
            pair_active_pre = np.asarray(all_pair_terms["pair_mask"]) > 0.5
            p2_log = float(np.mean(p2[pair_active_pre])) if np.any(pair_active_pre) else float(np.mean(p2))
            p2_nn = (
                float(np.mean(beta_nn[pair_active_pre])) if np.any(pair_active_pre) else float(np.mean(beta_nn))
            )
            if fallback_used:
                p2_fallback = p2_log

        (
            u_nom,
            A_full,
            b_full,
            mask_full,
            ddq_safe,
            qp_solved,
            qp_status,
            solve_time_s,
        ) = build_and_solve_hard_qp_for_p12(
            q_arm=q_arm,
            dq_arm=dq_arm,
            ddq_nominal=ddq_nominal,
            all_pair_terms=all_pair_terms,
            joint_limits=joint_limits,
            p1=p1,
            p2=p2,
            dt=dt,
            eps_abs=args.eps_abs,
            eps_rel=args.eps_rel,
            max_iter=args.max_iter,
        )

        if qp_solved:
            hard_viol, min_qp_margin = hard_violation_numpy(ddq_safe, A_full, b_full)

            # Signed QP margins:
            # margin >= 0 means the row constraint is satisfied.
            qp_margin = A_full @ ddq_safe - b_full

            # First NUM_PAIR_ROWS rows are HOCBF/collision rows.
            pair_active = mask_full[:NUM_PAIR_ROWS] > 0.5
            if np.any(pair_active):
                min_g2 = float(np.min(qp_margin[:NUM_PAIR_ROWS][pair_active]))
            else:
                min_g2 = float("nan")

            # Remaining rows are joint acceleration / velocity / position bound rows.
            bound_active = mask_full[NUM_PAIR_ROWS:] > 0.5
            if np.any(bound_active):
                min_bound_margin = float(np.min(qp_margin[NUM_PAIR_ROWS:][bound_active]))
            else:
                min_bound_margin = float("nan")

            next_dq_arm = dq_arm + ddq_safe * dt
            next_q_arm = q_arm + dq_arm * dt + 0.5 * ddq_safe * (dt ** 2)

        else:
            # Infeasible hard QP: terminate immediately, no recovery hold
            # (paper Sec. III-C).
            qp_fail_count += 1
            final_status = "QP_FAIL"
            break

        next_q_full = q_full.copy()
        next_dq_full = dq_full.copy()
        next_q_full[:NUM_ARM_JOINTS] = next_q_arm
        next_dq_full[:NUM_ARM_JOINTS] = next_dq_arm

        # Metrics on the post-state and next obstacle state.
        metrics = compute_barrier_metrics(
            model=model,
            data=data,
            q_full=next_q_full,
            dq_full=next_dq_full,
            obstacles=next_obstacles,
            active_links=active_links,
            self_collision_link_pair=self_collision_link_pair,
            d_margin=d_margin,
            gamma=p1,
        )

        ee_pos = ee_position(model, data, ee_frame_id, next_q_full, next_dq_full)
        dist_to_goal = float(np.linalg.norm(ee_pos - goal))

        min_h = float(metrics["min_h"])
        min_psi = float(metrics["min_psi"])
        min_dist = float(metrics["min_dist"])

        min_h_overall = min(min_h_overall, min_h)
        min_psi_overall = min(min_psi_overall, min_psi)
        min_dist_overall = min(min_dist_overall, min_dist)
        if np.isfinite(hard_viol):
            hard_viol_max_overall = max(hard_viol_max_overall, hard_viol)

        intervention_l2 = float(np.linalg.norm(ddq_safe - ddq_nominal)) if qp_solved else float("nan")
        if np.isfinite(intervention_l2):
            sum_intervention += intervention_l2
        sum_solve_time += solve_time_s
        path_length_q += float(np.linalg.norm(next_q_arm - q_arm))

        if prev_ddq_safe is not None and qp_solved:
            jerk_sq_values.append(float(np.mean(((ddq_safe - prev_ddq_safe) / dt) ** 2)))
        if qp_solved:
            prev_ddq_safe = ddq_safe.copy()

        active_pairs = int(np.sum(all_pair_terms["pair_mask"] > 0.5))
        min_g1_pre = min_g1_current(
            all_pair_terms=all_pair_terms,
            p1=p1,
            pair_scope=args.g12_fallback_pair_scope,
        )
        step_elapsed_s = time.time() - step_t0

        rows.append(
            {
                "scenario_path": scenario_path,
                "scenario": Path(scenario_path).stem,
                "mode": mode,
                "step": k,
                "time": k * dt,
                "qp_solved": bool(qp_solved),
                "qp_status": qp_status,
                "p1": float(p1),
                "p2": p2_log,
                "p1_nn": p1_nn,
                "p2_nn": p2_nn,
                "p1_fallback": p1_fallback,
                "p2_fallback": p2_fallback,
                "fallback_used": bool(fallback_used),
                "fallback_reason": fallback_reason,
                "min_g1_candidate": min_g1_candidate,
                "min_g2_candidate": min_g2_candidate,
                "max_gamma_required": max_gamma_required,
                "max_beta_required": max_beta_required,
                "min_g1_pre": min_g1_pre,
                "min_h": min_h,
                "min_psi": min_psi,
                "min_dist": min_dist,
                "dist_to_goal": dist_to_goal,
                "hard_qp_violation": hard_viol,
                "min_qp_margin": min_qp_margin,
                "min_g2": min_g2,
                "min_bound_margin": min_bound_margin,
                "intervention_l2": intervention_l2,
                "ddq_nom_norm": float(np.linalg.norm(ddq_nominal)),
                "ddq_safe_norm": float(np.linalg.norm(ddq_safe)) if qp_solved else float("nan"),
                "solve_time_s": solve_time_s,
                "step_elapsed_s": step_elapsed_s,
                "active_pairs": active_pairs,
                # pre-state, post-state, and controls for validation against saved rollout NPZ
                "q_pre0": float(q_arm[0]),
                "q_pre1": float(q_arm[1]),
                "q_pre2": float(q_arm[2]),
                "q_pre3": float(q_arm[3]),
                "q_pre4": float(q_arm[4]),
                "q_pre5": float(q_arm[5]),
                "q_pre6": float(q_arm[6]),
                "dq_pre0": float(dq_arm[0]),
                "dq_pre1": float(dq_arm[1]),
                "dq_pre2": float(dq_arm[2]),
                "dq_pre3": float(dq_arm[3]),
                "dq_pre4": float(dq_arm[4]),
                "dq_pre5": float(dq_arm[5]),
                "dq_pre6": float(dq_arm[6]),
                "q_post0": float(next_q_arm[0]),
                "q_post1": float(next_q_arm[1]),
                "q_post2": float(next_q_arm[2]),
                "q_post3": float(next_q_arm[3]),
                "q_post4": float(next_q_arm[4]),
                "q_post5": float(next_q_arm[5]),
                "q_post6": float(next_q_arm[6]),
                "dq_post0": float(next_dq_arm[0]),
                "dq_post1": float(next_dq_arm[1]),
                "dq_post2": float(next_dq_arm[2]),
                "dq_post3": float(next_dq_arm[3]),
                "dq_post4": float(next_dq_arm[4]),
                "dq_post5": float(next_dq_arm[5]),
                "dq_post6": float(next_dq_arm[6]),
                "ddq_nom0": float(ddq_nominal[0]),
                "ddq_nom1": float(ddq_nominal[1]),
                "ddq_nom2": float(ddq_nominal[2]),
                "ddq_nom3": float(ddq_nominal[3]),
                "ddq_nom4": float(ddq_nominal[4]),
                "ddq_nom5": float(ddq_nominal[5]),
                "ddq_nom6": float(ddq_nominal[6]),
                "ddq_safe0": float(ddq_safe[0]),
                "ddq_safe1": float(ddq_safe[1]),
                "ddq_safe2": float(ddq_safe[2]),
                "ddq_safe3": float(ddq_safe[3]),
                "ddq_safe4": float(ddq_safe[4]),
                "ddq_safe5": float(ddq_safe[5]),
                "ddq_safe6": float(ddq_safe[6]),
            }
        )

        if pp_step is not None and pp_log is not None:
            pair_active_pre = np.asarray(all_pair_terms["pair_mask"]) > 0.5
            pp_log.append(
                {
                    "scenario": Path(scenario_path).stem,
                    "mode": mode,
                    "step": k,
                    "gamma": float(p1),
                    "gamma_nn": p1_nn,
                    "fallback_used": bool(fallback_used),
                    "gamma_raised": bool(p1 > p1_nn),
                    "gamma_cap_hit": bool(pp_step["gamma_cap_hit"]),
                    "n_active": int(np.sum(pair_active_pre)),
                    "n_fb_pairs": int(np.sum(pp_step["fb_pairs"])),
                    "fb_pairs": _pairs_str(pp_step["fb_pairs"]),
                    "n_lowered_pairs": int(np.sum(pp_step["lowered_pairs"])),
                    "lowered_pairs": _pairs_str(pp_step["lowered_pairs"]),
                    "n_beta_cap_pairs": int(np.sum(pp_step["beta_cap_pairs"])),
                    "beta_cap_pairs": _pairs_str(pp_step["beta_cap_pairs"]),
                    "beta_active": _pair_values_str(p2, pair_active_pre),
                    "beta_nn_active": _pair_values_str(beta_nn, pair_active_pre),
                }
            )

        if dist_to_goal < float(scenario["goal_tolerance"]):
            at_goal_timer += dt
            if at_goal_timer >= float(scenario["goal_settle_time_s"]):
                success = True
                final_status = "SUCCESS"
                q_full = next_q_full
                dq_full = next_dq_full
                obstacles = next_obstacles
                target_prev = target_next
                break
        else:
            at_goal_timer = 0.0

        q_full = next_q_full
        dq_full = next_dq_full
        obstacles = next_obstacles
        target_prev = target_next

    num_steps = len(rows)
    task_completion_time_s = float(num_steps * dt) if success else float("nan")
    final_q_arm = q_full[:NUM_ARM_JOINTS].copy()
    direct_q_dist = float(np.linalg.norm(final_q_arm - q0_arm))
    path_inefficiency = float(path_length_q / max(direct_q_dist, 1e-12))
    mean_squared_jerk = float(np.mean(jerk_sq_values)) if jerk_sq_values else 0.0
    mean_intervention = float(sum_intervention / max(1, num_steps - qp_fail_count))
    mean_solve_time = float(sum_solve_time / max(1, num_steps))

    if not success and qp_fail_count > 0:
        final_status = "QP_FAIL_OR_MAX_STEPS"

    rollout_elapsed_s = time.time() - t_rollout_start

    final_ee_pos = ee_position(model, data, ee_frame_id, q_full, dq_full)
    final_dist_to_goal = float(np.linalg.norm(final_ee_pos - goal))

    summary = {
        "scenario_path": scenario_path,
        "scenario": Path(scenario_path).stem,
        "mode": mode,
        "status": final_status,
        "success": bool(success),
        "steps": int(num_steps),
        "sim_time_s": float(num_steps * dt),
        "task_completion_time_s": task_completion_time_s,
        "rollout_elapsed_s": rollout_elapsed_s,
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(1, num_steps)),
        "qp_fail_count": int(qp_fail_count),
        "qp_fail_rate": float(qp_fail_count / max(1, num_steps)),
        "min_h": float(min_h_overall),
        "min_psi": float(min_psi_overall),
        "min_dist": float(min_dist_overall),
        "max_hard_qp_violation": float(hard_viol_max_overall),
        "final_dist_to_goal": final_dist_to_goal,
        "goal_tolerance": float(scenario["goal_tolerance"]),
        "path_length_q": float(path_length_q),
        "path_inefficiency": path_inefficiency,
        "mean_squared_jerk": mean_squared_jerk,
        "mean_intervention_l2": mean_intervention,
        "mean_solve_time_s": mean_solve_time,
        "p1_mean": float(np.mean([r["p1"] for r in rows])) if rows else float("nan"),
        "p1_max": float(np.max([r["p1"] for r in rows])) if rows else float("nan"),
        "p2_mean": float(np.mean([r["p2"] for r in rows])) if rows else float("nan"),
        "p2_max": float(np.max([r["p2"] for r in rows])) if rows else float("nan"),
        "initial_gamma": initial_gamma,
        "initial_beta": initial_beta,
        "d_margin": d_margin,
    }

    return summary, rows


# ---------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="Scenario YAML(s), directories, globs, or txt files")
    parser.add_argument("--modes", type=str, default="original,davide_online,fixed", help="Comma-separated modes. Valid: original,fixed,davide_online,nn,nn_g12_fallback,nn_pp_beta,nn_pp_beta_fallback")

    parser.add_argument("--p1-fixed", type=float, default=2.0)
    parser.add_argument("--p2-fixed", type=float, default=3.0)

    parser.add_argument("--model", type=str, default=None, help="NN checkpoint for modes nn and nn_g12_fallback")
    parser.add_argument("--stats", type=str, default=None, help="Feature stats .npz for NN modes. Optional if checkpoint contains feat_mean/feat_std")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--p1-floor", type=float, default=1e-3)
    parser.add_argument("--p2-floor", type=float, default=1e-3)

    parser.add_argument(
        "--online-pair-scope",
        choices=["obstacle", "all"],
        default="all",
        help="Pair rows used by the Davide online (Fixed + Fallback) gain update.",
    )
    parser.add_argument(
        "--nn-g12-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For mode nn_g12_fallback, test the NN candidate with current G1/G2 and fall back to independent minimal G12 gains if it fails. Pure mode nn never uses fallback.",
    )
    parser.add_argument(
        "--g12-fallback-pair-scope",
        choices=["obstacle", "all"],
        default="all",
        help="Pair rows monitored and used by the independent NN G12 fallback.",
    )
    parser.add_argument(
        "--g12-fallback-tol",
        type=float,
        default=1e-9,
        help="Tolerance for NN G12 fallback. Fallback if min G1 or min G2 is below -tol.",
    )
    parser.add_argument("--g12-fallback-p1-floor", type=float, default=1e-3)
    parser.add_argument("--g12-fallback-p2-floor", type=float, default=1e-3)

    parser.add_argument(
        "--pp-model",
        type=str,
        default=None,
        help="PerPairBetaNet checkpoint for modes nn_pp_beta and nn_pp_beta_fallback. An old P12ParamNet "
        "checkpoint is broadcast to beta_j = beta_NN (sanity check).",
    )
    parser.add_argument(
        "--pp-fallback-reduce",
        choices=["per_pair", "scalar"],
        default="per_pair",
        help="nn_pp_beta_fallback: per_pair applies beta_req_j to the flagged pairs only; scalar reproduces the "
        "existing fallback (max over pairs, applied to all pairs).",
    )
    parser.add_argument(
        "--pp-unflagged",
        choices=["keep", "lower_pair", "cap_old"],
        default="keep",
        help="nn_pp_beta_fallback, per_pair: beta of pairs that pass the G2 re-check on fallback steps. "
        "keep: beta_NN_j; lower_pair: min(beta_NN_j, beta_req_j + buffer); "
        "cap_old: min(beta_NN_j, existing scalar fallback beta).",
    )

    parser.add_argument("--dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--max-steps", type=int, default=None, help="Debug cap. Default uses scenario max_sim_duration_s")
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--move-obstacles", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eps-abs", type=float, default=1e-5)
    parser.add_argument("--eps-rel", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=25000)

    parser.add_argument("--out-prefix", type=str, default="closed_loop_compare")

    parser.add_argument("--p1-max", type=float, default=200.0,
                        help="Maximum p1/gamma used by Davide online and G12 fallback.")
    parser.add_argument("--p2-max", type=float, default=250.0,
                        help="Maximum p2/beta used by Davide online and G12 fallback.")

    args = parser.parse_args()

    global GAMMA_MAX_LIMIT, BETA_MAX_LIMIT
    GAMMA_MAX_LIMIT = float(args.p1_max)
    BETA_MAX_LIMIT = float(args.p2_max)

    scenario_paths = expand_scenario_inputs(args.scenarios)
    if args.limit_scenarios is not None:
        scenario_paths = scenario_paths[: int(args.limit_scenarios)]
    if not scenario_paths:
        raise RuntimeError("No scenario YAML files found.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"original", "fixed", "davide_online", "nn", "nn_g12_fallback", "nn_pp_beta", "nn_pp_beta_fallback"}
    for m in modes:
        if m not in valid_modes:
            raise ValueError(f"Unknown mode {m}. Valid: {sorted(valid_modes)}")

    nn_pack = None
    if any(m in modes for m in ("nn", "nn_g12_fallback")):
        if args.model is None:
            raise ValueError("Modes nn and nn_g12_fallback require --model")
        nn_pack = load_nn_policy(
            model_path=args.model,
            stats_path=args.stats,
            hidden_dim=args.hidden_dim,
            p1_floor=args.p1_floor,
            p2_floor=args.p2_floor,
        )
        print(f"loaded NN model = {args.model}")
        print(f"loaded NN stats = {args.stats}")

    pp_pack = None
    if any(m in modes for m in ("nn_pp_beta", "nn_pp_beta_fallback")):
        if args.pp_model is None:
            raise ValueError("Modes nn_pp_beta and nn_pp_beta_fallback require --pp-model")
        pp_pack = load_pp_pack(args.pp_model, args.stats, args)
        print(f"loaded per-pair model = {args.pp_model} ({pp_pack[0]})")
        print(f"pp fallback reduce = {args.pp_fallback_reduce}, unflagged = {args.pp_unflagged}")

    print(f"num_scenarios = {len(scenario_paths)}")
    for p in scenario_paths[:10]:
        print("  ", p)
    if len(scenario_paths) > 10:
        print("  ...")
    print("modes =", modes)

    context = make_rollout_context()

    all_summaries: list[dict] = []
    all_rows: list[dict] = []
    all_pp_rows: list[dict] = []

    total_t0 = time.time()
    for scen_i, scenario_path in enumerate(scenario_paths):
        for mode in modes:
            print(f"\n[{scen_i + 1}/{len(scenario_paths)}] {Path(scenario_path).name} | mode={mode}")
            summary, rows = rollout_one_policy(
                context=context,
                scenario_path=scenario_path,
                mode=mode,
                args=args,
                nn_pack=nn_pack,
                pp_pack=pp_pack,
                pp_log=all_pp_rows,
            )
            all_summaries.append(summary)
            all_rows.extend(rows)
            print(
                f"  status={summary['status']} success={summary['success']} "
                f"steps={summary['steps']} t_task={summary['task_completion_time_s']:.2f}s "
                f"qp_fail={summary['qp_fail_count']} fallback={summary['fallback_count']} "
                f"min_h={summary['min_h']:.3e} min_psi={summary['min_psi']:.3e} "
                f"goal_err={summary['final_dist_to_goal']:.3e} "
                f"p1_mean={summary['p1_mean']:.3e} p2_mean={summary['p2_mean']:.3e}"
            )

    elapsed_s = time.time() - total_t0

    summary_csv = f"{args.out_prefix}_summary.csv"
    rows_csv = f"{args.out_prefix}_steps.csv"
    summary_json = f"{args.out_prefix}_summary.json"

    write_csv(summary_csv, all_summaries)
    write_csv(rows_csv, all_rows)
    if all_pp_rows:
        pp_rows_csv = f"{args.out_prefix}_pp_steps.csv"
        write_csv(pp_rows_csv, all_pp_rows)
        print(f"wrote pp step CSV:  {pp_rows_csv}")
    write_json(
        summary_json,
        {
            "elapsed_s": elapsed_s,
            "elapsed_min": elapsed_s / 60.0,
            "num_scenarios": len(scenario_paths),
            "modes": modes,
            "summaries": all_summaries,
        },
    )

    print("\n=== CLOSED-LOOP COMPARISON DONE ===")
    print(f"elapsed_min = {elapsed_s / 60.0:.2f}")
    print(f"wrote summary CSV:  {summary_csv}")
    print(f"wrote step CSV:     {rows_csv}")
    print(f"wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()