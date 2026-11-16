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
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

from fr3_hocbf.common_io import expand_scenario_inputs, write_csv, write_json
from fr3_hocbf.sim.rollout import (
    ee_position,
    NUM_ARM_JOINTS,
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
from fr3_hocbf.evaluation import gain_policies
from fr3_hocbf.evaluation.gain_policies import (
    davide_online_p12_current,
    g12_check_current,
    g12_values_hold,
    load_nn_policy,
    load_pp_pack,
    min_g1_current,
    minimal_g12_p12_current,
    pp_g12_fallback_current,
    predict_p12_nn_current,
    predict_pp_current,
)


OPTIMAL_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


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
# per-pair step log helpers
# ---------------------------------------------------------------------


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

    # The fallback caps live in gain_policies; set them on that module.
    gain_policies.GAMMA_MAX_LIMIT = float(args.p1_max)
    gain_policies.BETA_MAX_LIMIT = float(args.p2_max)

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