#!/usr/bin/env python3
from __future__ import annotations

"""
PD-only closed-loop evaluator for the FR3 generated scenario set.

This script runs the nominal Cartesian PD controller directly:
  - no safety QP
  - no HOCBF p1/p2
  - no fallback
  - no clipping/saturation of ddq_nominal

Failure classification:
  - task_success: reaches the scenario goal tolerance for the required settle time
  - safety_ok: post-step minimum geometric barrier h never drops below -h_fail_tol
  - pd_failed: (not task_success) OR (not safety_ok)

Outputs:
  <out_prefix>_summary.csv
  <out_prefix>_summary.json
  <out_prefix>_failed_scenarios.txt
  <out_prefix>_bad_safety_scenarios.txt
  <out_prefix>_bad_goal_scenarios.txt
  <out_prefix>_bad_safety_and_goal_scenarios.txt
"""

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin

from fr3_rollout import (
    NUM_ARM_JOINTS,
    make_rollout_context,
    load_scenario_yaml,
    get_default_initial_state,
    clone_obstacles,
    update_obstacles,
    compute_barrier_metrics,
)
from fr3_nominal_controller import nominal_controller_js_standalone


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

    out = [
        os.path.abspath(os.path.expanduser(p))
        for p in out
        if p.endswith((".yaml", ".yml"))
    ]
    return sorted(list(dict.fromkeys(out)))


def ee_position(model, data, ee_frame_id: int, q_full: np.ndarray, dq_full: np.ndarray | None = None) -> np.ndarray:
    if dq_full is None:
        pin.forwardKinematics(model, data, q_full)
    else:
        pin.forwardKinematics(model, data, q_full, dq_full, np.zeros(model.nv))
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_frame_id].translation.copy()


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_path_list(path: str, scenario_paths: list[str]) -> None:
    with open(path, "w") as f:
        for p in scenario_paths:
            f.write(str(p) + "\n")


def rollout_pd_only(
    context: dict,
    scenario_path: str,
    *,
    dt: float,
    max_steps_override: int | None,
    move_obstacles: bool,
    h_fail_tol: float,
) -> dict:
    model = context["model"]
    data = model.createData()
    ee_frame_id = context["ee_frame_id"]
    active_links = context["active_links"]
    self_collision_link_pair = context["self_collision_link_pair"]

    scenario = load_scenario_yaml(scenario_path)

    q_full, dq_full = get_default_initial_state(model)
    obstacles = clone_obstacles(scenario["obstacles"])
    goal = np.asarray(scenario["goal_ee_pos"], dtype=float)
    d_margin = float(scenario["d_margin"])
    gamma_for_metrics = float(scenario.get("gamma", 10.0))

    max_steps_from_scenario = int(float(scenario["max_sim_duration_s"]) / dt)
    max_steps = max_steps_from_scenario if max_steps_override is None else min(int(max_steps_override), max_steps_from_scenario)

    target_prev = ee_position(model, data, ee_frame_id, q_full, dq_full)

    at_goal_timer = 0.0
    task_success = False
    first_goal_step = None
    first_safety_violation_step = None

    min_h_overall = float("inf")
    min_psi_overall = float("inf")
    min_dist_overall = float("inf")
    final_dist_to_goal = float("nan")
    path_length_q = 0.0
    ddq_nom_norm_sum = 0.0
    ddq_nom_norm_max = 0.0
    steps_done = 0

    t0 = time.time()

    for k in range(max_steps):
        steps_done = k + 1
        q_arm = q_full[:NUM_ARM_JOINTS].copy()
        dq_arm = dq_full[:NUM_ARM_JOINTS].copy()

        next_obstacles = update_obstacles(obstacles, dt) if move_obstacles else clone_obstacles(obstacles)

        ddq_nominal, target_next = nominal_controller_js_standalone(
            model=model,
            data=data,
            ee_frame_id=ee_frame_id,
            q_arm_curr=q_arm,
            dq_arm_curr=dq_arm,
            target_ee_pos_cartesian=goal,
            target=target_prev,
        )
        ddq_nominal = np.asarray(ddq_nominal, dtype=float).reshape(NUM_ARM_JOINTS)

        ddq_norm = float(np.linalg.norm(ddq_nominal))
        ddq_nom_norm_sum += ddq_norm
        ddq_nom_norm_max = max(ddq_nom_norm_max, ddq_norm)

        next_dq_arm = dq_arm + ddq_nominal * dt
        next_q_arm = q_arm + dq_arm * dt + 0.5 * ddq_nominal * (dt ** 2)

        next_q_full = q_full.copy()
        next_dq_full = dq_full.copy()
        next_q_full[:NUM_ARM_JOINTS] = next_q_arm
        next_dq_full[:NUM_ARM_JOINTS] = next_dq_arm

        metrics = compute_barrier_metrics(
            model=model,
            data=data,
            q_full=next_q_full,
            dq_full=next_dq_full,
            obstacles=next_obstacles,
            active_links=active_links,
            self_collision_link_pair=self_collision_link_pair,
            d_margin=d_margin,
            gamma=gamma_for_metrics,
        )

        min_h = float(metrics["min_h"])
        min_psi = float(metrics["min_psi"])
        min_dist = float(metrics["min_dist"])

        min_h_overall = min(min_h_overall, min_h)
        min_psi_overall = min(min_psi_overall, min_psi)
        min_dist_overall = min(min_dist_overall, min_dist)

        if first_safety_violation_step is None and min_h < -float(h_fail_tol):
            first_safety_violation_step = int(k)

        ee_pos = ee_position(model, data, ee_frame_id, next_q_full, next_dq_full)
        dist_to_goal = float(np.linalg.norm(ee_pos - goal))
        final_dist_to_goal = dist_to_goal

        path_length_q += float(np.linalg.norm(next_q_arm - q_arm))

        if dist_to_goal < float(scenario["goal_tolerance"]):
            at_goal_timer += dt
            if at_goal_timer >= float(scenario["goal_settle_time_s"]):
                task_success = True
                first_goal_step = int(k)
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

    safety_ok = bool(min_h_overall >= -float(h_fail_tol))
    pd_failed = bool((not task_success) or (not safety_ok))

    if task_success and safety_ok:
        status = "SUCCESS_SAFE"
    elif task_success and not safety_ok:
        status = "SUCCESS_BUT_UNSAFE"
    elif (not task_success) and safety_ok:
        status = "GOAL_FAIL_BUT_SAFE"
    else:
        status = "GOAL_FAIL_AND_UNSAFE"

    task_completion_time_s = float(steps_done * dt) if task_success else float("nan")

    return {
        "scenario_path": scenario_path,
        "scenario": Path(scenario_path).stem,
        "status": status,
        "pd_failed": pd_failed,
        "task_success": bool(task_success),
        "safety_ok": safety_ok,
        "steps": int(steps_done),
        "sim_time_s": float(steps_done * dt),
        "task_completion_time_s": task_completion_time_s,
        "first_goal_step": first_goal_step if first_goal_step is not None else -1,
        "first_safety_violation_step": first_safety_violation_step if first_safety_violation_step is not None else -1,
        "first_safety_violation_time_s": (
            float(first_safety_violation_step * dt)
            if first_safety_violation_step is not None
            else float("nan")
        ),
        "min_h": float(min_h_overall),
        "min_psi_metric_gamma": float(min_psi_overall),
        "min_dist": float(min_dist_overall),
        "final_dist_to_goal": float(final_dist_to_goal),
        "goal_tolerance": float(scenario["goal_tolerance"]),
        "path_length_q": float(path_length_q),
        "ddq_nom_norm_mean": float(ddq_nom_norm_sum / max(1, steps_done)),
        "ddq_nom_norm_max": float(ddq_nom_norm_max),
        "rollout_elapsed_s": float(time.time() - t0),
        "h_fail_tol": float(h_fail_tol),
        "d_margin": float(d_margin),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="+", help="Scenario YAML(s), directories, globs, or txt files")
    parser.add_argument("--dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional debug cap; default uses each scenario duration.")
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--move-obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h-fail-tol", type=float, default=0.0, help="Safety fail if min_h < -tol.")
    parser.add_argument("--out-prefix", type=str, default="pd_only_all_scenarios")
    args = parser.parse_args()

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

    context = make_rollout_context()
    summaries: list[dict] = []

    t0 = time.time()
    for i, scenario_path in enumerate(scenario_paths, start=1):
        summary = rollout_pd_only(
            context=context,
            scenario_path=scenario_path,
            dt=float(args.dt),
            max_steps_override=args.max_steps,
            move_obstacles=bool(args.move_obstacles),
            h_fail_tol=float(args.h_fail_tol),
        )
        summaries.append(summary)
        print(
            f"[{i}/{len(scenario_paths)}] {Path(scenario_path).name} | "
            f"status={summary['status']} pd_failed={summary['pd_failed']} "
            f"min_h={summary['min_h']:.3e} goal_err={summary['final_dist_to_goal']:.3e}"
        )

    elapsed_s = float(time.time() - t0)

    failed = [r["scenario_path"] for r in summaries if bool(r["pd_failed"])]
    bad_safety = [r["scenario_path"] for r in summaries if not bool(r["safety_ok"])]
    bad_goal = [r["scenario_path"] for r in summaries if not bool(r["task_success"])]
    bad_both = [
        r["scenario_path"]
        for r in summaries
        if (not bool(r["safety_ok"])) and (not bool(r["task_success"]))
    ]

    summary_csv = f"{args.out_prefix}_summary.csv"
    summary_json = f"{args.out_prefix}_summary.json"
    failed_txt = f"{args.out_prefix}_failed_scenarios.txt"
    bad_safety_txt = f"{args.out_prefix}_bad_safety_scenarios.txt"
    bad_goal_txt = f"{args.out_prefix}_bad_goal_scenarios.txt"
    bad_both_txt = f"{args.out_prefix}_bad_safety_and_goal_scenarios.txt"

    write_csv(summary_csv, summaries)
    write_json(
        summary_json,
        {
            "elapsed_s": elapsed_s,
            "elapsed_min": elapsed_s / 60.0,
            "num_scenarios": len(scenario_paths),
            "num_failed": len(failed),
            "num_bad_safety": len(bad_safety),
            "num_bad_goal": len(bad_goal),
            "num_bad_safety_and_goal": len(bad_both),
            "summaries": summaries,
        },
    )
    write_path_list(failed_txt, failed)
    write_path_list(bad_safety_txt, bad_safety)
    write_path_list(bad_goal_txt, bad_goal)
    write_path_list(bad_both_txt, bad_both)

    print("\n=== PD-ONLY EVALUATION DONE ===")
    print(f"elapsed_min = {elapsed_s / 60.0:.2f}")
    print(f"num_scenarios = {len(scenario_paths)}")
    print(f"num_pd_failed = {len(failed)}")
    print(f"num_bad_safety = {len(bad_safety)}")
    print(f"num_bad_goal = {len(bad_goal)}")
    print(f"num_bad_safety_and_goal = {len(bad_both)}")
    print(f"wrote summary CSV: {summary_csv}")
    print(f"wrote summary JSON: {summary_json}")
    print(f"wrote failed list: {failed_txt}")
    print(f"wrote bad safety list: {bad_safety_txt}")
    print(f"wrote bad goal list: {bad_goal_txt}")
    print(f"wrote bad both list: {bad_both_txt}")


if __name__ == "__main__":
    main()
