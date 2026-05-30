#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

from fr3_rollout import (
    make_rollout_context,
    run_rollout_from_scenario_with_context,
    save_dataset_npz,
)


_WORKER_CONTEXT = None


def _init_worker(urdf_path: str | None):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = make_rollout_context(urdf_path=urdf_path)


def _classify_rollout(summary: dict, goal_tol_override: float | None = None):
    """
    Labels:
      - good
      - bad_qp
      - bad_safety
      - bad_goal
    """
    final_qp_solved = bool(summary.get("final_qp_solved", False))
    final_min_h = float(summary.get("final_min_h", np.nan))
    final_min_psi = float(summary.get("final_min_psi", np.nan))
    final_dist_to_goal = float(summary.get("final_dist_to_goal", np.nan))

    scenario_data = summary.get("scenario_data", {})
    scenario_goal_tol = scenario_data.get("goal_tolerance", None)

    if goal_tol_override is not None:
        goal_tol = float(goal_tol_override)
    elif scenario_goal_tol is not None:
        goal_tol = float(scenario_goal_tol)
    else:
        goal_tol = 0.02

    if not final_qp_solved:
        return "bad_qp", "final_qp_solved_false"

    if not np.isfinite(final_min_h):
        return "bad_safety", "final_min_h_not_finite"

    if not np.isfinite(final_min_psi):
        return "bad_safety", "final_min_psi_not_finite"

    if final_min_h < 0.0:
        return "bad_safety", "final_min_h_negative"

    if final_min_psi < 0.0:
        return "bad_safety", "final_min_psi_negative"

    if not np.isfinite(final_dist_to_goal):
        return "bad_goal", "final_dist_to_goal_not_finite"

    if final_dist_to_goal > goal_tol:
        return "bad_goal", f"final_dist_to_goal_above_tol({final_dist_to_goal:.6e}>{goal_tol:.6e})"

    return "good", "ok"


def _process_one(task):
    global _WORKER_CONTEXT

    scenario_path_str, output_dir_str, skip_existing, goal_tol_override = task
    scenario_path = Path(scenario_path_str)
    output_dir = Path(output_dir_str)
    output_npz = output_dir / f"{scenario_path.stem}_offline_rollout.npz"

    if skip_existing and output_npz.exists():
        return {
            "status": "skipped_existing",
            "scenario_path": str(scenario_path),
            "output_path": str(output_npz),
            "reason": "output_exists",
            "elapsed_s": 0.0,
        }

    t0 = time.time()
    try:
        summary = run_rollout_from_scenario_with_context(_WORKER_CONTEXT, str(scenario_path))
        save_dataset_npz(summary["dataset"], str(output_npz))
        elapsed_s = time.time() - t0

        status, reason = _classify_rollout(summary, goal_tol_override=goal_tol_override)

        scenario_data = summary.get("scenario_data", {})

        return {
            "status": status,
            "scenario_path": str(scenario_path),
            "output_path": str(output_npz),
            "reason": reason,
            "elapsed_s": elapsed_s,
            "summary": {
                "scenario": summary.get("scenario"),
                "steps": int(summary.get("steps", 0)),
                "final_qp_solved": bool(summary.get("final_qp_solved", False)),
                "final_min_h": float(summary.get("final_min_h", np.nan)),
                "final_min_psi": float(summary.get("final_min_psi", np.nan)),
                "final_min_dist": float(summary.get("final_min_dist", np.nan)),
                "final_dist_to_goal": float(summary.get("final_dist_to_goal", np.nan)),
                "goal_tolerance": float(scenario_data.get("goal_tolerance", 0.02)),
                "gamma": float(scenario_data.get("gamma", np.nan)),
                "beta": float(scenario_data.get("beta", np.nan)),
                "d_margin": float(scenario_data.get("d_margin", np.nan)),
            },
        }

    except Exception as exc:
        elapsed_s = time.time() - t0
        return {
            "status": "bad_runtime",
            "scenario_path": str(scenario_path),
            "output_path": str(output_npz),
            "reason": repr(exc),
            "elapsed_s": elapsed_s,
            "traceback": traceback.format_exc(),
        }


def _write_list(path: Path, lines: list[str]):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def _append_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-dir",
        type=str,
        required=True,
        help="Directory containing scenario_XXXX.yaml files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where rollout npz files will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="scenario_*.yaml",
        help="Scenario filename pattern.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of scenarios to process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenarios whose output npz already exists.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=None,
        help="Optional URDF path override.",
    )
    parser.add_argument(
        "--goal-tol-override",
        type=float,
        default=None,
        help="Optional global goal tolerance override for labeling.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional log directory. Default: sibling folder 'logs' next to output-dir.",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.log_dir is None:
        log_dir = output_dir.parent / "logs"
    else:
        log_dir = Path(args.log_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scenario_paths = sorted(scenario_dir.glob(args.pattern))
    if args.limit is not None:
        scenario_paths = scenario_paths[: args.limit]

    print(f"scenario_dir        = {scenario_dir}")
    print(f"output_dir          = {output_dir}")
    print(f"log_dir             = {log_dir}")
    print(f"num_scenarios       = {len(scenario_paths)}")
    print(f"workers             = {args.workers}")
    print(f"skip_existing       = {args.skip_existing}")
    print(f"goal_tol_override   = {args.goal_tol_override}")

    tasks = [
        (str(p), str(output_dir), args.skip_existing, args.goal_tol_override)
        for p in scenario_paths
    ]

    manifest_rows = []

    good = []
    bad_runtime = []
    bad_qp = []
    bad_safety = []
    bad_goal = []
    skipped = []

    t_global_0 = time.time()

    if args.workers <= 1:
        _init_worker(args.urdf_path)
        results_iter = map(_process_one, tasks)
    else:
        mp_ctx = mp.get_context("spawn")
        pool = mp_ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.urdf_path,),
        )
        results_iter = pool.imap_unordered(_process_one, tasks)

    try:
        for idx, result in enumerate(results_iter, start=1):
            status = result["status"]
            scenario_path = result["scenario_path"]
            output_path = result["output_path"]
            elapsed_s = result["elapsed_s"]
            reason = result.get("reason", "")

            row = {
                "scenario_path": scenario_path,
                "output_path": output_path,
                "status": status,
                "reason": reason,
                "elapsed_s": elapsed_s,
            }

            if "summary" in result:
                row.update(result["summary"])

            manifest_rows.append(row)

            if status == "good":
                print(
                    f"[{idx}/{len(tasks)}] GOOD  {Path(scenario_path).name} "
                    f"-> {Path(output_path).name} "
                    f"({elapsed_s:.1f}s, min_h={row['final_min_h']:.4e}, "
                    f"min_psi={row['final_min_psi']:.4e}, "
                    f"goal={row['final_dist_to_goal']:.4e})"
                )
                good.append(output_path)

            elif status == "bad_qp":
                print(
                    f"[{idx}/{len(tasks)}] BQP   {Path(scenario_path).name} "
                    f"-> {Path(output_path).name} "
                    f"({elapsed_s:.1f}s, reason={reason})"
                )
                bad_qp.append(json.dumps(row))

            elif status == "bad_safety":
                print(
                    f"[{idx}/{len(tasks)}] SAFE  {Path(scenario_path).name} "
                    f"-> {Path(output_path).name} "
                    f"({elapsed_s:.1f}s, reason={reason}, "
                    f"min_h={row.get('final_min_h', np.nan):.4e}, "
                    f"min_psi={row.get('final_min_psi', np.nan):.4e})"
                )
                bad_safety.append(json.dumps(row))

            elif status == "bad_goal":
                print(
                    f"[{idx}/{len(tasks)}] GOAL  {Path(scenario_path).name} "
                    f"-> {Path(output_path).name} "
                    f"({elapsed_s:.1f}s, reason={reason})"
                )
                bad_goal.append(json.dumps(row))

            elif status == "bad_runtime":
                print(
                    f"[{idx}/{len(tasks)}] RUN   {Path(scenario_path).name} "
                    f"({elapsed_s:.1f}s, reason={reason})"
                )
                if "traceback" in result:
                    row["traceback"] = result["traceback"]
                bad_runtime.append(json.dumps(row))

            elif status == "skipped_existing":
                print(
                    f"[{idx}/{len(tasks)}] SKIP  {Path(scenario_path).name} "
                    f"-> {Path(output_path).name}"
                )
                skipped.append(output_path)

    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    total_elapsed_s = time.time() - t_global_0

    _append_jsonl(log_dir / "manifest.jsonl", manifest_rows)
    _write_list(log_dir / "good_rollouts.txt", good)
    _write_list(log_dir / "bad_runtime.txt", bad_runtime)
    _write_list(log_dir / "bad_qp.txt", bad_qp)
    _write_list(log_dir / "bad_safety.txt", bad_safety)
    _write_list(log_dir / "bad_goal.txt", bad_goal)
    _write_list(log_dir / "skipped_existing_files.txt", skipped)

    summary = {
        "scenario_dir": str(scenario_dir),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "workers": args.workers,
        "skip_existing": args.skip_existing,
        "goal_tol_override": args.goal_tol_override,
        "counts": {
            "total_requested": len(tasks),
            "good": len(good),
            "bad_runtime": len(bad_runtime),
            "bad_qp": len(bad_qp),
            "bad_safety": len(bad_safety),
            "bad_goal": len(bad_goal),
            "skipped_existing": len(skipped),
        },
        "total_elapsed_s": total_elapsed_s,
        "total_elapsed_min": total_elapsed_s / 60.0,
    }

    with open(log_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()