#!/usr/bin/env python3
"""
Serial ROS/Gazebo evaluation runner for the FR3 safety-filter project.

This replaces the parallel benchmark runner for the final ROS evaluation.

Design goal:
    correctness and reproducibility first, not throughput.

Important properties:
    - No workers
    - No ROS_DOMAIN_ID isolation
    - No IGN_PARTITION isolation
    - No ThreadPoolExecutor / concurrent launches
    - Exactly one ROS/Gazebo launch alive at a time
    - Full manual-style cleanup before and after every job
    - Structured output under fr3_ros_eval_runs/<run-name>/

The cleanup intentionally mirrors the manual pkill block that reproduced the
expected Davide-mode behavior in our debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_MODES = ("pd_only", "davide_online", "nn", "nn_g12_fallback")

CLEANUP_PATTERNS = (
    "ros2 launch cbf_safety_filter",
    "dv_simulation_hocbf_ros.py",
    "simulation_HOCBF",
    "ign gazebo",
    "ruby /usr/bin/ign gazebo",
    "controller_manager",
    "robot_state_publisher",
    "joint_state_publisher",
    "spawner",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bool_launch(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"
    raise ValueError(f"Could not parse boolean-like launch value: {value!r}")


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def full_cleanup(verbose: bool = True, settle_s: float = 5.0) -> None:
    """Manual-style ROS/Gazebo cleanup used before and after every serial job."""
    if verbose:
        print("Running full stale-process cleanup...")

    for pattern in CLEANUP_PATTERNS:
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as exc:
            if verbose:
                print(f"  cleanup warning for pattern {pattern!r}: {exc}")

    time.sleep(max(0.0, float(settle_s)))


def terminate_process_group(proc: subprocess.Popen[Any], graceful_s: float = 10.0) -> None:
    """Best-effort termination of the single active ros2 launch process group."""
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=graceful_s)
        return
    except Exception:
        pass

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5.0)
        return
    except Exception:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass


def scenario_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"scenario_(\d+)", path.stem)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def scenario_index(path: Path) -> int | None:
    match = re.search(r"scenario_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def discover_scenarios(
    scenario_dir: Path,
    *,
    start_index: int | None,
    limit: int | None,
    scenario_numbers: list[int] | None,
) -> list[Path]:
    all_paths = sorted(scenario_dir.glob("scenario_*.yaml"), key=scenario_sort_key)

    if scenario_numbers:
        wanted = set(int(number) for number in scenario_numbers)
        selected = [path for path in all_paths if scenario_index(path) in wanted]
        found = {idx for path in selected if (idx := scenario_index(path)) is not None}
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(
                f"Could not find scenario YAML(s) for indices {missing} in {scenario_dir}"
            )
        return selected

    selected = all_paths
    if start_index is not None:
        selected = [
            path for path in selected
            if (idx := scenario_index(path)) is not None and idx >= int(start_index)
        ]

    if limit is not None:
        selected = selected[: int(limit)]

    return selected


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("At least one mode is required.")

    invalid = [mode for mode in modes if mode not in SUPPORTED_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported mode(s): {invalid}. Supported modes: {list(SUPPORTED_MODES)}"
        )
    return modes


def build_launch_command(
    *,
    scenario_path: Path,
    result_dir: Path,
    mode: str,
    headless: str,
    save_plots: str,
    model_path: Path | None,
    stats_path: Path | None,
    g12_fallback_tol: float,
    g12_fallback_pair_scope: str,
    p1_max: float,
    p2_max: float,
) -> list[str]:
    command = [
        "ros2",
        "launch",
        "cbf_safety_filter",
        "gazebo_simulation.launch.py",
        "load_gripper:=true",
        "franka_hand:=franka_hand",
        f"headless:={headless}",
        "use_sim_time:=false",
        f"scenario_config_file:={scenario_path}",
        f"result_dir:={result_dir}",
        f"save_plots:={save_plots}",
    ]

    if mode == "pd_only":
        command += [
            "use_hocbf_filter:=false",
            "p12_mode:=davide_online",
        ]
    elif mode == "davide_online":
        command += [
            "use_hocbf_filter:=true",
            "p12_mode:=davide_online",
        ]
    elif mode in {"nn", "nn_g12_fallback"}:
        if model_path is None:
            raise ValueError(f"--model is required for mode {mode!r}.")

        command += [
            "use_hocbf_filter:=true",
            f"p12_mode:={mode}",
            f"nn_model_path:={model_path}",
            f"g12_fallback_tol:={g12_fallback_tol}",
            f"g12_fallback_pair_scope:={g12_fallback_pair_scope}",
            f"p1_max:={p1_max}",
            f"p2_max:={p2_max}",
        ]

        if stats_path is not None:
            command.append(f"nn_feature_stats_path:={stats_path}")
    else:
        raise ValueError(f"Unexpected mode: {mode!r}")

    return command


def record_paths(run_dir: Path, mode: str, scenario_name: str) -> tuple[Path, Path]:
    log_path = run_dir / "logs" / mode / f"{scenario_name}.log"
    summary_path = run_dir / "raw" / mode / scenario_name / "run_summary.json"
    return log_path, summary_path


def run_one_job(
    *,
    run_dir: Path,
    scenario_path: Path,
    mode: str,
    job_index: int,
    job_total: int,
    headless: str,
    save_plots: str,
    model_path: Path | None,
    stats_path: Path | None,
    timeout_s: float,
    cleanup_settle_s: float,
    g12_fallback_tol: float,
    g12_fallback_pair_scope: str,
    p1_max: float,
    p2_max: float,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    scenario_name = scenario_path.stem
    result_dir = ensure_dir(run_dir / "raw" / mode)
    log_path, summary_path = record_paths(run_dir, mode, scenario_name)
    ensure_dir(log_path.parent)

    command = build_launch_command(
        scenario_path=scenario_path,
        result_dir=result_dir,
        mode=mode,
        headless=headless,
        save_plots=save_plots,
        model_path=model_path,
        stats_path=stats_path,
        g12_fallback_tol=g12_fallback_tol,
        g12_fallback_pair_scope=g12_fallback_pair_scope,
        p1_max=p1_max,
        p2_max=p2_max,
    )

    record: dict[str, Any] = {
        "job_index": int(job_index),
        "job_total": int(job_total),
        "scenario": scenario_name,
        "scenario_path": str(scenario_path),
        "mode": mode,
        "command": command,
        "log_path": str(log_path),
        "run_summary": str(summary_path),
        "skipped": False,
        "dry_run": bool(dry_run),
        "returncode": None,
        "timed_out": False,
        "summary_exists": summary_path.exists(),
        "elapsed_wall_s": 0.0,
        "cleanup_settle_s": float(cleanup_settle_s),
        "started_at_utc": None,
        "finished_at_utc": None,
    }

    if resume and summary_path.exists():
        record["skipped"] = True
        record["summary_exists"] = True
        print(f"[{job_index}/{job_total}] SKIP {scenario_name} | {mode} (summary exists)")
        return record

    print(f"[{job_index}/{job_total}] RUN {scenario_name} | {mode}")

    if dry_run:
        print("    DRY-RUN:", shell_join(command))
        return record

    # Reproduce the proven clean manual start before every single rollout.
    full_cleanup(verbose=True, settle_s=cleanup_settle_s)

    start_wall = time.time()
    record["started_at_utc"] = utc_now_iso()
    process: subprocess.Popen[Any] | None = None

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(shell_join(command) + "\n\n")
        log_file.flush()

        try:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )

            try:
                returncode = process.wait(timeout=float(timeout_s))
                record["returncode"] = int(returncode)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                record["returncode"] = -999
                log_file.write(
                    f"\n[SERIAL_RUNNER] Wall timeout after {timeout_s:.1f} s. "
                    "Terminating launch group.\n"
                )
                log_file.flush()
                terminate_process_group(process)
        finally:
            if process is not None:
                terminate_process_group(process)

            record["elapsed_wall_s"] = float(time.time() - start_wall)
            record["summary_exists"] = summary_path.exists()
            record["finished_at_utc"] = utc_now_iso()

    # Leave the machine clean before the next serial rollout.
    full_cleanup(verbose=True, settle_s=cleanup_settle_s)

    print(
        f"    DONE [{job_index}/{job_total}] {scenario_name} | {mode}: "
        f"returncode={record['returncode']} timed_out={record['timed_out']} "
        f"summary_exists={record['summary_exists']} "
        f"elapsed_wall_s={record['elapsed_wall_s']:.1f}"
    )
    return record


def write_manifest(run_dir: Path, records: list[dict[str, Any]]) -> None:
    manifest_dir = ensure_dir(run_dir / "manifests")
    json_path = manifest_dir / "batch_manifest.json"
    csv_path = manifest_dir / "batch_manifest.csv"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    fieldnames = [
        "job_index",
        "job_total",
        "scenario",
        "mode",
        "skipped",
        "dry_run",
        "returncode",
        "timed_out",
        "summary_exists",
        "elapsed_wall_s",
        "cleanup_settle_s",
        "started_at_utc",
        "finished_at_utc",
        "scenario_path",
        "log_path",
        "run_summary",
        "command",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for record in records:
            row = dict(record)
            row["command"] = shell_join(record.get("command", []))
            writer.writerow(row)

    print(f"Wrote manifest JSON: {json_path}")
    print(f"Wrote manifest CSV:  {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serial FR3 ROS/Gazebo evaluation runner. No parallel execution by design."
    )
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scenario-numbers",
        type=str,
        default=None,
        help="Optional comma-separated explicit scenario indices, e.g. 3,10,87.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--modes",
        default="pd_only,davide_online,nn,nn_g12_fallback",
        help=f"Comma-separated subset of: {','.join(SUPPORTED_MODES)}",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--cleanup-settle-s",
        type=float,
        default=5.0,
        help="Seconds to wait after each full cleanup. Default: 5.0.",
    )
    parser.add_argument("--headless", default="true")
    parser.add_argument("--save-plots", default="false")
    parser.add_argument("--g12-fallback-tol", type=float, default=1e-9)
    parser.add_argument("--g12-fallback-pair-scope", default="all")
    parser.add_argument("--p1-max", type=float, default=200.0)
    parser.add_argument("--p2-max", type=float, default=250.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = parse_modes(args.modes)

    scenario_numbers = None
    if args.scenario_numbers:
        scenario_numbers = [
            int(item.strip())
            for item in args.scenario_numbers.split(",")
            if item.strip()
        ]

    scenario_dir = args.scenario_dir.expanduser().resolve()
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory does not exist: {scenario_dir}")

    model_path = args.model.expanduser().resolve() if args.model is not None else None
    stats_path = args.stats.expanduser().resolve() if args.stats is not None else None

    if any(mode in {"nn", "nn_g12_fallback"} for mode in modes):
        if model_path is None:
            raise ValueError("--model is required because an NN mode was requested.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    if stats_path is not None and not stats_path.exists():
        raise FileNotFoundError(f"Feature stats file does not exist: {stats_path}")

    scenarios = discover_scenarios(
        scenario_dir,
        start_index=args.start_index,
        limit=args.limit,
        scenario_numbers=scenario_numbers,
    )
    if not scenarios:
        raise RuntimeError("No scenarios selected.")

    source_dir = Path(__file__).resolve().parent
    run_dir = ensure_dir(source_dir / "fr3_ros_eval_runs" / args.run_name)

    jobs: list[tuple[Path, str]] = []
    for scenario_path in scenarios:
        for mode in modes:
            jobs.append((scenario_path, mode))

    print(f"run_dir = {run_dir}")
    print(f"num_scenarios = {len(scenarios)}")
    print(f"modes = {modes}")
    print("execution = serial only; no workers, no ROS_DOMAIN_ID, no IGN_PARTITION")
    print(f"model = {model_path if model_path is not None else '<unused>'}")
    print(f"stats = {stats_path if stats_path is not None else '<checkpoint or unused>'}")
    print(f"cleanup_settle_s = {float(args.cleanup_settle_s)}")

    records: list[dict[str, Any]] = []
    for job_index, (scenario_path, mode) in enumerate(jobs, start=1):
        record = run_one_job(
            run_dir=run_dir,
            scenario_path=scenario_path,
            mode=mode,
            job_index=job_index,
            job_total=len(jobs),
            headless=bool_launch(args.headless),
            save_plots=bool_launch(args.save_plots),
            model_path=model_path,
            stats_path=stats_path,
            timeout_s=float(args.timeout_s),
            cleanup_settle_s=float(args.cleanup_settle_s),
            g12_fallback_tol=float(args.g12_fallback_tol),
            g12_fallback_pair_scope=str(args.g12_fallback_pair_scope),
            p1_max=float(args.p1_max),
            p2_max=float(args.p2_max),
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
        )
        records.append(record)

    write_manifest(run_dir, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
