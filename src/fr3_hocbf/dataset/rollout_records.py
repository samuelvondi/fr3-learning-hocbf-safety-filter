"""Rollout NPZ records used by training: labeled JSONL lists, critical windows, start-state sampling."""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from fr3_hocbf.common_io import read_list_file


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
