#!/usr/bin/env python3
from __future__ import annotations

"""
Select hard-good FR3 rollouts from good_rollouts.txt.

Hard-good = rollout is labelled good/successful, but comes close to the
geometric barrier boundary h >= 0 or the first-order HOCBF condition psi >= 0.

This script only labels/selects rollouts. It does NOT create train/val/test
splits and it does NOT create critical start windows.

Typical use from src/:

  python make_hard_good_list.py \
    --good-rollouts ../data/rollouts/logs/good_rollouts.txt \
    --exclude-files ../data/rollout_splits/good_200_seed0/train_files.txt \
                    ../data/rollout_splits/good_200_seed0/val_files.txt \
    --top-k 150 \
    --out ../data/rollout_splits/hard_good_top150_not_good200.jsonl \
    --write-all

Main output rows contain e.g.:
  {
    "category": "good_hard",
    "rollout_path": ".../scenario_0181_offline_rollout.npz",
    "scenario": "scenario_0181",
    "global_min_h": 1.2e-04,
    "global_min_psi": 8.5e-04,
    "difficulty_rank": 3
  }
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def scenario_name_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    if stem.endswith("_offline_rollout"):
        stem = stem[: -len("_offline_rollout")]
    return stem


def read_path_file(path: str | Path) -> list[str]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    out: list[str] = []
    with open(p, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{"):
                row = json.loads(s)
                s = row.get("output_path") or row.get("rollout_path") or row.get("path") or row.get("scenario_path")
                if not s:
                    raise KeyError(f"JSON row in {p} has no usable path key")
            out.append(os.path.abspath(os.path.expanduser(str(s))))
    return out


def read_excluded_scenarios(files: list[str] | None) -> set[str]:
    excluded: set[str] = set()
    if not files:
        return excluded
    for f in files:
        for p in read_path_file(f):
            excluded.add(scenario_name_from_path(p))
    return excluded


def rollout_length(z: np.lib.npyio.NpzFile) -> int:
    for key in ["min_h", "h_all", "q_arm", "q", "dq_arm", "dq"]:
        if key in z.files:
            arr = np.asarray(z[key])
            if arr.ndim > 0:
                return int(arr.shape[0])
    raise KeyError(f"Could not infer rollout length. Available keys: {z.files}")


def active_mask(z: np.lib.npyio.NpzFile, T: int, N: int) -> np.ndarray:
    if "pair_mask" not in z.files:
        return np.ones((T, N), dtype=bool)

    mask = np.asarray(z["pair_mask"])
    if mask.ndim == 1 and mask.shape[0] == N:
        return np.broadcast_to(mask.reshape(1, N) > 0.5, (T, N)).copy()

    if mask.ndim >= 2:
        if mask.shape[0] == T and mask.shape[1] == N:
            return mask[:T, :N] > 0.5
        if mask.shape[-1] == N:
            m = mask.reshape(-1, N)
            if m.shape[0] >= T:
                return m[:T] > 0.5

    return np.ones((T, N), dtype=bool)


def broadcast_p1_or_gamma(z: np.lib.npyio.NpzFile, T: int, N: int) -> np.ndarray:
    if "gamma" in z.files:
        g = np.asarray(z["gamma"], dtype=float)
    elif "p1" in z.files:
        g = np.asarray(z["p1"], dtype=float)
    else:
        g = np.asarray(10.0, dtype=float)

    g = np.squeeze(g)
    if g.ndim == 0:
        return np.full((T, N), float(g), dtype=float)
    if g.shape[0] == T:
        return np.broadcast_to(g.reshape(T, 1), (T, N)).astype(float)
    return np.full((T, N), float(np.ravel(g)[0]), dtype=float)


def min_h_and_min_psi_series(rollout_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(rollout_path, allow_pickle=False) as z:
        T = rollout_length(z)

        direct_h = None
        if "min_h" in z.files:
            a = np.asarray(z["min_h"], dtype=float).reshape(-1)
            if a.shape[0] >= T:
                direct_h = a[:T]

        direct_psi = None
        if "min_psi" in z.files:
            a = np.asarray(z["min_psi"], dtype=float).reshape(-1)
            if a.shape[0] >= T:
                direct_psi = a[:T]

        if direct_h is not None and direct_psi is not None:
            return direct_h, direct_psi

        if "h_all" not in z.files:
            raise KeyError(f"{rollout_path} has no min_h and no h_all. Available keys: {z.files}")

        h_all = np.asarray(z["h_all"], dtype=float)
        if h_all.ndim == 1:
            h_all = h_all.reshape(-1, 1)
        h_all = h_all[:T]
        N = int(h_all.shape[1])
        active = active_mask(z, T, N)

        if direct_h is None:
            h_masked = np.where(active, h_all, np.inf)
            min_h = np.min(h_masked, axis=1)
            min_h[~np.isfinite(min_h)] = np.nan
        else:
            min_h = direct_h

        if direct_psi is None:
            if "Lf_h_all" not in z.files:
                raise KeyError(f"{rollout_path} has no min_psi and no Lf_h_all. Available keys: {z.files}")
            Lf_h_all = np.asarray(z["Lf_h_all"], dtype=float)
            if Lf_h_all.ndim == 1:
                Lf_h_all = Lf_h_all.reshape(-1, 1)
            Lf_h_all = Lf_h_all[:T, :N]
            p1 = broadcast_p1_or_gamma(z, T, N)
            psi_all = Lf_h_all + p1 * h_all
            psi_masked = np.where(active, psi_all, np.inf)
            min_psi = np.min(psi_masked, axis=1)
            min_psi[~np.isfinite(min_psi)] = np.nan
        else:
            min_psi = direct_psi

        return min_h, min_psi


def arg_nanmin(a: np.ndarray) -> int | None:
    finite = np.isfinite(a)
    if not np.any(finite):
        return None
    aa = np.where(finite, a, np.inf)
    return int(np.argmin(aa))


def scan_rollout(path: str) -> dict[str, Any]:
    min_h, min_psi = min_h_and_min_psi_series(path)
    T = int(len(min_h))

    step_min_h = arg_nanmin(min_h)
    step_min_psi = arg_nanmin(min_psi)
    global_min_h = float(np.nanmin(min_h)) if np.any(np.isfinite(min_h)) else float("nan")
    global_min_psi = float(np.nanmin(min_psi)) if np.any(np.isfinite(min_psi)) else float("nan")

    return {
        "category": "good_candidate",
        "rollout_path": path,
        "scenario": scenario_name_from_path(path),
        "rollout_length": T,
        "global_min_h": global_min_h,
        "global_min_psi": global_min_psi,
        "step_min_h": step_min_h,
        "step_min_psi": step_min_psi,
        "num_h_viol_steps": int(np.sum(np.isfinite(min_h) & (min_h < 0.0))),
        "num_psi_viol_steps": int(np.sum(np.isfinite(min_psi) & (min_psi < 0.0))),
    }


def rank_and_select(rows: list[dict[str, Any]], top_k: int, allow_negative: bool, h_tol: float, psi_tol: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [r for r in rows if r.get("error") is None]
    if not allow_negative:
        valid = [
            r for r in valid
            if float(r["global_min_h"]) >= -abs(h_tol)
            and float(r["global_min_psi"]) >= -abs(psi_tol)
        ]

    by_h = sorted(valid, key=lambda r: (float(r["global_min_h"]), r["scenario"]))
    by_psi = sorted(valid, key=lambda r: (float(r["global_min_psi"]), r["scenario"]))
    rank_h = {id(r): i + 1 for i, r in enumerate(by_h)}
    rank_psi = {id(r): i + 1 for i, r in enumerate(by_psi)}

    for r in rows:
        r.setdefault("rank_h", None)
        r.setdefault("rank_psi", None)
        r.setdefault("difficulty_rank", None)
        r.setdefault("difficulty_source", None)
        r.setdefault("selected_hard_good", False)

    for r in valid:
        rh = rank_h[id(r)]
        rp = rank_psi[id(r)]
        r["rank_h"] = int(rh)
        r["rank_psi"] = int(rp)
        r["difficulty_rank"] = int(min(rh, rp))
        r["difficulty_source"] = "h" if rh <= rp else "psi"

    selected = sorted(valid, key=lambda r: (int(r["difficulty_rank"]), r["scenario"]))[: int(top_k)]
    selected_ids = {id(r) for r in selected}
    for r in rows:
        if id(r) in selected_ids:
            r["selected_hard_good"] = True
            r["category"] = "good_hard"

    return rows, selected


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good-rollouts", required=True)
    parser.add_argument("--exclude-files", nargs="*", default=[], help="Files with rollout paths or scenario YAMLs to exclude, e.g. good200 train/val lists")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--out", required=True, help="Selected hard-good JSONL")
    parser.add_argument("--allow-negative", action="store_true", help="Allow tiny/actual negative h or psi in selected hard-good rows")
    parser.add_argument("--h-tol", type=float, default=1e-9)
    parser.add_argument("--psi-tol", type=float, default=1e-9)
    parser.add_argument("--write-all", action="store_true")
    args = parser.parse_args()

    good_paths = read_path_file(args.good_rollouts)
    excluded_scenarios = read_excluded_scenarios(args.exclude_files)

    good_paths = [p for p in good_paths if scenario_name_from_path(p) not in excluded_scenarios]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, path in enumerate(good_paths, start=1):
        try:
            rows.append(scan_rollout(path))
        except Exception as exc:
            row = {
                "category": "good_candidate",
                "rollout_path": path,
                "scenario": scenario_name_from_path(path),
                "error": repr(exc),
                "selected_hard_good": False,
            }
            rows.append(row)
            errors.append(row)
        if i % 50 == 0:
            print(f"scanned {i}/{len(good_paths)}")

    rows, selected = rank_and_select(
        rows,
        top_k=args.top_k,
        allow_negative=args.allow_negative,
        h_tol=args.h_tol,
        psi_tol=args.psi_tol,
    )

    write_jsonl(args.out, selected)
    out_path = Path(args.out).expanduser()
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "good_rollouts_file": str(Path(args.good_rollouts).expanduser()),
        "exclude_files": [str(Path(p).expanduser()) for p in args.exclude_files],
        "num_excluded_scenarios": len(excluded_scenarios),
        "num_scanned_after_exclusion": len(good_paths),
        "num_errors": len(errors),
        "top_k": args.top_k,
        "num_selected_hard_good": len(selected),
        "allow_negative": bool(args.allow_negative),
        "h_tol": args.h_tol,
        "psi_tol": args.psi_tol,
        "out": str(out_path),
    }
    if selected:
        summary.update({
            "selected_min_global_min_h": float(min(r["global_min_h"] for r in selected)),
            "selected_max_global_min_h": float(max(r["global_min_h"] for r in selected)),
            "selected_min_global_min_psi": float(min(r["global_min_psi"] for r in selected)),
            "selected_max_global_min_psi": float(max(r["global_min_psi"] for r in selected)),
            "num_selected_by_h": sum(1 for r in selected if r.get("difficulty_source") == "h"),
            "num_selected_by_psi": sum(1 for r in selected if r.get("difficulty_source") == "psi"),
        })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.write_all:
        all_jsonl = out_path.with_name(out_path.stem + "_all_ranked.jsonl")
        all_csv = out_path.with_name(out_path.stem + "_all_ranked.csv")
        write_jsonl(all_jsonl, rows)
        write_csv(all_csv, rows)
        summary["all_ranked_jsonl"] = str(all_jsonl)
        summary["all_ranked_csv"] = str(all_csv)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote:", out_path)
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
