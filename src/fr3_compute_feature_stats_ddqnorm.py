#!/usr/bin/env python3
from __future__ import annotations

"""
Compute feature-normalization statistics for the FR3 p12 rollout NN.

This version is intended for the 114D feature vector:
    q_arm,
    dq_arm,
    ddq_nominal_arm,
    ddq_nominal_arm / ddq_scale,
    h_all,
    Lf_h_all

It imports build_raw_feature_matrix from fr3_train_rollout.py, so run this
after replacing that training file with the updated 114D version.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fr3_train_rollout import build_raw_feature_matrix


def read_jsonl_rollout_paths(path: str) -> list[str]:
    paths: list[str] = []
    with open(os.path.expanduser(path), "r") as f:
        for line_i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            rp = row.get("rollout_path") or row.get("output_path") or row.get("path")
            if rp is None:
                raise KeyError(f"{path}:{line_i} has no rollout_path/output_path/path: {row}")
            paths.append(os.path.abspath(os.path.expanduser(str(rp))))
    return list(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-labeled",
        required=True,
        help="Training labeled JSONL. Statistics are computed from these rollout NPZs only.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .npz stats path, containing feat_mean and feat_std.",
    )
    args = parser.parse_args()

    rollout_paths = read_jsonl_rollout_paths(args.rollout_labeled)
    if not rollout_paths:
        raise RuntimeError(f"No rollout paths found in {args.rollout_labeled}")

    count = 0
    feat_sum = None
    feat_sq_sum = None
    feature_dim = None

    for i, rp in enumerate(rollout_paths, start=1):
        if not os.path.isfile(rp):
            raise FileNotFoundError(rp)
        data = np.load(rp)
        z_raw = np.asarray(build_raw_feature_matrix(data), dtype=float)
        if z_raw.ndim != 2:
            raise RuntimeError(f"Expected 2D feature matrix from {rp}, got {z_raw.shape}")

        if feature_dim is None:
            feature_dim = int(z_raw.shape[1])
            feat_sum = np.zeros(feature_dim, dtype=float)
            feat_sq_sum = np.zeros(feature_dim, dtype=float)
        elif int(z_raw.shape[1]) != feature_dim:
            raise RuntimeError(
                f"Feature dim mismatch in {rp}: got {z_raw.shape[1]}, expected {feature_dim}"
            )

        feat_sum += z_raw.sum(axis=0)
        feat_sq_sum += np.square(z_raw).sum(axis=0)
        count += int(z_raw.shape[0])

        if i <= 5 or i % 25 == 0 or i == len(rollout_paths):
            print(f"[{i}/{len(rollout_paths)}] {Path(rp).name} rows={z_raw.shape[0]} total_rows={count}")

    if count <= 0 or feat_sum is None or feat_sq_sum is None:
        raise RuntimeError("No feature rows accumulated.")

    feat_mean = feat_sum / float(count)
    feat_var = feat_sq_sum / float(count) - np.square(feat_mean)
    feat_var = np.maximum(feat_var, 0.0)
    feat_std = np.sqrt(feat_var)
    feat_std = np.clip(feat_std, 1e-6, None)

    out = Path(os.path.expanduser(args.out))
    if str(out.parent) not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        feat_mean=feat_mean,
        feat_std=feat_std,
        mean=feat_mean,
        std=feat_std,
        feature_dim=np.asarray([feature_dim], dtype=int),
        num_rows=np.asarray([count], dtype=int),
        num_rollouts=np.asarray([len(rollout_paths)], dtype=int),
        feature_version=np.asarray(["q_dq_ddqnom_ddqnomnorm_h_Lfh"]),
    )

    print("\n=== FEATURE STATS DONE ===")
    print("out =", out)
    print("num_rollouts =", len(rollout_paths))
    print("num_rows =", count)
    print("feature_dim =", feature_dim)


if __name__ == "__main__":
    main()
