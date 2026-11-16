#!/usr/bin/env python3
from __future__ import annotations

"""
Feature-normalization statistics for the gain policies, from the training rollout NPZs.

--kind global (default): mean/std of the 114D policy input (policy/features.py)
    q_arm, dq_arm, ddq_nominal_arm, ddq_nominal_arm / ddq_scale, h_all, Lf_h_all
    -> feat_mean, feat_std (stats .npz for --stats-in)

--kind per_pair: mean/std of the 8 per-pair features of PerPairBetaNet (policy/pp_beta.py)
    over all ACTIVE pair rows. psi and Lf_psi need a gamma; the rollout's own per-step
    gamma is used (the gain the rollout was generated with), which only sets the scale.
    -> pp_mean, pp_std (stats .npz for --pp-stats-in)

Examples:
    python -m fr3_hocbf.dataset.compute_feature_stats \
      --rollout-labeled <split_dir>/fr3_fullpool_train_weighted.jsonl --out fr3_feature_stats_v10.npz
    python -m fr3_hocbf.dataset.compute_feature_stats --kind per_pair \
      --rollout-labeled ../data/fr3_fullpool_train_weighted_local.jsonl --out fr3_pp_feature_stats_v10.npz
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from fr3_hocbf.dataset.rollout_records import expand_labeled_rollout_inputs
from fr3_hocbf.policy.features import build_raw_feature_matrix
from fr3_hocbf.policy.pp_beta import PP_FEATURE_NAMES, kinematic_ddq_bounds_torch, pp_features_torch
from fr3_hocbf.sim.qp_data import get_joint_limits
from fr3_hocbf.train.config import DT_DEFAULT


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


def compute_global_stats(args) -> None:
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


def compute_per_pair_stats(args) -> None:
    records = expand_labeled_rollout_inputs([args.rollout_labeled])
    paths = sorted({r["rollout_path"] for r in records})
    limits_t = {k: torch.tensor(v, dtype=torch.double) for k, v in get_joint_limits().items()}

    chunks = []
    for p in paths:
        z = np.load(p)
        t = {k: torch.tensor(z[k], dtype=torch.double) for k in
             ("q_arm", "dq_arm", "ddq_nominal_arm", "h_all", "Lf_h_all", "vrel_sq2_all", "Lg_psi_all", "pair_mask")}
        n = t["q_arm"].shape[0]
        gamma = torch.tensor(np.broadcast_to(np.asarray(z["gamma"], dtype=float).reshape(-1), (n,)).copy()).view(n, 1)
        ddq_min, ddq_max = kinematic_ddq_bounds_torch(t["q_arm"], t["dq_arm"], limits_t, args.dt)
        x = pp_features_torch(gamma, t["h_all"], t["Lf_h_all"], t["vrel_sq2_all"], t["Lg_psi_all"],
                              t["ddq_nominal_arm"], ddq_min, ddq_max)
        chunks.append(x[t["pair_mask"] > 0.5].numpy())
        print(f"{p}: steps={n} active_rows={chunks[-1].shape[0]}")

    x_all = np.concatenate(chunks, axis=0)
    mean = x_all.mean(axis=0)
    std = np.clip(x_all.std(axis=0), 1e-6, None)
    np.savez(args.out, pp_mean=mean, pp_std=std, feature_names=np.array(PP_FEATURE_NAMES),
             num_rows=x_all.shape[0], num_rollouts=len(paths))
    for name, m, s in zip(PP_FEATURE_NAMES, mean, std):
        print(f"  {name:14s} mean={m: .4e} std={s:.4e}")
    print(f"wrote {args.out} ({x_all.shape[0]} active rows from {len(paths)} rollouts)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=["global", "per_pair"],
        default="global",
        help="global: 114D policy input stats (feat_mean/feat_std). per_pair: PerPairBetaNet per-pair feature stats (pp_mean/pp_std).",
    )
    parser.add_argument(
        "--rollout-labeled",
        required=True,
        help="Training labeled JSONL. Statistics are computed from these rollout NPZs only.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .npz stats path.",
    )
    parser.add_argument("--dt", type=float, default=DT_DEFAULT, help="per_pair: control period for the kinematic acceleration bounds")
    args = parser.parse_args()

    if args.kind == "per_pair":
        compute_per_pair_stats(args)
    else:
        compute_global_stats(args)


if __name__ == "__main__":
    main()
