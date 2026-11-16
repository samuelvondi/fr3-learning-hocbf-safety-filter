#!/usr/bin/env python3
from __future__ import annotations

"""
Per-pair feature statistics for PerPairBetaNet (fr3_hocbf/policy/pp_beta.py).

Mean/std of the 8 per-pair features over all ACTIVE pair rows of the training
rollout NPZs. psi and Lf_psi need a gamma; the rollout's own per-step gamma is
used (the gain the rollout was generated with), which only sets the scale.

Example:
    python -m fr3_hocbf.dataset.compute_pp_feature_stats \
      --rollout-labeled ../data/fr3_fullpool_train_weighted_local.jsonl \
      --out fr3_pp_feature_stats_v10.npz
"""

import argparse

import numpy as np
import torch

from fr3_hocbf.policy.pp_beta import PP_FEATURE_NAMES, kinematic_ddq_bounds_torch, pp_features_torch
from fr3_hocbf.sim.qp_data import get_joint_limits
from fr3_hocbf.dataset.rollout_records import expand_labeled_rollout_inputs
from fr3_hocbf.train.train_rollout import DT_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-labeled", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
