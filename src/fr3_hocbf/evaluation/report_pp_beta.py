#!/usr/bin/env python3
"""
Results table for the per-pair beta comparison, and the exact-reproduction check.

Reads summary/steps CSVs of fr3_hocbf/evaluation/compare_rollout_policies.py (shards allowed via
globs) and reports per mode, as % of scenarios with counts in brackets:
safe success, violations (min_h < -tol), QP-fail stops; and over all control
steps: fallback rate, % of active pairs on fallback, cap hits.

--check-equal A=B asserts that modes A and B produced bit-identical step logs
(e.g. nn=nn_pp_beta and nn_g12_fallback=nn_pp_beta_fallback when the per-pair
modes run the old checkpoint with --pp-fallback-reduce scalar).

Example:
    python -m fr3_hocbf.evaluation.report_pp_beta --prefix '../out_pp/shards/part_*.txt' \
      --check-equal nn=nn_pp_beta nn_g12_fallback=nn_pp_beta_fallback
"""

import argparse
import glob
import sys

import numpy as np
import pandas as pd

GAMMA_CAP = 200.0
BETA_CAP = 250.0
BUFFER = 0.5


def read_concat(pattern: str, exclude: str | None = None) -> pd.DataFrame:
    paths = sorted(p for p in glob.glob(pattern) if exclude is None or exclude not in p)
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)


def pct(n: int, total: int) -> str:
    return f"{100.0 * n / max(1, total):5.1f}% ({n})"


def mode_table(summary: pd.DataFrame, steps: pd.DataFrame, pp_steps: pd.DataFrame, tol: float) -> pd.DataFrame:
    summary = summary.copy()
    summary["success"] = summary["success"].astype(str).str.lower().isin(["true", "1", "1.0"])
    summary["unsafe"] = pd.to_numeric(summary["min_h"], errors="coerce") < -tol
    summary["qp_fail"] = summary["qp_fail_count"] > 0

    out = []
    for mode, s in summary.groupby("mode", sort=False):
        n = len(s)
        n_steps = int(s["steps"].sum())
        row = {
            "mode": mode,
            "scenarios": n,
            "safe success": pct(int((s.success & ~s.unsafe).sum()), n),
            "violations": pct(int(s.unsafe.sum()), n),
            "QP-fail stops": pct(int(s.qp_fail.sum()), n),
            # Same definition as Table I: fallback_count over all control steps.
            "fallback rate (steps)": f"{100.0 * s.fallback_count.sum() / max(1, n_steps):5.2f}%",
            "gamma/beta mean": f"{(s.p1_mean * s.steps).sum() / max(1, n_steps):.1f}/{(s.p2_mean * s.steps).sum() / max(1, n_steps):.1f}",
            "pairs on fallback": "-",
            "cap-hit steps": "-",
            "beta cap pair-steps": "-",
        }
        st = steps[steps["mode"] == mode] if len(steps) else steps
        if mode in ("nn_g12_fallback", "davide_online") and len(st):
            fb = st["fallback_used"].astype(str).str.lower().isin(["true", "1", "1.0"])
            cap = (st["max_gamma_required"] + BUFFER > GAMMA_CAP) | (st["max_beta_required"] + BUFFER > BETA_CAP)
            row["cap-hit steps"] = pct(int((fb & cap).sum()), len(st))
            if mode == "nn_g12_fallback":
                # Scalar fallback replaces beta of every active pair.
                row["pairs on fallback"] = f"{100.0 * st.loc[fb, 'active_pairs'].sum() / max(1, st['active_pairs'].sum()):5.2f}%"
        ps = pp_steps[pp_steps["mode"] == mode] if len(pp_steps) else pp_steps
        if len(ps):
            cap = ps["gamma_cap_hit"].astype(bool) | (ps["n_beta_cap_pairs"] > 0)
            # Pairs whose beta the fallback changed: flagged (raised) + lowered.
            changed = ps["n_fb_pairs"].sum() + ps["n_lowered_pairs"].sum()
            row["pairs on fallback"] = f"{100.0 * changed / max(1, ps['n_active'].sum()):5.2f}%"
            row["cap-hit steps"] = pct(int(cap.sum()), len(ps))
            row["beta cap pair-steps"] = str(int(ps["n_beta_cap_pairs"].sum()))
        out.append(row)
    return pd.DataFrame(out)


def check_equal(steps: pd.DataFrame, a: str, b: str) -> bool:
    ignore = {"mode", "solve_time_s", "step_elapsed_s"}
    # Per-pair modes log p2/p2_nn as the mean over active beta_j; for 43 equal values
    # np.mean rounds in the last bits. All control quantities are compared bitwise.
    mean_logged = {"p2", "p2_nn", "p2_fallback"}
    sa = steps[steps["mode"] == a].reset_index(drop=True)
    sb = steps[steps["mode"] == b].reset_index(drop=True)
    if len(sa) != len(sb) or len(sa) == 0:
        print(f"[check] {a} vs {b}: FAIL, {len(sa)} vs {len(sb)} logged steps")
        return False
    cols = [c for c in sa.columns if c not in ignore]
    bad = []
    for c in cols:
        x, y = sa[c], sb[c]
        if c in mean_logged:
            same = np.isclose(x.to_numpy(), y.to_numpy(), rtol=1e-12, atol=0.0, equal_nan=True)
        elif pd.api.types.is_numeric_dtype(x):
            same = (x.to_numpy() == y.to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
        else:
            same = (x.astype(str) == y.astype(str)).to_numpy()
        if not same.all():
            bad.append(f"{c} ({int((~same).sum())} rows)")
    print(f"[check] {a} vs {b}: {len(sa)} steps, {len(cols)} columns -> " + ("IDENTICAL" if not bad else "DIFFER: " + ", ".join(bad)))
    return not bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="out-prefix (glob allowed) of the comparison runs")
    parser.add_argument("--tol", type=float, default=1e-4, help="Safety tolerance on min_h, as in Table I")
    parser.add_argument("--check-equal", nargs="*", default=[], help="Pairs A=B of modes whose step logs must be identical")
    args = parser.parse_args()

    summary = read_concat(f"{args.prefix}_summary.csv")
    steps = read_concat(f"{args.prefix}_steps.csv", exclude="_pp_steps.csv")
    pp_steps = read_concat(f"{args.prefix}_pp_steps.csv")
    if summary.empty:
        sys.exit(f"no summary CSVs for {args.prefix}")

    table = mode_table(summary, steps, pp_steps, args.tol)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.to_string(index=False))

    ok = all([check_equal(steps, *pair.split("=")) for pair in args.check_equal])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
