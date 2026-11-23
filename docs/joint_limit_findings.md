# Joint-limit infeasibility (`box_only`) in QP-fail stops — findings

2026-11-23. Part A: offline analysis of existing runs. Part B: experimental `--joint-limit-mode brake`. Scripts, runs and full tables are in `data/joint_limits/`. All percentages are of the 460 benchmark runs.

## Part B summary

- **The brake rows remove all `box_only` stops** (margin ≥ 0.05) and change nothing else. Safe success: NN + FB 80.4 → 83.0 %, recover B = 200 86.7 → 90.0 %. No previously safe run is lost.
- **Gain vs. Part A ceiling:** NN + FB 12 of 20 runs, recover 15 of 29. The other rescued stops end in a timeout, a later HOCBF-row stop, or (NN + FB only) a violation.
- **NN + FB: 3 former QP-fail runs now end in a violation** (12.2 % vs. 11.5 %). The stop had only hidden a later collision. Recover stays at 0 violations.
- **The brake rows hardly conflict with the collision rows.** In 1 run (margin 0.2) the QP would have been feasible without them. The remaining stops are HOCBF-row conflicts (NN + FB 4.1 %, recover 6.1 %), mostly in runs that used to stop earlier on the box.
- **The margin barely matters.** At margin 0 two `box_only` stops remain, because the braking set has no slack. Margins 0.05 and 0.2 give the same result within ±1 run. The default is 0.05.
- **`cbf` mode (joint HOCBF rows with fixed gains) not implemented.** Part A showed a one-step-ahead condition suffices, and `brake` already removes every `box_only` stop.

Details below Part A, in the "Part B" section.

# Part A — offline analysis

## Summary

- **Most QP-fail stops are caused by the joint box alone.** For NN + Fallback, 72 % of the stops (31 of 43) are `box_only`; for recover B = 200, 59 % (29 of 49); for Fixed 10/15, 94 % (29 of 31).
- **The conflict is always the same.** The discrete position bound conflicts with the acceleration limit: the joint is at or within 0.02 rad of its limit and still moves towards it at 0.45–1.7 rad/s. Even full braking (40 rad/s²) cannot keep it inside the limit on the next step.
  - In all but one case it is the lower limit (`pos-lo vs acc-hi`); the exception is scenario_0658, joint 1 upper.
  - The joints involved are mostly joint 6 (18 of 31 for NN + FB), then joints 1 and 4.
- **The state becomes unrecoverable only 0–1 steps before the stop.** At almost every stop, the braking-distance bound |dq| ≤ √(2·40·d) is violated for the first time at the failing state itself. The exact discrete check (simulate full braking with the rollout's Euler update) gives the same result. One step earlier, a braking input still existed, but the QP chose an acceleration that drove the joint into the limit.
  - The cause is that the one-step position box only requires q_{k+1} to stay inside the limit. It allows q_{k+1} = q_min with a large velocity still pointing outwards, and the next step is then infeasible.
  - A velocity bound on the next state therefore only has to act one step ahead. No long look-ahead is needed.
- **Ceiling for joint-limit CBFs:** 4.3 % (20) of runs for NN + FB and 6.3 % (29) for recover B = 200. These runs end in a `box_only` stop without an earlier violation. Without that stop they could finish safely, but they could also time out, violate later, or stop on a HOCBF-row conflict.
- **Recover's extra QP-fail stops are not caused by the joint box.** Recover has slightly fewer `box_only` stops than NN + FB (6.3 % vs. 6.7 %) but more stops of other causes (4.3 % vs. 2.6 %). Those come from the β = 200 HOCBF rows, and joint-limit CBFs will not remove them.

## Method

- **QP variable:** u = ddq of the 7 arm joints.
- **Joint box:** separable per joint, with limits from `sim.qp_data.get_joint_limits()` and dt = 0.02 (the rollout default; matches sim_time / steps).
  - lower = max(acc, vel, pos) lower bounds and upper = min(acc, vel, pos) upper bounds, built as in `kinematic_ddq_bounds`.
  - A stop is `box_only` iff lower > upper for at least one joint.
- **Failing state:** the rollout breaks before it logs the failing step. The state at that step is therefore `q_post`/`dq_post` of the last logged row. Propagation q_pre[k+1] = q_post[k] holds exactly (error 0) in all runs.
- **Cross-check:**
  - Fixed 10/15 rerun (`data/od_hocbf/runs/fixed`): 29 of 31 stops are `box_only`, as reported in `od_hocbf_findings.md`.
  - OD p_ω = 10⁸, ω ≥ 0: the analytic test agrees with the logged `od_stop_cause` on 30 of 30 stops.
- **"Violation before stop":** the run's min h < −10⁻⁴, the same rule as `make_final_metric_tables.py`.

## Results

| Controller | QP-fail stop | box_only | other | Violation before stop | Ceiling (box_only, no prior violation) |
|---|---|---|---|---|---|
| NN + Fallback (`out_nnfb/`) | 9.3 % (43) | 6.7 % (31) | 2.6 % (12) | 2.8 % (13) | 4.3 % (20) |
| NN + FB recover B = 200 (`data/fb_recover/b200/`) | 10.7 % (49) | 6.3 % (29) | 4.3 % (20) | 0.0 % (0) | 6.3 % (29) |
| Fixed 10/15 (`data/od_hocbf/runs/fixed/`) | 6.7 % (31) | 6.3 % (29) | 0.4 % (2) | 4.6 % (21) | 2.2 % (10) |

- **"QP-fail stop" counts every run that stops on a QP failure.** The disjoint outcome tables count a run with an earlier violation as a violation instead. That is why NN + FB shows 6.5 % QP fail there: 43 − 13 = 30 runs.
- **Bound pair** for the `box_only` stops of all three controllers: pos-lo vs. acc-hi in all but one; acc-lo vs. pos-hi once (scenario_0658, joint 1).
- **At the failing state:** distance to the limit 0–0.021 rad; |dq| 0.45–1.68 rad/s.
- **Look-ahead:** the final stretch of states outside the braking set before the stop is 0 steps in 29 of 31 stops (NN + FB) and 28 of 29 (recover); otherwise 1–2 steps. Earlier, transient violations of the braking bound occur only in single runs (scenario_0250 NN + FB, 74 steps before the stop; scenario_0642 Fixed, 19 steps before).

## Implications for Part B

- A `brake` row on the next state, dq_{k+1} ≤ √(2·a_brake·(q_max − q_{k+1})) and its mirror, targets exactly this failure. A one-step-ahead viability condition is enough.
- **Open risk:** the brake row acts when the joint is close to its limit, which is also when the collision rows are often tight. New conflicts between the brake row and the collision rows are the main thing to measure in Part B.

# Part B — experimental mode `--joint-limit-mode brake`

## Controller

- **Flags:** `--joint-limit-mode {box,brake}` (default `box`, unchanged) and `--brake-margin m` (default 0.05). They act only in `nn_g12_fallback` and `nn_g12_fallback_recover`. NN gains, fallback and recover rule are unchanged; no retraining.
- **Brake bound:** per joint, a bound on u such that after this step full braking at a = (1 − m)·40 rad/s² keeps the joint inside [q_min, q_max] at every later step. It uses the rollout's exact update, including the ½u·dt² term.
  - After n braking steps from (q₁, dq₁): q_n = q₁ + n·dq₁·dt − ½·a·dt²·n². This is linear in u for each n, so the bound is a closed-form min (upper) / max (lower) over n = 0…100.
  - n = 0 is the existing one-step position box, so the bound only tightens the pos rows. The QP keeps its size; the bound stays a linear box row.
  - If no braking-viable u exists inside the old box, the bound is clipped to the old box (strongest braking the box allows). The brake rows therefore never empty the box on their own.
- **Stop cause logged for every QP fail** (both modes, also under `box`) in `*_jl.csv`:
  - `box_only`: the joint rows alone (incl. brake) are infeasible;
  - `brake_vs_hocbf`: feasible without the brake tightening;
  - `hocbf_rows`: otherwise.
- Code: `brake_ddq_bounds`, `apply_brake_rows`, `jl_stop_cause` in `evaluation/compare_rollout_policies.py`.

## Checks

- **`box` reproduces HEAD bit-identically** on 3 scenarios (0000 success, 0045 `box_only` stop, 0102 HOCBF-row stop), for both `nn_g12_fallback` and `nn_g12_fallback_recover` B = 200. All step and summary columns except timing are identical.
- **Recursive feasibility of the brake set** (random walks, 120k steps per margin, u always at an end of the box): never an empty box, never a joint-limit violation, for margins 0, 0.05 and 0.2.
- **The `box` rerun on the 460 benchmark reproduces the archived results exactly:** 80.4 / 11.5 / 6.5 / 1.5 % and 86.7 / 0.0 / 10.7 / 2.6 %.

## Results (460 benchmark)

Runs: `data/joint_limits/runs460/` (`run_sweep.sh`, `report_sweep.py`, `report460.md`). A 20-scenario pre-run on Part A `box_only` stops is in `runs20/`, `report20.md`.

| Controller | Joint limits | Safe success | Violation | QP fail | Timeout | QP-fail stops: box_only / hocbf_rows / brake_vs_hocbf | mean intervention_l2 |
|---|---|---|---|---|---|---|---|
| NN + FB | box | 80.4 % | 11.5 % | 6.5 % | 1.5 % | 6.7 / 2.6 / 0.0 % | 20.4 |
| NN + FB | brake, m = 0 | 82.6 % | 12.6 % | 3.0 % | 1.7 % | 0.4 / 3.9 / 0.0 % | 33.6 |
| NN + FB | brake, m = 0.05 | **83.0 %** | 12.2 % | 3.0 % | 1.7 % | 0.0 / 4.1 / 0.0 % | 39.1 |
| NN + FB | brake, m = 0.2 | 82.8 % | 12.4 % | 3.0 % | 1.7 % | 0.0 / 3.9 / 0.2 % | 35.9 |
| Recover B = 200 | box | 86.7 % | 0.0 % | 10.7 % | 2.6 % | 6.3 / 4.3 / 0.0 % | 18.4 |
| Recover B = 200 | brake, m = 0 | **90.0 %** | 0.0 % | 5.7 % | 4.3 % | 0.4 / 5.2 / 0.0 % | 34.7 |
| Recover B = 200 | brake, m = 0.05 | **90.0 %** | 0.0 % | 6.1 % | 3.9 % | 0.0 / 6.1 / 0.0 % | 29.6 |
| Recover B = 200 | brake, m = 0.2 | **90.0 %** | 0.0 % | 5.9 % | 4.1 % | 0.0 / 5.9 / 0.0 % | 28.0 |

- **Disjoint outcomes** as in `make_final_metric_tables.py`. The stop-cause columns count every QP-fail stop, including runs that had a violation before.
- **Only runs that stopped under `box` change outcome.** Of the ~415 runs without a QP stop under `box`, 0 (m = 0), 1 (m = 0.05) and 3 (m = 0.2) differ at all. Their intervention is unchanged (NN + FB 19.51 vs. 19.51 at m = 0.05) and no new QP stops occur. The brake rows cut into the box on only 0.5 % of all steps.
- **The higher mean intervention_l2 comes from the rescued runs.** They now continue through hard situations (e.g. recover m = 0.05: 21 runs, 24.6k steps, mean 101) instead of stopping. It is not a cost on ordinary runs.
- **Where the former QP-fail runs go** (m = 0.05):
  - NN + FB (30): 12 safe success, 1 timeout, 3 violation, 14 still QP fail (all `hocbf_rows`);
  - recover (49): 15 safe success, 6 timeout, 28 still QP fail.
- **Timeouts rise slightly for recover** (2.6 → 3.9 %); for NN + FB +1 run.

## Open

- The remaining stops are HOCBF-row conflicts (NN + FB 4.1 %, recover 6.1 %). They are not addressable by joint limits.
- Retraining with the same brake rows in `qp_rows.py`, so that the training QP matches the evaluation QP (not done).
- The G1/G2 fallback still computes its bounds on β/γ with the old joint box (`kinematic_ddq_bounds`). This has no visible effect here.
