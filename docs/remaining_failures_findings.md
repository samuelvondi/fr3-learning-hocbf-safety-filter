# Remaining QP fails and timeouts — findings

2026-11-23. Analysis of the best configuration, recover B = 200 + `--joint-limit-mode brake --brake-margin 0.05`, with NN + FB + brake 0.05 as a cross-check. Percentages are of the 460 benchmark runs.

Scripts and data are in `data/joint_limits/`:
- `analyze_qpfail.py` and `report_qpfail.md`, with the per-stop table;
- `qpfail_dump/`, a rerun of all 28 scenarios with a QP stop using `--dump-failed-qp`.

## Summary

- **QP fails, recover 6.1 % (28):**
  - **1.3 % (6) are solver failures, not infeasibility.** OSQP hits its iteration limit (`user_limit`, max_iter = 25 000) on a feasible QP. Clarabel solves all of them. In 3 of the 4 tested, OSQP itself succeeds with 10⁵ iterations.
  - **4.8 % (22) are truly infeasible**, collision rows against the joint box. The decay rate β is not the cause: with β = 0 on every collision row (only ψ̇ ≥ 0 required) none becomes feasible. The recover boost (β = 200) causes 1 of them.
  - **The rows ask for more acceleration than the joints have.** With twice the acceleration limit, 13 of 22 become feasible. A different γ (½× or 2×) makes 9 more of them feasible in some combination (17 of 22 fixed by acc×2 or one γ change).
- **Timeouts, recover 3.9 % (18).** Three patterns, often combined:
  - **Joint 7 spins at its velocity limit (±3 rad/s)** in 8 runs. The nominal controller tracks only the end-effector position, so joint 7 gets zero nominal acceleration and no damping.
  - **The nominal pushes a joint into its limit** in 7 runs. The brake rows hold the joint there, so the arm makes no progress.
  - **The arm hovers 1.7–2.7 cm from the goal** without settling (tolerance 2 cm) in 5 runs.
  - One run is a pure deadlock: the QP cancels the whole nominal and the arm stands still.
- **Fallback (recover + brake 0.05):** used in 33.0 % (152) of runs, on 3.2 % of all steps. Most runs that end in a timeout (15 of 18) or a QP stop (26 of 28) used it; of the safe successes, 111 of 414 did.

## QP fails

Method: the failing QP (rows A u ≥ b, u = ddq) is dumped at the failing step. Then:
- **Depth:** a max-margin LP on row-normalised constraints. A positive margin means the QP is feasible.
- **IIS:** the minimal infeasible row set, found with a deletion filter.
- **Counterfactuals:** collision rows rebuilt with changed gains, or acceleration rows scaled.

The reruns reproduce the sweep exactly (19 NN + FB, 28 recover stops).

| | NN + FB | Recover B = 200 |
|---|---|---|
| QP-fail stops | 4.1 % (19) | 6.1 % (28) |
| OSQP iteration limit on a feasible QP | 0.9 % (4) | 1.3 % (6) |
| Truly infeasible | 3.3 % (15) | 4.8 % (22) |
| … feasible with β = 0 on all collision rows | 0 | 0 |
| … feasible without the recover boost | 0 | 1 |
| … feasible with acc limit ×2 | 7 | 13 |
| … feasible with acc ×2, γ/2 or γ×2 | 11 | 17 |
| … fallback active at the failing step | 9 | 18 |
| … self-collision row in IIS | 7 | 9 |
| … IIS with 2 collision rows | 7 | 9 |

- **Typical IIS:** 2–3 collision rows plus a few acceleration/velocity/position rows. Often the same link sits against two obstacles (7 of 22), or a link is caught between an obstacle and a self-collision pair.
- **ψ is positive in these rows.** β only scales ψ, so the infeasibility comes from the part of the row that does not depend on β: −L_fψ = −(‖v_rel‖²-term + γ·L_f h). This is a closing speed that the available acceleration cannot reverse in time. The fallback, which only changes γ/β, can therefore not fix it.
- **Solver failures:** the dumped QPs have constants up to |b| ≈ 10³ next to row norms ≈ 1, a scaling OSQP converges slowly on.
  - More iterations or a different solver (Clarabel) would remove up to 1.3 % of stops.
  - This needs a new flag; it would change existing modes if made the default.

## Timeouts

Classified over the last 5 s of each run (`dist_to_goal`, joint velocities, ddq_nom near joint limits).

| Pattern | Recover (18) | NN + FB (8) |
|---|---|---|
| Joint 7 at its velocity limit | 8 | 4 |
| Nominal pushing a joint into its limit | 7 | 2 |
| Hovering 1.7–2.7 cm from goal, not settling | 5 | 3 |
| Deadlock (arm at rest, QP cancels nominal) | 2 | 1 |
| Unclear | 2 | 0 |

The patterns overlap, so the columns sum to more than the run count.

- **Joint 7 spinning is systemic, not only a timeout effect.** Share of runs where |dq₇| ever exceeds 2.7 rad/s:
  - recover: 61 % of timeouts, 18 % of QP fails, 4 % of safe successes;
  - NN + FB: 29 % of violations.

  The cause is in `sim/nominal_controller.py`. It is a 3-D position task, so the 4-D null space has no velocity damping, and joint 7 does not move the end-effector position (nominal ddq₇ = 0). Velocity the QP gives joint 7, for example to turn the hand away from an obstacle, is never removed.
- **Limit push:** the nominal drives a joint (1, 2, 5 or 6) into its limit with up to −280 rad/s². The brake rows hold the joint and the arm chatters at the limit. The nominal's own joint-limit term is too weak and acts only in the outer 10 % of the range.

## Null-space damping in the nominal (pre-run only)

Flag `--nominal-ns-damping Kd` (default 0; all modes). It adds −K_d·(I − J⁺J)·dq to the nominal acceleration, damping the 4-D null space of the position task.
- The column of the position Jacobian for joint 7 is exactly 0, so joint 7 lies entirely in the null space and is damped directly.
- With K_d = 0 the code reproduces the reference runs bit-identically on 3 scenarios, for both modes.
- The NN still sees the changed nominal as an input; the NN was not retrained.

**Pre-run** with brake 0.05 on 24 hard scenarios (`data/joint_limits/runs_ns24/`): the 18 recover timeouts plus 6 NN + FB violation runs. These are counts on this selection, not benchmark percentages.

| K_d | NN + FB: safe / viol / QP fail / timeout | Recover B = 200: safe / viol / QP fail / timeout | Steps with \|dq₇\| > 2.7 (FB / rec) |
|---|---|---|---|
| 0 (brake 0.05) | 0 / 16 / 0 / 8 | 4 / 0 / 2 / 18 | 25.5 % / 20.2 % |
| 2 | 4 / 14 / 0 / 6 | 8 / 0 / 4 / 12 | 0.1 % / 0.0 % |
| 5 | 5 / 12 / 1 / 6 | 11 / 0 / 2 / 11 | 0 / 0 |
| 10 | 6 / 11 / 2 / 5 | 12 / 0 / 2 / 10 | 0 / 0 |
| 20 | 6 / 11 / 0 / 7 | 12 / 0 / 0 / 12 | 0 / 0 |

- **Joint 7 stops spinning from K_d = 2 on.** Mean intervention on these runs stays about the same (82–98 vs. 89–91 at K_d = 0).
- **Recover:** safe successes rise from 4 to 12 of 24 at K_d = 10–20; most come from former timeouts. NN + FB: violations fall from 16 to 11.

**Full 460 run** with brake 0.05 (`data/joint_limits/runs460/brake05_ns10`, `brake05_ns20`; `report460_ns.md`):

| Controller | K_d | Safe success | Violation | QP fail | Timeout | mean intervention_l2 |
|---|---|---|---|---|---|---|
| NN + FB | 0 | 83.0 % | 12.2 % | 3.0 % | 1.7 % | 39.1 |
| NN + FB | 10 | **85.2 %** | **9.8 %** | 3.0 % | 2.0 % | 32.0 |
| NN + FB | 20 | 85.0 % | 10.9 % | 1.5 % | 2.6 % | 38.4 |
| Recover B = 200 | 0 | 90.0 % | 0.0 % | 6.1 % | 3.9 % | 29.6 |
| Recover B = 200 | 10 | 92.2 % | 0.0 % | 4.1 % | 3.7 % | 31.9 |
| Recover B = 200 | 20 | **92.6 %** | 0.0 % | **2.8 %** | 4.6 % | 36.4 |

- **The joint-7 spin is gone on the whole benchmark.** Runs with \|dq₇\| > 2.7 rad/s: 36 (NN + FB) and 33 (recover) at K_d = 0; 0–1 at K_d = 10 and 20.
- **Recover gains 2.2–2.6 points of safe success** with still 0 violations. The QP-fail stops drop to 2.8 % at K_d = 20, but timeouts rise slightly (4.6 %). K_d = 10 and 20 differ by 2 runs.
- **NN + FB gains 2.2 points of safe success,** and violations fall from 12.2 to 9.8 % at K_d = 10.
- **Unlike the brake rows, the damping changes every run,** because the nominal changes everywhere. So there are losses as well as gains. At K_d = 10:
  - recover: 16 runs become safe successes, 6 former safe successes become 2 QP fails and 4 timeouts;
  - NN + FB: 26 gains, 16 losses (9 of them become violations).
- **The NN was not retrained.** It now receives the damped nominal as an input, so these numbers are a lower bound for a retrained policy.

## Remaining failures with damping (K_d = 10 vs. 20, recover B = 200 + brake 0.05)

Same methods as above:
- QP fails: reruns with `--dump-failed-qp` (`data/joint_limits/qpfail_dump_ns10`, `_ns20`; `report_qpfail_ns10.md`, `_ns20.md`).
- Timeouts: `classify_timeouts.py` (`report_timeouts_ns.md`).

Both reruns reproduce the sweep stop counts exactly.

| | K_d = 10 | K_d = 20 |
|---|---|---|
| QP fail (all collision rows vs. joint box) | 4.1 % (19) | 2.8 % (13) |
| … OSQP iteration limit on a feasible QP | 0 | 0 |
| … feasible with acc ×2 or γ ½×/2× | 14 | 10 |
| … feasible without recover boost / with β = 0 | 2 / 1 | 1 / 1 |
| Timeout | 3.7 % (17) | 4.6 % (21) |
| … deadlock at an obstacle (no progress over 30 s, min dist ≈ 0, QP cancels the nominal) | 16 | 20 |
| … nominal also pushes a joint into its limit | 6 | 6 |
| … no obstacle contact, joint limit only (scenario_0864) | 1 | 1 |

- **The solver failures are gone with damping.** The OSQP retry is no longer worth doing.
- **The timeouts are now all deadlocks, with no spinning left.** The arm stands at the safety boundary while the nominal pulls through the obstacle; ‖u − u_nom‖ ≈ ‖u_nom‖. The goal distance stays constant over the last 30 s, so more time would not help. This is a local minimum of nominal + CBF, which the safety filter cannot resolve alone.
- **The failure sets overlap strongly:** 29 runs fail at both K_d.
  - All 17 timeouts at K_d = 10 are also timeouts at K_d = 20.
  - K_d = 20 turns 7 of the 19 QP stops of K_d = 10 into safe successes and 2 into timeouts.
- **Improvement potential:**
  - QP stops: K_d = 10 has more left (14 runs fixable in principle by earlier/stronger braking, vs. 10).
  - Deadlocks: K_d = 20 has slightly more (20 vs. 16), and they are the larger remaining block for both.
  - Overall both end up at about the same count of failures (36 vs. 34 runs); K_d = 20 is the better baseline.

## Possible next steps (not done)

- Retraining with brake rows and the damped nominal, so that training matches evaluation.
- A solver fallback on `user_limit` (more iterations or Clarabel) as a flag: up to 1.3 % of runs.
- The truly infeasible stops need earlier action (larger effective safety distance or a lower γ for fast closing speeds). Gain changes at the failing step only rarely help.
