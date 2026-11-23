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

**Pre-run** with brake 0.05 on 24 hard scenarios (`data/joint_limits/runs_ns24/`): the 18 recover timeouts plus 6 NN + FB violation runs. These are counts on this selection, not benchmark percentages; the full 460 run is still open.

| K_d | NN + FB: safe / viol / QP fail / timeout | Recover B = 200: safe / viol / QP fail / timeout | Steps with \|dq₇\| > 2.7 (FB / rec) |
|---|---|---|---|
| 0 (brake 0.05) | 0 / 16 / 0 / 8 | 4 / 0 / 2 / 18 | 25.5 % / 20.2 % |
| 2 | 4 / 14 / 0 / 6 | 8 / 0 / 4 / 12 | 0.1 % / 0.0 % |
| 5 | 5 / 12 / 1 / 6 | 11 / 0 / 2 / 11 | 0 / 0 |
| 10 | 6 / 11 / 2 / 5 | 12 / 0 / 2 / 10 | 0 / 0 |
| 20 | 6 / 11 / 0 / 7 | 12 / 0 / 0 / 12 | 0 / 0 |

- **Joint 7 stops spinning from K_d = 2 on.** Mean intervention on these runs stays about the same (82–98 vs. 89–91 at K_d = 0).
- **Recover:** safe successes rise from 4 to 12 of 24 at K_d = 10–20; most come from former timeouts. NN + FB: violations fall from 16 to 11.
- **Open:** the full 460 run with K_d = 10 and 20, to check that the many runs that already succeed are not hurt.

## Possible next steps (not done)

- Full 460 run of the null-space damping (see above).
- A solver fallback on `user_limit` (more iterations or Clarabel) as a flag: up to 1.3 % of runs.
- The truly infeasible stops need earlier action (larger effective safety distance or a lower γ for fast closing speeds). Gain changes at the failing step only rarely help.
