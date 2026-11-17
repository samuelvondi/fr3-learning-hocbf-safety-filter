# Optimal-decay HOCBF — findings

Evaluation on 2026-11-17 of the new standalone controller `od_hocbf` (no NN, no fallback) on the 460-scenario PD-failed benchmark. Outcomes are disjoint (safe success / violation / QP fail / timeout; violation = min h < −10⁻⁴ at any point; same logic as `make_final_metric_tables.py`, table 0). Runs, driver and report in `data/od_hocbf/` (`run_all.sh`, `analyze.py`, `report.md`).

## Summary

- **Optimal decay does not help on this benchmark.** For large p_ω (≥ 10²) the controller gives the Fixed result (62.2 / 35.2 / 2.2 / 0.4 %). For small p_ω it is clearly worse: down to 19.3 % safe success and 79.8 % violations.
- **It rescues none of the 31 runs where Fixed stops on a QP failure.** 29 of these stops are `box_only`: the joint bounds alone (acceleration, velocity and discrete position box) are infeasible, so no ω can help. 21 of the 31 runs already had a violation before the stop.
- **The 35 % violations of Fixed come from feasible QPs.** The QP is solvable at every step and h still drops below −10⁻⁴. Relaxing the decay rate can only make that worse.

## Controller

Zeng, Zhang, Li, Sreenath, *Safety-critical control using optimal-decay control barrier function with guaranteed point-wise feasibility*, ACC 2021, eq. (21). It is applied to the top-order HOCBF row of every collision pair j, with one decay variable ω_j per pair and ω₀ = 1:

    min_{u, ω}  ½‖u − u_nom‖² + p_ω Σ_j (ω_j − 1)²                (21a, paper weighting)
    s.t.        L_gψ_j u + L_fψ_j ≥ −ω_j β₀ ψ_j      active pairs     (21b, α(ψ) = β₀ψ)
                u in the joint bounds (unchanged hard rows)          (21c)
                ω_j ≥ ω_min                          only if finite

- ψ_j = L_f h_j + γ₀ h_j, with γ₀ = 10 and β₀ = 15, the Fixed baseline. γ₀ stays fixed: a pointwise γ would add a γ̇·h term that the row ignores.
- Rows with |ψ_j| < 10⁻⁹ are hard rows with ω_j = 1. Inactive and collided pairs are handled as in `fixed`.
- Two variants:
  - `--od-omega-min=0` (default): a row with ψ_j < 0 can at most relax to ψ̇_j ≥ 0.
  - `--od-omega-min=-inf`: unbounded, as in Remark 6. A row with ψ_j < 0 can then be relaxed without limit.
- Implementation: `OptimalDecayQP` in `evaluation/compare_rollout_policies.py`. It is a fixed-size parameterized cvxpy/OSQP problem with the same tolerances as the hard QP. A QP-fail stop is unchanged and logged with its cause: `box_only` / `psi_zero` / `psi_neg_bound` / `other`. The per-step ω log is `*_od_steps.csv`.

**Checks** (20 benchmark scenarios, `data/od_hocbf/checks/`):

- **ω frozen at 1** (`--od-freeze-omega`): bit-identical to `fixed --p1-fixed 10 --p2-fixed 15`, all step and summary columns except timing.
- **p_ω = 10⁸, both variants** (Remark 7): same status and step count as Fixed on all 19 runs without a QP failure. |Δ min h| ≤ 3·10⁻⁸ and |ω − 1| ≤ 2·10⁻⁶.
  - The Fixed QP failure on these 20 is `box_only`.
  - Over long runs, solver-tolerance differences can make the trajectories drift apart (scenario_0028: |Δu| reaches 15 after step 710), but the outcome is the same.
- **Baseline rerun:** Fixed 10/15, rerun here, reproduces the archived 62.2 % / 35.2 % exactly.

## Results

| Controller | p_ω | Safe success | Violation | QP fail | Timeout |
|---|---|---|---|---|---|
| NN + Fallback (reference, `out_nnfb/`) | | 80.4 % (370) | 11.5 % (53) | 6.5 % (30) | 1.5 % (7) |
| Fixed γ/β = 10/15 (rerun) | | 62.2 % (286) | 35.2 % (162) | 2.2 % (10) | 0.4 % (2) |
| OD, ω ≥ 0 | 10⁻² | 40.4 % (186) | 58.3 % (268) | 1.3 % (6) | 0.0 % (0) |
| | 1 | 50.7 % (233) | 47.4 % (218) | 1.7 % (8) | 0.2 % (1) |
| | 10² | 62.2 % (286) | 35.2 % (162) | 2.2 % (10) | 0.4 % (2) |
| | 10⁴ | 62.0 % (285) | 35.2 % (162) | 2.4 % (11) | 0.4 % (2) |
| | 10⁶, 10⁸ | 62.2 % (286) | 35.2 % (162) | 2.2 % (10) | 0.4 % (2) |
| OD, ω unbounded | 10⁻² | 19.3 % (89) | 79.8 % (367) | 0.9 % (4) | 0.0 % (0) |
| | 1 | 48.0 % (221) | 50.4 % (232) | 1.5 % (7) | 0.0 % (0) |
| | 10² | 62.2 % (286) | 35.2 % (162) | 2.2 % (10) | 0.4 % (2) |
| | 10⁴ | 62.0 % (285) | 35.2 % (162) | 2.4 % (11) | 0.4 % (2) |
| | 10⁶, 10⁸ | 62.2 % (286) | 35.2 % (162) | 2.2 % (10) | 0.4 % (2) |

- **At p_ω = 10² and 10⁴ the totals match Fixed, but individual runs differ.** 3–4 runs move from safe success to violation and 3–4 move the other way.
- **At p_ω ≥ 10⁶ the outcome of every run is identical to Fixed.**
- **Small p_ω mostly turns Fixed's safe successes into violations.** p_ω = 10⁻² with ω unbounded: 197 of 286.

**Scale:** when the QP intervenes, ½‖u − u_nom‖² has a median of about 5·10² and a q90 of about 3.6·10⁴ (Fixed step log). Moving ω by 1 costs p_ω. Relative to the cost scale, the paper's ACC values (p_ω = 10⁸ safe, 10² unsafe, with ½u² ≈ 10⁷) correspond to about 10⁴–10⁵ and 10⁻² here. The sweep covers both.

### Key question: Fixed's QP-fail stops

Fixed stops on a QP failure in 31 runs; 21 of them had a violation before the stop.

| p_ω (both ω variants) | Safe success | Violation | QP fail | Timeout |
|---|---|---|---|---|
| 10⁻² (ω ≥ 0 / unbounded) | 0 | 25 / 27 | 6 / 4 | 0 |
| 1 | 0 | 24 | 7 | 0 |
| 10² | 0 | 22 | 9 | 0 |
| ≥ 10⁴ | 0 | 21 | 10 | 0 |

- **No run is finished safely.** The 21 violations at p_ω ≥ 10⁴ are the runs that were already violated before Fixed stopped. The remaining 10 still stop.
- **At p_ω = 10⁸, 29 of the 31 stops are `box_only`.** The joint-limit rows alone have no solution, so the theorem's assumption of a non-empty, convex U_adm fails.
- **Only one Remark 7 event in 235k steps:** scenario_0185, step 149. There the HOCBF rows make the QP infeasible and ω relaxes (ω up to 7.1 on 3 pairs). The run had already violated (min h = −1.2·10⁻³) and then times out instead of stopping.

### How often ω leaves 1

Share of solved steps with |ω_j − 1| > 10⁻³ on any pair. The requested threshold of 10⁻⁶ mostly counts OSQP noise: at p_ω = 10⁶ it flags 17.8 % of steps while every ω lies in [0.999, 1].

| p_ω | ω_min | any ω ≠ 1 | any ω > 1 (relaxed, ψ > 0) | any ω < 1 | steps with a ψ_j < 0 row |
|---|---|---|---|---|---|
| 10⁻² | 0 | 34.0 % | 10.8 % | 28.7 % | 29.6 % |
| 1 | 0 | 33.9 % | 17.5 % | 23.3 % | 24.7 % |
| 10² | 0 | 28.0 % | 17.2 % | 15.7 % | 22.0 % |
| 10⁴ | 0 | 7.3 % | 3.8 % | 4.7 % | 20.9 % |
| 10⁶ | 0 | 0.1 % | 0.0 % | 0.1 % | 19.8 % |
| 10⁸ | 0 | 0.0 % (1 step) | 0.0 % | 0.0 % | 22.8 % |
| 10⁻² | −∞ | 39.0 % | 6.0 % | 36.2 % | 37.1 % |
| 1 | −∞ | 25.6 % | 13.4 % | 15.7 % | 17.1 % |
| 10² | −∞ | 23.7 % | 17.0 % | 10.8 % | 17.6 % |
| ≥ 10⁴ | −∞ | same as ω ≥ 0 | | | |

- **ω < 1 is never a tightening here.** For a row with ψ_j > 0, a larger ω only relaxes, so the optimum never chooses ω < 1. ω < 1 occurs on rows with ψ_j < 0, where it relaxes the row: down to ω = 0 (ψ̇ ≥ 0) with ω ≥ 0, and below 0 when ω is unbounded (min ω = −161 at p_ω = 10⁻²). This is exactly the "row effectively deleted" case.
- **ψ_j < 0 rows are common:** about 20 % of steps have at least one, even when the controller behaves like Fixed (p_ω ≥ 10⁶). These are states already outside the first-order set.
- **Ranges** (ω ≥ 0): ω_max up to 410 (p_ω = 10⁻²), 44 (1), 3.4 (10²) and 7.0 (10⁴). The q99 of ω_max is 12.1, 3.9, 1.19 and 1.01.

**QP-fail stop causes, OD runs** (box_only / psi_zero / psi_neg_bound / other): ω ≥ 0 at 10⁻² gives 23 / 0 / 16 / 0 and at 10⁸ gives 29 / 0 / 0 / 1. ω unbounded at 10⁻² gives 8 / 0 / 0 / 0, because those runs end in violations before they can stop. `psi_zero` never occurs.

### QP infeasibility is not reduced

QP-fail stops per configuration. Each run stops at most once, so these are runs out of 460, regardless of any earlier violation.

| Config | QP-fail stops | box_only | psi_neg_bound | other |
|---|---|---|---|---|
| Fixed 10/15 | 31 (6.7 %) | (29 of the 30 equivalent p_ω = 10⁸ stops) | | |
| OD p_ω ≥ 10⁴, both ω variants | 30–31 | 29 | 0–2 | 0–2 |
| OD p_ω = 10², ω ≥ 0 / unbounded | 27 / 25 | 26 / 25 | 1 / 0 | 0 |
| OD p_ω = 1, ω ≥ 0 / unbounded | 33 / 14 | 26 / 14 | 7 / 0 | 0 |
| OD p_ω = 10⁻², ω ≥ 0 / unbounded | 39 / 8 | 23 / 8 | 16 / 0 | 0 |

- **At p_ω ≥ 10⁴ the stops are the same as Fixed's.** The method removes exactly one: the single HOCBF-caused infeasibility (scenario_0185).
- **Fewer stops at small p_ω are not a fix.** With ω unbounded, the runs violate first and never reach the box-infeasible states (8 stops, 80 % violations).
- **With ω ≥ 0 and small p_ω the stops even increase.** The relaxed trajectories reach states where ψ_j < 0 rows conflict with the bounds (`psi_neg_bound`).

**Why Theorem 1 does not apply.** The theorem resolves the conflict *between the CBF row and U_adm*. It assumes U_adm(x) is convex **and non-empty**. In the paper's ACC example, U_adm is a constant force box, so it is never empty. Here, U_adm is the intersection of three state-dependent boxes on the same acceleration u:

- acceleration: ddq_min ≤ u ≤ ddq_max
- velocity, one step ahead: (dq_min − dq)/dt ≤ u ≤ (dq_max − dq)/dt
- position, one step ahead: 2(q_min − q − dq·dt)/dt² ≤ u ≤ 2(q_max − q − dq·dt)/dt²

A joint moving fast toward its limit needs more deceleration than ddq_max allows. Then U_adm = ∅ (`box_only`), whatever h, ψ and ω are. The joint position limit is effectively a relative-degree-2 state constraint enforced only one step ahead, not as a CBF, so nothing keeps the state in the region where braking is still possible. In addition, the paper's guarantee only holds pointwise, not persistently (Sec. V-C).

**Possible next step:** write the joint limits as CBFs, e.g. h_q = q_max − q with a braking-distance-aware velocity bound (dq ≤ sqrt(2·ddq_max·(q_max − q))). That keeps U_adm non-empty along the trajectory and would remove the `box_only` stops. Alternative: soften the position rows with a slack variable.

## Interpretation

Zeng et al. guarantee feasibility at each state only if the input set is non-empty and the state is inside the safe set (Theorem 1). They say explicitly that this guarantees neither persistent feasibility nor safety (Sec. V-C, Fig. 4b). On this benchmark neither failure mode of Fixed is the one optimal decay addresses:

1. **Infeasibility comes from the joint bounds, not from the HOCBF rows** (29/31 `box_only`). The discrete position box 2(q_max − q − q̇·dt)/dt² is out of reach of the acceleration limits. A fix would have to act on the bounds, e.g. a braking-distance-aware velocity bound, not on β.
2. **Violations happen with a feasible QP.** They come from the sampled-data implementation and from states already outside the first-order set (ψ_j < 0). Optimal decay can only relax rows, so it helps safety only through fewer QP stops, and there are none to gain. The rows it relaxes are exactly those with ψ_j < 0, which increases violations.

By contrast, the fallback mode with large β on ψ_j < 0 rows (`nn_g12_fallback_recover`, `fallback_findings.md`) tightens exactly those rows and removes violations.
