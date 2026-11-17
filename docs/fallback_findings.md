# Runtime fallback — findings

Analyses from 2026-11-16/17 of the NN + Fallback controller (`nn_g12_fallback`, published checkpoint epoch 22) on the 460-scenario benchmark, using the reference run in `out_nnfb/`. Outcomes are disjoint (safe success / violation / QP fail / timeout; violation = min h < −10⁻⁴ at any point; see `make_final_metric_tables.py`, table 0).

## 1. What the fallback does

- **Only G1 ever fires.** All 2,815 fallback steps (1.3 % of steps) are G1 triggers; G2 never fails because it lets every row use its own best-case u (`per_pair_beta_findings.md` §2).
- **Effect:** γ is raised to max(γ_NN, γ_req + 0.5) and β drops to 10⁻³…0.5, an emergency brake on ψ.
- **It is not a feasibility check.** It does not prevent QP infeasibility: QP-fail runs are 7.8 % for NN and 9.3 % for NN + Fallback. Only 8 of the 30 QP-fail runs without a violation had a fallback in their last 3 steps.

## 2. What happens after it fires

| Runs (row %) | Runs | Safe success | Violation | QP fail | Timeout |
|---|---|---|---|---|---|
| Fallback never fires | 316 | 94.0 % | 0.0 % | 5.4 % | 0.6 % |
| Fallback fires | 144 | 50.7 % | 36.8 % | 9.0 % | 3.5 % |

- **The trigger is a perfect danger detector:** it fired before the first violation in all 53 violation runs, and no run without a fallback had a violation.
- **It is not enough to recover:** only half of the triggered runs end safely.
- **Fallback episodes are short:** 50 % last a single step, the median is 2 steps and the 95th percentile is 8. The first violation follows a median of 4 steps after the last fallback step.

## 3. Holding the brake does not help

Experimental mode `nn_g12_fallback_hold --fallback-hold-steps K`: after a trigger, the fallback gains (recomputed each step) are kept for K more steps. With K = 0 it reproduces `nn_g12_fallback` exactly on 3 scenarios (3,804 steps; all columns except timing). Results in `data/fb_hold/` (`run_sweep.sh`, `report.py`).

| Hold | Safe success | Violation | QP fail | Timeout | Fallback steps |
|---|---|---|---|---|---|
| K = 0 (reference) | 80.4 % | 11.5 % | 6.5 % | 1.5 % | 1.3 % |
| K = 3 | 80.7 % | 11.5 % | 6.5 % | 1.3 % | 2.7 % |
| K = 5 | 80.2 % | 10.9 % | 7.6 % | 1.3 % | 2.8 % |
| K = 10 | 80.7 % | 10.9 % | 7.2 % | 1.3 % | 3.6 % |

- **The brake is not released too early, it is too weak.** With K = 10, the fallback is still active at the first violation step in 78 % of the violation runs.

## 4. Hypothesis: β ≈ 0 is wrong once ψ < 0

In the 5 steps before a violation (K = 10), the logged min ψ is negative (median −0.0036). With ψ_j < 0, the row ψ̇_j ≥ −β ψ_j with β ≈ 0 only stops ψ from falling further; a large β would force it back up. The fallback's β ≈ 0 may therefore be the wrong direction once ψ is already negative.

Checked: the logged ψ uses the applied γ, i.e. γ_final on fallback steps (`compare_rollout_policies.py`, `compute_barrier_metrics(..., gamma=p1)`), evaluated on the post-step state as the minimum over all pairs. So ψ stays negative even with the raised γ, and the hypothesis stands.

On fallback steps, ψ < 0 occurs before the step in 18 % of cases but after the step in 55 % (other steps: about 10 % each). A recovery rule must therefore act on every step, not only on fallback steps.

## 5. Large β on rows with ψ < 0 works

Experimental mode `nn_g12_fallback_recover --recover-beta B`: the fallback is unchanged. In addition, on every step, active rows with ψ_j < 0 (computed with the applied γ, before the step) get β_j = max(β, B) in the QP. With B = 0 it reproduces `nn_g12_fallback` exactly on 3 scenarios (3,804 steps; all columns except timing). Results in `data/fb_recover/` (`run_sweep.sh`, `report.py`).

| Recover β | Safe success | Violation | QP fail | Timeout | Fallback steps |
|---|---|---|---|---|---|
| B = 0 (reference) | 80.4 % | 11.5 % | 6.5 % | 1.5 % | 1.3 % |
| B = 10 | 81.3 % | 10.4 % | 6.7 % | 1.5 % | 1.3 % |
| B = 25 | 82.4 % | 8.7 % | 7.2 % | 1.7 % | 1.2 % |
| B = 50 | 84.1 % | 6.1 % | 7.6 % | 2.2 % | 1.2 % |
| B = 100 | 86.5 % | 1.3 % | 9.1 % | 3.0 % | 1.9 % |
| B = 200 | **86.7 %** | **0.0 %** | 10.7 % | 2.6 % | 1.9 % |

- **Violations fall steadily with B and reach zero at B = 200;** safe success rises by 6.3 points.
- **The cost is more QP-fail stops** (6.5 → 10.7 %) and slightly more timeouts.
- **Caveat:** a run that stops on QP failure cannot record a later violation, so part of the violation drop could move into the QP-fail column. Safe success, which rises steadily, is not affected by this.
- **Why it works:** for a row with ψ_j < 0, the HOCBF row ψ̇_j ≥ −β ψ_j = β |ψ_j| demands that ψ_j increase, faster for larger β. The existing fallback does the opposite (β ≈ 0), which only stops ψ from falling.

Open: B > 200 (the β cap is 250); the recovery rule without the G1 fallback; whether the extra QP failures can be reduced, e.g. with B scaled by |ψ_j|.
