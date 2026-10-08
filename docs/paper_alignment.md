# Aligning the code with the paper — changes and results

Status: 2026-10-08, branch `dev` up to commit `763fd02`.

This note records how the evaluation code (`src/fr3_compare_rollout_policies.py`
and the reporting scripts) was changed to follow the method as described in the
paper, what the re-run benchmark shows, and what this means for the paper text.

## 1. Code changes

| # | Change | Before | Now | Commit |
|---|---|---|---|---|
| 1 | NN gains as lower bound for γ (eq. 22) | Fallback replaced the NN gains by the minimal required ones; γ could in rare cases drop below γ_NN | γ_final = max(γ_NN, γ_req + 0.5) | `8055bf6` |
| 2 | Analytic pre-QP check (Sec. II-D) | Solved a candidate QP with the NN gains and repaired if ψ < 0 on any active pair or the QP was infeasible; the post-solve G2 check could never fail; 2 QP solves on fallback steps | `g12_check_current`: first-order check on active pairs with h > 0, S_sup check on pairs with ψ > 0; QP solved once | `39d0406` |
| 3 | Skip pairs in collision (eqs. 21/23) | h < 0 set γ to the cap 200, ψ < 0 set β to 250 — this made the QP infeasible (e.g. 2787 steps in scenario 0002) | Such pairs are excluded from γ_req / β_req, in both fallback modes | `39d0406` |
| 4 | All active pairs ("any active pair") | Published runs checked only the 40 robot–obstacle pairs | `--online-pair-scope` / `--g12-fallback-pair-scope` default to `all`; `obstacle` still available | `f6f6aeb` |
| 5 | Fixed + Fallback uses only active pairs | Used all pair rows, including pairs above the activation threshold | Same active-pair mask as the QP | `f6f6aeb` |
| 6 | Fixed + Fallback logs its activations | `fallback_used` was never set, so this code reported 0 % | A step counts when γ or β is raised above the base gains | `f6f6aeb` |
| 7 | Stop at the first infeasible QP (Sec. III-C) | Only with `--terminate-on-qp-fail`, which `reproduce_tables.sh` did not pass | Always; flag removed | `f6f6aeb` |
| 8 | Fallback rate and γ̄/β̄ over all control steps (Table II) | Table II mixed definitions: Fixed + FB used per-scenario means (18.4 %, 32.1/42.6), NN + FB the all-steps rate (1.4 %) | Both scripts use total fallback steps / total steps and average gains over every step | `6fb903c` |
| 9 | Conservative β in NN + Fallback | Paper rule β_final = max(β_NN, β_req) (introduced in #1) | β_final = max(10⁻³, β_req + 0.5), may lie below β_NN | `763fd02` |

Also: `reproduce_tables.sh` prints its total runtime (`7396eab`); the README
documents the changes.

**Why change #9 deviates from the paper.** With change #1 the fallback could only
raise gains. Raising γ and β loosens the HOCBF constraints, so it restores
feasibility but cannot make the controller more cautious. The original code
instead often lowered β, which is what made it safe. A small β means ψ may only
decrease slowly, so the robot brakes earlier. Change #9 restores this
deliberately, without the cap-at-contact artifact (change #3).

## 2. Results (PD-failed benchmark, 460 scenarios)

Full run in `out/` (paper version of the fallback), NN + Fallback re-run with
conservative β in `out_nnfb/`. Brackets: published values.

| Controller | Safe succ. | Unsafe succ. | Safe non-succ. | Unsafe non-succ. | Violations | Fallback rate | γ̄/β̄ |
|---|---|---|---|---|---|---|---|
| Fixed γ/β | 62.2 (62.2) | 27.2 | 2.6 | 8.0 | 35.2 (35.2) | – | 10.0/15.0 |
| Fixed + Fallback | 67.6 (77.8) | 21.3 | 3.5 | 7.6 | 28.9 (13.9) | 7.5 | 10.7/14.5 |
| Learned (NN) | 71.1 (70.9) | 16.7 | 5.7 | 6.5 | 23.3 (23.0) | – | 12.3/5.4 |
| NN + Fallback, paper rule | 73.9 | 13.9 | 7.0 | 5.2 | 19.1 | 1.6 | 12.7/5.3 |
| **NN + Fallback, conservative β** | **80.4 (86.5)** | **7.2** | **8.0** | **4.3** | **11.5 (0.7)** | **1.3** | **12.6/5.1** |

- **The pipeline reproduces.** Fixed γ/β matches the paper exactly, and Learned (NN) gives the same outcome as the published run in 455 of 460 scenarios.
- **Conservative β vs. the paper rule:** safety improved in 38 scenarios and worsened in 3, task success is unchanged (87.8 → 87.6 %), and the median completion time is unchanged (8.38 s). QP-failure runs rose from 36 to 43.
- **What the fallback does:** it fires on 1.3 % of steps, always because ψ < 0 on a pair with h > 0. On every fallback step β is lowered, with a median of 2.34 → 10⁻³, so it effectively acts as a brake on the HOCBF margin.
- **Gap to the published 0.7 %:** the old code also fired on pairs already in contact. Its jump to the 200/250 caps made the QP infeasible right at contact, so the rollout stopped before the violation exceeded the 10⁻⁴ tolerance. Of the old "safe non-success" runs that changed, 20 of 29 ended with min h between −10⁻⁴ and 0. This is a measurement artifact and was not restored.
- **Fixed + Fallback** no longer escalates its gains. The "too reactive, large late corrections" argument in the discussion no longer holds.
- **Step times** (22–43 ms mean) were measured with all cores busy and are not comparable to Table III. Runtime should be measured in a separate single-process run.

## 3. Consequences for the paper

- **Sec. II-D / eq. (24):** describe β_final = max(ε, β_req + δ) and why lowering β is conservative. Fig. 2 only covers raising γ.
- **Mention the buffer δ = 0.5 and the caps 200/250**, which the code still uses.
- **Abstract, discussion, conclusion:** 86.5 % / 0.7 % becomes 80.4 % / 11.5 %, and "safe in 99.3 % of cases" becomes 88.5 %.
- **Rewrite the Fixed + Fallback discussion** (Table II: 7.5 %, 10.7/14.5).
- **Table II caption:** the fallback rate and γ̄/β̄ are now computed over all control steps.

## 4. Open items

- 21 scenario YAMLs set γ/β = 2/3. Fixed + Fallback takes its base gains from the YAML, whereas Fixed uses `--p1-fixed 10 --p2-fixed 15`, so the baselines differ in those scenarios.
- `fallback_count` includes the final, unlogged step when the QP fails (a 0.01-percentage-point effect).
- Fig. 4 plots have not been regenerated for `out_nnfb/`.
- Paper vs. code: "forward Euler" in the paper vs. the ½·u·dt² position update in the code, and `d_margin` is not passed for self-collision pairs (no effect while `d_margin = 0`).
- Rebuild the Docker image before the next `reproduce_tables.sh` run. The last run used an older Table I script.

## 5. Reproducing the NN + Fallback re-run

From `/repo/src` in the dev container:

```bash
OUT=/repo/out_nnfb; mkdir -p $OUT
split -n l/$(nproc) -d -a 3 --additional-suffix=.txt /data/pd_failed_460.txt $OUT/part_
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for p in $OUT/part_*.txt; do
  python fr3_compare_rollout_policies.py "$p" --modes nn_g12_fallback \
    --model ../artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt \
    --out-prefix "$p" > "$p.log" 2>&1 &
done; wait
```

Then replace the `nn_g12_fallback` rows of `out/pd_failed_final_{summary,steps}.csv`
with the new shards and run the Table I block and the table script from
`docker/reproduce_tables.sh` with `PREFIX=/repo/out_nnfb/pd_failed_final`.
