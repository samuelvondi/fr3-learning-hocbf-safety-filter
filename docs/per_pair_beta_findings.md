# Per-pair β, fallback analysis and training speed — findings

Work from 2026-11-15/16 on branch `dev`. Independent of the paper notes in `paper_alignment.md`. The existing controllers, the published checkpoint and the reference results in `out/` and `out_nnfb/` (local, gitignored; from the 2026-10 alignment run) are unchanged, and the existing modes reproduce them exactly (§5.3). Environment, data folders and verification tools: `dev_notes.md`.

## 1. Summary

- **Per-pair β brings no improvement over a shared β.** Four seeds, each with two checkpoint-selection rules, were evaluated on the 460-scenario benchmark. None beats the existing NN + Fallback (80.4 % safe success, 11.5 % violations). Under the README selection rule, all four seeds are consistently worse: 77.3 ± 0.4 % / 14.9 ± 0.3 % with fallback.
- **The G2 check of the runtime fallback never fires.** All 2,815 fallback steps of the existing NN + Fallback are triggered by G1. In practice the fallback is "raise γ and set β ≈ 0 for all pairs", an emergency brake on ψ.
- **The README checkpoint rule picks barely trained per-pair networks** (epochs 2–14). Later epochs often learn β ≈ 0 themselves: very few violations, but many QP-fail stops and much less task success.
- **Training is single-core bound, and a GPU does not help.** Half the time went into link Jacobians computed with per-endpoint autograd. An analytic Jacobian (`differentiable/kinematics_fast.py`) makes training 4.7× faster per segment.
- **`src/` was restructured** into the `fr3_hocbf` package (see the README). All scripts reproduce their previous outputs bit for bit.

## 2. The runtime fallback: what G1 and G2 really do

For each active pair j with the NN gains γ, β (`evaluation/gain_policies.py`, `g12_check_current`):

- **G1** (pairs with h_j > 0): ψ_j = L_f h_j + γ h_j ≥ 0. It fails when the pair approaches too fast for its distance. Only raising γ repairs it.
- **G2** (pairs with ψ_j > 0): L_f ψ_j + S_sup,j + β ψ_j ≥ 0, with S_sup,j = sup_u L_g ψ_j · u over the kinematic acceleration box. It asks whether this row on its own could be satisfied with the bounded input. Raising β repairs it.

The fallback fires if any pair fails either check. It then sets γ = max(γ_NN, γ_req + 0.5) and β = max(10⁻³, β_req + 0.5), where β_req is the maximum over pairs (scalar).

**Measured in `out_nnfb/`** (218,373 steps, 2,815 fallback steps = 1.29 %):

| Quantity on fallback steps | 5 % | 25 % | median | 75 % | 95 % |
|---|---|---|---|---|---|
| β_NN | 1.9 | 2.12 | 2.34 | 24.2 | 32.5 |
| max_j β_req,j | −15.8 | −2.77 | −0.90 | −0.09 | 0 |
| β_final | 10⁻³ | 10⁻³ | 10⁻³ | 0.41 | 0.5 |
| min_j G2_j with NN gains | 0.69 | 1.45 | 2.74 | 8.6 | 17 |

- **All 2,815 fallback steps are G1 triggers.** G2 never fails, γ is raised on every fallback step, and β is lowered on every one.
- **Why G2 never fails:** S_sup,j is large because it adds the best case of all 7 joints (mean 6.9 vs. 0.09 for L_f ψ over active rows of the training rollouts). Each row also gets its own best-case u. The check ignores that all rows and the joint bounds must hold for the *same* u, so it is a weak necessary condition.
- **What the fallback therefore does:** β_req ≤ 0 on every fallback step, so β_final is set entirely by the floor and buffer (10⁻³ … 0.5). With β ≈ 0 each HOCBF row requires ψ̇_j ≥ −β ψ_j ≈ 0: ψ may no longer decrease on any active pair. This brake on 1.3 % of steps is what makes NN + Fallback better than NN (80.4 vs. 71.1 % safe success). The β_req formula is never binding on this benchmark.
- **Consequence:** a G2-based trigger cannot identify which pairs need help. Any per-pair fallback rule that relies on G2 does nothing (§3.3).

## 3. Per-pair β experiment

### 3.1 Design

- **Network** (`policy/pp_beta.py`, `PerPairBetaNet`):
  - The trunk is identical to P12ParamNet (114-D input with the published feature stats). It outputs a shared γ and a 64-D context.
  - The β head is a small MLP with weights shared across pairs, applied to each of the 43 pair rows. Its input is the context, 8 per-pair features (h, L_f h, v_rel², ψ, L_f ψ, S_sup, L_g ψ·u_nom, ‖L_g ψ‖, normalised with stats over active rows of the training rollouts), an activity bit, a pair-type one-hot (obstacle / self-collision) and an 8-D learned pair embedding.
  - Output: β_j = floor + (250 − floor)·sigmoid(·). 148k parameters, against 138k for the old network.
- **QP:** unchanged except that row j uses β_j. Inactive rows are zeroed as before.
- **Training** (`train/train_rollout.py --policy per_pair`): same differentiable bound-slack QP rollout, scenarios, losses and hyperparameters as README §5.5, trained from scratch. The only change is the p-regularisation, 0.5·(γ² + mean over active pairs of β_j²), which keeps the old scale.
- **Fallback** (`--modes nn_pp_beta_fallback`): same pre-QP check, with G2_j using β_NN,j. γ follows the existing rule. If the check fails, G2_j is re-checked with γ_final, and failing ("flagged") pairs get β_req,j + 0.5. The other pairs are set by `--pp-unflagged`:
  - `keep`: β_NN,j. In practice a γ-only fallback, since G2 never flags (§2).
  - `cap_old`: min(β_NN,j, B), with B the existing scalar fallback β. This is the per-pair form of the existing rule.
  - `lower_pair`: min(β_NN,j, β_req,j + 0.5). This is mostly the floor 10⁻³.
- `--pp-fallback-reduce scalar` reproduces the existing scalar fallback exactly (used for the checks below).

### 3.2 Correctness checks

- **Exact reproduction:** with the old checkpoint broadcast to β_j = β_NN for all pairs, `nn_pp_beta` and `nn_pp_beta_fallback --pp-fallback-reduce scalar` reproduce `nn` and `nn_g12_fallback` **bit for bit** on 20 benchmark scenarios (28k steps, 73 columns, all summaries). The only exception is the logged mean β, which differs by ≤ 7·10⁻¹⁶ (`np.mean` rounding).
- `cap_old` with the broadcast old checkpoint also reproduces `nn_g12_fallback` bit for bit, as expected because every fallback is a G1 trigger.
- The existing modes from the edited comparison script match the reference `out/` and `out_nnfb/` summaries exactly on those 20 scenarios.
- **Per-pair flagging:** checked against a brute-force reference on 240 real states, and on a synthetic case where G2 must fail for exactly one pair.

### 3.3 Training and checkpoint selection

- 4 seeds (0–3), all with identical settings: 50 epochs × 80 segments, horizon 50, validation every 2 epochs on 20 segments. About 4.5 h each with fast kinematics (~5 min per epoch incl. validation), run in parallel on one core each. For comparison, the published run took 20.8 h (training epochs only, from its log).
- Why several seeds: the seed sets the initial weights and which training windows are sampled. One run cannot separate an effect of the method from a lucky or unlucky seed; the spread over 4 seeds can. The old network was trained with a single seed only.
- **Rule 1 (README §6):** among the "safe" validation epochs (no failed segment, min_h ≥ −10⁻⁶), take the best `normalized_mean_goal_distance`.
- **Rule 2** (fixed before evaluation): among the safe epochs ≥ 20, take the highest validation min_h.
- **Why rule 1 is not ideal:**
  - Once an epoch passes the weak gate, only goal progress counts. That rewards aggressive gains.
  - 20 short validation segments let immature networks pass by chance.
  - The validation goal metric is noisy (0.76–1.4 between epochs).
  - It worked for the old network (epoch 22, well trained), but for the per-pair networks it picks epochs 2–14.
- For the old network only epoch 22 (its rule-1 choice) was archived. Its rule-2 choice would be epoch 24 (validation min_h 2.5·10⁻³), which does not exist anymore.

### 3.4 Results (460 PD-failed scenarios)

Columns: safe success / violations (min_h < −10⁻⁴) / QP-fail stops, in % of scenarios with counts. The fallback is `cap_old`. The reference is the existing network: Learned (NN) 71.1 (327) / 23.3 (107) / 7.8 (36); **NN + Fallback 80.4 (370) / 11.5 (53) / 9.3 (43)**, fallback rate 1.30 %, γ̄/β̄ 12.6/5.1.

| Seed, rule (epoch) | γ̄/β̄ | Per-pair NN | Per-pair NN + Fallback | FB rate |
|---|---|---|---|---|
| 0, 1 (8) | 15.5/21.7 | 64.1 (295) / 33.3 (153) / 6.1 (28) | 77.8 (358) / 14.6 (67) / 11.3 (52) | 5.7 % |
| 1, 1 (2) | 14.6/14.3 | 65.0 (299) / 31.7 (146) / 7.4 (34) | 77.0 (354) / 15.0 (69) / 12.0 (55) | 4.5 % |
| 2, 1 (14) | 25.5/13.3 | 68.3 (314) / 28.3 (130) / 8.3 (38) | 77.4 (356) / 15.2 (70) / 13.3 (61) | 1.3 % |
| 3, 1 (6) | 6.4/2.4 | 64.1 (295) / 31.7 (146) / 7.6 (35) | 77.2 (355) / 15.0 (69) / 10.2 (47) | 3.3 % |
| **Rule 1, mean ± sd** | | **65.4 ± 2.0 / 31.2 ± 2.1 / 7.3 ± 0.9** | **77.3 ± 0.4 / 14.9 ± 0.3 / 11.7 ± 1.3** | 3.7 % |
| 0, 2 (38) | 6.8/10.5 | 58.5 (269) / 23.0 (106) / 17.4 (80) | 71.1 (327) / 14.1 (65) / 15.9 (73) | 35.9 % |
| 1, 2 (26) | 13.6/0.002 | 31.5 (145) / 0.2 (1) / 31.7 (146) | 31.5 (145) / 0.2 (1) / 31.7 (146) | 0.0 % |
| 2, 2 (46) | 9.0/0.35 | 65.4 (301) / 3.5 (16) / 23.0 (106) | 66.5 (306) / 2.2 (10) / 22.8 (105) | 0.1 % |
| 3, 2 (30) | 4.4/0.83 | 68.9 (317) / 14.3 (66) / 15.4 (71) | 74.3 (342) / 5.0 (23) / 16.1 (74) | 1.4 % |
| **Rule 2, mean ± sd** | | 56.1 ± 16.9 / 10.3 ± 10.4 / 21.9 ± 7.3 | 60.9 ± 19.8 / 5.4 ± 6.2 / 21.6 ± 7.5 | 9.4 % |

Fallback variants for seed 0, rule 1:

| Variant | Safe success / violations / QP-fail stops | FB rate |
|---|---|---|
| `keep` | 69.1 (318) / 27.4 (126) / 6.7 (31) | 8.2 % |
| `cap_old` | 77.8 (358) / 14.6 (67) / 11.3 (52) | 5.7 % |
| `lower_pair` | 77.2 (355) / 12.8 (59) / 13.7 (63) | 6.0 % |

`keep` flagged no pair at all on any step, which confirms §2. Cap hits are rare and are only γ caps. Share of active pairs whose β the fallback changed: 1.59 % for the old NN + Fallback (scalar rule, all active pairs on fallback steps), 7.3 % for `cap_old` and `lower_pair` (seed 0, rule 1), 0 % for `keep`.

### 3.5 Findings

- **Rule 1: consistently worse than the shared β.** The spread over seeds is small, so this is not seed luck. The selected networks have higher gains (β̄ up to 22 vs. 5) and leave about 3× more work to the fallback.
- **Rule 2: networks with learned β ≈ 0.** They are almost never unsafe but often stop on QP failure (15–32 %) or don't reach the goal. Part of their low violation rate is a measurement artifact: a rollout that stops on an infeasible QP cannot record a later violation. The best trade-off (seed 3 + fallback: 74.3 / 5.0 / 16.1) has fewer violations than the old controller but less success and more QP-fail stops. No checkpoint dominates it.
- **The per-pair freedom is only partly used.** Within a step, the median coefficient of variation of β_j over the active pairs is 0.06–0.13 for 5 of 8 checkpoints, i.e. essentially one β. It is 0.48–0.59 for seed 3 rule 1 and seeds 2–3 rule 2. Seed 0 rule 2 splits by pair type (self-collision ≈ 2.5, obstacle ≈ 16).
- **Why the existing NN + Fallback still wins:** it applies β ≈ 0 only on the 1.3 % of steps where G1 fails, a state-dependent brake. The per-pair networks learn either a high β everywhere or β ≈ 0 everywhere.
- **Conclusion:** no evidence that per-pair β helps on this benchmark. The checkpoint rule changes the outcome more than per-pair vs. shared β does.

### 3.6 Open points

- A fair rule-2 comparison needs the old network's epoch-24 checkpoint, i.e. retraining the old network. Seeds 1–3 of the old network would also show how lucky its single seed was.
- A more promising direction than per-pair β is learning *when* to brake, i.e. a state-dependent β that drops towards 0 only in critical states, which is what the fallback does by hand. A per-pair trigger that actually fires (e.g. ψ_j close to 0, or ψ_j decreasing fast) would be the prerequisite for any per-pair fallback.
- A checkpoint rule that weighs safety margin, not just goal progress, would make selection less arbitrary for any network.

## 4. Training speed

**Profile of one 50-step training segment** (old network, published checkpoint, before the fix: 8–9 s per segment on the i7-12700H):

| Component | Share |
|---|---|
| Link Jacobians (16 endpoints, each with its own `autograd.functional.jacobian`) | ~49 % |
| Backward pass (mostly through those Jacobians) | ~29 % |
| QP forward (cvxpylayers) | ~7 % |
| Pair geometry | ~6 % |
| Network forward | < 1 % |
| Nominal controller (Pinocchio) | ~0 % |

- **GPU:** not useful. The work is about 10⁵ tiny operations per segment (7 joints, 43 pairs) with batch size 1, the 50 steps are sequential, the QP solver and Pinocchio are CPU-only, and everything is float64 (consumer GPUs run float64 at 1/32–1/64 speed). Kernel-launch overhead would exceed the math.
- **Threads:** 1 thread is as fast as 10 (8.7 vs. 9.1 s per segment). A training run uses one core. Use the other cores for parallel runs (seeds, variants) or evaluation shards.
- **Fast kinematics** (`differentiable/kinematics_fast.py`): one FK pass and the analytic geometric Jacobian J_f[:, i] = z_i × (p_f − o_i).
  - Positions agree to 8·10⁻¹⁶ m, Jacobians to 7·10⁻¹² and their q-derivatives to 3·10⁻¹¹. The 7·10⁻¹² comes mostly from the original dividing the joint axis by `norm + 1e-12`.
  - 37× faster per kinematics call and **4.7× faster per training segment** (8.1 → 1.7 s; about 2.8 s with the per-pair head).
  - Weight gradients differ by 10⁻⁴–10⁻². That is the same size as the effect of perturbing the start state by 10⁻¹² in the original code: the gradient through the QP is very sensitive near active constraints. A retraining with fast kinematics is therefore a new run, not a bitwise reproduction.
  - Opt-in via `--fast-kinematics` of `train/train_rollout.py` (default off, so the shared-gain training stays bit-identical to the published run). The per-pair runs above used it.
- **Profile with fast kinematics** (same 3 segments, 1 thread, 1.9 s per segment): backward pass ~30 %, QP forward (cvxpylayers) ~29 %, pair geometry ~25 %, link kinematics ~6 %, the rest < 2 %. A further speed-up would have to come from the QP layer or the pair geometry; both run per step on tiny tensors, so again not a GPU case.
- **Another machine:** single-core speed is what counts. A 9th-gen desktop i9 is probably no faster per core than this 12th-gen laptop, though better cooling avoids throttling. Measure with the one-segment profile before moving work.

## 5. Data and reproduction

### 5.1 Data

- **Training rollouts** (`data/rollouts/fullpool_candidates_seed20260520_n1500/`, gitignored, ~14 GB): 966 rollouts = 741 training + 225 validation scenarios of the 1,500-candidate pool (the training JSONL has 1,664 lines because of weighting).
  - The training uses them only for start states and feature statistics. They don't depend on the network, so they are **reusable for any training** with the same split.
  - Regenerate them only if the scenarios, the split or the simulation/controller code changes. With `--skip-existing`, only missing ones are generated (~110 s each, ~1.5 h with 16 workers).
- **Evaluation benchmark:** the 460 PD-failed scenarios of the separate 1,000-scenario evaluation set. It needs no stored rollouts. One evaluation of 2 controllers on all 460 scenarios takes ~13 min in 20 parallel shards (slower for cautious checkpoints, whose rollouts run longer).
- **Run outputs** (gitignored): `data/pp_beta_v10/` (seed 0, all evaluations, `selection.json`, `seed_results.csv`, driver scripts) and `data/pp_beta_v10_seed{1,2,3}/`.

### 5.2 Commands (from `src/`)

```bash
# local copies of the split JSONLs with rollout paths under data/ (see data/fr3_fullpool_*_local.jsonl)
python -m fr3_hocbf.dataset.generate_scenario_data --scenario-dir <dir of the 966 YAMLs> \
  --output-dir ../data/rollouts/fullpool_candidates_seed20260520_n1500 --workers 16 --skip-existing
python -m fr3_hocbf.dataset.compute_feature_stats --kind per_pair \
  --rollout-labeled ../data/fr3_fullpool_train_weighted_local.jsonl --out <pp_stats.npz>
# training: README section 5.5 arguments plus
python -m fr3_hocbf.train.train_rollout <README 5.5 args> --policy per_pair --fast-kinematics \
  --stats-in ../artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt --pp-stats-in <pp_stats.npz> --seed <s>
# evaluation
python -m fr3_hocbf.evaluation.compare_rollout_policies <scenario list> \
  --modes nn_pp_beta,nn_pp_beta_fallback --pp-unflagged cap_old --pp-model <checkpoint> --out-prefix <prefix>
python -m fr3_hocbf.evaluation.report_pp_beta --prefix '<prefix glob>'
# exact-reproduction check with the old checkpoint
python -m fr3_hocbf.evaluation.compare_rollout_policies <list> --modes nn,nn_g12_fallback,nn_pp_beta,nn_pp_beta_fallback \
  --model <old ckpt> --pp-model <old ckpt> --pp-fallback-reduce scalar --out-prefix <prefix>
python -m fr3_hocbf.evaluation.report_pp_beta --prefix '<prefix glob>' --check-equal nn=nn_pp_beta nn_g12_fallback=nn_pp_beta_fallback
```

The full drivers used here are `data/pp_beta_v10/run_pp_pipeline.sh`, `train_seed.sh` and `run_seed_evals.sh`.

### 5.3 Restructure check

After moving `src/` into the `fr3_hocbf` package (layout in the README):
- Short training runs of both training scripts give bit-identical epoch logs and weights.
- The table script, the 3 plots and the per-pair report are byte-identical.
- All 13 CLIs have the same options.
- The four existing controllers reproduce the reference `out/` and `out_nnfb/` summaries exactly on 20 benchmark scenarios.

The later refactor commits (shared helpers, `policy/` and `differentiable/qp_rows.py` split, training scripts merged into `train_rollout.py --policy`, gain policies moved to `evaluation/gain_policies.py`, combined feature-stats CLI, `build_fullpool_split` CLI) were checked the same way after every commit. A run with non-default `--p1-max/--p2-max` caps and the PD-only runner were checked too, and `build_fullpool_split` gives identical output files on the 966 local rollouts.
