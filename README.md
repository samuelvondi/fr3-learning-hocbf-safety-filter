# Learning Per-Timestep HOCBF Gains with an Analytical Runtime Feasibility Fallback

Code and artifacts for *"Learning Per-Timestep HOCBF Gains with an Analytical
Runtime Feasibility Fallback for Whole-Body Collision Avoidance on a 7-DOF
Manipulator"* (S. Marti, D. Paloschi, L. Gimeno — ETH Zürich).

A neural network predicts the two HOCBF class-K gains (γ, β) of a QP-based
collision-avoidance safety filter for the Franka FR3 at every control step,
trained end-to-end through a differentiable HOCBF-QP layer. Because a learned
policy carries no runtime guarantee, the proposed gains are checked against the
first-order and bounded-input HOCBF feasibility conditions and minimally
repaired by an analytical fallback before the hard QP is solved.

All reported results come from an **offline closed-loop simulator** with
forward-Euler integration (paper Sec. III-C) — no dynamics engine, no robot
middleware, no ROS 2, no physical robot.

> **Status.** All repo-side items are complete: every Table I / II / Fig. 1
> number is verified from committed data, the training split is reproducible from
> the committed catalog, and all scenarios, results, splits, and the model are
> committed. The one remaining **`⚠️ TODO`** is a *paper* edit (stating the full
> network depth), not a repository task.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Requirements](#2-requirements)
3. [Repository layout](#3-repository-layout)
4. [The two scenario pools](#4-the-two-scenario-pools)
5. [Paper ↔ code map](#5-paper--code-map)
6. [Full pipeline](#6-full-pipeline)
7. [Provenance of the published model](#7-provenance-of-the-published-model)
8. [Model selection rule](#8-model-selection-rule)
9. [Archived artifacts](#9-archived-artifacts)
10. [Configuration reference](#10-configuration-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Outstanding TODOs](#12-outstanding-todos)

---

## 1. Quick start

Reproduce Tables I–II and Fig. 1 from the archived checkpoint and benchmark,
without retraining:

```bash
pip install -r requirements.txt
cd src

python fr3_compare_rollout_policies.py \
  ../artifacts/results/fr3_pd_only_all1000_failed_scenarios.txt \
  --modes fixed,davide_online,nn,nn_g12_fallback \
  --p1-fixed 10.0 --p2-fixed 15.0 \
  --online-pair-scope obstacle --g12-fallback-pair-scope obstacle \
  --model ../artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt \
  --out-prefix pd_failed_final

python fr3_make_final_metric_tables_generic_fixed.py \
  --run pd_failed_final_summary.csv pd_failed_final_steps.csv \
        "fixed=Fixed gamma/beta" "davide_online=Fixed + Fallback" \
        "nn=Learned (NN)" "nn_g12_fallback=NN + Fallback" \
  --order "Fixed gamma/beta" "Fixed + Fallback" "Learned (NN)" "NN + Fallback" \
  --safety-tol 1e-4 --out-prefix fr3_final_metric_tables

python fr3_plot_compare_results.py \
  --steps pd_failed_final_steps.csv --summary pd_failed_final_summary.csv \
  --out-dir fr3_compare_plots --h-tol 1e-4
```

**No `--stats` argument is needed** — the checkpoint embeds `feat_mean`/`feat_std`.

**Pair scope.** The published runs used `--online-pair-scope obstacle
--g12-fallback-pair-scope obstacle` (the code defaults), verified to reproduce
Table I.

**Published run prefixes** (for reference — the numbers behind Table I came from
two runs):
- `fr3_final_pd_failed_v10ep22_davide_fixed` → the `fixed` + `davide_online` rows
- `fr3_final_pd_failed_v10ep22_vs_davide_terminal_qpfail` → the `nn` +
  `nn_g12_fallback` rows

Both are archived under `artifacts/results/`.

The 1000 evaluation scenario YAMLs referenced by
`fr3_pd_only_all1000_failed_scenarios.txt` are committed at
`artifacts/scenarios/generated_scenarios_eval.zip` — unzip them into
`config/generated_scenarios/` before running (or point `--scenario-root` at the
unzipped folder).

The *Fixed γ/β* row is reproduced by mode `fixed` (constant gains held at
`--p1-fixed 10.0 --p2-fixed 15.0`); its run is archived at
`artifacts/results/fr3_final_pd_failed_v10ep22_davide_fixed_summary.csv` and its
numbers (89.3 / 62.2 / 35.2 / 6.7, γ̄/β̄ = 10.0/15.0) are verified against Table I.
The corresponding step log is archived slim-and-gzipped (see §9).

---

## 2. Requirements

| Component | Requirement |
|---|---|
| Python | 3.10+ (uses `list[str]` / `X \| None` syntax) |
| Hardware | CPU is sufficient. A GPU speeds up training only. |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Complete dependency set, verified by scanning every import in `src/`:
`numpy`, `pandas`, `matplotlib`, `torch`, `cvxpy`, `cvxpylayers`,
`pin` (Pinocchio, imported as `pinocchio`), `pyyaml`.

> If the Pinocchio wheel fails: `conda install -c conda-forge pinocchio`

**Run all scripts from inside `src/`** — the pipeline uses flat local imports.

---

## 3. Repository layout

```
.
├── include/urdf/fr3_robot.urdf     # FR3 model, loaded by Pinocchio everywhere
├── requirements.txt
├── src/                            # see the pipeline section for the full list
└── artifacts/                      # published run artifacts (see §9)
    ├── checkpoints/
    │   └── epoch_0022_v10_fullpool_g300.pt          ✅ committed
    ├── training_logs/
    │   ├── fr3_v10_fullpool_g300_..._log.jsonl       ✅ committed
    │   ├── safe_candidates_..._summary.txt           ✅ committed
    │   └── fr3_fullpool_split_summary.json           ✅ committed
    ├── splits/
    │   ├── fr3_fullpool_catalog.csv                  ✅ committed (labels + min_h/min_psi for all 1500)
    │   ├── fr3_fullpool_train_weighted.jsonl         ✅ committed (labeled manifest training reads)
    │   ├── fr3_fullpool_train_weighted.txt           ✅ committed
    │   ├── fr3_fullpool_train_unique.csv             ✅ committed (the 741)
    │   ├── fr3_fullpool_val_unique.jsonl             ✅ committed (the 225)
    │   └── fr3_fullpool_val_unique.csv               ✅ committed
    ├── scenarios/
    │   ├── generated_scenarios_fullpool_candidates.zip  ✅ committed (1500 YAMLs, 892 KB)
    │   └── generated_scenarios_eval.zip                 ✅ committed (1000 YAMLs, 554 KB)
    └── results/
        ├── fr3_pd_only_all1000_summary.csv           ✅ committed
        ├── fr3_pd_only_all1000_failed_scenarios.txt  ✅ committed (the 460)
        ├── fr3_final_pd_failed_..._davide_fixed_summary.csv        ✅ committed (Fixed + Fixed+FB rows)
        ├── fr3_final_pd_failed_..._davide_fixed_steps_slim.csv.gz  ✅ committed (9.9 MB; verifies FB rate)
        ├── fr3_final_pd_failed_..._qpfail_summary.json ✅ committed (NN + NN+FB rows)
        └── fr3_all1000_..._summary.csv               ✅ committed
```

`src/` contents:

```
src/
│ # engine (library modules)
├── fr3_geometry.py  fr3_torch_geometry.py  fr3_torch_kinematics.py
├── fr3_qp_solver.py  fr3_qp_data.py  fr3_nominal_controller.py
├── fr3_cvx_layer_bound_slack.py  fr3_rollout.py
│ # training-data pipeline
├── fr3_generate_candidate_scenarios.py  fr3_generate_scenario_data.py
├── fr3_build_fullpool_split.py          # the split used for the paper (§7)
│ # training
├── fr3_train_rollout.py  fr3_compute_feature_stats_ddqnorm.py
│ # evaluation
├── fr3_run_pd_only_scenarios.py  fr3_compare_rollout_policies.py
├── fr3_make_final_metric_tables_generic_fixed.py  fr3_plot_compare_results.py
│ # alternative split path (not used for the published model)
└── fr3_make_split.py  fr3_make_hard_good_list.py
```

---

## 4. The two scenario pools

Training and evaluation use **two independently generated** scenario sets. This
is a stronger separation than a train/test split of one pool: there is no shared
generation, so no leakage is possible (paper Sec. III-D, "held out from
training").

| | Training pool | Evaluation pool |
|---|---|---|
| Name | "full pool" | the "1000" |
| Count | 1500 candidates | 1000 |
| Directory | `config/generated_scenarios_fullpool_candidates/` | `config/generated_scenarios/` |
| Seed | 20260520 | not recorded — see note below |

The training pool's seed (20260520) is preserved in its folder name and
`generation_summary.json`. The **evaluation** pool was generated earlier (before
seeds were logged in folder names) and its seed was never persisted. This does
not affect reproducibility: the 1000 evaluation scenario YAMLs are committed
directly under `artifacts/scenarios/generated_scenarios_eval.zip`, so the exact
benchmark is reproduced from the files themselves, not by re-seeding the
generator.
| Role | source for the train/val split | source of the 460 PD-failed benchmark |

"Full pool" does **not** mean all 1500 scenarios train the network. The pool is
catalogued, split, then filtered: from `fr3_fullpool_split_summary.json`,
1500 candidates (1216 good / 146 bad_safety / 98 bad_qp / 40 bad_goal) are split
85/15 *before* selection, then the training set is built from the train side only
as 300 good + 200 hard_good + all failures = **741 unique**, oversampled to
**1664 weighted** samples (good×1, hard_good×2, bad_*×4). Validation is 225
unique, never duplicated; train/val overlap is 0.

Note the two pools have different difficulty by design: the 1500 training pool is
81% "good", whereas the 1000 evaluation pool is 46% PD-failed (460/1000). The
oversampling of failure cases is what compensates for the easy training
distribution.

---

## 5. Paper ↔ code map

| Paper element | Implementation |
|---|---|
| Double integrator, capsule barrier (Eq. 2–4) | `fr3_geometry.py`, `fr3_torch_geometry.py` |
| ψ, second-order condition, hard QP (Eq. 5–10) | `fr3_qp_solver.py` |
| 43 collision rows = 8 links × 5 obstacles + 3 self-pairs | `fr3_qp_data.py` (`NUM_PAIR_ROWS=43`) |
| Bounded-input support, feasibility test (Eq. 12–13) | `davide_online_p12_current` in `fr3_compare_rollout_policies.py` |
| Gain policy π_θ, 114-D features (Eq. 14–17) | `fr3_train_rollout.py` |
| Normalization from training split only (Eq. 16) | `fr3_compute_feature_stats_ddqnorm.py` |
| Differentiable QP with bound slack (Eq. 21) | `fr3_cvx_layer_bound_slack.py` |
| Losses L_h, L_ψ, L_ξ, L_g, L_p (Eq. 18–22) | `fr3_train_rollout.py` |
| Split before oversampling (Sec. II-D) | `fr3_build_fullpool_split.py` |
| Fallback stages G1/G2 (Eq. 23–25) | `fr3_compare_rollout_policies.py` |
| 50 Hz, Δt = 0.02 s | `--dt` default `1/50` |
| OSQP runtime / cvxpylayers training | `cp.OSQP`; `fr3_cvx_layer_bound_slack.py` |
| 1000 eval scenarios, 2–5 obstacles (Sec. III-D) | `fr3_generate_candidate_scenarios.py` |
| PD-failed subset, 460 (Sec. III-F) | `fr3_run_pd_only_scenarios.py` |
| Violation threshold ε_h = 1e-4 m² (Eq. 26) | `--safety-tol` default |
| Appendix A approximation (Eq. 30) | `Lf_psi` = `2‖v_rel‖² + γ·Lf_h`; Hessian term omitted |

**Controller names** (paper → `--modes`):

| Paper (Table I) | `--modes` | Gain source |
|---|---|---|
| Fixed γ/β | `fixed` | constant γ/β from `--p1-fixed`/`--p2-fixed` (10.0 / 15.0) |
| Fixed + Fallback (Baseline) | `davide_online` | scenario-YAML γ/β as floor, raised by the fallback |
| Learned (NN) | `nn` | network output, no repair |
| NN + Fallback | `nn_g12_fallback` | network output, repaired by the G12 fallback |

> The archived *Fixed γ/β* run used mode `fixed` with `--p1-fixed 10.0
> --p2-fixed 15.0`, which holds the gains perfectly constant (verified: every
> step has `p1 = 10.0`, `p2 = 15.0`). A related mode `original` instead reads
> constant γ/β from each scenario YAML; with the archived scenarios (γ=10/β=15
> for 439 of 460) it yields nearly the same row.

---

## 6. Full pipeline

Only needed to retrain from scratch. To reproduce the published tables, use §1.

### 6.1 Generate the training pool (1500)

```bash
python fr3_generate_candidate_scenarios.py \
  --num-scenarios 1500 --seed 20260520 \
  --output-dir config/generated_scenarios_fullpool_candidates
```

### 6.2 Generate rollouts

```bash
python fr3_generate_scenario_data.py \
  --scenario-dir config/generated_scenarios_fullpool_candidates \
  --output-dir data/rollouts/fullpool_candidates_seed20260520_n1500 \
  --workers 8 --skip-existing
```

The rollout simulator has no randomness, so this reproduces bit-identically from
the same YAMLs.

### 6.3 Build the split (the one used for the paper)

```bash
python fr3_build_fullpool_split.py
```

⚠️ This script has a hardcoded path — see §9 and fix line 14 first. It implements
paper Sec. II-D exactly and writes `fr3_fullpool_train_weighted.jsonl` and
`fr3_fullpool_val_unique.jsonl`.

**The 741-scenario selection is reproducible without re-running any rollouts.**
The committed `artifacts/splits/fr3_fullpool_catalog.csv` records, per scenario,
the outcome label (`good`/`bad_safety`/`bad_qp`/`bad_goal`) and the
`min_h_rollout`/`min_psi_rollout` values used to rank the hard-good cases. The
split itself is pure pandas seeded by `SEED = 20260520` (stratified
`sample(frac=1, random_state=SEED)` + a stable sort on
`[min_h_rollout, min_psi_rollout, steps]` for the 200 hard-good, then
`sample(n=300, random_state=SEED)` for the good leftovers). Running that logic on
the catalog reproduces the committed 741 unique scenarios **exactly**, including
the exact 200 hard-good — verified. This matters because the labels themselves
depend on solver numerics at the `min_h ≈ 0` boundary; shipping the catalog makes
the split reproducible independently of anyone's OSQP/SciPy versions. To rebuild
the split from the catalog alone (no rollouts needed):

```python
import pandas as pd
SEED, TRAIN_FRAC = 20260520, 0.85
df = pd.read_csv("artifacts/splits/fr3_fullpool_catalog.csv")
train = []
for _, g in df.groupby("status"):
    g = g.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train.append(g.iloc[:round(TRAIN_FRAC*len(g))])
tc = pd.concat(train, ignore_index=True)
good = tc[tc.status=="good"].sort_values(
    ["min_h_rollout","min_psi_rollout","steps"], ascending=[True,True,False], kind="stable")
hard_good = good.iloc[:200]
good_300  = good.iloc[200:].sample(n=300, random_state=SEED)
train_unique = pd.concat([good_300, hard_good,
    tc[tc.status.isin(["bad_safety","bad_qp","bad_goal"])]])   # -> 741 rows
```

### 6.4 Feature statistics

```bash
python fr3_compute_feature_stats_ddqnorm.py \
  --rollout-labeled <split_dir>/fr3_fullpool_train_weighted.jsonl \
  --out fr3_feature_stats_v10.npz
```

### 6.5 Train

Exact configuration of the published model, recovered from the checkpoint's
embedded `cfg`:

```bash
python fr3_train_rollout.py \
  config/generated_scenarios_fullpool_candidates \
  --rollout-labeled <split_dir>/fr3_fullpool_train_weighted.jsonl \
  --val-rollout-labeled <split_dir>/fr3_fullpool_val_unique.jsonl \
  --stats-in fr3_feature_stats_v10.npz \
  --sample-start-from-rollout \
  --epochs 50 --horizon 50 --segments-per-epoch 80 \
  --start-step-min 0 --start-step-max 700 \
  --val-every 2 --val-segments 20 \
  --lr 1e-3 --weight-decay 0.0 --grad-clip 5.0 --seed 0 \
  --hidden-dim 256 --rho 1e5 \
  --init-p1 10.0 --init-p2 15.0 \
  --p1-floor 1e-3 --p2-floor 1e-3 --p1-max 200.0 --p2-max 250.0 \
  --w-safety 100.0 --w-g1-safety 20.0 --w-goal 0.01 \
  --w-slack 0.01 --w-preg 5e-7 --w-viol-freq 0.0 \
  --safety-topk-frac 1.0 --slack-huber-delta 1.0 \
  --track-metric normalized_mean_goal_distance \
  --safe-min-h-tol 1e-6 --checkpoint-tol 1e-9 \
  --no-move-obstacles \
  --model-out fr3_v10_fullpool_g300_..._latest.pt \
  --log-jsonl fr3_v10_fullpool_g300_..._log.jsonl \
  --safe-candidate-dir safe_candidates_v10_fullpool_g300_...
```

Several values differ from the script defaults: `--w-g1-safety 20.0` (default
10.0), `--safety-topk-frac 1.0` (default 0.05), `--slack-huber-delta 1.0`
(default 0.0, which disables Huber), `--no-move-obstacles` (default is on).

### 6.6 The evaluation benchmark (1000)

The evaluation scenarios are committed directly, so you do **not** regenerate
them (the generation seed was not recorded — see §4). Unzip them and derive the
PD-failed subset:

```bash
unzip artifacts/scenarios/generated_scenarios_eval.zip -d config/   # -> config/generated_scenarios/

python fr3_run_pd_only_scenarios.py \
  config/generated_scenarios --out-prefix fr3_pd_only_all1000
```

`fr3_pd_only_all1000_failed_scenarios.txt` is the 460-scenario benchmark (also
committed under `artifacts/results/`). The scenario argument is **positional**.

If you ever do want to generate a *fresh* 1000-scenario pool (not the paper's),
`fr3_generate_candidate_scenarios.py --num-scenarios 1000 --seed <any>` works —
but it will not reproduce the published benchmark, which is why the original
YAMLs are shipped.

### 6.7 Evaluate and tabulate

See §1.

---

## 7. Provenance of the published model

**Checkpoint:** `artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt`
(run `v10_fullpool_g300_track001_slack001_huberslack_preg5e7`, epoch 22).

Read directly from the checkpoint:

| Field | Value |
|---|---|
| `input_dim` / `hidden_dim` | 114 / 256 |
| `feature_version` | `q_dq_ddqnom_ddqnomnorm_h_Lfh` |
| `init_p1` / `init_p2` | 10.0 / 15.0 |
| `p1_max` / `p2_max` | 200.0 / 250.0 |
| `dt` | 0.02 |
| `num_pair_rows` / `num_bound_rows` | 43 / 42 |
| `rho` | 1e5 |
| `feat_mean`, `feat_std` | embedded, 114-D |
| `epoch` | 22 |
| `metric_value` | 0.7825233 |

Validation at epoch 22: `min_h` = +4.85e-05, `min_ψ` = +7.83e-04,
`p1_mean` = 12.27, `p2_mean` = 5.12, `num_failed_segments` = 0. The mean gains
12.27 / 5.12 match Table I's *Learned (NN)* 12.3 / 5.2.

**Network architecture** (from the state dict): `114 → 256 → 256 → 128 → 64 → 2`
with LayerNorm on the first three blocks. The paper text says "hidden width 256";
consider stating the full layer widths so the network can be rebuilt from the
text alone. **⚠️ TODO (paper edit, not repo).**

---

## 8. Model selection rule

Epoch 22 was selected mechanically, and the choice is reproducible from
`artifacts/training_logs/`:

1. Validate every 2 epochs (25 validation points over 50 epochs).
2. **Safety gate:** keep checkpoints with `num_failed_segments == 0` **and**
   `min_h ≥ -1e-6`. This admits exactly nine epochs (8, 20, 22, 24, 28, 34, 38,
   42, 44), matching `safe_candidates_..._summary.txt`.
3. **Rank** survivors by `normalized_mean_goal_distance`; epoch 22 ranks first at
   0.7825.

Epochs 2 (0.7589) and 16 (0.7756) had better goal distance but were correctly
rejected by the safety gate (`min_h` < 0). Epoch 22 is additionally the only
epoch whose validation satisfies both `min_h > 0` and `min_ψ > 0`
simultaneously — the cleanest justification for the selected checkpoint.

---

## 9. Archived artifacts

Committed under `artifacts/` so the published results reproduce without
regenerating anything expensive.

**Committed now:**

- `checkpoints/epoch_0022_v10_fullpool_g300.pt` — the published model (~1.1 MB)
- `training_logs/*_log.jsonl` — 50-epoch train+val curve
- `training_logs/safe_candidates_*_summary.txt` — the 9 gated candidates
- `training_logs/fr3_fullpool_split_summary.json` — dataset composition (verified
  internally consistent: duplication counts and 741→1664 arithmetic all check out)
- `splits/fr3_fullpool_val_unique.jsonl` — 225 validation records
- `splits/fr3_fullpool_train_weighted.txt` — 1664 weighted training paths
- `results/fr3_pd_only_all1000_summary.csv` — PD-only run over the 1000
- `results/fr3_pd_only_all1000_failed_scenarios.txt` — **the 460-scenario benchmark**
- `results/fr3_final_pd_failed_v10ep22_davide_fixed_summary.csv` — the run behind
  the *Fixed γ/β* and *Fixed + Fallback* rows of Table I
- `results/fr3_final_pd_failed_v10ep22_davide_fixed_steps_slim.csv.gz` — a slim,
  gzipped per-step log (10 columns) from that run; the full 873 MB step log is
  not committable, but this 9.9 MB extract still recomputes every step-level
  number, including the 18.4% fallback rate
- `results/fr3_final_pd_failed_..._qpfail_summary.json` — an earlier comparison
  run covering `nn` / `nn_g12_fallback` (see the note below)
- `results/fr3_all1000_..._summary.csv` — all-1000 counterpart

**Verified against the paper** — every cell of Table I, all of Table II, and
Fig. 1 reproduce from committed data:

| Row | Succ | Safe | Viol | QPfail | FB | γ̄/β̄ | Source |
|---|---|---|---|---|---|---|---|
| Fixed γ/β | 89.3 | 62.2 | 35.2 | 6.7 | — | 10.0/15.0 | davide_fixed summary |
| Fixed + Fallback | 82.2 | 77.8 | 13.9 | 17.0 | 18.4 | 32.1/42.6 | davide_fixed summary + slim steps |
| Learned (NN) | 87.0 | 70.9 | 23.0 | 7.8 | — | 12.3/5.2 | qpfail summary |
| NN + Fallback | 86.5 | 86.5 | 0.7 | 11.3 | 1.4 | 12.7/5.6 | qpfail summary |

The 18.4% fallback rate is the **mean over the 460 scenarios of each scenario's
own elevation fraction** (fraction of that scenario's steps where the fallback
raised γ or β above the nominal floor) — not the global step fraction, which is
48.8%. State this definition if the Table I caption is ambiguous.

> **Two `davide_online` runs exist.** The `davide_fixed` files reproduce the
> paper's *Fixed + Fallback* row exactly (γ̄/β̄ = 32.1/42.6, QPfail 17.0%). An
> earlier `..._qpfail_summary.json` reports the same mode with different mean
> gains (11.3/16.2) and QPfail 16.7% — a differently configured run. Treat the
> **`davide_fixed`** files as canonical for Table I; the `qpfail` file remains
> useful only for the `nn` / `nn_g12_fallback` rows.

- `splits/fr3_fullpool_catalog.csv` — per-scenario labels + `min_h`/`min_psi`;
  regenerates the 741-scenario training set exactly (verified) without re-running
  rollouts
- `splits/fr3_fullpool_train_weighted.jsonl` — the labeled manifest training reads
- `splits/fr3_fullpool_train_unique.csv`, `splits/fr3_fullpool_val_unique.csv` —
  the 741 / 225 membership lists
- `scenarios/generated_scenarios_fullpool_candidates.zip` — 1500 training YAMLs (892 KB)
- `scenarios/generated_scenarios_eval.zip` — 1000 evaluation YAMLs (554 KB)

The rollout `.npz` files (~22 GB) are deliberately **not** committed: the
simulator is deterministic, so they regenerate bit-identically from the scenario
YAMLs, and the split does not need them (the catalog already carries the labels).

**Deliberately not archived** (large and fully regenerable): rollout `.npz`
datasets (the simulator is deterministic — regenerate from scenario YAMLs) and
`fr3_feature_stats_*.npz` (embedded in the checkpoint).

### `.gitignore` note

The repository ignores `src/*.pt`, `src/*.jsonl`, `src/*.txt`, `src/*.npz`,
`src/*.csv`, `data/`, and `config/generated_scenarios*/`. Keeping everything under
top-level `artifacts/` avoids all of those rules. **Do not** place artifacts under
`src/` — they will be silently ignored.

---

## 10. Configuration reference

✓ marks a value stated in the paper.

| Parameter | Value | Location |
|---|---|---|
| Control step Δt | 0.02 s (50 Hz) ✓ | `--dt` |
| Arm joints | 7 ✓ | `NUM_ARM_JOINTS` |
| Link capsules | 8 ✓ | `ACTIVE_LINKS_DEF` |
| Max obstacles | 5 ✓ | `MAX_OBSTACLES` |
| Total pair rows | 43 ✓ | `NUM_PAIR_ROWS` |
| Obstacles per scenario | 2–5 ✓ | `NUM_OBSTACLES_RANGE` |
| Feature dimension | 114 ✓ | checkpoint `input_dim` |
| Network | 114→256→256→128→64→2, LayerNorm | checkpoint state dict |
| γ max / β max | 200.0 / 250.0 | `--p1-max` / `--p2-max` |
| γ / β init | 10.0 / 15.0 | `--init-p1` / `--init-p2` |
| Violation threshold ε_h | 1e-4 m² ✓ | `--safety-tol` |
| QP solver | OSQP, eps 1e-5, max_iter 25000 ✓ | `solve_hard_qp_osqp` |
| Training split | 85/15 stratified, seed 20260520 | `fr3_fullpool_split_summary.json` |
| Oversampling weights | good×1, hard_good×2, bad_*×4 | `fr3_build_fullpool_split.py` |

On hard-QP failure the rollout terminates immediately with no recovery hold
(Sec. III-C).

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fr3_rollout` | Run from inside `src/` |
| `import pinocchio` fails after `pip install pin` | `conda install -c conda-forge pinocchio` |
| `KeyError: ... has no rollout_path` | `--rollout-labeled` expects a JSONL with a `rollout_path` key |
| Baseline gains look wrong | Mode `original`/`davide_online` read γ/β from the scenario YAML, not from CLI flags |
| Artifacts not committing | They must live in `artifacts/`, not `src/` — see §9 |
| Table generator errors on `--run` | Syntax is `SUMMARY.csv STEPS.csv mode=Name ...`; `--out-prefix` required |
| `fr3_build_fullpool_split.py` cannot find data | Fix the hardcoded path at line 14 |

---

## 12. Outstanding TODOs

Everything still open, in one place.

**Files to add to the repo**

None outstanding — all data, model, results, scenarios, and split files are
committed. Every Table I / II / Fig. 1 number is verified from committed data,
and the 741-scenario training split is reproducible from the committed catalog.
The only large data (~22 GB of rollout `.npz`) is intentionally excluded because
it regenerates deterministically from the committed scenario YAMLs.

**Values — now resolved**

| Item | Resolution |
|---|---|
| Evaluation-pool seed | Not recorded (pool predates seed-logging); moot — the 1000 YAMLs are shipped directly |
| `--online-pair-scope` | `obstacle` (code default), verified against Table I |
| Published comparison `--out-prefix` | `fr3_final_pd_failed_v10ep22_davide_fixed` (fixed/davide) and `..._vs_davide_terminal_qpfail` (nn/nn+fb) |

**Known data caveats to decide how to report**

- **Mixed baseline gains.** In the archived benchmark, 439/460 scenarios carry
  γ=10/β=15 but **21 (4.6%) carry γ=2/β=3** (the generator default). Decide
  whether to regenerate those 21 or note the inhomogeneity in the paper.
- **Table I fallback-rate definition.** The 18.4% is a mean of per-scenario
  elevation fractions (each scenario weighted equally), not the global
  step-fraction (48.8%). Consider stating this in the Table I caption.

**Paper edit (not repo)**

- State the full network architecture (114→256→256→128→64→2, LayerNorm), not just
  "hidden width 256".

---

## License

Apache 2.0.
