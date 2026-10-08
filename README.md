# Learning Per-Timestep HOCBF Gains with an Analytical Runtime Feasibility Fallback

Code and artifacts for *"Learning Per-Timestep HOCBF Gains with an Analytical
Runtime Feasibility Fallback for Whole-Body Collision Avoidance on a 7-DOF
Manipulator"* (S. Marti, L. Gimeno, D. Paloschi — ETH Zürich).

---

## Contents

1. [Quick start](#1-quick-start)
2. [Requirements](#2-requirements)
3. [Scenario pools](#3-scenario-pools)
4. [Paper ↔ code map](#4-paper--code-map)
5. [Full pipeline](#5-full-pipeline)
6. [Published model](#6-published-model)

---

## 1. Quick start

Reproduces Tables I–III and Fig. 4 from the committed checkpoint and benchmark,
without retraining. Runtime is roughly 2 h on CPU (460 scenarios × 4 modes).

```bash
pip install -r requirements.txt

# 1. Unpack the 1000 evaluation scenarios.
unzip artifacts/scenarios/generated_scenarios_eval.zip -d config/
#    -> config/generated_scenarios/

# 2. Rewrite the benchmark list to local paths (see note below).
python - <<'PY'
from pathlib import Path
root = Path("config/generated_scenarios").resolve()
names = [Path(l.strip()).name
         for l in Path("artifacts/results/fr3_pd_only_all1000_failed_scenarios.txt")
                   .read_text().splitlines() if l.strip()]
Path("pd_failed_460.txt").write_text("\n".join(str(root / n) for n in names) + "\n")
print(len(names), "scenarios")
PY

# 3. Run the four controllers.
cd src
python fr3_compare_rollout_policies.py ../pd_failed_460.txt \
  --modes fixed,davide_online,nn,nn_g12_fallback \
  --p1-fixed 10.0 --p2-fixed 15.0 \
  --model ../artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt \
  --out-prefix pd_failed_final
```

This writes `pd_failed_final_summary.csv` and `pd_failed_final_steps.csv`.

**Table I** (safe/unsafe × success/non-success) is derived from the summary:

```bash
python - <<'PY'
import pandas as pd
TOL = 1e-4
d = pd.read_csv("pd_failed_final_summary.csv")
d["success"] = d["success"].astype(str).str.lower().isin(["true", "1", "1.0"])
d["unsafe"]  = pd.to_numeric(d["min_h"], errors="coerce") < -TOL
for m, label in [("fixed", "Fixed g/b"), ("davide_online", "Fixed + FB"),
                 ("nn", "Learned (NN)"), ("nn_g12_fallback", "NN + Fallback")]:
    s = d[d["mode"] == m]
    print(f"{label:14s} {100*( s.success & ~s.unsafe).mean():5.1f} "
          f"{100*( s.success &  s.unsafe).mean():5.1f} "
          f"{100*(~s.success & ~s.unsafe).mean():5.1f} "
          f"{100*(~s.success &  s.unsafe).mean():5.1f}   "
          f"viol {100*s.unsafe.mean():5.1f}   "
          f"FB {100*s.fallback_rate.mean():5.1f}   "
          f"g/b {s.p1_mean.mean():.1f}/{s.p2_mean.mean():.1f}")
PY
```

**Tables II–III and Fig. 4** come from the two reporting scripts:

```bash
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

### Controller names

| Paper | `--modes` | Gain source |
|---|---|---|
| Fixed γ/β | `fixed` | constant γ/β from `--p1-fixed` / `--p2-fixed` |
| Fixed + Fallback | `davide_online` | scenario-YAML γ/β as floor, raised by the fallback |
| Learned (NN) | `nn` | network output, no repair |
| NN + Fallback | `nn_g12_fallback` | network output, repaired by the fallback |

### Notes

- **Benchmark list paths.** `fr3_pd_only_all1000_failed_scenarios.txt` stores
  absolute paths from the original machine. Step 2 above rewrites them; there is
  no `--scenario-root` flag.
- **No `--stats` argument is needed** — the checkpoint embeds `feat_mean` / `feat_std`.
- **Run scripts from inside `src/`** — the pipeline uses flat local imports.
- **Pair scope.** The published runs used the code defaults
  (`--online-pair-scope obstacle`, `--g12-fallback-pair-scope obstacle`).

### Verified against the paper

Recomputed from the committed archives under `artifacts/results/`:

| Row | Safe succ. | Unsafe succ. | Safe non-succ. | Unsafe non-succ. | γ̄/β̄ |
|---|---|---|---|---|---|
| Fixed γ/β | 62.2 | 27.2 | 2.6 | 8.0 | 10.0/15.0 |
| Fixed + Fallback | 77.8 | 4.6 | 8.0 | 9.6 | 32.1/42.6 |
| Learned (NN) | 70.9 | 16.1 | 6.1 | 7.0 | 12.3/5.2 |
| NN + Fallback | 86.5 | 0.0 | 12.8 | 0.7 | 12.7/5.6 |

The `fixed` and `davide_online` rows come from
`fr3_final_pd_failed_v10ep22_davide_fixed_summary.csv`; the `nn` and
`nn_g12_fallback` rows from
`fr3_final_pd_failed_v10ep22_vs_davide_terminal_qpfail_summary.json`. Both files
contain a `davide_online` mode from differently configured runs — the
`davide_fixed` file is canonical for that row.

Table II's fallback-activation column reports the mean of each scenario's own
elevation fraction (18.4% for Fixed + Fallback). The global step fraction is a
different statistic (48.8% for Fixed + Fallback, 1.4% for NN + Fallback).

---

## 2. Requirements

### Docker (recommended)

The `Dockerfile` pins Python 3.10.12 with CPU-only torch and unpacks the
evaluation scenarios at build time. Building runs no simulation.

```bash
docker build -t fr3-hocbf .
# smoke test: first 20 benchmark scenarios
docker run --rm -e LIMIT=20 -v "$PWD/out_smoke:/out" fr3-hocbf /repo/docker/reproduce_tables.sh
# full run: Tables I-III + Fig. 4 into ./out (shards across all cores)
docker run --rm -v "$PWD/out:/out" fr3-hocbf /repo/docker/reproduce_tables.sh
```

`docker/reproduce_tables.sh` runs §1 steps 3+ in parallel shards (`JOBS`,
default `nproc`), merges them, and writes `table1.txt`, the Table II–III files
and the Fig. 4 plots. `LIMIT` restricts it to the first N benchmark scenarios,
and `G12_SCOPE` sets `--g12-fallback-pair-scope` (default `obstacle`). The
image unpacks the evaluation scenarios to `/data/generated_scenarios/` and the
benchmark list to `/data/pd_failed_460.txt`.

**Dev container (for development).** `.devcontainer/devcontainer.json` reuses
the same `Dockerfile`. In VS Code run *Dev Containers: Reopen in Container*;
the repo is mounted live at `/repo`, so code edits need no rebuild. From the
integrated terminal, `bash /repo/docker/reproduce_tables.sh` writes to
`/repo/out/`. Rebuild the container only after changing `Dockerfile` or
`requirements.txt`.

### Local install

Python 3.10.x, target **3.10.12** — the Python version that produced the
published tables. The lower bound comes from `list[str]` / `X | None` syntax
(needs ≥ 3.10). numpy is dictated by Pinocchio: `pin == 3.9.0` requires
`cmeel-boost ~= 1.89.0`, whose wheels require `numpy >= 2.2, < 2.3` on Python
3.10 and `numpy >= 2.3, < 2.4` on 3.11+. The original machine likely had
numpy 1.26.4 with a non-pip Pinocchio; that combination is not installable via
pip, so tiny numerical differences near `min_h ≈ 0` are possible. A clean
interpreter is easiest via conda:

```bash
conda create -n fr3 python=3.10 && conda activate fr3
```

CPU is sufficient; a GPU speeds up training only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is pinned and covers every import in `src/`: `numpy`,
`pandas`, `matplotlib`, `torch`, `cvxpy`, `cvxpylayers`, `pin` (Pinocchio,
imported as `pinocchio`), `pyyaml`, plus the pinned solver stack (`osqp`,
`clarabel`, `scs`, `scipy`).

> If the Pinocchio wheel fails: `conda install -c conda-forge pinocchio`

### `.gitignore` note

The repository ignores `src/*.pt`, `src/*.jsonl`, `src/*.txt`, `src/*.npz`,
`src/*.csv`, `data/`, and `config/generated_scenarios*/`. Keeping everything
under top-level `artifacts/` avoids all of those rules. **Do not** place
artifacts under `src/` — they will be silently ignored.

---

## 3. Scenario pools

Training and evaluation use two independently generated scenario sets, so no
leakage is possible (paper Sec. III-D).

| | Training pool | Evaluation pool |
|---|---|---|
| Count | 1500 candidates | 1000 |
| Archive | `artifacts/scenarios/generated_scenarios_fullpool_candidates.zip` | `artifacts/scenarios/generated_scenarios_eval.zip` |
| Seed | 20260520 | not recorded — YAMLs are shipped instead |
| Role | source of the train/val split | source of the 460 PD-failed benchmark |

Not all 1500 scenarios train the network. The pool is catalogued, split 85/15
*before* selection, and the training set is then built from the train side only
as 300 good + 200 hard-good + all failures = **741 unique**, oversampled to
**1664 weighted** samples (good×1, hard_good×2, bad_*×4). Validation is 225
unique and never duplicated; train/val overlap is 0.

---

## 4. Paper ↔ code map

| Paper element | File |
|---|---|
| Double integrator, capsule barrier (Eq. 2–4) | `fr3_geometry.py`, `fr3_torch_geometry.py` |
| ψ, second-order condition, hard QP (Eq. 5–10) | `fr3_qp_solver.py` |
| 43 collision rows | `fr3_qp_data.py` |
| Gain policy π_θ, 114-D features (Eq. 11–14) | `fr3_train_rollout.py` |
| Feature normalization (Eq. 13) | `fr3_compute_feature_stats_ddqnorm.py` |
| Differentiable QP with bound slack (Eq. 18–19) | `fr3_cvx_layer_bound_slack.py` |
| Losses (Eq. 15–20) | `fr3_train_rollout.py` |
| Split before oversampling (Sec. II-C) | `fr3_build_fullpool_split.py` |
| Runtime fallback (Eq. 21–24) | `fr3_compare_rollout_policies.py` |
| Closed-loop rollout, 50 Hz (Sec. III-C) | `fr3_rollout.py` |
| Nominal controller (Sec. III-B) | `fr3_nominal_controller.py` |
| Scenario generation (Sec. III-D) | `fr3_generate_candidate_scenarios.py` |
| PD-failed subset, 460 (Sec. III-F) | `fr3_run_pd_only_scenarios.py` |
| Tables I–III | `fr3_make_final_metric_tables_generic_fixed.py` |
| Fig. 4 | `fr3_plot_compare_results.py` |
| Appendix A kinematics / Jacobians | `fr3_torch_kinematics.py` |

---

## 5. Full pipeline

Only needed to retrain from scratch. To reproduce the published tables, use §1.

### Hardcoded paths

| File | Line | What to change |
|---|---|---|
| `fr3_build_fullpool_split.py` | 14 | `SRC = Path("/home/simon/davide_fr3_ws/src/cbf_safety_filter")` → your repo root |
| `artifacts/results/fr3_pd_only_all1000_failed_scenarios.txt` | all | absolute scenario paths — rewrite as in §1 step 2 |

Docstring examples in `fr3_compare_rollout_policies.py` and
`fr3_train_rollout.py` also show `~/davide_fr3_ws/...` paths; these are comments
only and do not affect execution. The URDF is resolved relative to the repository
root, so it needs no configuration.

### 5.1 Generate the training pool

```bash
python fr3_generate_candidate_scenarios.py \
  --num-scenarios 1500 --seed 20260520 \
  --output-dir config/generated_scenarios_fullpool_candidates
```

### 5.2 Generate rollouts

```bash
python fr3_generate_scenario_data.py \
  --scenario-dir config/generated_scenarios_fullpool_candidates \
  --output-dir data/rollouts/fullpool_candidates_seed20260520_n1500 \
  --workers 8 --skip-existing
```

The rollout `.npz` files (~22 GB) are deliberately not committed: the simulator
is deterministic, so they regenerate bit-identically from the scenario YAMLs, and
the split does not need them (the catalog already carries the labels).
`fr3_feature_stats_*.npz` is likewise not archived — it is embedded in the
checkpoint.

### 5.3 Build the split

```bash
python fr3_build_fullpool_split.py     # fix the hardcoded path first
```

The 741-scenario selection is reproducible **without re-running any rollouts**.
`artifacts/splits/fr3_fullpool_catalog.csv` records the outcome label and the
`min_h_rollout` / `min_psi_rollout` values per scenario, and the split is pure
pandas seeded by `SEED = 20260520`. This matters because the labels themselves
depend on solver numerics near `min_h ≈ 0`; shipping the catalog makes the split
independent of anyone's OSQP/SciPy versions.

```python
import pandas as pd
SEED, TRAIN_FRAC = 20260520, 0.85
df = pd.read_csv("artifacts/splits/fr3_fullpool_catalog.csv")
train = []
for _, g in df.groupby("status"):
    g = g.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train.append(g.iloc[:round(TRAIN_FRAC * len(g))])
tc = pd.concat(train, ignore_index=True)
good = tc[tc.status == "good"].sort_values(
    ["min_h_rollout", "min_psi_rollout", "steps"],
    ascending=[True, True, False], kind="stable")
hard_good = good.iloc[:200]
good_300  = good.iloc[200:].sample(n=300, random_state=SEED)
train_unique = pd.concat([good_300, hard_good,
    tc[tc.status.isin(["bad_safety", "bad_qp", "bad_goal"])]])   # -> 741 rows
```

### 5.4 Feature statistics

```bash
python fr3_compute_feature_stats_ddqnorm.py \
  --rollout-labeled <split_dir>/fr3_fullpool_train_weighted.jsonl \
  --out fr3_feature_stats_v10.npz
```

### 5.5 Train

Configuration of the published model, recovered from the checkpoint's embedded
`cfg`:

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
  --model-out fr3_v10_fullpool_g300_latest.pt \
  --log-jsonl fr3_v10_fullpool_g300_log.jsonl \
  --safe-candidate-dir safe_candidates_v10_fullpool_g300
```

Several values differ from the script defaults: `--w-g1-safety 20.0` (default
10.0), `--safety-topk-frac 1.0` (default 0.05), `--slack-huber-delta 1.0`
(default 0.0, which disables Huber), `--no-move-obstacles` (default is on).

### 5.6 Rebuild the evaluation benchmark

The evaluation scenarios are committed directly, so they are not regenerated.

```bash
unzip artifacts/scenarios/generated_scenarios_eval.zip -d config/

python fr3_run_pd_only_scenarios.py \
  config/generated_scenarios --out-prefix fr3_pd_only_all1000
```

`fr3_pd_only_all1000_failed_scenarios.txt` is the 460-scenario benchmark. The
scenario argument is positional.

### 5.7 Evaluate

See §1.

---

## 6. Published model

`artifacts/checkpoints/epoch_0022_v10_fullpool_g300.pt` (run
`v10_fullpool_g300_track001_slack001_huberslack_preg5e7`, epoch 22).

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
12.27 / 5.12 match Table II's *Learned (NN)* row (12.3 / 5.2).

Epoch 22 was selected mechanically from `artifacts/training_logs/`: validate
every 2 epochs, keep checkpoints with `num_failed_segments == 0` and
`min_h ≥ -1e-6` (nine epochs survive), then rank by
`normalized_mean_goal_distance`. Epoch 22 ranks first and is the only epoch whose
validation satisfies both `min_h > 0` and `min_ψ > 0`.

---

## License

Apache 2.0.
