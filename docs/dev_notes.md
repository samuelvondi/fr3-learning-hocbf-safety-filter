# Development notes

Practical notes on the environment, the data folders and how changes were verified. Written 2026-11-16.

## 1. What survives a dev-container rebuild

| Location | Survives? | Why |
|---|---|---|
| `/repo` (code, `docs/`, and the gitignored `data/`, `config/`, `out*/`) | yes | bind mount of the host folder (`.devcontainer/devcontainer.json`) |
| `/root/.claude` (Claude Code memory and session transcripts) | yes | named volume `claude-code-config` |
| `/data` (evaluation scenarios, `pd_failed_460.txt`) | rebuilt | baked into the image by the Dockerfile |
| `/tmp`, pip packages installed by hand | no | container filesystem |

After a rebuild: `pip install -r requirements-dev.txt` for pyflakes. The image already sets `PYTHONPATH=/repo/src`.

## 2. Local data (gitignored)

| Path | Contents |
|---|---|
| `config/generated_scenarios/`, `config/generated_scenarios_fullpool_candidates/` | scenario YAMLs unpacked from `artifacts/scenarios/*.zip` |
| `data/rollouts/fullpool_candidates_seed20260520_n1500/` | 966 regenerated training/validation rollouts (~14 GB), reusable for any training with the same split |
| `data/fr3_fullpool_{train_weighted,val_unique}_local.jsonl` | the split JSONLs from `artifacts/splits/` with rollout paths pointing to the folder above |
| `data/pd_failed_460.txt` | the 460-scenario benchmark list with local paths |
| `data/smoke/` | 14-rollout smoke subset, smoke stats/trainings, 20-scenario check outputs |
| `data/pp_beta_v10*/` | per-pair β runs (seed 0–3), evaluations, `selection.json`, `seed_results.csv`, driver scripts |
| `data/dev_tools/` | verification tools and baselines (§4) |
| `data/od_hocbf/` | optimal-decay sweep: `run_all.sh` (260 jobs, ~40 min on 20 cores), `status.sh` (progress), `analyze.py` → `report.md`, `runs/`, `checks/` |
| `out/`, `out_nnfb/` | reference results of the 2026-10 alignment run (see `paper_alignment.md`) |

## 3. Environment quirks

- **Line endings:** the working copy has CRLF in files from the Windows checkout, git stores LF (`.gitattributes`: `* text=auto eol=lf`). Compare files with `tr -d '\r'` or through git, not with plain `cmp`.
- **No `watch`:** use `while true; do clear; <cmd>; sleep 30; done`.
- **Negative CLI values:** argparse reads `-inf` as a flag; write `--od-omega-min=-inf`.
- **Solver time:** cvxpy recompiling the problem every step dominates the hard-QP solve (~5 ms vs. ~1.4 ms for the parameterized `OptimalDecayQP`).
- **No `pgrep`/`ps`** in the container. To find processes, scan `/proc/*/cmdline`; use a `[x]yz` pattern so the `grep` does not match itself.
- **torch is the CPU-only build.** A GPU would not help anyway (`per_pair_beta_findings.md` §4).
- **Run scripts from `src/` as modules** (`python -m fr3_hocbf.<stage>.<script>`); running a file inside the package directly breaks its imports.
- `docker/reproduce_tables.sh` also runs in the dev container (it uses `/data`); `LIMIT=1 JOBS=1 OUT=<dir>` gives a one-scenario smoke run in ~20 s.

## 4. Verifying that a change keeps results identical

Training results are extremely sensitive: changing the start state by 10⁻¹² already changes the weight gradients by 10⁻⁴–10⁻². A refactor is therefore only "behaviour-preserving" if trainings stay **bit-identical**. Tools in `data/dev_tools/` (they take paths from there; outputs go to `data/dev_tools/verify_*`):

- `verify.sh quick` (~2 min): compile + import every module; short trainings of both policies (epoch logs and weights bit-identical to `baseline/`); table script, plots, per-pair report, both feature-stats modes, seeded scenario generation and the PD-only runner identical to the baselines.
- `verify.sh eval` (~12 min): the four published controllers on 20 benchmark scenarios identical to `out/` / `out_nnfb/`; per-pair checkpoint identical to its full evaluation; exact per-pair ↔ shared equivalence (`report_pp_beta --check-equal`); a run with non-default `--p1-max/--p2-max` identical to `refactor_baseline/` (catches module-level cap globals that are not set).
- `verify.sh record`: re-records `refactor_baseline/` from the current tree (only after a deliberate behaviour change).
- `smoke_pipeline.sh` (~3 min): every entry point once on minimal input, including `reproduce_tables.sh` with `LIMIT=1`.
- `profile_segment.py [--fast-kinematics]`: where the time of a training segment goes; run it on another machine to compare per-core speed.
- `defs.py`: cut/show top-level definitions of a Python file by name (used to move code between modules without hand edits).

`baseline/` was recorded before the 2026-11-16 restructure, `refactor_baseline/` after it. Both stay valid as long as behaviour is unchanged. `verify.sh` defaults `PP_TRAIN` to `train_rollout --policy per_pair --fast-kinematics`.

## 5. Known small issues (not fixed)

- `dataset/generate_scenario_data.py`: `global _WORKER_CONTEXT` in `_process_one` is redundant (pyflakes warning, harmless).
- `sim/qp_solver.py` and `evaluation/compare_rollout_policies.py` each build/solve a hard QP, with different interfaces (geometry-driven vs. prebuilt matrices); not merged.
- Docstring examples in a few scripts still show `~/davide_fr3_ws/...` paths (comments only).
