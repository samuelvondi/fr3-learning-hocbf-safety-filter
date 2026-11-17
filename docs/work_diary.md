# Work diary

High-level log of what was done. Results and findings are in the topic documents:
`per_pair_beta_findings.md` (experiments and analyses), `fallback_findings.md` (runtime fallback), `od_hocbf_findings.md` (optimal-decay controller), `dev_notes.md` (environment, tooling, verification),
`paper_alignment.md` (the 2026-10 alignment of the code with the paper).

## 2026-11-17 — Optimal-decay HOCBF controller

- Added the standalone controller mode `od_hocbf` (Zeng et al., ACC 2021, eq. 21, one decay variable ω_j per pair, fixed γ₀/β₀; flags `--od-*`). It is built once as a parameterized QP. The existing modes are unchanged.
- Checked it: frozen ω reproduces `fixed` 10/15 bit-identically; p_ω = 10⁸ gives the same outcomes as Fixed on 20 scenarios.
- Swept p_ω ∈ {10⁻², 1, 10², 10⁴, 10⁶, 10⁸} × ω ≥ 0 / unbounded on the 460 benchmark, plus a Fixed rerun.
- Analysed why the QP stops remain: they come from the joint bounds being infeasible on their own, outside the scope of Zeng's Theorem 1. Noted joint-limit CBFs with a braking-distance velocity bound as a possible next step.
- Findings in `od_hocbf_findings.md`.

## 2026-11-17 — Training behaviour, outcome table, fallback analysis

**Training behaviour**
- Added `train/plot_gain_evolution.py`: mean γ, β and loss per epoch (training and validation) for any set of training logs, with the selected checkpoints marked. Plotted the published run and the four per-pair seeds (`data/plots/gain_evolution.png`).
- Finding: the training does not converge. The loss is lowest around epoch 1 and never trends down. γ and β swing by one to two orders of magnitude up to the last epoch, β collapses to about 0 in phases, and seeds 0 and 3 diverge late. Checkpoint selection therefore effectively picks one of many similar, partly random networks.

**Result tables**
- Added table "0. Outcome" to `evaluation/make_final_metric_tables.py`: disjoint categories (safe success / violation / QP fail / timeout) that sum to 100 %, plus `outcome_*` columns in the values CSV. The existing tables are unchanged. Before, overlapping flags summed to 101.2 % for NN + Fallback.

**Fallback analysis**
- Analysed what happens after the fallback fires, and added the experimental mode `nn_g12_fallback_hold` (`--fallback-hold-steps K`). Checked it against the existing fallback with K = 0 and swept K = 3, 5, 10 on the 460 benchmark.
- Added the experimental mode `nn_g12_fallback_recover` (`--recover-beta B`): large β on rows with ψ_j < 0. Checked it with B = 0 and swept B = 10–200. At B = 200: 86.7 % safe success, 0 % violations (reference 80.4 / 11.5), at the cost of more QP-fail stops (10.7 vs. 6.5 %).
- Findings in `fallback_findings.md`.

## 2026-11-15/16 — Per-pair β experiment, code reorganisation

**Per-pair β controller**
- Explored the training and evaluation code and planned a per-pair β variant of the NN gains: shared γ, one β per collision pair from a small shared-weight network.
- Implemented the network, the per-pair feature statistics, the training path and the controller modes `nn_pp_beta` / `nn_pp_beta_fallback`, with three variants of the per-pair fallback.
- Checked that the new code path reproduces the existing NN and NN + fallback controllers exactly when every pair gets the shared β.
- Analysed how the existing runtime fallback actually triggers (G1/G2 checks).
- Trainings, all with the published hyperparameters (README §5.5):
  - smoke training (2 epochs on 14 rollouts) to test the pipeline end to end;
  - main training, seed 0, 50 epochs;
  - three more seeds (1–3), run in parallel, to separate method effects from seed luck;
  - short verification trainings (a few segments) after each code change, compared bit for bit with a baseline.
- Evaluated all four seeds on the 460-scenario benchmark under two checkpoint-selection rules.
- Regenerated the 966 training/validation rollouts locally (they were not archived).

**Training speed**
- Profiled a training segment and asked whether a GPU (or another machine) would help.
- Added an analytic fast-kinematics option for training (opt-in, `--fast-kinematics`).

**Reorganisation of the code**
- Moved all scripts from the flat `src/` folder into the package `src/fr3_hocbf/` (`sim`, `differentiable`, `dataset`, `policy`, `train`, `evaluation`); scripts now run as `python -m fr3_hocbf.<stage>.<script>`.
- Split the two large scripts (training, controller comparison) into focused modules, merged duplicated helpers, merged the two training scripts (`--policy shared|per_pair`) and the two feature-stats scripts (`--kind`), gave `build_fullpool_split` a CLI instead of a hardcoded path, removed the superseded legacy split scripts and unused imports.
- Updated README, Docker script and docs; added `requirements-dev.txt` (pyflakes).
- Rebuilt the uncommitted work as small commits, one logical change each.

**Testing**
- Recorded baselines before the reorganisation and verified every commit against them: bit-identical trainings, identical tables/plots/reports, the four published controllers identical to the reference results on 20 benchmark scenarios.
- Final smoke test of every pipeline entry point (data generation, training, all 7 controllers, tables, plots, `reproduce_tables.sh`) on minimal input.

**Documentation**
- `docs/per_pair_beta_findings.md`, `docs/dev_notes.md`, this diary.
