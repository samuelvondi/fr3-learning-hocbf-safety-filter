# Work diary

High-level log of what was done. Results and findings are in the topic documents:
`per_pair_beta_findings.md` (experiments and analyses), `dev_notes.md` (environment, tooling, verification),
`paper_alignment.md` (the 2026-10 alignment of the code with the paper).

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
