# Training stability — findings

2026-11-23. Why the rollout training (README §5.5) does not converge, and a first stabilised variant. Scripts and outputs are in `data/train_stab/`. The published checkpoint was only read; a backup with its SHA-256 is in `data/train_stab/published_backup/`.

## Summary

- **Cause: exploding gradients through the unrolled rollout.** The per-segment weight gradient grows by about 25× per step through the stiff CBF-QP closed loop. At the published horizon of 50 steps it reaches up to 10⁶⁸.
  - 14 % of segments exceed the gradient clip (5). `clip_grad_norm` keeps their direction, so about every seventh update of the old training was a random step at maximum step size.
  - This explains the jumps of γ/β by orders of magnitude between epochs.
- **Fix: truncated backpropagation.** The loss still covers 50 steps, but the gradient flows through the state only over K = 1–2 steps.
  - The gradients become finite and consistent: max norm 3.4, median cosine between segment gradients 0.98–0.99.
  - About 8 segments per update are enough (noise scale 3–11).
- **The stabilised training is stable** (pilot, 10 epochs). γ and β change smoothly, with no loss spikes and no failed segments.
- **But the training loss has almost no signal.** On a fixed validation set, the untrained, the published and the three pilot networks reach almost the same loss (3.1–4.3·10⁻⁴).
  - About 60 % of it is the goal term, which the gains hardly change.
  - In closed loop the gains clearly matter: NN 80.4 % vs. Fixed 10/15 62.2 % safe success. The 50-step training loss therefore does not measure what decides the benchmark.

## Gradient measurements

Published checkpoint, 160 training segments (README §5.5 loss weights, fast kinematics). Scripts `grad_noise.py`, `grad_horizon.py`, `grad_window.py`; outputs `*.txt`.

| Horizon H (full backprop) | median ‖g‖ | max ‖g‖ | segments > clip 5 |
|---|---|---|---|
| 1 | 6.7·10⁻⁴ | 12 | 2 |
| 5 | 7.0·10⁻⁴ | 2.4·10⁷ | 7 |
| 10 | 7.1·10⁻⁴ | 1.4·10¹⁸ | 11 |
| 20 | 7.1·10⁻⁴ | 8.6·10³⁶ | 16 |
| 50 | 7.2·10⁻⁴ | 1.4·10⁶⁸ | 23 |

At H = 50 with full backprop, one segment carries 100 % of ‖g‖², so averaging over a batch would not help on its own.

| H = 50, window K | max ‖g‖ | > clip 5 | median cos(gᵢ, mean of others) | noise scale B (with per-segment clip) |
|---|---|---|---|---|
| 1 | 3.4 | 0 | 0.99 | 73 (3) |
| 2 | 3.3 | 0 | 0.98 | 107 (11) |
| 5 | 3.8·10⁶ | 12 | 0.60 | 116 (11) |

## New training flags (defaults unchanged)

All in `train/train_rollout.py` / `train/parallel.py`. Default: the original training. `data/dev_tools/verify.sh quick` confirms bit-identical weights and epoch logs. The parallel path with 1 segment per update reproduces the original training bit-identically.

- `--batch-segments B`, `--workers W`: average the gradients of B segments per update, computed in W processes.
- `--state-grad-window K`: truncated backprop (detach q, dq every K steps).
- `--seg-grad-clip c`: clip each segment gradient before averaging; non-finite ones are dropped.
- `--lr-schedule cosine --lr-min`, `--ema d`: LR decay; validation and checkpoints use the EMA weights.
- `--val-fixed`: the same validation segments at every validation.

## Pilot (10 epochs × 128 segments, B = 8, LR 3·10⁻⁴ cosine, EMA 0.95, 60 fixed validation segments)

Script `run_pilot.sh`, outputs `pilot/`. Wall time 32–41 min, three runs in parallel on 20 cores.

| Variant | γ over epochs | β over epochs | Train loss |
|---|---|---|---|
| Published run, epochs 1–10 | 17 → 148 → 83 | 29 → 190 → 109 | 8·10⁻⁴ … 3.5·10⁻¹ |
| K = 1 | 7.4 – 8.4 | 11.5 – 13.8 | 3.4 – 7.2·10⁻⁴ |
| K = 1 + segment clip 0.01 | 7.0 → 4.7 | 10.7 → 4.7 | 3.5 – 8.7·10⁻⁴ |
| K = 2 | 9.1 → 13.9 | 14.0 → 22.4 | 4.3 – 8.0·10⁻⁴ |

**Fixed validation set, per network** (`val_compare.py`):

| Network | Loss | 0.01 · goal | Mean γ / β | Segments with min h < −10⁻⁶ |
|---|---|---|---|---|
| Untrained | 3.45·10⁻⁴ | 1.99·10⁻⁴ | 8.9 / 14.9 | 6 |
| Published | 3.14·10⁻⁴ | 2.01·10⁻⁴ | 12.3 / 4.8 | 4 |
| K = 1 | 3.13·10⁻⁴ | 1.99·10⁻⁴ | 7.4 / 12.3 | 6 |
| K = 1 + clip | 3.59·10⁻⁴ | 2.00·10⁻⁴ | 4.6 / 4.8 | 5 |
| K = 2 | 4.34·10⁻⁴ | 1.99·10⁻⁴ | 13.2 / 21.7 | 7 |

- **One validation segment** (scenario_1359, start step 122) **violates for every network** (min h ≈ −4.8·10⁻³). With it in the set, the safe-candidate rule (min h ≥ −10⁻⁶) never saves a checkpoint. In the old training, the random validation sets made this luck.

## Closed-loop evaluation of the pilot networks

The three pilot checkpoints (EMA weights after 10 epochs) were run on the 460 benchmark as pure NN (`nn`) and NN + Fallback (`nn_g12_fallback`). Outputs are in `data/train_stab/eval_pilot/`; the published rows come from `out_nnfb/`.

| Network | NN: safe success / violation / QP fail / timeout | NN + FB: safe success / violation / QP fail / timeout | Mean γ / β |
|---|---|---|---|
| Published | **71.1 / 23.3 / 4.6 / 1.1 %** | **80.4 / 11.5 / 6.5 / 1.5 %** | 12.3 / 5.3 |
| Pilot K = 1 | 63.5 / 33.9 / 2.2 / 0.4 % | 77.2 / 15.4 / 6.7 / 0.7 % | 8.1 / 13.4 |
| Pilot K = 1 + clip | 59.3 / 36.7 / 3.0 / 0.9 % | 76.7 / 15.7 / 6.7 / 0.9 % | 4.8 / 5.1 |
| Pilot K = 2 | 65.9 / 31.1 / 2.6 / 0.4 % | 77.2 / 15.2 / 7.0 / 0.7 % | 13.4 / 21.6 |
| Fixed 10/15 (no NN) | 62.2 / 35.2 / 2.2 / 0.4 % | – | 10 / 15 |

- **The stable pilot networks are close to the Fixed level and clearly behind the published network.** After 10 epochs they have barely moved from their initialisation (γ ≈ 9, β ≈ 15, close to Fixed 10/15), because the training loss gives almost no signal.
- **The gain combination matters strongly in closed loop.** The published network sits at γ ≈ 12 with low β ≈ 5. The pilot with low β but also low γ (4.8 / 5.1) is the worst.
- **Interpretation:** the old training's exploding gradients threw the network across a wide gain range (γ up to 150), and selection by validation metric picked a good region. It acted more like a random search with selection than like gradient descent. This matches the strong seed and selection dependence seen before.

## How state-dependent is the published network?

Raw network outputs `p1_nn` / `p2_nn` were logged at every step. Plot: `data/train_stab/gain_example.png`.

- **γ is practically constant.** Over all steps of all 460 runs it lies between 11.9 and 12.8 (10th–90th percentile), median 12.3.
- **β has a median of 2.2.** After the first second its 10th–90th percentile is 1.9–4.6.
  - In 42 % of runs it is high (up to about 31) during the first 0.5 s, when the arm starts from rest; there are also short spikes in a few runs while the arm hovers at the goal.
  - Over all steps β is uncorrelated with min h, ψ and obstacle distance (|ρ| < 0.06).
  - The mean of 5.2 reported earlier is inflated by the start-up spikes.
- In the first second, steps with β > 10 have almost no QP intervention (9 % of steps vs. 50 % with β ≤ 10), at similar distance.

**Direct test: fixed gains at the network's median.** Fixed 12.3 / 2.2 as a pure controller (`fixed`), and with the G1/G2 fallback through a constant-output checkpoint (`data/train_stab/const_12.3_2.2.pt`: the published weights with the output layer zeroed and the bias set to 12.3 / 2.2). Outputs are in `data/train_stab/eval_fixed_12.3_2.2/`.

| Controller | Safe success | Violation | QP fail | Timeout | Median task time |
|---|---|---|---|---|---|
| Fixed 10/15 | 62.2 % | 35.2 % | 2.2 % | 0.4 % | 8.26 s |
| Published NN | 71.1 % | 23.3 % | 4.6 % | 1.1 % | 8.30 s |
| Fixed 12.3 / 2.2 | 72.2 % | 20.0 % | 6.5 % | 1.3 % | 8.30 s |
| Published NN + FB | 80.4 % | 11.5 % | 6.5 % | 1.5 % | 8.35 s |
| Fixed 12.3 / 2.2 + FB | 79.3 % | 12.0 % | 7.4 % | 1.3 % | 8.34 s |

- **Fixed 12.3 / 2.2 matches the published network within noise:** +1.1 points without fallback, −1.1 points with it. About 20–30 runs switch outcome in each direction.
- **Task time is identical** (median difference 0.00 s over runs safe in both). The start-up β boost does not make the task measurably faster.
- **Conclusion:** the published network's advantage over Fixed 10/15 (+9 points) comes from the gain level, mainly the low β (2.2 instead of 15), not from state dependence.

**Fixed gains in the best configuration** (brake 0.05 + null-space damping K_d = 20; outputs in `data/train_stab/eval_fixed_12.3_2.2_best/`; published rows from `data/joint_limits/runs460/brake05_ns20/`):

| Controller | Safe success | Violation | QP fail | Timeout | Runs with fallback | Median task time |
|---|---|---|---|---|---|---|
| Published NN + FB | 85.0 % | 10.9 % | 1.5 % | 2.6 % | 33.9 % | 8.40 s |
| Fixed 12.3 / 2.2 + FB | 85.4 % | 8.7 % | 2.8 % | 3.0 % | 31.3 % | 8.40 s |
| Published NN + recover B = 200 | 92.6 % | 0.0 % | 2.8 % | 4.6 % | 34.3 % | 8.42 s |
| Fixed 12.3 / 2.2 + recover B = 200 | 92.0 % | 0.0 % | 3.0 % | 5.0 % | 31.5 % | 8.42 s |

- **Within noise again.** Recover −0.6 points (10 runs lost, 7 gained); FB +0.4 points, with fewer violations for the fixed gains.
- **Fallback and recover act the same with fixed gains.** The fallback fires in a similar share of runs, recover reaches 0 % violations, and the remaining QP fails are all `hocbf_rows`.
- **Theory:** neither mechanism depends on where the gains come from.
  - With fixed gains, recover turns the top-order row into one piecewise-linear extended class-K function, α(ψ) = β·ψ for ψ ≥ 0 and B·ψ for ψ < 0. That is continuous, strictly increasing and defined on ℝ, so outside the safe set the row drives ψ back while the QP is feasible.
  - The G1 repair (raising γ) changes ψ itself and is not covered by the fixed-γ HOCBF theory. That holds with or without the network.
- **The best controller so far needs no network:** fixed γ = 12.3, β = 2.2 + G1/G2 fallback + recover + brake rows + null-space damping, at 92.0 % safe success and 0 % violations.

## Open

- A direct closed-loop search over (γ, β), ideally on scenarios not in the 460 benchmark, instead of network training.
- Whether a learned policy can add something with a training signal that reflects closed-loop outcomes.
- **A training signal that reflects the benchmark:** longer effective horizons without exploding gradients, a loss weighting in which the gains matter, and a closed-loop validation for checkpoint selection.
