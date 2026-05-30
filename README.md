# Learning-Based HOCBF Safety Filter for Franka FR3

This repository contains the code used for training and evaluating a learning-based HOCBF safety filter for collision avoidance on the Franka FR3.

The system learns HOCBF parameters for a QP-based safety filter and compares the learned policy against Davide's online HOCBF baseline. Evaluation scripts support offline rollout benchmarking, safety metrics, and paper-style result tables.

## Main components

- `src/fr3_train_rollout.py` — rollout-level training of the learned HOCBF parameter policy
- `src/fr3_compare_rollout_policies.py` — offline comparison of Davide, NN, and NN+fallback policies
- `src/fr3_make_final_metric_tables_generic_fixed.py` — generic final metric table generator
- `src/fr3_plot_compare_results.py` — plotting script for rollout comparison results
- `src/fr3_geometry.py`, `src/fr3_torch_geometry.py`, `src/fr3_torch_kinematics.py` — geometry and kinematics utilities
- `src/fr3_qp_solver.py`, `src/fr3_cvx_layer_bound_slack.py` — QP and differentiable QP utilities

## Notes

Large artifacts such as rollout datasets, model checkpoints, step-level CSV logs, plots, and result tables are intentionally excluded from Git. They should be stored separately or regenerated from the scripts.

## Typical workflow

1. Generate or load rollout data.
2. Compute feature statistics.
3. Train the NN parameter policy with `fr3_train_rollout.py`.
4. Evaluate policies with `fr3_compare_rollout_policies.py`.
5. Generate final tables with `fr3_make_final_metric_tables_generic_fixed.py`.
