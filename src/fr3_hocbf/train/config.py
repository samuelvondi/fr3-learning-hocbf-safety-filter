"""Training configuration of the differentiable-rollout training."""

from dataclasses import dataclass


DT_DEFAULT = 1.0 / 50.0


@dataclass
class RolloutTrainCfg:
    seed: int = 0
    epochs: int = 3
    segments_per_epoch: int = 1
    horizon: int = 10

    # If enabled, each segment starts from q,dq sampled from a generated good rollout NPZ.
    # The NPZ is used only for initial state/window sampling, not as labels.
    sample_from_rollouts: bool = False
    start_step_min: int = 0
    start_step_max: int | None = None

    lr: float = 1e-3
    weight_decay: float = 0.0

    hidden_dim: int = 256
    p1_floor: float = 1e-3
    p2_floor: float = 1e-3
    p1_max: float = 200.0
    p2_max: float = 250.0

    # Initial NN output at zero normalized input.
    # This makes an untrained network behave like the original/Davide baseline
    # instead of starting near half of the p caps.
    init_p1: float = 10.0
    init_p2: float = 15.0

    dt: float = DT_DEFAULT

    # Outer losses.
    # Safety is post-step geometric h plus post-step G1/psi violation.
    # Both safety terms use a top-k loss over actual positive violations so rare
    # dangerous pair/timestep violations are not diluted by many safe rows.
    # Goal is disabled by default for safety-first training.
    w_safety: float = 100.0          # post-step h top-k loss weight
    w_g1_safety: float = 10.0       # post-step G1/psi top-k loss weight
    w_viol_freq: float = 0.0        # post-step h violation-frequency loss weight
    viol_freq_eps: float = 1e-6     # sharpness scale for violation-frequency loss
    w_slack: float = 1.0
    w_goal: float = 0.0
    w_preg: float = 0.0

    # Post-step safety margins.
    # h loss:  topk(ReLU(h_safety_margin  - h_next)^2)
    # G1 loss: topk(ReLU(g1_safety_margin - psi_next)^2)
    h_safety_margin: float = 0.0
    g1_safety_margin: float = 0.0
    safety_topk_frac: float = 0.05
    slack_huber_delta: float = 0.0

    # Training-time DQP.
    # HOCBF/pair rows are hard; bound/input rows are softened with slack.
    rho: float = 1e5
    dqp_row_norm_eps: float = 1e-6

    # Rollout behavior.
    move_obstacles: bool = True
    detach_nominal: bool = True
    grad_clip: float = 5.0

    # Validation / checkpointing.
    val_every: int = 1
    val_segments: int = 5
    val_rollout_dir: str | None = None
    # safe_pair_viol_tol is kept for backwards-compatible CLIs/logs,
    # but checkpointing no longer uses pair-row violation.
    safe_pair_viol_tol: float = 1e-6
    safe_min_h_tol: float = 1e-6
    checkpoint_tol: float = 1e-9
    track_metric: str = "normalized_mean_goal_distance"
    goal_progress_eps: float = 1e-3

    # Logging / checkpoints.
    model_out: str = "fr3_p12_rollout_latest.pt"
    safe_candidate_dir: str = "safe_candidates"
    safe_candidate_csv: str = "safe_candidates_summary.csv"
    safe_candidate_txt: str = "safe_candidates_summary.txt"
    log_jsonl: str = "fr3_train_rollout_log.jsonl"
