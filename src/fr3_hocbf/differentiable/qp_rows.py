"""Batched torch construction of the full QP rows A u >= b (43 HOCBF pair rows + 42 joint-bound rows).

Shared gains: p12 (B, 2). Per-pair beta: gamma (B, 1), beta (B, 43).
"""

import torch

from fr3_hocbf.sim.qp_data import NUM_ARM_JOINTS, NUM_BOUND_ROWS


def build_collision_rows_torch(
    p12: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
    vrel_sq2_all: torch.Tensor,
    Lg_psi_all: torch.Tensor,
    pair_mask: torch.Tensor,
):
    """
    Build A_coll u >= b_coll for a whole batch.

    Shapes:
      p12         : (B, 2)
      h_all       : (B, 43)
      Lf_h_all    : (B, 43)
      vrel_sq2_all: (B, 43)
      Lg_psi_all  : (B, 43, 7)
      pair_mask   : (B, 43)
    """
    p1 = p12[:, 0:1]
    p2 = p12[:, 1:2]

    psi_all = Lf_h_all + p1 * h_all
    Lf_psi_all = vrel_sq2_all + p1 * Lf_h_all

    A_coll = Lg_psi_all
    b_coll = -Lf_psi_all - p2 * psi_all
    mask_coll = pair_mask

    return A_coll, b_coll, mask_coll


def build_bound_rows_torch(
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    dt: float,
    ddq_min_arm: torch.Tensor,
    ddq_max_arm: torch.Tensor,
    dq_min_arm: torch.Tensor,
    dq_max_arm: torch.Tensor,
    q_min_arm: torch.Tensor,
    q_max_arm: torch.Tensor,
):
    """
    Build fixed 42 joint-bound rows for the whole batch.

    Returns:
      A_bounds: (B, 42, 7)
      b_bounds: (B, 42)
      mask_bounds: (B, 42)
    """
    B = q_arm.shape[0]
    device = q_arm.device
    dtype = q_arm.dtype

    I = torch.eye(NUM_ARM_JOINTS, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    b_acc_low = ddq_min_arm.unsqueeze(0).expand(B, -1)
    b_acc_high = -ddq_max_arm.unsqueeze(0).expand(B, -1)

    b_vel_low = (dq_min_arm.unsqueeze(0) - dq_arm) / dt
    b_vel_high = -(dq_max_arm.unsqueeze(0) - dq_arm) / dt

    b_pos_low = 2.0 * (q_min_arm.unsqueeze(0) - q_arm - dq_arm * dt) / (dt ** 2)
    b_pos_high = -2.0 * (q_max_arm.unsqueeze(0) - q_arm - dq_arm * dt) / (dt ** 2)

    A_bounds = torch.cat(
        [I, -I, I, -I, I, -I],
        dim=1,
    )
    b_bounds = torch.cat(
        [b_acc_low, b_acc_high, b_vel_low, b_vel_high, b_pos_low, b_pos_high],
        dim=1,
    )
    mask_bounds = torch.ones(B, NUM_BOUND_ROWS, device=device, dtype=dtype)

    return A_bounds, b_bounds, mask_bounds


def build_full_qp_torch_batch(
    p12: torch.Tensor,
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
    vrel_sq2_all: torch.Tensor,
    Lg_psi_all: torch.Tensor,
    pair_mask: torch.Tensor,
    dt: float,
    ddq_min_arm: torch.Tensor,
    ddq_max_arm: torch.Tensor,
    dq_min_arm: torch.Tensor,
    dq_max_arm: torch.Tensor,
    q_min_arm: torch.Tensor,
    q_max_arm: torch.Tensor,
):
    A_coll, b_coll, mask_coll = build_collision_rows_torch(
        p12=p12,
        h_all=h_all,
        Lf_h_all=Lf_h_all,
        vrel_sq2_all=vrel_sq2_all,
        Lg_psi_all=Lg_psi_all,
        pair_mask=pair_mask,
    )

    A_bounds, b_bounds, mask_bounds = build_bound_rows_torch(
        q_arm=q_arm,
        dq_arm=dq_arm,
        dt=dt,
        ddq_min_arm=ddq_min_arm,
        ddq_max_arm=ddq_max_arm,
        dq_min_arm=dq_min_arm,
        dq_max_arm=dq_max_arm,
        q_min_arm=q_min_arm,
        q_max_arm=q_max_arm,
    )

    A_full = torch.cat([A_coll, A_bounds], dim=1)
    b_full = torch.cat([b_coll, b_bounds], dim=1)
    mask_full = torch.cat([mask_coll, mask_bounds], dim=1)

    # zero inactive rows so they become 0 >= 0
    A_full = A_full * mask_full.unsqueeze(-1)
    b_full = b_full * mask_full

    return A_full, b_full, mask_full


def build_collision_rows_torch_pp(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
    vrel_sq2_all: torch.Tensor,
    Lg_psi_all: torch.Tensor,
    pair_mask: torch.Tensor,
):
    """Same rows as build_collision_rows_torch, with beta (B, 43) instead of a shared scalar."""
    psi_all = Lf_h_all + gamma * h_all
    Lf_psi_all = vrel_sq2_all + gamma * Lf_h_all
    return Lg_psi_all, -Lf_psi_all - beta * psi_all, pair_mask


def build_full_qp_torch_batch_pp(gamma, beta, q_arm, dq_arm, pair_terms_b, dt, limits_t):
    """build_full_qp_torch_batch with per-pair beta."""
    A_coll, b_coll, mask_coll = build_collision_rows_torch_pp(
        gamma=gamma,
        beta=beta,
        h_all=pair_terms_b["h_all"],
        Lf_h_all=pair_terms_b["Lf_h_all"],
        vrel_sq2_all=pair_terms_b["vrel_sq2_all"],
        Lg_psi_all=pair_terms_b["Lg_psi_all"],
        pair_mask=pair_terms_b["pair_mask"],
    )
    A_bounds, b_bounds, mask_bounds = build_bound_rows_torch(
        q_arm=q_arm,
        dq_arm=dq_arm,
        dt=dt,
        ddq_min_arm=limits_t["ddq_min_arm"],
        ddq_max_arm=limits_t["ddq_max_arm"],
        dq_min_arm=limits_t["dq_min_arm"],
        dq_max_arm=limits_t["dq_max_arm"],
        q_min_arm=limits_t["q_min_arm"],
        q_max_arm=limits_t["q_max_arm"],
    )
    A_full = torch.cat([A_coll, A_bounds], dim=1)
    b_full = torch.cat([b_coll, b_bounds], dim=1)
    mask_full = torch.cat([mask_coll, mask_bounds], dim=1)
    A_full = A_full * mask_full.unsqueeze(-1)
    b_full = b_full * mask_full
    return A_full, b_full, mask_full
