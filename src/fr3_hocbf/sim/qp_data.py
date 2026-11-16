#!/usr/bin/env python3

import numpy as np

NUM_ARM_JOINTS = 7
NUM_PAIR_ROWS = 43
NUM_BOUND_ROWS = 42
NUM_TOTAL_ROWS = NUM_PAIR_ROWS + NUM_BOUND_ROWS


def get_joint_limits():
    ddq_max_scalar = 40.0
    ddq_max_arm = np.full(NUM_ARM_JOINTS, ddq_max_scalar)
    ddq_min_arm = np.full(NUM_ARM_JOINTS, -ddq_max_scalar)

    dq_max_arm = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0], dtype=float)
    dq_min_arm = -dq_max_arm

    q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=float)
    q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=float)

    return {
        "ddq_max_arm": ddq_max_arm,
        "ddq_min_arm": ddq_min_arm,
        "dq_max_arm": dq_max_arm,
        "dq_min_arm": dq_min_arm,
        "q_max_arm": q_max_arm,
        "q_min_arm": q_min_arm,
    }



def build_collision_rows_numpy(
    h_all,
    Lf_h_all,
    vrel_sq2_all,
    Lg_psi_all,
    pair_mask,
    p1,
    p2,
):
    """
    Build collision/self-collision block:

        A_coll u >= b_coll

    from exported all-pair rollout tensors.
    """
    h_all = np.asarray(h_all, dtype=float)
    Lf_h_all = np.asarray(Lf_h_all, dtype=float)
    vrel_sq2_all = np.asarray(vrel_sq2_all, dtype=float)
    Lg_psi_all = np.asarray(Lg_psi_all, dtype=float)
    pair_mask = np.asarray(pair_mask, dtype=float)

    assert h_all.shape == (NUM_PAIR_ROWS,)
    assert Lf_h_all.shape == (NUM_PAIR_ROWS,)
    assert vrel_sq2_all.shape == (NUM_PAIR_ROWS,)
    assert Lg_psi_all.shape == (NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    assert pair_mask.shape == (NUM_PAIR_ROWS,)

    psi_all = Lf_h_all + p1 * h_all
    Lf_psi_all = vrel_sq2_all + p1 * Lf_h_all

    A_coll = Lg_psi_all.copy()
    b_coll = -Lf_psi_all - p2 * psi_all
    mask_coll = pair_mask.copy()

    return A_coll, b_coll, mask_coll


def build_bound_rows_numpy(
    q_arm,
    dq_arm,
    dt,
    ddq_min_arm,
    ddq_max_arm,
    dq_min_arm,
    dq_max_arm,
    q_min_arm,
    q_max_arm,
):
    """
    Build joint bound block in the form:

        A_bounds u >= b_bounds
    """
    q_arm = np.asarray(q_arm, dtype=float)
    dq_arm = np.asarray(dq_arm, dtype=float)

    ddq_min_arm = np.asarray(ddq_min_arm, dtype=float)
    ddq_max_arm = np.asarray(ddq_max_arm, dtype=float)
    dq_min_arm = np.asarray(dq_min_arm, dtype=float)
    dq_max_arm = np.asarray(dq_max_arm, dtype=float)
    q_min_arm = np.asarray(q_min_arm, dtype=float)
    q_max_arm = np.asarray(q_max_arm, dtype=float)

    I = np.eye(NUM_ARM_JOINTS, dtype=float)

    A_blocks = []
    b_blocks = []

    # accel lower:  u >= ddq_min
    A_blocks.append(I)
    b_blocks.append(ddq_min_arm)

    # accel upper:  u <= ddq_max  ->  -u >= -ddq_max
    A_blocks.append(-I)
    b_blocks.append(-ddq_max_arm)

    # vel lower: dq + u dt >= dq_min
    A_blocks.append(I)
    b_blocks.append((dq_min_arm - dq_arm) / dt)

    # vel upper: dq + u dt <= dq_max
    A_blocks.append(-I)
    b_blocks.append(-(dq_max_arm - dq_arm) / dt)

    # pos lower: q + dq dt + 0.5 u dt^2 >= q_min
    A_blocks.append(I)
    b_blocks.append(2.0 * (q_min_arm - q_arm - dq_arm * dt) / (dt ** 2))

    # pos upper: q + dq dt + 0.5 u dt^2 <= q_max
    A_blocks.append(-I)
    b_blocks.append(-2.0 * (q_max_arm - q_arm - dq_arm * dt) / (dt ** 2))

    A_bounds = np.vstack(A_blocks)
    b_bounds = np.concatenate(b_blocks)
    mask_bounds = np.ones(NUM_BOUND_ROWS, dtype=float)

    assert A_bounds.shape == (NUM_BOUND_ROWS, NUM_ARM_JOINTS)
    assert b_bounds.shape == (NUM_BOUND_ROWS,)
    assert mask_bounds.shape == (NUM_BOUND_ROWS,)

    return A_bounds, b_bounds, mask_bounds


def build_full_qp_numpy(
    q_arm,
    dq_arm,
    u_nom,
    h_all,
    Lf_h_all,
    vrel_sq2_all,
    Lg_psi_all,
    pair_mask,
    p1,
    p2,
    dt,
    ddq_min_arm,
    ddq_max_arm,
    dq_min_arm,
    dq_max_arm,
    q_min_arm,
    q_max_arm,
):
    """
    Build full fixed-size QP data:

        min 0.5 ||u - u_nom||^2
        s.t. A_full u >= b_full
    """
    u_nom = np.asarray(u_nom, dtype=float)

    A_coll, b_coll, mask_coll = build_collision_rows_numpy(
        h_all=h_all,
        Lf_h_all=Lf_h_all,
        vrel_sq2_all=vrel_sq2_all,
        Lg_psi_all=Lg_psi_all,
        pair_mask=pair_mask,
        p1=p1,
        p2=p2,
    )

    A_bounds, b_bounds, mask_bounds = build_bound_rows_numpy(
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

    A_full = np.vstack([A_coll, A_bounds])
    b_full = np.concatenate([b_coll, b_bounds])
    mask_full = np.concatenate([mask_coll, mask_bounds])

    # zero-out inactive collision rows so they become 0 >= 0
    inactive = mask_full < 0.5
    A_full[inactive, :] = 0.0
    b_full[inactive] = 0.0

    assert u_nom.shape == (NUM_ARM_JOINTS,)
    assert A_full.shape == (NUM_TOTAL_ROWS, NUM_ARM_JOINTS)
    assert b_full.shape == (NUM_TOTAL_ROWS,)
    assert mask_full.shape == (NUM_TOTAL_ROWS,)

    return u_nom, A_full, b_full, mask_full