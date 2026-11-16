"""Runtime gain policies of the controller comparison.

Fixed / scenario gains are set inline in compare_rollout_policies. This module holds the rest:
  - the trained NN gains (P12ParamNet, shared gamma/beta; PerPairBetaNet, per-pair beta)
  - the Davide online gain update (Fixed + Fallback)
  - the analytic pre-QP G1/G2 check and the minimal-gain G12 fallback, scalar and per pair

GAMMA_MAX_LIMIT / BETA_MAX_LIMIT are module-level caps; compare_rollout_policies.main sets them
from --p1-max / --p2-max (assign gain_policies.GAMMA_MAX_LIMIT, not a copied name).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from fr3_hocbf.policy.features import normalize_features_numpy
from fr3_hocbf.policy.p12_net import P12ParamNet
from fr3_hocbf.policy.pp_beta import PP_ARCH, load_pp_policy
from fr3_hocbf.sim.qp_data import NUM_ARM_JOINTS, NUM_PAIR_ROWS
from fr3_hocbf.sim.rollout import MAX_OBSTACLES

# Davide dynamic CBF parameter constants.
EPSILON_DENOMINATOR = 1e-6


GAMMA_ADJUST_BUFFER = 0.5


GAMMA_MAX_LIMIT = 200.0


BETA_ADJUST_BUFFER = 0.5


BETA_MAX_LIMIT = 250.0


NUM_ROBOT_OBSTACLE_ROWS = 8 * MAX_OBSTACLES  # 8 active links * 5 obstacle slots = 40


def load_nn_policy(model_path: str, stats_path: str | None, hidden_dim: int, p1_floor: float, p2_floor: float):
    ckpt = torch.load(os.path.expanduser(model_path), map_location="cpu")

    if stats_path is not None:
        stats = np.load(os.path.expanduser(stats_path))
        if "feat_mean" in stats.files:
            feat_mean = stats["feat_mean"].astype(float)
            feat_std = stats["feat_std"].astype(float)
        elif "mean" in stats.files:
            feat_mean = stats["mean"].astype(float)
            feat_std = stats["std"].astype(float)
        else:
            raise KeyError(f"Could not find feat_mean/feat_std in {stats_path}")
    elif isinstance(ckpt, dict) and "feat_mean" in ckpt and "feat_std" in ckpt:
        feat_mean = np.asarray(ckpt["feat_mean"], dtype=float)
        feat_std = np.asarray(ckpt["feat_std"], dtype=float)
    else:
        raise ValueError("NN mode requires --stats unless the checkpoint contains feat_mean/feat_std")

    input_dim = int(feat_mean.shape[0])

    if isinstance(ckpt, dict):
        hidden_dim = int(ckpt.get("hidden_dim", hidden_dim))
        p1_floor = float(ckpt.get("p1_floor", p1_floor))
        p2_floor = float(ckpt.get("p2_floor", p2_floor))
        p1_max = float(ckpt.get("p1_max", 200.0))
        p2_max = float(ckpt.get("p2_max", 250.0))
    else:
        p1_max = 200.0
        p2_max = 250.0

    model = P12ParamNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        p1_floor=p1_floor,
        p2_floor=p2_floor,
        p1_max=p1_max,
        p2_max=p2_max,
    ).double()

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model, feat_mean, feat_std


@torch.no_grad()
def predict_p12_nn_current(
    model: P12ParamNet,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    ddq_nominal_arm: np.ndarray,
    joint_limits: dict,
    h_all: np.ndarray,
    Lf_h_all: np.ndarray,
) -> tuple[float, float]:
    ddq_nominal_arm = np.asarray(ddq_nominal_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    ddq_scale = np.maximum(
        np.abs(np.asarray(joint_limits["ddq_min_arm"], dtype=float).reshape(NUM_ARM_JOINTS)),
        np.abs(np.asarray(joint_limits["ddq_max_arm"], dtype=float).reshape(NUM_ARM_JOINTS)),
    )
    ddq_scale = np.clip(ddq_scale, 1e-6, None)
    ddq_nominal_norm = ddq_nominal_arm / ddq_scale

    feature_dim = int(np.asarray(feat_mean).reshape(-1).shape[0])
    old_dim = 7 + 7 + 7 + NUM_PAIR_ROWS + NUM_PAIR_ROWS
    new_dim = 7 + 7 + 7 + 7 + NUM_PAIR_ROWS + NUM_PAIR_ROWS

    if feature_dim == old_dim:
        # Backwards compatibility for existing 107D checkpoints/stats.
        z_parts = [
            np.asarray(q_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            np.asarray(dq_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            ddq_nominal_arm,
            np.asarray(h_all, dtype=float).reshape(NUM_PAIR_ROWS),
            np.asarray(Lf_h_all, dtype=float).reshape(NUM_PAIR_ROWS),
        ]
    elif feature_dim == new_dim:
        # New 114D feature vector with 3D analogue of u_nom/u_max.
        z_parts = [
            np.asarray(q_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            np.asarray(dq_arm, dtype=float).reshape(NUM_ARM_JOINTS),
            ddq_nominal_arm,
            ddq_nominal_norm,
            np.asarray(h_all, dtype=float).reshape(NUM_PAIR_ROWS),
            np.asarray(Lf_h_all, dtype=float).reshape(NUM_PAIR_ROWS),
        ]
    else:
        raise ValueError(
            f"Unsupported feature dimension {feature_dim}; expected old {old_dim}D or new {new_dim}D stats."
        )

    z_raw = np.concatenate(z_parts, axis=0)[None, :]

    z_norm = normalize_features_numpy(z_raw, feat_mean, feat_std)
    z_t = torch.tensor(z_norm, dtype=torch.double)
    p12 = model(z_t).detach().cpu().numpy()[0]
    return float(p12[0]), float(p12[1])


def kinematic_ddq_bounds(q_arm: np.ndarray, dq_arm: np.ndarray, limits: dict, dt: float) -> tuple[np.ndarray, np.ndarray]:
    q_arm = np.asarray(q_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    dq_arm = np.asarray(dq_arm, dtype=float).reshape(NUM_ARM_JOINTS)

    ddq_max_arm = limits["ddq_max_arm"]
    ddq_min_arm = limits["ddq_min_arm"]
    dq_max_arm = limits["dq_max_arm"]
    dq_min_arm = limits["dq_min_arm"]
    q_max_arm = limits["q_max_arm"]
    q_min_arm = limits["q_min_arm"]

    ddq_max_from_vel = (dq_max_arm - dq_arm) / dt
    ddq_min_from_vel = (dq_min_arm - dq_arm) / dt
    ddq_max_from_pos = 2.0 * (q_max_arm - q_arm - dq_arm * dt) / (dt ** 2)
    ddq_min_from_pos = 2.0 * (q_min_arm - q_arm - dq_arm * dt) / (dt ** 2)

    ddq_max_kinematic = np.minimum(ddq_max_arm, np.minimum(ddq_max_from_vel, ddq_max_from_pos))
    ddq_min_kinematic = np.maximum(ddq_min_arm, np.maximum(ddq_min_from_vel, ddq_min_from_pos))
    return ddq_min_kinematic, ddq_max_kinematic


def max_gamma_required_over_pairs(h: np.ndarray, Lf_h: np.ndarray) -> float:
    """
    Largest gamma that the first-order condition G1 = Lf_h + gamma*h >= 0 requires:
    max over pairs with h > 0 of -Lf_h/h; -inf if no pair qualifies.

    h <= 0: already in collision. Raising p1 only makes psi more negative and the
    QP row harder, so such pairs are skipped (eq. 21).
    """
    max_gamma_required = -float("inf")
    for hi, lfi in zip(h, Lf_h):
        if hi > EPSILON_DENOMINATOR:
            gamma_needed = -float(lfi) / float(hi)
            max_gamma_required = max(max_gamma_required, gamma_needed)
    return max_gamma_required


def max_beta_required_over_pairs(
    psi: np.ndarray,
    Lf_psi: np.ndarray,
    Lg: np.ndarray,
    ddq_min_kin: np.ndarray,
    ddq_max_kin: np.ndarray,
) -> float:
    """
    Largest beta that G2 = Lf_psi + S_sup + beta*psi >= 0 requires, with S_sup the
    largest Lg_psi u over the kinematic acceleration box: max over pairs with psi > 0
    of -(Lf_psi + S_sup)/psi; -inf if no pair qualifies.

    psi <= 0: raising p2 only tightens the QP row, so such pairs are skipped (eq. 23).
    """
    max_beta_required = -float("inf")
    for psii, lfpsii, lgi in zip(psi, Lf_psi, Lg):
        sup_lg_u = float(np.sum(np.where(lgi >= 0.0, lgi * ddq_max_kin, lgi * ddq_min_kin)))
        S_sup = float(lfpsii) + sup_lg_u

        if psii > EPSILON_DENOMINATOR:
            beta_needed = -S_sup / float(psii)
            max_beta_required = max(max_beta_required, beta_needed)
    return max_beta_required


def davide_online_p12_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    initial_gamma: float,
    initial_beta: float,
    pair_scope: str = "all",
) -> tuple[float, float, float, float]:
    """
    Single-step Davide-style dynamic gamma/beta calculation.

    This mirrors the two-pass logic:
      gamma from -Lf_h/h, then beta from -S_sup/psi.
    Only active pairs (below the activation threshold) are considered.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)

    h = h_all[active]
    Lf_h = Lf_h_all[active]
    vrel_sq2 = vrel_sq2_all[active]
    Lg = Lg_psi_all[active, :]

    max_gamma_required = max_gamma_required_over_pairs(h, Lf_h)

    gamma = float(initial_gamma)
    if max_gamma_required > -float("inf"):
        adjusted_gamma = max(0.0, max_gamma_required + GAMMA_ADJUST_BUFFER)
        gamma = max(float(initial_gamma), min(adjusted_gamma, GAMMA_MAX_LIMIT))

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)

    psi = Lf_h + gamma * h
    Lf_psi = vrel_sq2 + gamma * Lf_h

    max_beta_required = max_beta_required_over_pairs(psi, Lf_psi, Lg, ddq_min_kin, ddq_max_kin)

    beta = float(initial_beta)
    if max_beta_required > -float("inf"):
        adjusted_beta = max(0.0, max_beta_required + BETA_ADJUST_BUFFER)
        beta = max(float(initial_beta), min(adjusted_beta, BETA_MAX_LIMIT))

    return gamma, beta, float(max_gamma_required), float(max_beta_required)


def pair_scope_mask(pair_mask: np.ndarray, pair_scope: str) -> np.ndarray:
    """Return the active pair mask restricted to obstacle rows or all pair rows."""
    pair_mask = np.asarray(pair_mask, dtype=float).reshape(NUM_PAIR_ROWS) > 0.5
    if pair_scope == "obstacle":
        scoped = np.zeros(NUM_PAIR_ROWS, dtype=bool)
        scoped[:NUM_ROBOT_OBSTACLE_ROWS] = pair_mask[:NUM_ROBOT_OBSTACLE_ROWS]
        return scoped
    if pair_scope == "all":
        return pair_mask
    raise ValueError(f"Unknown pair_scope: {pair_scope}")


def min_g1_current(all_pair_terms: dict, p1: float, pair_scope: str) -> float:
    """
    Current-state G1/psi diagnostic:
        G1 = Lf_h + p1*h

    This is computed before integration, on the same state used to build the QP.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)
    if not np.any(active):
        return float("nan")
    g1_all = Lf_h_all + float(p1) * h_all
    return float(np.min(g1_all[active]))


def g12_check_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    p1: float,
    p2: float,
    pair_scope: str,
) -> tuple[float, float]:
    """
    Pre-QP analytic feasibility check of candidate p1,p2 (paper Sec. II-D).

    G1 over active pairs with h > 0:
        G1 = Lf_h + p1*h
    G2 over active pairs with psi > 0, using the largest contribution the
    bounded input can make (S_sup = sup_{u in U} Lg_psi u):
        G2 = Lf_psi + S_sup + p2*psi

    Returns (min G1, min G2); nan if no pair qualifies. A negative value means
    p1 resp. p2 is too small and the fallback has to repair it.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)

    psi_all = Lf_h_all + float(p1) * h_all
    g1_rows = active & (h_all > EPSILON_DENOMINATOR)
    min_g1 = float(np.min(psi_all[g1_rows])) if np.any(g1_rows) else float("nan")

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)
    Lf_psi_all = vrel_sq2_all + float(p1) * Lf_h_all
    S_sup_all = np.sum(np.where(Lg_psi_all >= 0.0, Lg_psi_all * ddq_max_kin, Lg_psi_all * ddq_min_kin), axis=1)
    g2_all = Lf_psi_all + S_sup_all + float(p2) * psi_all
    g2_rows = active & (psi_all > EPSILON_DENOMINATOR)
    min_g2 = float(np.min(g2_all[g2_rows])) if np.any(g2_rows) else float("nan")

    return min_g1, min_g2


def g12_values_hold(min_g1: float, min_g2: float, tol: float) -> bool:
    """Accept nan margins when there are no active rows, otherwise require >= -tol."""
    g1_ok = (not np.isfinite(min_g1)) or (min_g1 >= -float(tol))
    g2_ok = (not np.isfinite(min_g2)) or (min_g2 >= -float(tol))
    return bool(g1_ok and g2_ok)


def minimal_g12_p12_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    p1_floor: float = 1e-3,
    p2_floor: float = 1e-3,
    pair_scope: str = "all",
) -> tuple[float, float, float, float]:
    """
    Compute an independent fallback p1,p2 from the current G1/G2 conditions.

    This deliberately does NOT use the scenario YAML gamma/beta values as a
    baseline/floor. It computes the smallest current-state gains suggested by
    the G1/G2 inequalities, then clips them to the same caps used elsewhere.
    The NN fallback passes gamma_NN as p1_floor, which gives
    gamma_final = max(gamma_NN, gamma_req), and a small constant p2_floor, so
    beta_final is the minimal (conservative) required beta. psi is evaluated
    with gamma_final, so beta is solved against gamma_final.

    G1 condition:
        G1 = Lf_h + p1*h >= 0
        p1 >= -Lf_h/h, if h > 0

    G2 condition after p1 is selected:
        G2 = Lf_psi + Lg_psi*u + p2*psi >= 0

    For G2, we use the same acceleration-bound support estimate as the Davide
    online calculation. This is not a QP over p1,p2; it is the analytic/current
    state gain calculation analogous to the 2D G12 fallback.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)

    if not np.any(active):
        return float(p1_floor), float(p2_floor), float("nan"), float("nan")

    h = h_all[active]
    Lf_h = Lf_h_all[active]
    vrel_sq2 = vrel_sq2_all[active]
    Lg = Lg_psi_all[active, :]

    max_gamma_required = max_gamma_required_over_pairs(h, Lf_h)

    if max_gamma_required > -float("inf"):
        gamma = max(float(p1_floor), max(0.0, max_gamma_required + GAMMA_ADJUST_BUFFER))
        gamma = min(gamma, GAMMA_MAX_LIMIT)
    else:
        gamma = float(p1_floor)

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)

    psi = Lf_h + gamma * h
    Lf_psi = vrel_sq2 + gamma * Lf_h

    max_beta_required = max_beta_required_over_pairs(psi, Lf_psi, Lg, ddq_min_kin, ddq_max_kin)

    if max_beta_required > -float("inf"):
        beta = max(float(p2_floor), max(0.0, max_beta_required + BETA_ADJUST_BUFFER))
        beta = min(beta, BETA_MAX_LIMIT)
    else:
        beta = float(p2_floor)

    return float(gamma), float(beta), float(max_gamma_required), float(max_beta_required)


def load_pp_pack(model_path: str, stats_path: str | None, args):
    """
    PerPairBetaNet checkpoint -> ("pp", model, feat_mean, feat_std).

    An old P12ParamNet checkpoint is accepted as well -> ("broadcast", nn_pack):
    beta_j = shared beta_NN for all pairs. Used for the sanity check that the
    per-pair path reproduces nn / nn_g12_fallback exactly.
    """
    ckpt = torch.load(os.path.expanduser(model_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("arch") == PP_ARCH:
        model, feat_mean, feat_std = load_pp_policy(os.path.expanduser(model_path))
        return ("pp", model, feat_mean, feat_std)
    return ("broadcast", load_nn_policy(model_path, stats_path, args.hidden_dim, args.p1_floor, args.p2_floor))


@torch.no_grad()
def predict_pp_current(
    pp_pack,
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    ddq_nominal_arm: np.ndarray,
    joint_limits: dict,
    all_pair_terms: dict,
    dt: float,
) -> tuple[float, np.ndarray]:
    """Return gamma (scalar) and beta (43,) for the current state."""
    if pp_pack[0] == "broadcast":
        nn_model, feat_mean, feat_std = pp_pack[1]
        p1, p2 = predict_p12_nn_current(
            model=nn_model,
            feat_mean=feat_mean,
            feat_std=feat_std,
            q_arm=q_arm,
            dq_arm=dq_arm,
            ddq_nominal_arm=ddq_nominal_arm,
            joint_limits=joint_limits,
            h_all=all_pair_terms["h_all"].copy(),
            Lf_h_all=all_pair_terms["Lf_h_all"].copy(),
        )
        return p1, np.full(NUM_PAIR_ROWS, p2, dtype=float)

    _, model, feat_mean, feat_std = pp_pack
    ddq_nominal_arm = np.asarray(ddq_nominal_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    ddq_scale = np.clip(
        np.maximum(np.abs(np.asarray(joint_limits["ddq_min_arm"], dtype=float)), np.abs(np.asarray(joint_limits["ddq_max_arm"], dtype=float))),
        1e-6,
        None,
    )
    z_raw = np.concatenate(
        [q_arm, dq_arm, ddq_nominal_arm, ddq_nominal_arm / ddq_scale, all_pair_terms["h_all"], all_pair_terms["Lf_h_all"]]
    )[None, :]
    z_t = torch.tensor(normalize_features_numpy(z_raw, feat_mean, feat_std), dtype=torch.double)
    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, joint_limits, dt)

    def t(x):
        return torch.tensor(np.asarray(x, dtype=float), dtype=torch.double).unsqueeze(0)

    gamma, beta = model(
        z_t,
        {
            "h_all": t(all_pair_terms["h_all"]),
            "Lf_h_all": t(all_pair_terms["Lf_h_all"]),
            "vrel_sq2_all": t(all_pair_terms["vrel_sq2_all"]),
            "Lg_psi_all": t(all_pair_terms["Lg_psi_all"]),
            "pair_mask": t(all_pair_terms["pair_mask"]),
            "u_nom": t(ddq_nominal_arm),
            "ddq_min_kin": t(ddq_min_kin),
            "ddq_max_kin": t(ddq_max_kin),
        },
    )
    return float(gamma[0, 0]), beta[0].numpy().astype(float)


def pp_g12_fallback_current(
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    all_pair_terms: dict,
    limits: dict,
    dt: float,
    gamma_nn: float,
    beta_nn: np.ndarray,
    pair_scope: str,
    tol: float,
    p1_floor: float,
    p2_floor: float,
    reduce: str = "per_pair",
    unflagged: str = "keep",
) -> dict:
    """
    G12 fallback for a per-pair beta vector.

    Pre-QP check: same G1/G2 rows as g12_check_current, but G2_j uses beta_NN_j.
    If it holds, the NN gains are kept. Otherwise:
      gamma: unchanged rule, gamma_final = max(gamma_NN, gamma_req + 0.5) (capped).
      reduce="scalar": exactly the existing fallback (beta_req reduced by max over
          pairs, applied to every pair). Sanity-check path.
      reduce="per_pair": G2_j is re-checked with gamma_final and beta_NN_j; pairs that
          fail get beta_j = max(p2_floor, beta_req_j + 0.5) (capped). Unflagged pairs:
            unflagged="keep":       beta_NN_j
            unflagged="lower_pair": min(beta_NN_j, max(p2_floor, beta_req_j + 0.5)) for psi_j > 0
            unflagged="cap_old":    min(beta_NN_j, B), B = the existing scalar fallback beta
                                    (max over pairs of beta_req + 0.5). Equals the existing
                                    fallback when all beta_NN_j are equal and B <= beta_NN.
    """
    h_all = np.asarray(all_pair_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(all_pair_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(all_pair_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_psi_all = np.asarray(all_pair_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)
    active = pair_scope_mask(all_pair_terms["pair_mask"], pair_scope)
    beta_nn = np.asarray(beta_nn, dtype=float).reshape(NUM_PAIR_ROWS)

    ddq_min_kin, ddq_max_kin = kinematic_ddq_bounds(q_arm, dq_arm, limits, dt)
    S_sup_all = np.sum(np.where(Lg_psi_all >= 0.0, Lg_psi_all * ddq_max_kin, Lg_psi_all * ddq_min_kin), axis=1)

    psi_nn = Lf_h_all + float(gamma_nn) * h_all
    g1_rows = active & (h_all > EPSILON_DENOMINATOR)
    g2_rows = active & (psi_nn > EPSILON_DENOMINATOR)
    g2_nn = vrel_sq2_all + float(gamma_nn) * Lf_h_all + S_sup_all + beta_nn * psi_nn
    min_g1 = float(np.min(psi_nn[g1_rows])) if np.any(g1_rows) else float("nan")
    min_g2 = float(np.min(g2_nn[g2_rows])) if np.any(g2_rows) else float("nan")

    no_pairs = np.zeros(NUM_PAIR_ROWS, dtype=bool)
    out = {
        "gamma": float(gamma_nn),
        "beta": beta_nn.copy(),
        "fallback_used": False,
        "fallback_reason": "",
        "min_g1_candidate": min_g1,
        "min_g2_candidate": min_g2,
        "max_gamma_required": float("nan"),
        "max_beta_required": float("nan"),
        "fb_pairs": no_pairs.copy(),
        "lowered_pairs": no_pairs.copy(),
        "beta_cap_pairs": no_pairs.copy(),
        "gamma_cap_hit": False,
    }
    if g12_values_hold(min_g1=min_g1, min_g2=min_g2, tol=tol):
        return out

    out["fallback_used"] = True
    out["fallback_reason"] = "G1_negative" if (np.isfinite(min_g1) and min_g1 < -float(tol)) else "G2_sup_negative"

    gamma, beta_scalar, max_gamma_required, max_beta_required = minimal_g12_p12_current(
        q_arm=q_arm,
        dq_arm=dq_arm,
        all_pair_terms=all_pair_terms,
        limits=limits,
        dt=dt,
        p1_floor=max(float(p1_floor), float(gamma_nn)),
        p2_floor=float(p2_floor),
        pair_scope=pair_scope,
    )
    out["gamma"] = float(gamma)
    out["max_gamma_required"] = float(max_gamma_required)
    out["gamma_cap_hit"] = bool(np.isfinite(max_gamma_required) and max_gamma_required + GAMMA_ADJUST_BUFFER > GAMMA_MAX_LIMIT)

    if reduce == "scalar":
        out["beta"] = np.full(NUM_PAIR_ROWS, float(beta_scalar), dtype=float)
        out["max_beta_required"] = float(max_beta_required)
        out["fb_pairs"] = active.copy()
        cap = bool(np.isfinite(max_beta_required) and max_beta_required + BETA_ADJUST_BUFFER > BETA_MAX_LIMIT)
        out["beta_cap_pairs"] = active & cap
        return out
    if reduce != "per_pair":
        raise ValueError(f"Unknown pp fallback reduce: {reduce}")

    # Per pair, against gamma_final. psi_j <= 0 rows are skipped as in eq. 23.
    psi = Lf_h_all + gamma * h_all
    Lf_psi = vrel_sq2_all + gamma * Lf_h_all
    rows = active & (psi > EPSILON_DENOMINATOR)
    beta_req = np.full(NUM_PAIR_ROWS, -np.inf)
    beta_req[rows] = -(Lf_psi[rows] + S_sup_all[rows]) / psi[rows]
    g2 = Lf_psi + S_sup_all + beta_nn * psi
    flagged = rows & (g2 < -float(tol))
    target = np.minimum(np.maximum(float(p2_floor), np.maximum(0.0, beta_req + BETA_ADJUST_BUFFER)), BETA_MAX_LIMIT)

    beta = beta_nn.copy()
    beta[flagged] = target[flagged]
    lowered = no_pairs.copy()
    if unflagged == "lower_pair":
        lowered = rows & ~flagged & (target < beta_nn)
        beta[lowered] = target[lowered]
    elif unflagged == "cap_old":
        lowered = active & ~flagged & (float(beta_scalar) < beta_nn)
        beta[lowered] = float(beta_scalar)
    elif unflagged != "keep":
        raise ValueError(f"Unknown pp unflagged rule: {unflagged}")

    out["beta"] = beta
    out["fb_pairs"] = flagged
    out["lowered_pairs"] = lowered
    out["beta_cap_pairs"] = flagged & (beta_req + BETA_ADJUST_BUFFER > BETA_MAX_LIMIT)
    out["max_beta_required"] = float(np.max(beta_req[rows])) if np.any(rows) else float("nan")
    return out
