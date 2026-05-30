#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from fr3_torch_kinematics import (
    NUM_ARM_JOINTS,
    ACTIVE_LINKS_DEF,
    EE_FRAME_NAME,
)


MAX_OBSTACLES = 5
SELF_COLLISION_LINK_PAIRS = [
    ("link2", "hand"),
    ("link2", "end_effector"),
    ("link2", "forearm1"),
]

NUM_PAIR_ROWS = len(ACTIVE_LINKS_DEF) * MAX_OBSTACLES + len(SELF_COLLISION_LINK_PAIRS)
EPS = 1e-9


Tensor = torch.Tensor


@dataclass
class TorchCapsuleKinematics:
    p0: Tensor                    # (3,)
    p1: Tensor                    # (3,)
    radius: float
    J0: Optional[Tensor] = None   # (3,7)
    J1: Optional[Tensor] = None   # (3,7)
    linear_velocity: Optional[Tensor] = None  # (3,)
    name: str = ""


def _ensure_vec3(x: Tensor, name: str) -> Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(x.shape) != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(x.shape)}")
    return x


def _ensure_mat37(x: Tensor, name: str) -> Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(x.shape) != (3, NUM_ARM_JOINTS):
        raise ValueError(f"{name} must have shape (3,{NUM_ARM_JOINTS}), got {tuple(x.shape)}")
    return x


def _zeros3_like(x: Tensor) -> Tensor:
    return torch.zeros(3, dtype=x.dtype, device=x.device)


def _zeros7_like(x: Tensor) -> Tensor:
    return torch.zeros(NUM_ARM_JOINTS, dtype=x.dtype, device=x.device)


def _optional_subtract(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    if a is None and b is None:
        return None
    if a is None:
        return -b
    if b is None:
        return a
    return a - b


def _scalar_tensor(val: float, ref: Tensor) -> Tensor:
    return torch.tensor(val, dtype=ref.dtype, device=ref.device)


def _clamp01(x: Tensor) -> Tensor:
    return torch.clamp(x, 0.0, 1.0)


def closest_points_between_segments_torch(
    p1: Tensor,
    p2: Tensor,
    q1: Tensor,
    q2: Tensor,
    eps: float = EPS,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Torch port of the current robust segment-segment closest-point solver.

    Inputs:
        p1, p2, q1, q2: (3,)

    Returns:
        c1, c2, s, t
        where:
            c1 = p1 + s * (p2 - p1)
            c2 = q1 + t * (q2 - q1)
            s,t in [0,1]
    """
    p1 = _ensure_vec3(p1, "p1")
    p2 = _ensure_vec3(p2, "p2")
    q1 = _ensure_vec3(q1, "q1")
    q2 = _ensure_vec3(q2, "q2")

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = torch.dot(u, u)
    b = torch.dot(u, v)
    c = torch.dot(v, v)
    d = torch.dot(u, w)
    e = torch.dot(v, w)
    D = a * c - b * b

    eps_t = _scalar_tensor(eps, p1)
    zero = _scalar_tensor(0.0, p1)
    one = _scalar_tensor(1.0, p1)

    a_f = float(a.detach().cpu())
    c_f = float(c.detach().cpu())
    D_f = float(D.detach().cpu())

    # both degenerate to points
    if a_f <= eps and c_f <= eps:
        s = zero
        t = zero
        c1 = p1
        c2 = q1
        return c1, c2, s, t

    # first degenerates to a point
    if a_f <= eps:
        s = zero
        t = _clamp01(e / (c + eps_t)) if c_f > eps else zero
        c1 = p1
        c2 = q1 + t * v
        return c1, c2, s, t

    # second degenerates to a point
    if c_f <= eps:
        t = zero
        s = _clamp01((-d) / (a + eps_t))
        c1 = p1 + s * u
        c2 = q1
        return c1, c2, s, t

    sN = zero
    sD = D
    tN = zero
    tD = D

    if D_f <= eps:
        # nearly parallel
        sN = zero
        sD = one
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d

        sN_f = float(sN.detach().cpu())
        sD_f = float(sD.detach().cpu())

        if sN_f < 0.0:
            sN = zero
            tN = e
            tD = c
        elif sN_f > sD_f:
            sN = sD
            tN = e + b
            tD = c

    tN_f = float(tN.detach().cpu())
    tD_f = float(tD.detach().cpu())

    if tN_f < 0.0:
        tN = zero
        neg_d_f = float((-d).detach().cpu())
        a_now_f = float(a.detach().cpu())

        if neg_d_f < 0.0:
            sN = zero
        elif neg_d_f > a_now_f:
            sN = sD
        else:
            sN = -d
            sD = a

    elif tN_f > tD_f:
        tN = tD
        expr = -d + b
        expr_f = float(expr.detach().cpu())
        a_now_f = float(a.detach().cpu())

        if expr_f < 0.0:
            sN = zero
        elif expr_f > a_now_f:
            sN = sD
        else:
            sN = expr
            sD = a

    sN_f = float(torch.abs(sN).detach().cpu())
    tN_f = float(torch.abs(tN).detach().cpu())

    s = zero if sN_f <= eps else sN / (sD + eps_t)
    t = zero if tN_f <= eps else tN / (tD + eps_t)

    s = _clamp01(s)
    t = _clamp01(t)

    c1 = p1 + s * u
    c2 = q1 + t * v
    return c1, c2, s, t


def interpolate_point_on_segment_torch(p0: Tensor, p1: Tensor, s: Tensor) -> Tensor:
    p0 = _ensure_vec3(p0, "p0")
    p1 = _ensure_vec3(p1, "p1")
    return (1.0 - s) * p0 + s * p1


def interpolate_segment_jacobian_torch(J0: Tensor, J1: Tensor, s: Tensor) -> Tensor:
    J0 = _ensure_mat37(J0, "J0")
    J1 = _ensure_mat37(J1, "J1")
    return (1.0 - s) * J0 + s * J1


def point_kinematics_on_segment_torch(
    capsule: TorchCapsuleKinematics,
    s: Tensor,
    dq: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Tensor]:
    """
    Returns:
        J_c, v_c

    If J0/J1 exist, interpolate Jacobian and compute v = J dq.
    Otherwise use capsule.linear_velocity if provided, else zero.
    """
    ref = capsule.p0
    J_c = None
    v_c = _zeros3_like(ref)

    if capsule.J0 is not None and capsule.J1 is not None:
        J_c = interpolate_segment_jacobian_torch(capsule.J0, capsule.J1, s)
        if dq is not None:
            if tuple(dq.shape) != (NUM_ARM_JOINTS,):
                raise ValueError(f"dq must have shape ({NUM_ARM_JOINTS},), got {tuple(dq.shape)}")
            v_c = J_c @ dq
            return J_c, v_c

    if capsule.linear_velocity is not None:
        v_c = _ensure_vec3(capsule.linear_velocity, "linear_velocity")

    return J_c, v_c


def barrier_value_from_rel_position_torch(
    p_rel: Tensor,
    radius_a: float,
    radius_b: float,
    d_margin: float = 0.0,
) -> Tensor:
    p_rel = _ensure_vec3(p_rel, "p_rel")
    r_eff = radius_a + radius_b + d_margin
    r_eff_t = _scalar_tensor(r_eff, p_rel)
    return torch.dot(p_rel, p_rel) - r_eff_t * r_eff_t


def surface_distance_from_rel_position_torch(
    p_rel: Tensor,
    radius_a: float,
    radius_b: float,
) -> Tensor:
    p_rel = _ensure_vec3(p_rel, "p_rel")
    return torch.linalg.norm(p_rel) - _scalar_tensor(radius_a + radius_b, p_rel)


def lf_h_torch(p_rel: Tensor, v_rel: Tensor) -> Tensor:
    p_rel = _ensure_vec3(p_rel, "p_rel")
    v_rel = _ensure_vec3(v_rel, "v_rel")
    return 2.0 * torch.dot(p_rel, v_rel)


def lg_psi_torch(J_closest: Tensor, p_rel: Tensor, num_arm_joints: int = NUM_ARM_JOINTS) -> Tensor:
    J_closest = _ensure_mat37(J_closest, "J_closest")
    p_rel = _ensure_vec3(p_rel, "p_rel")
    J_arm = J_closest[:, :num_arm_joints]
    return 2.0 * (p_rel.unsqueeze(0) @ J_arm).squeeze(0)


def compute_pair_geometry_torch(
    capsule_a: TorchCapsuleKinematics,
    capsule_b: TorchCapsuleKinematics,
    dq: Optional[Tensor] = None,
    d_margin: float = 0.0,
) -> Dict[str, Optional[Tensor]]:
    """
    Torch mirror of the current NumPy compute_pair_geometry().
    """
    c_a, c_b, s_a, s_b = closest_points_between_segments_torch(
        capsule_a.p0, capsule_a.p1,
        capsule_b.p0, capsule_b.p1,
    )

    p_rel = c_a - c_b
    dist_centers = torch.linalg.norm(p_rel)
    dist_surfaces = dist_centers - _scalar_tensor(capsule_a.radius + capsule_b.radius, p_rel)
    effective_radius = _scalar_tensor(capsule_a.radius + capsule_b.radius + d_margin, p_rel)
    h_val = barrier_value_from_rel_position_torch(
        p_rel,
        capsule_a.radius,
        capsule_b.radius,
        d_margin=d_margin,
    )

    J_a, v_a = point_kinematics_on_segment_torch(capsule_a, s_a, dq)
    J_b, v_b = point_kinematics_on_segment_torch(capsule_b, s_b, dq)

    J_rel = _optional_subtract(J_a, J_b)
    v_rel = v_a - v_b

    return {
        "c_a": c_a,
        "c_b": c_b,
        "s_a": s_a,
        "s_b": s_b,
        "p_rel": p_rel,
        "dist_centers": dist_centers,
        "dist_surfaces": dist_surfaces,
        "effective_radius": effective_radius,
        "h": h_val,
        "J_a": J_a,
        "J_b": J_b,
        "J_rel": J_rel,
        "v_a": v_a,
        "v_b": v_b,
        "v_rel": v_rel,
    }


def _make_obstacle_capsule(obs: Dict, ref: Tensor) -> TorchCapsuleKinematics:
    """
    Expected obstacle dict:
    {
        "pose_start": (3,) tensor or array-like
        "pose_end":   (3,) tensor or array-like
        "radius":     float
        "velocity":   (3,) tensor or array-like
    }
    """
    def _to_torch_vec3(x):
        if isinstance(x, torch.Tensor):
            t = x.to(dtype=ref.dtype, device=ref.device)
        else:
            t = torch.tensor(x, dtype=ref.dtype, device=ref.device)
        return _ensure_vec3(t, "obstacle vec3")

    return TorchCapsuleKinematics(
        p0=_to_torch_vec3(obs["pose_start"]),
        p1=_to_torch_vec3(obs["pose_end"]),
        radius=float(obs["radius"]),
        linear_velocity=_to_torch_vec3(obs["velocity"]),
        name=obs.get("name", ""),
    )


def _make_robot_capsule(link: Dict) -> TorchCapsuleKinematics:
    return TorchCapsuleKinematics(
        p0=_ensure_vec3(link["p0"], "link p0"),
        p1=_ensure_vec3(link["p1"], "link p1"),
        radius=float(link["radius"]),
        J0=_ensure_mat37(link["J0"], "link J0"),
        J1=_ensure_mat37(link["J1"], "link J1"),
        name=link["name"],
    )


def compute_all_pair_terms_torch(
    robot_links: List[Dict],
    obstacles: List[Dict],
    self_collision_link_pairs: Optional[List[Tuple[str, str]]] = None,
    dq: Optional[Tensor] = None,
    d_margin: float = 0.0,
) -> Dict[str, Tensor]:
    """
    Torch mirror of p12_rollout.compute_all_pair_terms().

    Fixed order:
    1) robot-obstacle rows:
       for each active link, obstacle slots 0..MAX_OBSTACLES-1
    2) self-collision rows:
       fixed order from self_collision_link_pairs
    """
    if self_collision_link_pairs is None:
        self_collision_link_pairs = SELF_COLLISION_LINK_PAIRS

    if len(robot_links) != len(ACTIVE_LINKS_DEF):
        raise ValueError(f"Expected {len(ACTIVE_LINKS_DEF)} robot links, got {len(robot_links)}")

    ref = robot_links[0]["p0"]

    rows_h: List[Tensor] = []
    rows_Lf_h: List[Tensor] = []
    rows_vrel_sq2: List[Tensor] = []
    rows_Lg: List[Tensor] = []
    rows_mask: List[Tensor] = []

    robot_capsules = [_make_robot_capsule(link) for link in robot_links]

    # ------------------------------------------------------------------
    # robot-obstacle rows
    # ------------------------------------------------------------------
    for robot_capsule in robot_capsules:
        for obs_idx in range(MAX_OBSTACLES):
            if obs_idx < len(obstacles):
                obs_capsule = _make_obstacle_capsule(obstacles[obs_idx], ref)

                pair = compute_pair_geometry_torch(
                    robot_capsule,
                    obs_capsule,
                    dq=dq,
                    d_margin=d_margin,
                )

                h = pair["h"]
                v_rel = pair["v_rel"]
                J_a = pair["J_a"]
                if J_a is None:
                    raise RuntimeError("Robot-obstacle pair unexpectedly has no J_a.")

                Lf_h_val = lf_h_torch(pair["p_rel"], v_rel)
                vrel_sq2_val = 2.0 * torch.dot(v_rel, v_rel)
                Lg_val = lg_psi_torch(J_a, pair["p_rel"], NUM_ARM_JOINTS)
                active = (h < _scalar_tensor(0.07, ref)).to(dtype=ref.dtype)

                rows_h.append(h)
                rows_Lf_h.append(Lf_h_val)
                rows_vrel_sq2.append(vrel_sq2_val)
                rows_Lg.append(Lg_val)
                rows_mask.append(active)
            else:
                rows_h.append(_scalar_tensor(0.0, ref))
                rows_Lf_h.append(_scalar_tensor(0.0, ref))
                rows_vrel_sq2.append(_scalar_tensor(0.0, ref))
                rows_Lg.append(_zeros7_like(ref))
                rows_mask.append(_scalar_tensor(0.0, ref))

    # ------------------------------------------------------------------
    # self-collision rows
    # ------------------------------------------------------------------
    capsule_by_name = {cap.name: cap for cap in robot_capsules}

    for link1, link2 in self_collision_link_pairs:
        if link1 not in capsule_by_name or link2 not in capsule_by_name:
            rows_h.append(_scalar_tensor(0.0, ref))
            rows_Lf_h.append(_scalar_tensor(0.0, ref))
            rows_vrel_sq2.append(_scalar_tensor(0.0, ref))
            rows_Lg.append(_zeros7_like(ref))
            rows_mask.append(_scalar_tensor(0.0, ref))
            continue

        pair = compute_pair_geometry_torch(
            capsule_by_name[link1],
            capsule_by_name[link2],
            dq=dq,
            d_margin=0.0,
        )

        h = pair["h"]
        v_rel = pair["v_rel"]
        J_rel = pair["J_rel"]
        if J_rel is None:
            raise RuntimeError("Self-collision pair unexpectedly has no J_rel.")

        Lf_h_val = lf_h_torch(pair["p_rel"], v_rel)
        vrel_sq2_val = 2.0 * torch.dot(v_rel, v_rel)
        Lg_val = lg_psi_torch(J_rel, pair["p_rel"], NUM_ARM_JOINTS)
        active = (h < _scalar_tensor(0.03, ref)).to(dtype=ref.dtype)

        rows_h.append(h)
        rows_Lf_h.append(Lf_h_val)
        rows_vrel_sq2.append(vrel_sq2_val)
        rows_Lg.append(Lg_val)
        rows_mask.append(active)

    h_all = torch.stack(rows_h, dim=0)
    Lf_h_all = torch.stack(rows_Lf_h, dim=0)
    vrel_sq2_all = torch.stack(rows_vrel_sq2, dim=0)
    Lg_psi_all = torch.stack(rows_Lg, dim=0)
    pair_mask = torch.stack(rows_mask, dim=0)

    if tuple(h_all.shape) != (NUM_PAIR_ROWS,):
        raise RuntimeError(f"h_all shape mismatch: {tuple(h_all.shape)}")
    if tuple(Lf_h_all.shape) != (NUM_PAIR_ROWS,):
        raise RuntimeError(f"Lf_h_all shape mismatch: {tuple(Lf_h_all.shape)}")
    if tuple(vrel_sq2_all.shape) != (NUM_PAIR_ROWS,):
        raise RuntimeError(f"vrel_sq2_all shape mismatch: {tuple(vrel_sq2_all.shape)}")
    if tuple(Lg_psi_all.shape) != (NUM_PAIR_ROWS, NUM_ARM_JOINTS):
        raise RuntimeError(f"Lg_psi_all shape mismatch: {tuple(Lg_psi_all.shape)}")
    if tuple(pair_mask.shape) != (NUM_PAIR_ROWS,):
        raise RuntimeError(f"pair_mask shape mismatch: {tuple(pair_mask.shape)}")

    return {
        "h_all": h_all,
        "Lf_h_all": Lf_h_all,
        "vrel_sq2_all": vrel_sq2_all,
        "Lg_psi_all": Lg_psi_all,
        "pair_mask": pair_mask,
    }


def compute_critical_pair_features_torch(
    robot_links: List[Dict],
    obstacles: List[Dict],
    self_collision_link_pairs: Optional[List[Tuple[str, str]]] = None,
    dq: Optional[Tensor] = None,
    d_margin: float = 0.0,
) -> Dict[str, Tensor]:
    """
    Torch mirror of p12_rollout.compute_critical_pair_features().
    """
    if self_collision_link_pairs is None:
        self_collision_link_pairs = SELF_COLLISION_LINK_PAIRS

    if len(robot_links) == 0:
        raise ValueError("robot_links must be non-empty")

    ref = robot_links[0]["p0"]
    robot_capsules = [_make_robot_capsule(link) for link in robot_links]
    capsule_by_name = {cap.name: cap for cap in robot_capsules}

    best = None

    # robot-obstacle pairs
    for robot_capsule in robot_capsules:
        for obs in obstacles:
            obs_capsule = _make_obstacle_capsule(obs, ref)

            pair = compute_pair_geometry_torch(
                robot_capsule,
                obs_capsule,
                dq=dq,
                d_margin=d_margin,
            )

            p_rel = pair["p_rel"]
            norm_p = torch.linalg.norm(p_rel)
            n_crit = p_rel / (norm_p + _scalar_tensor(1e-8, ref))

            d_crit = pair["dist_surfaces"] - _scalar_tensor(d_margin, ref)
            d_dot_crit = torch.dot(n_crit, pair["v_rel"])

            h_crit = pair["h"]
            Lf_h_crit = lf_h_torch(pair["p_rel"], pair["v_rel"])
            vrel_sq2_crit = 2.0 * torch.dot(pair["v_rel"], pair["v_rel"])
            J_a = pair["J_a"]
            if J_a is None:
                raise RuntimeError("Robot-obstacle critical pair unexpectedly has no J_a.")
            Lg_psi_crit = lg_psi_torch(J_a, pair["p_rel"], NUM_ARM_JOINTS)

            tau_crit = _scalar_tensor(0.0, ref)

            candidate = {
                "n_crit": n_crit,
                "d_crit": d_crit,
                "d_dot_crit": d_dot_crit,
                "tau_crit": tau_crit,
                "h_crit": h_crit,
                "Lf_h_crit": Lf_h_crit,
                "vrel_sq2_crit": vrel_sq2_crit,
                "Lg_psi_crit": Lg_psi_crit,
            }

            if best is None or float(candidate["d_crit"].detach().cpu()) < float(best["d_crit"].detach().cpu()):
                best = candidate

    # self-collision pairs
    for link1, link2 in self_collision_link_pairs:
        if link1 not in capsule_by_name or link2 not in capsule_by_name:
            continue

        pair = compute_pair_geometry_torch(
            capsule_by_name[link1],
            capsule_by_name[link2],
            dq=dq,
            d_margin=0.0,
        )

        p_rel = pair["p_rel"]
        norm_p = torch.linalg.norm(p_rel)
        n_crit = p_rel / (norm_p + _scalar_tensor(1e-8, ref))

        d_crit = pair["dist_surfaces"]
        d_dot_crit = torch.dot(n_crit, pair["v_rel"])

        h_crit = pair["h"]
        Lf_h_crit = lf_h_torch(pair["p_rel"], pair["v_rel"])
        vrel_sq2_crit = 2.0 * torch.dot(pair["v_rel"], pair["v_rel"])
        J_rel = pair["J_rel"]
        if J_rel is None:
            raise RuntimeError("Self-collision critical pair unexpectedly has no J_rel.")
        Lg_psi_crit = lg_psi_torch(J_rel, pair["p_rel"], NUM_ARM_JOINTS)

        tau_crit = _scalar_tensor(1.0, ref)

        candidate = {
            "n_crit": n_crit,
            "d_crit": d_crit,
            "d_dot_crit": d_dot_crit,
            "tau_crit": tau_crit,
            "h_crit": h_crit,
            "Lf_h_crit": Lf_h_crit,
            "vrel_sq2_crit": vrel_sq2_crit,
            "Lg_psi_crit": Lg_psi_crit,
        }

        if best is None or float(candidate["d_crit"].detach().cpu()) < float(best["d_crit"].detach().cpu()):
            best = candidate

    if best is None:
        return {
            "n_crit": _zeros3_like(ref),
            "d_crit": _scalar_tensor(0.0, ref),
            "d_dot_crit": _scalar_tensor(0.0, ref),
            "tau_crit": _scalar_tensor(0.0, ref),
            "h_crit": _scalar_tensor(0.0, ref),
            "Lf_h_crit": _scalar_tensor(0.0, ref),
            "vrel_sq2_crit": _scalar_tensor(0.0, ref),
            "Lg_psi_crit": _zeros7_like(ref),
        }

    return best


def obstacles_to_torch(
    obstacles: List[Dict],
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Convenience helper to convert scenario/rollout obstacle dicts into torch-friendly dicts.
    """
    out = []
    for obs in obstacles:
        out.append(
            {
                "pose_start": torch.tensor(obs["pose_start"], dtype=dtype, device=device),
                "pose_end": torch.tensor(obs["pose_end"], dtype=dtype, device=device),
                "radius": float(obs["radius"]),
                "velocity": torch.tensor(obs["velocity"], dtype=dtype, device=device),
                "name": obs.get("name", ""),
            }
        )
    return out