from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


Array = np.ndarray
EPS = 1e-9


def _as_vec3(x) -> Array:
    arr = np.asarray(x, dtype=float).reshape(3)
    return arr


def _as_matrix(x) -> Array:
    return np.asarray(x, dtype=float)


def _optional_subtract(a: Optional[Array], b: Optional[Array]) -> Optional[Array]:
    if a is None and b is None:
        return None
    if a is None:
        return -np.asarray(b, dtype=float)
    if b is None:
        return np.asarray(a, dtype=float)
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


@dataclass
class CapsuleKinematics:
    """
    World-frame capsule / segment description.

    p0, p1:
        Segment endpoints in world frame.

    radius:
        Capsule radius.

    J0, J1:
        Optional endpoint Jacobians (3 x nv) in world frame.
        Use these for robot links.

    linear_velocity:
        Optional constant world-frame linear velocity for the whole segment.
        Use this for obstacles if needed.
    """
    p0: Array
    p1: Array
    radius: float
    J0: Optional[Array] = None
    J1: Optional[Array] = None
    linear_velocity: Optional[Array] = None
    name: str = ""

    def __post_init__(self) -> None:
        self.p0 = _as_vec3(self.p0)
        self.p1 = _as_vec3(self.p1)
        self.radius = float(self.radius)

        if self.J0 is not None:
            self.J0 = _as_matrix(self.J0)
        if self.J1 is not None:
            self.J1 = _as_matrix(self.J1)
        if self.linear_velocity is not None:
            self.linear_velocity = _as_vec3(self.linear_velocity)


@dataclass
class PairGeometry:
    """
    Geometry / kinematics bundle for one capsule-capsule pair.
    """
    c_a: Array
    c_b: Array
    s_a: float
    s_b: float

    p_rel: Array
    dist_centers: float
    dist_surfaces: float
    effective_radius: float
    h: float

    J_a: Optional[Array]
    J_b: Optional[Array]
    J_rel: Optional[Array]

    v_a: Array
    v_b: Array
    v_rel: Array


def closest_points_between_segments(
    p1: Array,
    p2: Array,
    q1: Array,
    q2: Array,
    eps: float = EPS,
) -> Tuple[Array, Array, float, float]:
    """
    Robust closest-points solver for two 3D line segments.

    Returns:
        c1, c2, s, t

    where:
        c1 = p1 + s * (p2 - p1),  s in [0, 1]
        c2 = q1 + t * (q2 - q1),  t in [0, 1]

    This version handles:
        - general skew segments
        - parallel segments
        - degenerate point-segments
        - point-point case
    """
    p1 = _as_vec3(p1)
    p2 = _as_vec3(p2)
    q1 = _as_vec3(q1)
    q2 = _as_vec3(q2)

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b

    # both segments degenerate to points
    if a <= eps and c <= eps:
        s = 0.0
        t = 0.0
        c1 = p1.copy()
        c2 = q1.copy()
        return c1, c2, s, t

    # first segment degenerates to a point
    if a <= eps:
        s = 0.0
        t = np.clip(e / c, 0.0, 1.0) if c > eps else 0.0
        c1 = p1.copy()
        c2 = q1 + t * v
        return c1, c2, s, float(t)

    # second segment degenerates to a point
    if c <= eps:
        t = 0.0
        s = np.clip(-d / a, 0.0, 1.0)
        c1 = p1 + s * u
        c2 = q1.copy()
        return c1, c2, float(s), t

    # general case
    sN = 0.0
    sD = D
    tN = 0.0
    tD = D

    if D <= eps:
        # nearly parallel
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)

        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    s = 0.0 if abs(sN) <= eps else sN / sD
    t = 0.0 if abs(tN) <= eps else tN / tD

    s = float(np.clip(s, 0.0, 1.0))
    t = float(np.clip(t, 0.0, 1.0))

    c1 = p1 + s * u
    c2 = q1 + t * v
    return c1, c2, s, t


def interpolate_point_on_segment(p0: Array, p1: Array, s: float) -> Array:
    p0 = _as_vec3(p0)
    p1 = _as_vec3(p1)
    return (1.0 - s) * p0 + s * p1


def interpolate_segment_jacobian(J0: Array, J1: Array, s: float) -> Array:
    J0 = _as_matrix(J0)
    J1 = _as_matrix(J1)
    if J0.shape != J1.shape:
        raise ValueError(f"Jacobian shape mismatch: {J0.shape} vs {J1.shape}")
    return (1.0 - s) * J0 + s * J1


def point_kinematics_on_segment(
    capsule: CapsuleKinematics,
    s: float,
    dq: Optional[Array] = None,
) -> Tuple[Optional[Array], Array]:
    """
    Returns:
        J_c, v_c

    J_c is None if no endpoint Jacobians were provided.
    v_c is always returned. If no motion information is given, it defaults to zero.
    """
    J_c = None
    v_c = np.zeros(3, dtype=float)

    if capsule.J0 is not None and capsule.J1 is not None:
        J_c = interpolate_segment_jacobian(capsule.J0, capsule.J1, s)
        if dq is not None:
            v_c = J_c @ np.asarray(dq, dtype=float)
            return J_c, v_c

    if capsule.linear_velocity is not None:
        v_c = capsule.linear_velocity.copy()

    return J_c, v_c


def barrier_value_from_rel_position(
    p_rel: Array,
    radius_a: float,
    radius_b: float,
    d_margin: float = 0.0,
) -> float:
    p_rel = np.asarray(p_rel, dtype=float)
    r_eff = float(radius_a) + float(radius_b) + float(d_margin)
    return float(np.dot(p_rel, p_rel) - r_eff * r_eff)


def surface_distance_from_rel_position(
    p_rel: Array,
    radius_a: float,
    radius_b: float,
) -> float:
    p_rel = np.asarray(p_rel, dtype=float)
    return float(np.linalg.norm(p_rel) - (float(radius_a) + float(radius_b)))


def compute_pair_geometry(
    capsule_a: CapsuleKinematics,
    capsule_b: CapsuleKinematics,
    dq: Optional[Array] = None,
    d_margin: float = 0.0,
) -> PairGeometry:
    """
    Generic pair extractor for:
        - robot link vs obstacle
        - robot link vs robot link
        - point-sphere / point-capsule special cases

    If endpoint Jacobians are present, closest-point Jacobians are interpolated.
    If dq is given, closest-point velocities are computed from those Jacobians.
    Otherwise linear_velocity is used when provided.
    """
    c_a, c_b, s_a, s_b = closest_points_between_segments(
        capsule_a.p0, capsule_a.p1,
        capsule_b.p0, capsule_b.p1,
    )

    p_rel = c_a - c_b
    dist_centers = float(np.linalg.norm(p_rel))
    dist_surfaces = dist_centers - (capsule_a.radius + capsule_b.radius)
    effective_radius = capsule_a.radius + capsule_b.radius + d_margin
    h_val = barrier_value_from_rel_position(
        p_rel,
        capsule_a.radius,
        capsule_b.radius,
        d_margin=d_margin,
    )

    J_a, v_a = point_kinematics_on_segment(capsule_a, s_a, dq)
    J_b, v_b = point_kinematics_on_segment(capsule_b, s_b, dq)

    J_rel = _optional_subtract(J_a, J_b)
    v_rel = v_a - v_b

    return PairGeometry(
        c_a=c_a,
        c_b=c_b,
        s_a=s_a,
        s_b=s_b,
        p_rel=p_rel,
        dist_centers=dist_centers,
        dist_surfaces=dist_surfaces,
        effective_radius=effective_radius,
        h=h_val,
        J_a=J_a,
        J_b=J_b,
        J_rel=J_rel,
        v_a=v_a,
        v_b=v_b,
        v_rel=v_rel,
    )