from __future__ import annotations

"""
Training-time input/bound-slack CVXPYLayer for the FR3 p12 / HOCBF safety filter.

Philosophy:
    - HOCBF / collision rows stay hard.
    - Joint acceleration / velocity / position bound rows get slack.
    - This mirrors the 2D mass-damper training idea more closely:
      the safe interval / HOCBF requirement stays fixed, but training can use
      extra input when the required safe input is outside the hard input limits.

QP:
    min_{u,s_b} 0.5 ||u - u_nom||^2 + 0.5 rho ||s_b||^2
    s.t.        A_pair  u       >= b_pair
                A_bound u + s_b >= b_bound
                s_b >= 0

Runtime/final evaluation should still use the original hard QP + fallback.
"""

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


# ---------------------------------------------------------------------
# layer construction
# ---------------------------------------------------------------------
def make_bound_slack_qp_layer(
    num_u: int,
    num_pair_rows: int,
    num_bound_rows: int,
    rho: float = 1e5,
) -> CvxpyLayer:
    """
    Build a reusable CVXPYLayer for one sample.

    Parameters are already row-normalized before being passed into this layer.
    """
    u = cp.Variable(num_u)
    s_bound = cp.Variable(num_bound_rows, nonneg=True)

    u_nom = cp.Parameter(num_u)
    A_pair = cp.Parameter((num_pair_rows, num_u))
    b_pair = cp.Parameter(num_pair_rows)
    A_bound = cp.Parameter((num_bound_rows, num_u))
    b_bound = cp.Parameter(num_bound_rows)

    objective = cp.Minimize(
        0.5 * cp.sum_squares(u - u_nom)
        + 0.5 * float(rho) * cp.sum_squares(s_bound)
    )

    constraints = [
        A_pair @ u >= b_pair,                 # hard HOCBF rows
        A_bound @ u + s_bound >= b_bound,     # soft bound/input rows
    ]

    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise RuntimeError("Bound-slack QP layer is not DPP.")

    return CvxpyLayer(
        problem,
        parameters=[u_nom, A_pair, b_pair, A_bound, b_bound],
        variables=[u, s_bound],
    )


# ---------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------
def _normalize_rows(A: torch.Tensor, b: torch.Tensor, eps: float):
    """
    Row-normalize linear inequalities A u >= b.

    A: (..., m, n)
    b: (..., m)
    """
    row_norm = torch.linalg.norm(A, dim=-1).clamp_min(float(eps))
    A_n = A / row_norm.unsqueeze(-1)
    b_n = b / row_norm
    return A_n, b_n, row_norm


def split_and_normalize_full_qp(
    A_full: torch.Tensor,
    b_full: torch.Tensor,
    num_pair_rows: int,
    eps: float = 1e-6,
):
    """
    Split full 85-row QP into pair rows and bound rows, then row-normalize both.

    A_full: (B, 85, 7)
    b_full: (B, 85)
    """
    A_pair = A_full[:, :num_pair_rows, :]
    b_pair = b_full[:, :num_pair_rows]
    A_bound = A_full[:, num_pair_rows:, :]
    b_bound = b_full[:, num_pair_rows:]

    A_pair_n, b_pair_n, pair_norm = _normalize_rows(A_pair, b_pair, eps=eps)
    A_bound_n, b_bound_n, bound_norm = _normalize_rows(A_bound, b_bound, eps=eps)

    return A_pair_n, b_pair_n, A_bound_n, b_bound_n, pair_norm, bound_norm


# ---------------------------------------------------------------------
# batched solve wrapper
# ---------------------------------------------------------------------
def solve_bound_slack_qp_batch(
    layer: CvxpyLayer,
    u_nom: torch.Tensor,
    A_full: torch.Tensor,
    b_full: torch.Tensor,
    num_pair_rows: int,
    eps: float = 1e-6,
):
    """
    Solve the bound-slack QP for a batch by looping over batch dimension.

    Returns:
        u_safe:       (B, 7)
        s_bound:      (B, num_bound_rows), normalized-row slack
        pair_norm:    (B, num_pair_rows)
        bound_norm:   (B, num_bound_rows)
    """
    if u_nom.ndim != 2:
        raise ValueError(f"u_nom must be (B, n), got {tuple(u_nom.shape)}")
    if A_full.ndim != 3:
        raise ValueError(f"A_full must be (B, m, n), got {tuple(A_full.shape)}")
    if b_full.ndim != 2:
        raise ValueError(f"b_full must be (B, m), got {tuple(b_full.shape)}")

    A_pair_n, b_pair_n, A_bound_n, b_bound_n, pair_norm, bound_norm = split_and_normalize_full_qp(
        A_full=A_full,
        b_full=b_full,
        num_pair_rows=num_pair_rows,
        eps=eps,
    )

    B = int(u_nom.shape[0])
    u_list = []
    s_list = []

    for i in range(B):
        u_i, s_i = layer(
            u_nom[i],
            A_pair_n[i],
            b_pair_n[i],
            A_bound_n[i],
            b_bound_n[i],
        )
        u_list.append(u_i)
        s_list.append(s_i)

    u_safe = torch.stack(u_list, dim=0)
    s_bound = torch.stack(s_list, dim=0)
    return u_safe, s_bound, pair_norm, bound_norm


# ---------------------------------------------------------------------
# losses / diagnostics
# ---------------------------------------------------------------------
def hard_violation_batch(
    u: torch.Tensor,
    A_full_raw: torch.Tensor,
    b_full_raw: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample max hard QP violation on the original unnormalized rows.

    Returns:
        violation: (B,)
    """
    lhs = torch.einsum("bmn,bn->bm", A_full_raw, u)
    violation = torch.clamp(b_full_raw - lhs, min=0.0)
    return violation.max(dim=1).values


def pair_violation_batch(
    u: torch.Tensor,
    A_full_raw: torch.Tensor,
    b_full_raw: torch.Tensor,
    num_pair_rows: int,
) -> torch.Tensor:
    """
    Per-sample max HOCBF/pair-row violation on the original unnormalized pair rows.
    This should be tiny because pair rows are hard in this layer.
    """
    A_pair = A_full_raw[:, :num_pair_rows, :]
    b_pair = b_full_raw[:, :num_pair_rows]
    lhs = torch.einsum("bmn,bn->bm", A_pair, u)
    violation = torch.clamp(b_pair - lhs, min=0.0)
    return violation.max(dim=1).values


def bound_violation_batch(
    u: torch.Tensor,
    A_full_raw: torch.Tensor,
    b_full_raw: torch.Tensor,
    num_pair_rows: int,
) -> torch.Tensor:
    """
    Per-sample max bound-row violation on the original unnormalized bound rows.
    Bound rows are softened during training, so this can be nonzero.
    """
    A_bound = A_full_raw[:, num_pair_rows:, :]
    b_bound = b_full_raw[:, num_pair_rows:]
    lhs = torch.einsum("bmn,bn->bm", A_bound, u)
    violation = torch.clamp(b_bound - lhs, min=0.0)
    return violation.max(dim=1).values


def mean_sq_bound_slack(s_bound: torch.Tensor) -> torch.Tensor:
    return (s_bound ** 2).mean()


# ---------------------------------------------------------------------
# Aliases for backward compatibility with pair_slack naming
# ---------------------------------------------------------------------
make_pair_slack_qp_layer = make_bound_slack_qp_layer
solve_pair_slack_qp_batch = solve_bound_slack_qp_batch
mean_sq_pair_slack = mean_sq_bound_slack
