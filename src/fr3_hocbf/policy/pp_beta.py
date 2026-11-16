#!/usr/bin/env python3
from __future__ import annotations

"""
Per-pair beta policy for FR3 HOCBF rows ("NN per-pair beta").

The existing P12ParamNet predicts one shared gamma and one shared beta. Here
gamma stays a shared scalar from a trunk identical to P12ParamNet, while beta
becomes a vector beta_j, one per collision pair row j, predicted by a small MLP
with weights shared across pairs:

    gamma, c  = trunk(z_norm)                       z_norm: (B, 114), c: (B, h4)
    x_j       = pair features of row j (uses gamma) x:      (B, 43, 8)
    beta_j    = floor + (cap - floor) * sigmoid(head([c, x_j_norm, active_j, type_j, emb_j]))

The HOCBF row is unchanged except for the index:
    A_j = Lg_psi_j,  b_j = -(vrel_sq2_j + gamma*Lf_h_j) - beta_j*psi_j

Inactive rows are zeroed in the QP as before, so their beta_j is unused.
"""

import numpy as np
import torch
import torch.nn as nn

from fr3_hocbf.sim.qp_data import NUM_ARM_JOINTS, NUM_PAIR_ROWS

NUM_ROBOT_OBSTACLE_ROWS = 40  # 8 active links * 5 obstacle slots; rows 40..42 are self-collision
PP_FEATURE_NAMES = (
    "h",
    "Lf_h",
    "vrel_sq2",
    "psi",          # Lf_h + gamma*h
    "Lf_psi",       # vrel_sq2 + gamma*Lf_h
    "S_sup",        # sup_{u in U_kin} Lg_psi u
    "Lg_psi_u_nom",
    "Lg_psi_norm",
)
PP_FEATURE_DIM = len(PP_FEATURE_NAMES)
PP_ARCH = "pp_beta_v1"


def pair_type_onehot() -> torch.Tensor:
    """(43, 2): column 0 = robot-obstacle row, column 1 = self-collision row."""
    t = torch.zeros(NUM_PAIR_ROWS, 2, dtype=torch.double)
    t[:NUM_ROBOT_OBSTACLE_ROWS, 0] = 1.0
    t[NUM_ROBOT_OBSTACLE_ROWS:, 1] = 1.0
    return t


def kinematic_ddq_bounds_torch(q_arm, dq_arm, limits_t: dict, dt: float):
    """Torch version of kinematic_ddq_bounds in fr3_hocbf/evaluation/compare_rollout_policies.py. q_arm, dq_arm: (B, 7)."""
    ddq_max_from_vel = (limits_t["dq_max_arm"] - dq_arm) / dt
    ddq_min_from_vel = (limits_t["dq_min_arm"] - dq_arm) / dt
    ddq_max_from_pos = 2.0 * (limits_t["q_max_arm"] - q_arm - dq_arm * dt) / (dt ** 2)
    ddq_min_from_pos = 2.0 * (limits_t["q_min_arm"] - q_arm - dq_arm * dt) / (dt ** 2)
    ddq_max = torch.minimum(limits_t["ddq_max_arm"].expand_as(q_arm), torch.minimum(ddq_max_from_vel, ddq_max_from_pos))
    ddq_min = torch.maximum(limits_t["ddq_min_arm"].expand_as(q_arm), torch.maximum(ddq_min_from_vel, ddq_min_from_pos))
    return ddq_min, ddq_max


def pp_features_torch(
    gamma: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
    vrel_sq2_all: torch.Tensor,
    Lg_psi_all: torch.Tensor,
    u_nom: torch.Tensor,
    ddq_min_kin: torch.Tensor,
    ddq_max_kin: torch.Tensor,
) -> torch.Tensor:
    """
    Raw per-pair features, (B, 43, 8) in PP_FEATURE_NAMES order.

    Shapes: gamma (B, 1), h/Lf_h/vrel_sq2 (B, 43), Lg_psi (B, 43, 7),
            u_nom/ddq_min_kin/ddq_max_kin (B, 7).
    """
    psi = Lf_h_all + gamma * h_all
    Lf_psi = vrel_sq2_all + gamma * Lf_h_all
    S_sup = torch.where(
        Lg_psi_all >= 0.0,
        Lg_psi_all * ddq_max_kin.unsqueeze(1),
        Lg_psi_all * ddq_min_kin.unsqueeze(1),
    ).sum(dim=-1)
    Lg_u_nom = torch.einsum("bpj,bj->bp", Lg_psi_all, u_nom)
    Lg_norm = torch.linalg.norm(Lg_psi_all, dim=-1)
    return torch.stack([h_all, Lf_h_all, vrel_sq2_all, psi, Lf_psi, S_sup, Lg_u_nom, Lg_norm], dim=-1)


class PerPairBetaNet(nn.Module):
    """
    Shared-gamma trunk (same layers as P12ParamNet) + shared-weight per-pair beta head.

    forward(z_norm, pp) -> (gamma (B, 1), beta (B, 43))
    pp is a dict of batched tensors: h_all, Lf_h_all, vrel_sq2_all, Lg_psi_all,
    pair_mask, u_nom, ddq_min_kin, ddq_max_kin.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        pair_embed_dim: int = 8,
        head_hidden: int = 64,
        p1_floor: float = 1e-3,
        p2_floor: float = 1e-3,
        p1_max: float = 200.0,
        p2_max: float = 250.0,
        init_p1: float = 10.0,
        init_p2: float = 15.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.p1_floor = float(p1_floor)
        self.p2_floor = float(p2_floor)
        self.p1_max = float(p1_max)
        self.p2_max = float(p2_max)

        h1 = int(hidden_dim)
        h2 = max(64, h1)
        h3 = max(64, h1 // 2)
        h4 = max(32, h1 // 4)
        self.context_dim = h4

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, h3),
            nn.LayerNorm(h3),
            nn.SiLU(),
            nn.Linear(h3, h4),
            nn.SiLU(),
        )
        self.gamma_layer = nn.Linear(h4, 1)

        self.pair_embed = nn.Embedding(NUM_PAIR_ROWS, int(pair_embed_dim))
        head_in = h4 + PP_FEATURE_DIM + 1 + 2 + int(pair_embed_dim)
        self.beta_out = nn.Linear(int(head_hidden), 1)
        self.beta_head = nn.Sequential(
            nn.Linear(head_in, int(head_hidden)),
            nn.SiLU(),
            nn.Linear(int(head_hidden), int(head_hidden)),
            nn.SiLU(),
            self.beta_out,
        )

        self.register_buffer("pair_type", pair_type_onehot())
        self.register_buffer("pair_index", torch.arange(NUM_PAIR_ROWS, dtype=torch.long))
        # Per-pair feature normalization, filled by set_pp_stats().
        self.register_buffer("pp_mean", torch.zeros(PP_FEATURE_DIM, dtype=torch.double))
        self.register_buffer("pp_std", torch.ones(PP_FEATURE_DIM, dtype=torch.double))

        self.double()
        self._initialize_outputs(float(init_p1), float(init_p2))

    def set_pp_stats(self, mean, std):
        with torch.no_grad():
            self.pp_mean.copy_(torch.as_tensor(mean, dtype=torch.double))
            self.pp_std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.double), min=1e-6))

    @staticmethod
    def _target_to_logit(target: float, floor: float, cap: float) -> float:
        y = (float(target) - float(floor)) / (float(cap) - float(floor))
        y = min(max(y, 1e-6), 1.0 - 1e-6)
        return float(np.log(y / (1.0 - y)))

    def _initialize_outputs(self, init_p1: float, init_p2: float):
        """Shift final biases so z_norm=0 and x_pp_norm=0 give gamma≈init_p1, mean beta≈init_p2."""
        with torch.no_grad():
            z0 = torch.zeros(1, self.input_dim, dtype=torch.double)
            c0 = self.trunk(z0)
            g_raw = self.gamma_layer(c0).view(-1)
            self.gamma_layer.bias.add_(self._target_to_logit(init_p1, self.p1_floor, self.p1_max) - g_raw)

            x0 = torch.zeros(1, NUM_PAIR_ROWS, PP_FEATURE_DIM, dtype=torch.double)
            active0 = torch.ones(1, NUM_PAIR_ROWS, dtype=torch.double)
            b_raw = self._beta_raw(c0, x0, active0).mean()
            self.beta_out.bias.add_(self._target_to_logit(init_p2, self.p2_floor, self.p2_max) - b_raw)

    def _beta_raw(self, ctx, x_norm, active):
        B = ctx.shape[0]
        head_in = torch.cat(
            [
                ctx.unsqueeze(1).expand(B, NUM_PAIR_ROWS, ctx.shape[-1]),
                x_norm * active.unsqueeze(-1),  # inactive rows see zeros; their beta is unused
                active.unsqueeze(-1),
                self.pair_type.unsqueeze(0).expand(B, -1, -1),
                self.pair_embed(self.pair_index).unsqueeze(0).expand(B, -1, -1),
            ],
            dim=-1,
        )
        return self.beta_head(head_in).squeeze(-1)

    def forward(self, z_norm: torch.Tensor, pp: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.trunk(z_norm)
        gamma = self.p1_floor + (self.p1_max - self.p1_floor) * torch.sigmoid(self.gamma_layer(ctx))

        x_raw = pp_features_torch(
            gamma=gamma,
            h_all=pp["h_all"],
            Lf_h_all=pp["Lf_h_all"],
            vrel_sq2_all=pp["vrel_sq2_all"],
            Lg_psi_all=pp["Lg_psi_all"],
            u_nom=pp["u_nom"],
            ddq_min_kin=pp["ddq_min_kin"],
            ddq_max_kin=pp["ddq_max_kin"],
        )
        x_norm = (x_raw - self.pp_mean) / self.pp_std
        active = (pp["pair_mask"] > 0.5).to(x_norm.dtype)
        beta = self.p2_floor + (self.p2_max - self.p2_floor) * torch.sigmoid(self._beta_raw(ctx, x_norm, active))
        return gamma, beta


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


def load_pp_policy(path: str) -> PerPairBetaNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") != PP_ARCH:
        raise ValueError(f"{path} is not a {PP_ARCH} checkpoint (arch={ckpt.get('arch')})")
    model = PerPairBetaNet(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        pair_embed_dim=int(ckpt["pair_embed_dim"]),
        head_hidden=int(ckpt["pp_head_hidden"]),
        p1_floor=float(ckpt["p1_floor"]),
        p2_floor=float(ckpt["p2_floor"]),
        p1_max=float(ckpt["p1_max"]),
        p2_max=float(ckpt["p2_max"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, np.asarray(ckpt["feat_mean"], dtype=float), np.asarray(ckpt["feat_std"], dtype=float)
