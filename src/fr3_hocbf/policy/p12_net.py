"""Shared-gain policy: P12ParamNet predicts one gamma (p1) and one beta (p2) per control step."""

import numpy as np
import torch
import torch.nn as nn


class P12ParamNet(nn.Module):
    """
    Deeper NN for FR3 p12/gamma-beta prediction.

    Input is already normalized outside the model.
    Output is bounded positive:
        p1 in [p1_floor, p1_max]
        p2 in [p2_floor, p2_max]

    The final layer is initialized so that a zero normalized input predicts
    approximately the original/Davide baseline gains, e.g. p1=10, p2=15.
    This avoids starting at the middle of the output caps, which would be
    around p1=100, p2=125 for caps 200/250.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
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
        self.init_p1 = float(init_p1)
        self.init_p2 = float(init_p2)

        h1 = int(hidden_dim)
        h2 = max(64, h1)
        h3 = max(64, h1 // 2)
        h4 = max(32, h1 // 4)

        self.output_layer = nn.Linear(h4, 2)
        self.net = nn.Sequential(
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
            self.output_layer,
        )

        self.initialize_output_to_nominal(self.init_p1, self.init_p2)

    @staticmethod
    def _target_to_logit(target: float, floor: float, cap: float) -> float:
        """Convert desired bounded output value into the required sigmoid logit."""
        if cap <= floor:
            raise ValueError(f"cap must be larger than floor, got floor={floor}, cap={cap}")
        y = (float(target) - float(floor)) / (float(cap) - float(floor))
        y = min(max(y, 1e-6), 1.0 - 1e-6)
        return float(np.log(y / (1.0 - y)))

    def initialize_output_to_nominal(self, init_p1: float = 10.0, init_p2: float = 15.0):
        """
        Shift the final-layer bias so z_norm=0 gives the desired nominal p1,p2.

        We do not zero the final-layer weights, because keeping random weights
        gives the hidden layers gradients immediately. Instead, we compensate the
        current raw output at z_norm=0 by shifting only the final bias.
        """
        target_logits = torch.tensor(
            [
                self._target_to_logit(init_p1, self.p1_floor, self.p1_max),
                self._target_to_logit(init_p2, self.p2_floor, self.p2_max),
            ],
            dtype=self.output_layer.bias.dtype,
            device=self.output_layer.bias.device,
        )

        with torch.no_grad():
            z0 = torch.zeros(1, self.input_dim, dtype=self.output_layer.bias.dtype, device=self.output_layer.bias.device)
            raw0 = self.net(z0).view(-1)
            self.output_layer.bias.add_(target_logits - raw0)

    def forward(self, z_norm: torch.Tensor) -> torch.Tensor:
        raw_out = self.net(z_norm)

        p1 = self.p1_floor + (self.p1_max - self.p1_floor) * torch.sigmoid(raw_out[:, 0:1])
        p2 = self.p2_floor + (self.p2_max - self.p2_floor) * torch.sigmoid(raw_out[:, 1:2])

        return torch.cat([p1, p2], dim=1)
