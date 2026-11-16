"""114-D input features of the gain policies and their normalisation.

Feature order: q_arm (7), dq_arm (7), ddq_nominal (7), ddq_nominal / ddq_scale (7), h_all (43), Lf_h_all (43).
"""

import os

import numpy as np
import torch

from fr3_hocbf.sim.qp_data import NUM_ARM_JOINTS, NUM_PAIR_ROWS, get_joint_limits


INPUT_DIM = 7 + 7 + 7 + 7 + NUM_PAIR_ROWS + NUM_PAIR_ROWS  # q, dq, ddq_nominal, ddq_nominal/ddq_scale, h_all, Lf_h_all = 114


def nominal_ddq_scale_numpy() -> np.ndarray:
    """
    Per-joint nominal-acceleration normalization scale.

    This is the 3D analogue of the 2D u_nom / u_max feature.  Because the FR3
    acceleration limits may be asymmetric, we use the larger absolute limit per
    joint:
        scale_j = max(|ddq_min_j|, |ddq_max_j|)
    """
    limits = get_joint_limits()
    scale = np.maximum(
        np.abs(np.asarray(limits["ddq_min_arm"], dtype=float)),
        np.abs(np.asarray(limits["ddq_max_arm"], dtype=float)),
    )
    return np.clip(scale.reshape(NUM_ARM_JOINTS), 1e-6, None)


def build_raw_feature_matrix(data):
    """
    Build the 114D feature matrix from saved rollout NPZ data.

    Feature order:
        q_arm,
        dq_arm,
        ddq_nominal_arm,
        ddq_nominal_arm / ddq_scale,
        h_all,
        Lf_h_all

    The normalized nominal acceleration gives the NN direct information about
    how large the nominal input request is relative to the available joint
    acceleration authority, analogous to the 2D u_nom/u_max feature.

    Important: this intentionally uses h_all and Lf_h_all directly. It does not
    use the old d_all = h_all + d_margin feature, because h_all is a squared
    distance barrier and d_margin is a distance.
    """
    q_arm = data["q_arm"]
    dq_arm = data["dq_arm"]
    ddq_nominal_arm = data["ddq_nominal_arm"]
    ddq_nominal_norm = ddq_nominal_arm / nominal_ddq_scale_numpy()[None, :]
    h_all = data["h_all"]
    Lf_h_all = data["Lf_h_all"]

    z_raw = np.concatenate(
        [
            q_arm,
            dq_arm,
            ddq_nominal_arm,
            ddq_nominal_norm,
            h_all,
            Lf_h_all,
        ],
        axis=1,
    )
    return z_raw


def compute_feature_stats(z_raw):
    feat_mean = z_raw.mean(axis=0)
    feat_std = z_raw.std(axis=0)
    feat_std = np.clip(feat_std, 1e-6, None)
    return feat_mean, feat_std


def normalize_features_numpy(z_raw, mean, std):
    std_safe = np.clip(std, 1e-6, None)
    return (z_raw - mean) / std_safe


def load_feature_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """114-D feature mean/std from a stats .npz, or embedded in a training checkpoint .pt."""
    if path.endswith(".pt"):
        ckpt = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
        mean_t = torch.as_tensor(ckpt["feat_mean"], dtype=torch.double)
        std_t = torch.clamp(torch.as_tensor(ckpt["feat_std"], dtype=torch.double), min=1e-6)
        if mean_t.numel() != INPUT_DIM:
            raise ValueError(f"Expected feature dim {INPUT_DIM}, got {mean_t.numel()} from {path}")
        return mean_t, std_t

    stats = np.load(os.path.expanduser(path))
    if "feat_mean" in stats.files:
        mean = stats["feat_mean"]
        std = stats["feat_std"]
    elif "mean" in stats.files:
        mean = stats["mean"]
        std = stats["std"]
    else:
        raise KeyError(f"Could not find feat_mean/feat_std in {path}")

    mean_t = torch.tensor(mean, dtype=torch.double)
    std_t = torch.tensor(std, dtype=torch.double)
    std_t = torch.clamp(std_t, min=1e-6)

    if mean_t.numel() != INPUT_DIM:
        raise ValueError(f"Expected feature dim {INPUT_DIM}, got {mean_t.numel()} from {path}")

    return mean_t, std_t


def normalize_features_torch(z_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (z_raw - mean) / torch.clamp(std, min=1e-6)


def build_feature_vector(
    q_arm: torch.Tensor,
    dq_arm: torch.Tensor,
    ddq_nominal: torch.Tensor,
    ddq_nominal_scale: torch.Tensor,
    h_all: torch.Tensor,
    Lf_h_all: torch.Tensor,
) -> torch.Tensor:
    """
    Build the 114D NN feature vector.

    Feature order:
        q_arm,
        dq_arm,
        ddq_nominal,
        ddq_nominal / ddq_scale,
        h_all,
        Lf_h_all
    """
    ddq_nominal_norm = ddq_nominal / torch.clamp(ddq_nominal_scale, min=1e-6)
    z = torch.cat([q_arm, dq_arm, ddq_nominal, ddq_nominal_norm, h_all, Lf_h_all], dim=0)
    if tuple(z.shape) != (INPUT_DIM,):
        raise RuntimeError(f"Feature shape mismatch: {tuple(z.shape)}")
    return z.unsqueeze(0)
