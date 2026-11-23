"""Parallel segment evaluation, LR schedule and weight EMA for train_rollout (experimental flags).

SegmentPool runs train_one_segment in worker processes. Each job gets the current weights and one
segment spec (scenario path, start state) and returns the flattened weight gradient (training) and the
segment stats. Results come back in job order, so a run is reproducible for a fixed number of segments
per update, independent of the worker count and scheduling.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp

import numpy as np
import torch

_W: dict = {}


def _init_worker(model, cfg, feat_mean, feat_std, fast_kinematics: bool):
    # Imports here so that the module can be imported by train_rollout without a cycle.
    from fr3_hocbf.differentiable.kinematics import load_fr3_torch_kinematics
    from fr3_hocbf.differentiable.kinematics_fast import load_fr3_torch_kinematics_fast
    from fr3_hocbf.differentiable.qp_layer import make_bound_slack_qp_layer
    from fr3_hocbf.sim.qp_data import NUM_BOUND_ROWS, NUM_PAIR_ROWS, get_joint_limits
    from fr3_hocbf.sim.rollout import NUM_ARM_JOINTS, make_rollout_context

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    _W.update(
        model=model,
        cfg=cfg,
        feat_mean=feat_mean,
        feat_std=feat_std,
        pin_context=make_rollout_context(),
        torch_kin=load_fr3_torch_kinematics_fast() if fast_kinematics else load_fr3_torch_kinematics(),
        layer=make_bound_slack_qp_layer(num_u=NUM_ARM_JOINTS, num_pair_rows=NUM_PAIR_ROWS,
                                        num_bound_rows=NUM_BOUND_ROWS, rho=cfg.rho),
        limits_t={k: torch.tensor(v, dtype=torch.double) for k, v in get_joint_limits().items()},
    )


def _run_job(job):
    from fr3_hocbf.train.train_rollout import train_one_segment

    state, scenario_path, start_state, train = job
    model = _W["model"]
    model.load_state_dict(state)
    stats = train_one_segment(
        scenario_path=scenario_path, pin_context=_W["pin_context"], torch_kin=_W["torch_kin"], model=model,
        bound_slack_layer=_W["layer"], optimizer=None, feat_mean=_W["feat_mean"], feat_std=_W["feat_std"],
        limits_t=_W["limits_t"], cfg=_W["cfg"], train=train, start_state=start_state,
    )
    grad = None
    if train and not stats.get("solver_failed", False):
        grad = flat_grad(model)
    return grad, stats


def flat_grad(model) -> np.ndarray:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1) for p in model.parameters()
    ]).cpu().numpy()


def set_flat_grad(model, g: np.ndarray):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = torch.as_tensor(g[i:i + n], dtype=p.dtype).reshape(p.shape).clone()
        i += n


class SegmentPool:
    def __init__(self, workers: int, model, cfg, feat_mean, feat_std, fast_kinematics: bool):
        ctx = mp.get_context("fork")
        self.pool = ctx.Pool(
            processes=int(workers), initializer=_init_worker,
            initargs=(copy.deepcopy(model), cfg, feat_mean, feat_std, bool(fast_kinematics)),
        )

    def run(self, model, specs: list[tuple], train: bool) -> list[tuple]:
        """specs: list of (scenario_path, start_state). Returns [(grad or None, stats)] in spec order."""
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return self.pool.map(_run_job, [(state, sp, st, train) for sp, st in specs], chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


def apply_batch_update(model, optimizer, results: list[tuple], grad_clip: float, seg_clip: float = 0.0) -> int:
    """Average the gradients of the valid segments (each first clipped to seg_clip if > 0), clip the mean,
    step. Non-finite segment gradients are dropped. Returns the number of segments used."""
    grads = [g for g, _ in results if g is not None and np.all(np.isfinite(g))]
    if not grads:
        return 0
    if seg_clip > 0.0:
        grads = [g * min(1.0, seg_clip / max(float(np.linalg.norm(g)), 1e-300)) for g in grads]
    optimizer.zero_grad()
    set_flat_grad(model, np.mean(np.stack(grads), axis=0))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    return len(grads)


def make_lr_scheduler(optimizer, kind: str, total_updates: int, lr_min: float):
    if kind == "constant":
        return None
    if kind == "cosine":
        lr0 = optimizer.param_groups[0]["lr"]
        ratio = lr_min / lr0

        def f(step):
            t = min(step, total_updates) / max(1, total_updates)
            return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    raise ValueError(kind)


class WeightEMA:
    """Exponential moving average of the weights; swap() exchanges live and averaged weights."""

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        for s, p in zip(self.shadow, model.parameters()):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)
