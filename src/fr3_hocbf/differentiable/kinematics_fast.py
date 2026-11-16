#!/usr/bin/env python3
"""
Faster drop-in for FR3TorchKinematics (fr3_hocbf/differentiable/kinematics.py).

The original computes each of the 16 active-link endpoint Jacobians with its own
torch.autograd.functional.jacobian call, i.e. one full forward-kinematics pass
plus 3 backward passes per endpoint. This subclass runs forward kinematics
once per call and uses the analytic geometric position Jacobian of a revolute
chain:

    J_f[:, i] = z_i x (p_f - o_i)   if arm joint i lies on the path root -> f
              = 0                   otherwise

with z_i the world joint axis and o_i the world joint origin. The result is an
ordinary differentiable function of q, so create_graph=True (double backward
through J in rollout training) keeps working. Same math as the original; values
agree to floating-point rounding.

Non-arm movable joints (the fingers) are held at q=0, as in the original.
fr3_hocbf/differentiable/kinematics.py itself is unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from fr3_hocbf.differentiable.kinematics import (
    ACTIVE_LINKS_DEF,
    NUM_ARM_JOINTS,
    RELEVANT_FRAME_NAMES,
    FR3TorchKinematics,
    _rpy_to_rot_const,
    _skew,
)


class FR3TorchKinematicsFast(FR3TorchKinematics):
    def __init__(self, urdf_path: Optional[str] = None):
        super().__init__(urdf_path=urdf_path)

        for jn in self.q_index:
            if self.joints[jn].joint_type not in ("revolute", "continuous"):
                raise NotImplementedError(f"Fast kinematics assumes revolute arm joints, {jn} is {self.joints[jn].joint_type}")

        # Joints in the same root-first traversal order as get_all_link_transforms_torch.
        self._order = []
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children_map.get(parent, []):
                self._order.append(joint)
                stack.append(joint.child)

        # Arm joint indices on the path root -> link.
        self._ancestors: Dict[str, List[int]] = {self.root_link: []}
        for joint in self._order:
            own = [self.q_index[joint.name]] if joint.name in self.q_index else []
            self._ancestors[joint.child] = self._ancestors[joint.parent] + own

        self._const_cache: Dict[tuple, dict] = {}

    def _consts(self, dtype, device) -> dict:
        key = (dtype, str(device))
        if key not in self._const_cache:
            consts = {}
            for joint in self._order:
                axis = torch.tensor(joint.axis, dtype=dtype, device=device)
                axis = axis / (torch.linalg.norm(axis) + 1e-12)
                K = _skew(axis)
                consts[joint.name] = {
                    "R_origin": _rpy_to_rot_const(joint.origin_rpy, dtype=dtype, device=device),
                    "t_origin": torch.tensor(joint.origin_xyz, dtype=dtype, device=device),
                    "axis": axis,
                    "K": K,
                    "KK": K @ K,
                }
            consts["I3"] = torch.eye(3, dtype=dtype, device=device)
            consts["zero3"] = torch.zeros(3, dtype=dtype, device=device)
            self._const_cache[key] = consts
        return self._const_cache[key]

    def _fk(self, q_arm: torch.Tensor):
        """One FK pass. Returns link -> (R_world (3,3), p_world (3,)) and arm joint index -> (o_i, z_i)."""
        if q_arm.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"Expected q_arm shape (7,), got {tuple(q_arm.shape)}")
        c = self._consts(q_arm.dtype, q_arm.device)
        frames = {self.root_link: (c["I3"], c["zero3"])}
        joint_axes = {}
        cos_q, sin_q = torch.cos(q_arm), torch.sin(q_arm)
        for joint in self._order:
            jc = c[joint.name]
            R_parent, p_parent = frames[joint.parent]
            R_joint_frame = R_parent @ jc["R_origin"]
            p_joint_frame = p_parent + R_parent @ jc["t_origin"]
            if joint.name in self.q_index:
                i = self.q_index[joint.name]
                R_motion = c["I3"] + sin_q[i] * jc["K"] + (1.0 - cos_q[i]) * jc["KK"]
                joint_axes[i] = (p_joint_frame, R_joint_frame @ jc["axis"])
                frames[joint.child] = (R_joint_frame @ R_motion, p_joint_frame)
            else:
                frames[joint.child] = (R_joint_frame, p_joint_frame)
        return frames, joint_axes

    def _jacobian(self, frame_name: str, p_f: torch.Tensor, joint_axes: dict) -> torch.Tensor:
        anc = set(self._ancestors[frame_name])
        zero = torch.zeros(3, dtype=p_f.dtype, device=p_f.device)
        cols = []
        for i in range(NUM_ARM_JOINTS):
            if i in anc:
                o_i, z_i = joint_axes[i]
                cols.append(torch.linalg.cross(z_i, p_f - o_i))
            else:
                cols.append(zero)
        return torch.stack(cols, dim=1)

    def get_relevant_frame_positions_torch(self, q_arm: torch.Tensor, frame_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        frames, _ = self._fk(q_arm)
        names = RELEVANT_FRAME_NAMES if frame_names is None else frame_names
        return {fn: frames[fn][1] for fn in names}

    def get_frame_position_jacobian_torch(self, q_arm: torch.Tensor, frame_name: str, create_graph: bool = True) -> torch.Tensor:
        frames, joint_axes = self._fk(q_arm)
        J = self._jacobian(frame_name, frames[frame_name][1], joint_axes)
        return J if create_graph else J.detach()

    def get_active_link_endpoint_jacobians_torch(self, q_arm: torch.Tensor, create_graph: bool = True) -> List[Dict]:
        return [
            {k: d[k] for k in ("name", "J0", "J1", "start_frame_name", "end_frame_name")}
            for d in self.get_active_link_endpoint_data_torch(q_arm, create_graph=create_graph)
        ]

    def get_active_link_endpoint_data_torch(self, q_arm: torch.Tensor, create_graph: bool = True) -> List[Dict]:
        frames, joint_axes = self._fk(q_arm)
        jac_cache: Dict[str, torch.Tensor] = {}

        def jac(fn: str) -> torch.Tensor:
            if fn not in jac_cache:
                J = self._jacobian(fn, frames[fn][1], joint_axes)
                # Original: autograd.functional.jacobian(create_graph=False) returns a J without graph.
                jac_cache[fn] = J if create_graph else J.detach()
            return jac_cache[fn]

        out = []
        for link_def in ACTIVE_LINKS_DEF:
            s, e = link_def["start_frame_name"], link_def["end_frame_name"]
            out.append(
                {
                    "name": link_def["name"],
                    "p0": frames[s][1],
                    "p1": frames[e][1],
                    "J0": jac(s),
                    "J1": jac(e),
                    "radius": float(link_def["radius"]),
                    "start_frame_name": s,
                    "end_frame_name": e,
                }
            )
        return out


def load_fr3_torch_kinematics_fast(urdf_path: Optional[str] = None) -> FR3TorchKinematicsFast:
    return FR3TorchKinematicsFast(urdf_path=urdf_path)
