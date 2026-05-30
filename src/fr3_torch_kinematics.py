#!/usr/bin/env python3
"""
Differentiable FR3 kinematics in torch, driven by the local URDF.

Goal:
    q_arm (7,) torch tensor
        -> world positions of relevant FR3 frames
        -> active-link endpoint positions
        -> active-link endpoint Jacobians via autograd

Important assumptions:
1) The target frame names used by the capsule model are URDF link names.
2) The FR3 arm is represented by joints:
       fr3_joint1 ... fr3_joint7
3) The URDF uses standard joint types:
       fixed, revolute, continuous, prismatic

If validation shows a mismatch because Pinocchio exposes extra frames that are not
plain URDF links, we can add a follow-up alias/attachment layer.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


NUM_ARM_JOINTS = 7
EE_FRAME_NAME = "fr3_hand_tcp"

ARM_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

# Must match your current trusted reference exactly.
ACTIVE_LINKS_DEF = [
    {"name": "link2_base",   "start_frame_name": "fr3_link2_offset1", "end_frame_name": "fr3_link2_offset2", "radius": 0.055},
    {"name": "link2",        "start_frame_name": "fr3_link2",         "end_frame_name": "fr3_link3",         "radius": 0.060},
    {"name": "joint4",       "start_frame_name": "fr3_link4",         "end_frame_name": "fr3_link5_offset1", "radius": 0.065},
    {"name": "forearm1",     "start_frame_name": "fr3_link5_offset2", "end_frame_name": "fr3_link5_offset3", "radius": 0.035},
    {"name": "forearm2",     "start_frame_name": "fr3_link5_offset3", "end_frame_name": "fr3_link5",         "radius": 0.050},
    {"name": "wrist",        "start_frame_name": "fr3_link7_offset1", "end_frame_name": "fr3_hand",          "radius": 0.055},
    {"name": "hand",         "start_frame_name": "fr3_hand_offset1",  "end_frame_name": "fr3_hand_offset2",  "radius": 0.030},
    {"name": "end_effector", "start_frame_name": EE_FRAME_NAME,       "end_frame_name": EE_FRAME_NAME,       "radius": 0.030},
]

RELEVANT_FRAME_NAMES = sorted(
    set(
        [EE_FRAME_NAME]
        + [x["start_frame_name"] for x in ACTIVE_LINKS_DEF]
        + [x["end_frame_name"] for x in ACTIVE_LINKS_DEF]
    )
)


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]


def get_default_urdf_path() -> str:
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, "include", "urdf", "fr3_robot.urdf")


def _parse_xyz(s: Optional[str]) -> Tuple[float, float, float]:
    if s is None or s.strip() == "":
        return (0.0, 0.0, 0.0)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected xyz triplet, got: {s}")
    return (vals[0], vals[1], vals[2])


def _parse_rpy(s: Optional[str]) -> Tuple[float, float, float]:
    if s is None or s.strip() == "":
        return (0.0, 0.0, 0.0)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected rpy triplet, got: {s}")
    return (vals[0], vals[1], vals[2])


def _parse_urdf_joints(urdf_path: str) -> Tuple[Dict[str, JointSpec], List[str], str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_names: List[str] = []
    joints: Dict[str, JointSpec] = {}

    for link_elem in root.findall("link"):
        name = link_elem.attrib["name"]
        link_names.append(name)

    child_links = set()

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib["type"]

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint {name} missing parent or child")

        parent = parent_elem.attrib["link"]
        child = child_elem.attrib["link"]
        child_links.add(child)

        origin_elem = joint_elem.find("origin")
        if origin_elem is None:
            origin_xyz = (0.0, 0.0, 0.0)
            origin_rpy = (0.0, 0.0, 0.0)
        else:
            origin_xyz = _parse_xyz(origin_elem.attrib.get("xyz"))
            origin_rpy = _parse_rpy(origin_elem.attrib.get("rpy"))

        axis_elem = joint_elem.find("axis")
        if axis_elem is None:
            axis = (0.0, 0.0, 1.0)
        else:
            axis = _parse_xyz(axis_elem.attrib.get("xyz"))

        joints[name] = JointSpec(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
        )

    root_links = [ln for ln in link_names if ln not in child_links]
    if len(root_links) != 1:
        raise ValueError(f"Expected exactly one URDF root link, found: {root_links}")

    return joints, link_names, root_links[0]


def _build_children_map(joints: Dict[str, JointSpec]) -> Dict[str, List[JointSpec]]:
    children_map: Dict[str, List[JointSpec]] = {}
    for j in joints.values():
        if j.parent not in children_map:
            children_map[j.parent] = []
        children_map[j.parent].append(j)
    return children_map


def _const_tensor3(xyz: Tuple[float, float, float], dtype, device) -> torch.Tensor:
    return torch.tensor([xyz[0], xyz[1], xyz[2]], dtype=dtype, device=device)


def _skew(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    row0 = torch.stack([zero, -v[2],  v[1]])
    row1 = torch.stack([v[2],  zero, -v[0]])
    row2 = torch.stack([-v[1], v[0], zero])
    return torch.stack([row0, row1, row2], dim=0)


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / (torch.linalg.norm(axis) + 1e-12)
    K = _skew(axis)
    I = torch.eye(3, dtype=axis.dtype, device=axis.device)
    c = torch.cos(angle)
    s = torch.sin(angle)
    return I + s * K + (1.0 - c) * (K @ K)


def _rpy_to_rot_const(rpy: Tuple[float, float, float], dtype, device) -> torch.Tensor:
    r, p, y = rpy

    cr = math.cos(r)
    sr = math.sin(r)
    cp = math.cos(p)
    sp = math.sin(p)
    cy = math.cos(y)
    sy = math.sin(y)

    Rx = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, cr, -sr],
         [0.0, sr, cr]],
        dtype=dtype,
        device=device,
    )
    Ry = torch.tensor(
        [[cp, 0.0, sp],
         [0.0, 1.0, 0.0],
         [-sp, 0.0, cp]],
        dtype=dtype,
        device=device,
    )
    Rz = torch.tensor(
        [[cy, -sy, 0.0],
         [sy, cy, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )

    # URDF uses fixed-axis roll-pitch-yaw, equivalent to Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return Rz @ Ry @ Rx


def _make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _origin_transform(joint: JointSpec, dtype, device) -> torch.Tensor:
    R = _rpy_to_rot_const(joint.origin_rpy, dtype=dtype, device=device)
    t = _const_tensor3(joint.origin_xyz, dtype=dtype, device=device)
    return _make_transform(R, t)


def _joint_motion_transform(joint: JointSpec, q_value: torch.Tensor, dtype, device) -> torch.Tensor:
    jt = joint.joint_type

    if jt == "fixed":
        R = torch.eye(3, dtype=dtype, device=device)
        t = torch.zeros(3, dtype=dtype, device=device)
        return _make_transform(R, t)

    axis = _const_tensor3(joint.axis, dtype=dtype, device=device)
    axis = axis / (torch.linalg.norm(axis) + 1e-12)

    if jt in ("revolute", "continuous"):
        R = _rodrigues(axis, q_value)
        t = torch.zeros(3, dtype=dtype, device=device)
        return _make_transform(R, t)

    if jt == "prismatic":
        R = torch.eye(3, dtype=dtype, device=device)
        t = axis * q_value
        return _make_transform(R, t)

    raise NotImplementedError(f"Unsupported joint type in torch FK: {jt}")


class FR3TorchKinematics:
    def __init__(self, urdf_path: Optional[str] = None):
        if urdf_path is None:
            urdf_path = get_default_urdf_path()

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at: {urdf_path}")

        self.urdf_path = urdf_path
        self.joints, self.link_names, self.root_link = _parse_urdf_joints(urdf_path)
        self.children_map = _build_children_map(self.joints)

        # Validate arm joint names exist
        for jn in ARM_JOINT_NAMES:
            if jn not in self.joints:
                raise ValueError(f"Required FR3 arm joint not found in URDF: {jn}")

        # Validate relevant frames exist as URDF links
        missing = [fn for fn in RELEVANT_FRAME_NAMES if fn not in self.link_names]
        if missing:
            raise ValueError(
                "These relevant frame names are not URDF links. "
                "This first torch FK version assumes they are link names. "
                f"Missing: {missing}"
            )

        self.q_index = {jn: i for i, jn in enumerate(ARM_JOINT_NAMES)}

    def _joint_value(self, joint_name: str, q_arm: torch.Tensor) -> torch.Tensor:
        if joint_name in self.q_index:
            return q_arm[self.q_index[joint_name]]
        return torch.zeros((), dtype=q_arm.dtype, device=q_arm.device)

    def get_all_link_transforms_torch(self, q_arm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns a dict:
            link_name -> T_world_link   (4,4) torch tensor
        """
        if q_arm.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"Expected q_arm shape (7,), got {tuple(q_arm.shape)}")

        dtype = q_arm.dtype
        device = q_arm.device

        transforms: Dict[str, torch.Tensor] = {
            self.root_link: torch.eye(4, dtype=dtype, device=device)
        }

        stack = [self.root_link]
        while stack:
            parent_link = stack.pop()
            T_world_parent = transforms[parent_link]

            for joint in self.children_map.get(parent_link, []):
                q_val = self._joint_value(joint.name, q_arm)
                T_parent_child = _origin_transform(joint, dtype=dtype, device=device) @ _joint_motion_transform(
                    joint, q_val, dtype=dtype, device=device
                )
                T_world_child = T_world_parent @ T_parent_child
                transforms[joint.child] = T_world_child
                stack.append(joint.child)

        return transforms

    def get_relevant_frame_positions_torch(
        self,
        q_arm: torch.Tensor,
        frame_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            frame_name -> position (3,)
        """
        if frame_names is None:
            frame_names = RELEVANT_FRAME_NAMES

        transforms = self.get_all_link_transforms_torch(q_arm)
        out: Dict[str, torch.Tensor] = {}
        for fn in frame_names:
            if fn not in transforms:
                raise KeyError(f"Frame/link not found in computed transforms: {fn}")
            out[fn] = transforms[fn][:3, 3]
        return out

    def get_frame_position_torch(self, q_arm: torch.Tensor, frame_name: str) -> torch.Tensor:
        return self.get_relevant_frame_positions_torch(q_arm, [frame_name])[frame_name]

    def get_frame_position_jacobian_torch(
        self,
        q_arm: torch.Tensor,
        frame_name: str,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """
        Returns:
            J_pos(frame_name) = d p_world(frame_name) / d q_arm
            shape (3, 7)
        """
        def f(q_local: torch.Tensor) -> torch.Tensor:
            return self.get_frame_position_torch(q_local, frame_name)

        J = torch.autograd.functional.jacobian(
            f,
            q_arm,
            create_graph=create_graph,
            strict=True,
            vectorize=False,
        )

        if J.shape != (3, NUM_ARM_JOINTS):
            raise RuntimeError(f"Unexpected Jacobian shape for {frame_name}: {tuple(J.shape)}")

        return J

    def get_active_link_endpoint_positions_torch(self, q_arm: torch.Tensor) -> List[Dict]:
        frame_pos = self.get_relevant_frame_positions_torch(q_arm)

        out = []
        for link_def in ACTIVE_LINKS_DEF:
            out.append(
                {
                    "name": link_def["name"],
                    "p0": frame_pos[link_def["start_frame_name"]],
                    "p1": frame_pos[link_def["end_frame_name"]],
                    "radius": float(link_def["radius"]),
                    "start_frame_name": link_def["start_frame_name"],
                    "end_frame_name": link_def["end_frame_name"],
                }
            )
        return out

    def get_active_link_endpoint_jacobians_torch(
        self,
        q_arm: torch.Tensor,
        create_graph: bool = True,
    ) -> List[Dict]:
        out = []
        for link_def in ACTIVE_LINKS_DEF:
            J0 = self.get_frame_position_jacobian_torch(
                q_arm,
                link_def["start_frame_name"],
                create_graph=create_graph,
            )
            J1 = self.get_frame_position_jacobian_torch(
                q_arm,
                link_def["end_frame_name"],
                create_graph=create_graph,
            )
            out.append(
                {
                    "name": link_def["name"],
                    "J0": J0,
                    "J1": J1,
                    "start_frame_name": link_def["start_frame_name"],
                    "end_frame_name": link_def["end_frame_name"],
                }
            )
        return out

    def get_active_link_endpoint_data_torch(
        self,
        q_arm: torch.Tensor,
        create_graph: bool = True,
    ) -> List[Dict]:
        """
        Returns one dict per active capsule:
        {
            "name": str,
            "p0": (3,),
            "p1": (3,),
            "J0": (3,7),
            "J1": (3,7),
            "radius": float,
            "start_frame_name": str,
            "end_frame_name": str,
        }
        """
        pos_data = self.get_active_link_endpoint_positions_torch(q_arm)
        jac_data = self.get_active_link_endpoint_jacobians_torch(q_arm, create_graph=create_graph)

        out = []
        for p, j in zip(pos_data, jac_data):
            out.append(
                {
                    "name": p["name"],
                    "p0": p["p0"],
                    "p1": p["p1"],
                    "J0": j["J0"],
                    "J1": j["J1"],
                    "radius": p["radius"],
                    "start_frame_name": p["start_frame_name"],
                    "end_frame_name": p["end_frame_name"],
                }
            )
        return out


def load_fr3_torch_kinematics(urdf_path: Optional[str] = None) -> FR3TorchKinematics:
    return FR3TorchKinematics(urdf_path=urdf_path)


if __name__ == "__main__":
    kin = load_fr3_torch_kinematics()
    q = torch.tensor([0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0], dtype=torch.double)
    data = kin.get_active_link_endpoint_data_torch(q, create_graph=False)
    print("Loaded URDF:", kin.urdf_path)
    print("Root link:", kin.root_link)
    print("Num active links:", len(data))
    for item in data:
        print(item["name"], item["start_frame_name"], item["end_frame_name"], tuple(item["p0"].shape), tuple(item["J0"].shape))