#!/usr/bin/env python3

import os
import sys
import yaml
import numpy as np
import pinocchio as pin

from fr3_geometry import CapsuleKinematics, compute_pair_geometry
from fr3_nominal_controller import nominal_controller_js_standalone
from fr3_qp_solver import (
    h_func,
    Lf_h,
    psi_func,
    solve_hocbf_qp_standalone,
    Lg_psi,
)


NUM_ARM_JOINTS = 7
EE_FRAME_NAME = "fr3_hand_tcp"
MAX_OBSTACLES = 5


def get_default_urdf_path():
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, "include", "urdf", "fr3_robot.urdf")


def load_fr3_model(urdf_path=None):
    if urdf_path is None:
        urdf_path = get_default_urdf_path()

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found at: {urdf_path}")

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    ee_frame_id = model.getFrameId(EE_FRAME_NAME)

    return model, data, ee_frame_id


def build_active_links(model):
    links_def = [
        {"name": "link2_base", "start_frame_name": "fr3_link2_offset1", "end_frame_name": "fr3_link2_offset2", "radius": 0.055},
        {"name": "link2", "start_frame_name": "fr3_link2", "end_frame_name": "fr3_link3", "radius": 0.06},
        {"name": "joint4", "start_frame_name": "fr3_link4", "end_frame_name": "fr3_link5_offset1", "radius": 0.065},
        {"name": "forearm1", "start_frame_name": "fr3_link5_offset2", "end_frame_name": "fr3_link5_offset3", "radius": 0.035},
        {"name": "forearm2", "start_frame_name": "fr3_link5_offset3", "end_frame_name": "fr3_link5", "radius": 0.05},
        {"name": "wrist", "start_frame_name": "fr3_link7_offset1", "end_frame_name": "fr3_hand", "radius": 0.055},
        {"name": "hand", "start_frame_name": "fr3_hand_offset1", "end_frame_name": "fr3_hand_offset2", "radius": 0.03},
        {"name": "end_effector", "start_frame_name": EE_FRAME_NAME, "end_frame_name": EE_FRAME_NAME, "radius": 0.03},
    ]

    active_links = []
    for link_def in links_def:
        active_links.append(
            {
                "name": link_def["name"],
                "start_frame_id": model.getFrameId(link_def["start_frame_name"]),
                "end_frame_id": model.getFrameId(link_def["end_frame_name"]),
                "radius": link_def["radius"],
            }
        )

    return active_links


def get_self_collision_link_pair():
    return [
        ("link2", "hand"),
        ("link2", "end_effector"),
        ("link2", "forearm1"),
    ]


def get_joint_limits():
    ddq_max_scalar = 40.0
    ddq_max_arm = np.full(NUM_ARM_JOINTS, ddq_max_scalar)
    ddq_min_arm = np.full(NUM_ARM_JOINTS, -ddq_max_scalar)

    dq_max_arm = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0])
    dq_min_arm = -dq_max_arm

    q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
    q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])

    return {
        "ddq_max_arm": ddq_max_arm,
        "ddq_min_arm": ddq_min_arm,
        "dq_max_arm": dq_max_arm,
        "dq_min_arm": dq_min_arm,
        "q_max_arm": q_max_arm,
        "q_min_arm": q_min_arm,
    }


def get_default_initial_q():
    q = np.zeros(7)
    q[:] = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
    return q


def get_default_initial_state(model):
    q_full_init = np.zeros(model.nq)
    dq_full_init = np.zeros(model.nv)
    q_full_init[:7] = get_default_initial_q()
    return q_full_init, dq_full_init

def make_rollout_context(urdf_path=None):
    """
    Load reusable FR3 rollout context once.

    We keep the loaded Pinocchio model and static metadata in the context.
    A fresh Pinocchio data object is created per scenario run, since data is
    mutable and cheap to create.
    """
    model, _, ee_frame_id = load_fr3_model(urdf_path)
    active_links = build_active_links(model)
    self_collision_link_pair = get_self_collision_link_pair()
    joint_limits = get_joint_limits()

    return {
        "model": model,
        "ee_frame_id": ee_frame_id,
        "active_links": active_links,
        "self_collision_link_pair": self_collision_link_pair,
        "joint_limits": joint_limits,
        "urdf_path": urdf_path,
    }

def load_scenario_yaml(scenario_path):
    with open(scenario_path, "r") as f:
        config = yaml.safe_load(f)

    hocbf_params = config.get("hocbf_controller", {}).get("ros__parameters", {})
    obstacle_params = config.get("obstacles", [])

    goal_ee_pos = np.array(hocbf_params.get("goal_ee_pos", [0.3, 0.0, 0.5]), dtype=float)
    gamma = float(hocbf_params.get("gamma_js", 2.0))
    beta = float(hocbf_params.get("beta_js", 3.0))
    d_margin = float(hocbf_params.get("d_margin", 0.0))
    goal_tolerance = float(hocbf_params.get("goal_tolerance_m", 0.02))
    goal_settle_time_s = float(hocbf_params.get("goal_settle_time_s", 2.0))
    max_sim_duration_s = float(hocbf_params.get("max_sim_duration_s", 60.0))
    output_basename = hocbf_params.get("output_data_basename", "unnamed_scenario")

    obstacles = []
    for obs in obstacle_params:
        pose_start = np.array(obs["pose_start"]["position"], dtype=float)
        pose_end = pose_start.copy()
        if "pose_end" in obs:
            pose_end = np.array(obs["pose_end"]["position"], dtype=float)

        obstacles.append(
            {
                "pose_start": pose_start,
                "pose_end": pose_end,
                "radius": float(obs["size"]["radius"]),
                "velocity": np.array(obs["velocity"]["linear"], dtype=float),
            }
        )

    return {
        "goal_ee_pos": goal_ee_pos,
        "gamma": gamma,
        "beta": beta,
        "d_margin": d_margin,
        "goal_tolerance": goal_tolerance,
        "goal_settle_time_s": goal_settle_time_s,
        "max_sim_duration_s": max_sim_duration_s,
        "output_basename": output_basename,
        "obstacles": obstacles,
    }


def clone_obstacles(obstacles):
    cloned = []
    for obs in obstacles:
        cloned.append(
            {
                "pose_start": np.array(obs["pose_start"], dtype=float).copy(),
                "pose_end": np.array(obs["pose_end"], dtype=float).copy(),
                "radius": float(obs["radius"]),
                "velocity": np.array(obs["velocity"], dtype=float).copy(),
            }
        )
    return cloned


def update_obstacles(obstacles, dt):
    updated = clone_obstacles(obstacles)
    for obs in updated:
        obs["pose_start"] = obs["pose_start"] + obs["velocity"] * dt
        obs["pose_end"] = obs["pose_end"] + obs["velocity"] * dt
    return updated


def compute_barrier_metrics(
    model,
    data,
    q_full,
    dq_full,
    obstacles,
    active_links,
    self_collision_link_pair,
    d_margin,
    gamma,
):
    pin.forwardKinematics(model, data, q_full, dq_full, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full)
    pin.updateFramePlacements(model, data)

    min_h = float("inf")
    min_psi = float("inf")
    min_dist = float("inf")

    for link_info in active_links:
        start_frame_id = link_info["start_frame_id"]
        end_frame_id = link_info["end_frame_id"]
        link_radius_val = link_info["radius"]

        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        J_start = pin.getFrameJacobian(
            model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J_end = pin.getFrameJacobian(
            model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        robot_capsule = CapsuleKinematics(
            p0=p1,
            p1=p2,
            radius=link_radius_val,
            J0=J_start,
            J1=J_end,
            name=link_info["name"],
        )

        for obs_idx, obs_item in enumerate(obstacles):
            obs_capsule = CapsuleKinematics(
                p0=obs_item["pose_start"],
                p1=obs_item["pose_end"],
                radius=obs_item["radius"],
                linear_velocity=obs_item["velocity"],
                name=f"obs_{obs_idx}",
            )

            pair = compute_pair_geometry(
                robot_capsule,
                obs_capsule,
                dq=dq_full,
                d_margin=d_margin,
            )

            min_h = min(min_h, pair.h)
            min_dist = min(min_dist, pair.dist_surfaces)

            v_rel = pair.v_rel
            Lf_h_val = Lf_h(pair.p_rel, v_rel)
            psi_val = psi_func(pair.h, Lf_h_val, gamma)
            min_psi = min(min_psi, psi_val)

    for link1, link2 in self_collision_link_pair:
        link1_info = next((l for l in active_links if l["name"] == link1), None)
        link2_info = next((l for l in active_links if l["name"] == link2), None)

        if link1_info is None or link2_info is None:
            continue

        start_frame_id_1 = link1_info["start_frame_id"]
        end_frame_id_1 = link1_info["end_frame_id"]
        link_radius_val_1 = link1_info["radius"]

        start_frame_id_2 = link2_info["start_frame_id"]
        end_frame_id_2 = link2_info["end_frame_id"]
        link_radius_val_2 = link2_info["radius"]

        p1_1 = data.oMf[start_frame_id_1].translation
        p2_1 = data.oMf[end_frame_id_1].translation
        p1_2 = data.oMf[start_frame_id_2].translation
        p2_2 = data.oMf[end_frame_id_2].translation

        J1_start = pin.getFrameJacobian(
            model, data, start_frame_id_1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J1_end = pin.getFrameJacobian(
            model, data, end_frame_id_1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        J2_start = pin.getFrameJacobian(
            model, data, start_frame_id_2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J2_end = pin.getFrameJacobian(
            model, data, end_frame_id_2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        link1_capsule = CapsuleKinematics(
            p0=p1_1,
            p1=p2_1,
            radius=link_radius_val_1,
            J0=J1_start,
            J1=J1_end,
            name=link1,
        )

        link2_capsule = CapsuleKinematics(
            p0=p1_2,
            p1=p2_2,
            radius=link_radius_val_2,
            J0=J2_start,
            J1=J2_end,
            name=link2,
        )

        pair = compute_pair_geometry(
            link1_capsule,
            link2_capsule,
            dq=dq_full,
        )

        min_h = min(min_h, pair.h)
        min_dist = min(min_dist, pair.dist_surfaces)

        v_rel = pair.v_rel
        Lf_h_val = Lf_h(pair.p_rel, v_rel)
        psi_val = psi_func(pair.h, Lf_h_val, gamma)
        min_psi = min(min_psi, psi_val)

    return {
        "min_h": min_h,
        "min_psi": min_psi,
        "min_dist": min_dist,
    }


def compute_all_pair_terms(
    model,
    data,
    q_full,
    dq_full,
    obstacles,
    active_links,
    self_collision_link_pair,
    d_margin,
):
    """
    Build all pairwise HOCBF row terms in a fixed deterministic order.

    Fixed order:
    1) robot-obstacle rows:
       for each active link, for obstacle slots 0..MAX_OBSTACLES-1
    2) self-collision rows:
       for each pair in self_collision_link_pair

    For missing obstacle slots, we pad with zero rows and mask=0.
    For real rows, mask follows Davide's activation logic:
      - robot-obstacle active iff h < 0.07
      - self-collision active iff h < 0.03
    """
    pin.forwardKinematics(model, data, q_full, dq_full, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full)
    pin.updateFramePlacements(model, data)

    rows_h = []
    rows_Lf_h = []
    rows_vrel_sq2 = []
    rows_Lg = []
    rows_mask = []

    # ------------------------------------------------------------------
    # robot-obstacle rows: fixed obstacle slots 0..MAX_OBSTACLES-1
    # ------------------------------------------------------------------
    for link_info in active_links:
        start_frame_id = link_info["start_frame_id"]
        end_frame_id = link_info["end_frame_id"]
        link_radius_val = link_info["radius"]

        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        J_start = pin.getFrameJacobian(
            model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J_end = pin.getFrameJacobian(
            model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        robot_capsule = CapsuleKinematics(
            p0=p1,
            p1=p2,
            radius=link_radius_val,
            J0=J_start,
            J1=J_end,
            name=link_info["name"],
        )

        for obs_idx in range(MAX_OBSTACLES):
            if obs_idx < len(obstacles):
                obs_item = obstacles[obs_idx]

                obs_capsule = CapsuleKinematics(
                    p0=obs_item["pose_start"],
                    p1=obs_item["pose_end"],
                    radius=obs_item["radius"],
                    linear_velocity=obs_item["velocity"],
                    name=f"obs_{obs_idx}",
                )

                pair = compute_pair_geometry(
                    robot_capsule,
                    obs_capsule,
                    dq=dq_full,
                    d_margin=d_margin,
                )

                rows_h.append(float(pair.h))
                rows_Lf_h.append(float(Lf_h(pair.p_rel, pair.v_rel)))
                rows_vrel_sq2.append(float(2.0 * np.dot(pair.v_rel, pair.v_rel)))
                rows_Lg.append(Lg_psi(pair.J_a, pair.p_rel, NUM_ARM_JOINTS).astype(float))

                # Davide robot-obstacle activation logic
                active = 1.0 if pair.h < 0.07 else 0.0
                rows_mask.append(active)

            else:
                # padded dummy obstacle slot
                rows_h.append(0.0)
                rows_Lf_h.append(0.0)
                rows_vrel_sq2.append(0.0)
                rows_Lg.append(np.zeros(NUM_ARM_JOINTS, dtype=float))
                rows_mask.append(0.0)

    # ------------------------------------------------------------------
    # self-collision rows: fixed order from self_collision_link_pair
    # ------------------------------------------------------------------
    for link1, link2 in self_collision_link_pair:
        link1_info = next((l for l in active_links if l["name"] == link1), None)
        link2_info = next((l for l in active_links if l["name"] == link2), None)

        if link1_info is None or link2_info is None:
            rows_h.append(0.0)
            rows_Lf_h.append(0.0)
            rows_vrel_sq2.append(0.0)
            rows_Lg.append(np.zeros(NUM_ARM_JOINTS, dtype=float))
            rows_mask.append(0.0)
            continue

        start_frame_id_1 = link1_info["start_frame_id"]
        end_frame_id_1 = link1_info["end_frame_id"]
        link_radius_val_1 = link1_info["radius"]

        start_frame_id_2 = link2_info["start_frame_id"]
        end_frame_id_2 = link2_info["end_frame_id"]
        link_radius_val_2 = link2_info["radius"]

        p1_1 = data.oMf[start_frame_id_1].translation
        p2_1 = data.oMf[end_frame_id_1].translation
        p1_2 = data.oMf[start_frame_id_2].translation
        p2_2 = data.oMf[end_frame_id_2].translation

        J1_start = pin.getFrameJacobian(
            model, data, start_frame_id_1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J1_end = pin.getFrameJacobian(
            model, data, end_frame_id_1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        J2_start = pin.getFrameJacobian(
            model, data, start_frame_id_2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J2_end = pin.getFrameJacobian(
            model, data, end_frame_id_2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        link1_capsule = CapsuleKinematics(
            p0=p1_1,
            p1=p2_1,
            radius=link_radius_val_1,
            J0=J1_start,
            J1=J1_end,
            name=link1,
        )

        link2_capsule = CapsuleKinematics(
            p0=p1_2,
            p1=p2_2,
            radius=link_radius_val_2,
            J0=J2_start,
            J1=J2_end,
            name=link2,
        )

        pair = compute_pair_geometry(
            link1_capsule,
            link2_capsule,
            dq=dq_full,
        )

        rows_h.append(float(pair.h))
        rows_Lf_h.append(float(Lf_h(pair.p_rel, pair.v_rel)))
        rows_vrel_sq2.append(float(2.0 * np.dot(pair.v_rel, pair.v_rel)))
        rows_Lg.append(Lg_psi(pair.J_rel, pair.p_rel, NUM_ARM_JOINTS).astype(float))

        # Davide self-collision activation logic
        active = 1.0 if pair.h < 0.03 else 0.0
        rows_mask.append(active)

    h_all = np.array(rows_h, dtype=float)
    Lf_h_all = np.array(rows_Lf_h, dtype=float)
    vrel_sq2_all = np.array(rows_vrel_sq2, dtype=float)
    Lg_psi_all = np.array(rows_Lg, dtype=float)
    pair_mask = np.array(rows_mask, dtype=float)

    expected_rows = len(active_links) * MAX_OBSTACLES + len(self_collision_link_pair)
    assert h_all.shape == (expected_rows,)
    assert Lf_h_all.shape == (expected_rows,)
    assert vrel_sq2_all.shape == (expected_rows,)
    assert Lg_psi_all.shape == (expected_rows, NUM_ARM_JOINTS)
    assert pair_mask.shape == (expected_rows,)

    return {
        "h_all": h_all,
        "Lf_h_all": Lf_h_all,
        "vrel_sq2_all": vrel_sq2_all,
        "Lg_psi_all": Lg_psi_all,
        "pair_mask": pair_mask,
    }


def compute_critical_pair_features(
    model,
    data,
    q_full,
    dq_full,
    obstacles,
    active_links,
    self_collision_link_pair,
    d_margin,
):
    pin.forwardKinematics(model, data, q_full, dq_full, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full)
    pin.updateFramePlacements(model, data)

    best = None

    # robot-obstacle pairs
    for link_info in active_links:
        start_frame_id = link_info["start_frame_id"]
        end_frame_id = link_info["end_frame_id"]
        link_radius_val = link_info["radius"]

        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        J_start = pin.getFrameJacobian(
            model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J_end = pin.getFrameJacobian(
            model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        robot_capsule = CapsuleKinematics(
            p0=p1,
            p1=p2,
            radius=link_radius_val,
            J0=J_start,
            J1=J_end,
            name=link_info["name"],
        )

        for obs_idx, obs_item in enumerate(obstacles):
            obs_capsule = CapsuleKinematics(
                p0=obs_item["pose_start"],
                p1=obs_item["pose_end"],
                radius=obs_item["radius"],
                linear_velocity=obs_item["velocity"],
                name=f"obs_{obs_idx}",
            )

            pair = compute_pair_geometry(
                robot_capsule,
                obs_capsule,
                dq=dq_full,
                d_margin=d_margin,
            )

            p_rel = pair.p_rel
            norm_p = np.linalg.norm(p_rel)
            n_crit = p_rel / (norm_p + 1e-8)

            d_crit = pair.dist_surfaces - d_margin
            d_dot_crit = float(n_crit @ pair.v_rel)

            h_crit = float(pair.h)
            Lf_h_crit = float(Lf_h(pair.p_rel, pair.v_rel))
            vrel_sq2_crit = float(2.0 * np.dot(pair.v_rel, pair.v_rel))
            Lg_psi_crit = Lg_psi(pair.J_a, pair.p_rel, NUM_ARM_JOINTS).astype(float)

            tau_crit = 0.0

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

            if best is None or candidate["d_crit"] < best["d_crit"]:
                best = candidate

    # self-collision pairs
    for link1, link2 in self_collision_link_pair:
        link1_info = next((l for l in active_links if l["name"] == link1), None)
        link2_info = next((l for l in active_links if l["name"] == link2), None)

        if link1_info is None or link2_info is None:
            continue

        start_frame_id_1 = link1_info["start_frame_id"]
        end_frame_id_1 = link1_info["end_frame_id"]
        link_radius_val_1 = link1_info["radius"]

        start_frame_id_2 = link2_info["start_frame_id"]
        end_frame_id_2 = link2_info["end_frame_id"]
        link_radius_val_2 = link2_info["radius"]

        p1_1 = data.oMf[start_frame_id_1].translation
        p2_1 = data.oMf[end_frame_id_1].translation
        p1_2 = data.oMf[start_frame_id_2].translation
        p2_2 = data.oMf[end_frame_id_2].translation

        J1_start = pin.getFrameJacobian(
            model, data, start_frame_id_1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J1_end = pin.getFrameJacobian(
            model, data, end_frame_id_1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        J2_start = pin.getFrameJacobian(
            model, data, start_frame_id_2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        J2_end = pin.getFrameJacobian(
            model, data, end_frame_id_2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]

        link1_capsule = CapsuleKinematics(
            p0=p1_1,
            p1=p2_1,
            radius=link_radius_val_1,
            J0=J1_start,
            J1=J1_end,
            name=link1,
        )

        link2_capsule = CapsuleKinematics(
            p0=p1_2,
            p1=p2_2,
            radius=link_radius_val_2,
            J0=J2_start,
            J1=J2_end,
            name=link2,
        )

        pair = compute_pair_geometry(
            link1_capsule,
            link2_capsule,
            dq=dq_full,
        )

        p_rel = pair.p_rel
        norm_p = np.linalg.norm(p_rel)
        n_crit = p_rel / (norm_p + 1e-8)

        d_crit = pair.dist_surfaces
        d_dot_crit = float(n_crit @ pair.v_rel)

        h_crit = float(pair.h)
        Lf_h_crit = float(Lf_h(pair.p_rel, pair.v_rel))
        vrel_sq2_crit = float(2.0 * np.dot(pair.v_rel, pair.v_rel))
        Lg_psi_crit = Lg_psi(pair.J_rel, pair.p_rel, NUM_ARM_JOINTS).astype(float)

        tau_crit = 1.0

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

        if best is None or candidate["d_crit"] < best["d_crit"]:
            best = candidate

    if best is None:
        return {
            "n_crit": np.zeros(3, dtype=float),
            "d_crit": 0.0,
            "d_dot_crit": 0.0,
            "tau_crit": 0.0,
            "h_crit": 0.0,
            "Lf_h_crit": 0.0,
            "vrel_sq2_crit": 0.0,
            "Lg_psi_crit": np.zeros(NUM_ARM_JOINTS, dtype=float),
        }

    return best


def rollout_step(
    model,
    data,
    ee_frame_id,
    active_links,
    self_collision_link_pair,
    joint_limits,
    q_full,
    dq_full,
    target_ee_pos_cartesian,
    target_prev,
    obstacles,
    dt,
    gamma,
    beta,
    d_margin,
    move_obstacles=True,
):
    q_full = np.array(q_full, dtype=float).copy()
    dq_full = np.array(dq_full, dtype=float).copy()
    target_prev = np.array(target_prev, dtype=float).copy()

    if move_obstacles:
        next_obstacles = update_obstacles(obstacles, dt)
    else:
        next_obstacles = clone_obstacles(obstacles)

    all_pair_terms = compute_all_pair_terms(
        model=model,
        data=data,
        q_full=q_full,
        dq_full=dq_full,
        obstacles=next_obstacles,
        active_links=active_links,
        self_collision_link_pair=self_collision_link_pair,
        d_margin=d_margin,
    )

    critical_pair = compute_critical_pair_features(
        model=model,
        data=data,
        q_full=q_full,
        dq_full=dq_full,
        obstacles=next_obstacles,
        active_links=active_links,
        self_collision_link_pair=self_collision_link_pair,
        d_margin=d_margin,
    )

    q_arm_curr = q_full[:NUM_ARM_JOINTS]
    dq_arm_curr = dq_full[:NUM_ARM_JOINTS]

    ddq_nominal_arm, target_next = nominal_controller_js_standalone(
        model=model,
        data=data,
        ee_frame_id=ee_frame_id,
        q_arm_curr=q_arm_curr,
        dq_arm_curr=dq_arm_curr,
        target_ee_pos_cartesian=target_ee_pos_cartesian,
        target=target_prev,
    )

    ddq_safe_arm, qp_solved = solve_hocbf_qp_standalone(
        model=model,
        data=data,
        num_arm_joints=NUM_ARM_JOINTS,
        dt_val=dt,
        q_full_curr=q_full,
        dq_full_curr=dq_full,
        ddq_nominal_arm_val=ddq_nominal_arm,
        current_gamma_js_val=gamma,
        current_beta_js_val=beta,
        d_margin=d_margin,
        current_obstacles_list_sim=next_obstacles,
        active_links_list=active_links,
        self_collision_link_pair=self_collision_link_pair,
        ddq_min_arm=joint_limits["ddq_min_arm"],
        ddq_max_arm=joint_limits["ddq_max_arm"],
        dq_min_arm=joint_limits["dq_min_arm"],
        dq_max_arm=joint_limits["dq_max_arm"],
        q_min_arm=joint_limits["q_min_arm"],
        q_max_arm=joint_limits["q_max_arm"],
    )

    if qp_solved:
        next_dq_arm = dq_arm_curr + ddq_safe_arm * dt
        next_q_arm = q_arm_curr + dq_arm_curr * dt + 0.5 * ddq_safe_arm * dt ** 2
    else:
        next_dq_arm = dq_arm_curr.copy()
        next_q_arm = q_arm_curr.copy()

    next_q_full = q_full.copy()
    next_dq_full = dq_full.copy()

    next_q_full[:NUM_ARM_JOINTS] = next_q_arm
    next_dq_full[:NUM_ARM_JOINTS] = next_dq_arm

    metrics = compute_barrier_metrics(
        model=model,
        data=data,
        q_full=next_q_full,
        dq_full=next_dq_full,
        obstacles=next_obstacles,
        active_links=active_links,
        self_collision_link_pair=self_collision_link_pair,
        d_margin=d_margin,
        gamma=gamma,
    )

    # raw geometry features for NN input
    # assuming h = d - d_margin  ->  d = h + d_margin
    # and d_dot = Lf_h
    d_all = all_pair_terms["h_all"] + d_margin
    d_dot_all = all_pair_terms["Lf_h_all"].copy()

    info = {
        "ddq_nominal_arm": ddq_nominal_arm.copy(),
        "ddq_safe_arm": ddq_safe_arm.copy(),
        "qp_solved": qp_solved,
        "target_prev": target_prev.copy(),
        "target_next": target_next.copy(),
        "obstacles_next": clone_obstacles(next_obstacles),
        "min_h": metrics["min_h"],
        "min_psi": metrics["min_psi"],
        "min_dist": metrics["min_dist"],
        "n_crit": critical_pair["n_crit"].copy(),
        "d_crit": critical_pair["d_crit"],
        "d_dot_crit": critical_pair["d_dot_crit"],
        "tau_crit": critical_pair["tau_crit"],
        "h_crit": critical_pair["h_crit"],
        "Lf_h_crit": critical_pair["Lf_h_crit"],
        "vrel_sq2_crit": critical_pair["vrel_sq2_crit"],
        "Lg_psi_crit": critical_pair["Lg_psi_crit"].copy(),
        "h_all": all_pair_terms["h_all"].copy(),
        "Lf_h_all": all_pair_terms["Lf_h_all"].copy(),
        "vrel_sq2_all": all_pair_terms["vrel_sq2_all"].copy(),
        "Lg_psi_all": all_pair_terms["Lg_psi_all"].copy(),
        "pair_mask": all_pair_terms["pair_mask"].copy(),
        "d_all": d_all.copy(),
        "d_dot_all": d_dot_all.copy(),
    }

    return next_q_full, next_dq_full, target_next, next_obstacles, info


def rollout_episode(
    model,
    data,
    ee_frame_id,
    active_links,
    self_collision_link_pair,
    joint_limits,
    q_full_init,
    dq_full_init,
    target_ee_pos_cartesian,
    obstacles_init,
    dt,
    gamma,
    beta,
    d_margin,
    num_steps,
    target_init=None,
    move_obstacles=True,
):
    q_full = np.array(q_full_init, dtype=float).copy()
    dq_full = np.array(dq_full_init, dtype=float).copy()
    obstacles = clone_obstacles(obstacles_init)

    if target_init is None:
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        target_prev = data.oMf[ee_frame_id].translation.copy()
    else:
        target_prev = np.array(target_init, dtype=float).copy()

    history = []

    for k in range(num_steps):
        next_q_full, next_dq_full, target_next, obstacles, info = rollout_step(
            model=model,
            data=data,
            ee_frame_id=ee_frame_id,
            active_links=active_links,
            self_collision_link_pair=self_collision_link_pair,
            joint_limits=joint_limits,
            q_full=q_full,
            dq_full=dq_full,
            target_ee_pos_cartesian=target_ee_pos_cartesian,
            target_prev=target_prev,
            obstacles=obstacles,
            dt=dt,
            gamma=gamma,
            beta=beta,
            d_margin=d_margin,
            move_obstacles=move_obstacles,
        )

        history.append(
            {
                "step": k,
                "time": k * dt,
                "q_full": q_full.copy(),
                "dq_full": dq_full.copy(),
                "q_arm": q_full[:NUM_ARM_JOINTS].copy(),
                "dq_arm": dq_full[:NUM_ARM_JOINTS].copy(),
                "target_prev": target_prev.copy(),
                "target_next": target_next.copy(),
                "ddq_nominal_arm": info["ddq_nominal_arm"].copy(),
                "ddq_safe_arm": info["ddq_safe_arm"].copy(),
                "qp_solved": info["qp_solved"],
                "min_h": info["min_h"],
                "min_psi": info["min_psi"],
                "min_dist": info["min_dist"],
                "gamma": gamma,
                "beta": beta,
                "d_margin": d_margin,
                "obstacles": clone_obstacles(obstacles),
                "next_q_full": next_q_full.copy(),
                "next_dq_full": next_dq_full.copy(),
                "n_crit": info["n_crit"].copy(),
                "d_crit": info["d_crit"],
                "d_dot_crit": info["d_dot_crit"],
                "tau_crit": info["tau_crit"],
                "h_crit": info["h_crit"],
                "Lf_h_crit": info["Lf_h_crit"],
                "vrel_sq2_crit": info["vrel_sq2_crit"],
                "Lg_psi_crit": info["Lg_psi_crit"].copy(),
                "h_all": info["h_all"].copy(),
                "Lf_h_all": info["Lf_h_all"].copy(),
                "vrel_sq2_all": info["vrel_sq2_all"].copy(),
                "Lg_psi_all": info["Lg_psi_all"].copy(),
                "pair_mask": info["pair_mask"].copy(),
                "d_all": info["d_all"].copy(),
                "d_dot_all": info["d_dot_all"].copy(),
            }
        )

        q_full = next_q_full
        dq_full = next_dq_full
        target_prev = target_next

    return history


def run_rollout_from_scenario_with_context(context, scenario_path):
    """
    Run one rollout using a preloaded reusable context.

    This is the function that parallel workers should call.
    """
    model = context["model"]
    ee_frame_id = context["ee_frame_id"]
    active_links = context["active_links"]
    self_collision_link_pair = context["self_collision_link_pair"]
    joint_limits = context["joint_limits"]

    # fresh mutable Pinocchio data per scenario run
    data = model.createData()

    scenario = load_scenario_yaml(scenario_path)
    q_full_init, dq_full_init = get_default_initial_state(model)

    dt = 1.0 / 50.0
    num_steps = int(scenario["max_sim_duration_s"] / dt)

    pin.forwardKinematics(model, data, q_full_init)
    pin.updateFramePlacements(model, data)
    target_init = data.oMf[ee_frame_id].translation.copy()

    history = rollout_episode(
        model=model,
        data=data,
        ee_frame_id=ee_frame_id,
        active_links=active_links,
        self_collision_link_pair=self_collision_link_pair,
        joint_limits=joint_limits,
        q_full_init=q_full_init,
        dq_full_init=dq_full_init,
        target_ee_pos_cartesian=scenario["goal_ee_pos"],
        obstacles_init=scenario["obstacles"],
        dt=dt,
        gamma=scenario["gamma"],
        beta=scenario["beta"],
        d_margin=scenario["d_margin"],
        num_steps=num_steps,
        target_init=target_init,
        move_obstacles=True,
    )

    dataset = history_to_dataset(history)

    last = history[-1]

    q_final = last["next_q_full"].copy()
    pin.forwardKinematics(model, data, q_final)
    pin.updateFramePlacements(model, data)
    ee_final = data.oMf[ee_frame_id].translation.copy()

    dist_to_goal = np.linalg.norm(ee_final - scenario["goal_ee_pos"])

    summary = {
        "scenario": scenario["output_basename"],
        "steps": len(history),
        "final_qp_solved": last["qp_solved"],
        "final_min_h": last["min_h"],
        "final_min_psi": last["min_psi"],
        "final_min_dist": last["min_dist"],
        "final_dist_to_goal": dist_to_goal,
        "final_q_arm": q_final[:7].copy(),
        "history": history,
        "scenario_data": scenario,
        "dataset": dataset,
    }

    return summary


def run_rollout_from_scenario(scenario_path, urdf_path=None):
    """
    Backward-compatible convenience wrapper for single-scenario use.
    """
    context = make_rollout_context(urdf_path=urdf_path)
    return run_rollout_from_scenario_with_context(context, scenario_path)

def history_to_dataset(history):
    dataset = {
        "q_arm": [],
        "dq_arm": [],
        "target_prev": [],
        "target_next": [],
        "ddq_nominal_arm": [],
        "ddq_safe_arm": [],
        "qp_solved": [],
        "min_h": [],
        "min_psi": [],
        "min_dist": [],
        "gamma": [],
        "beta": [],
        "d_margin": [],
        "time": [],
        "n_crit": [],
        "d_crit": [],
        "d_dot_crit": [],
        "tau_crit": [],
        "h_crit": [],
        "Lf_h_crit": [],
        "vrel_sq2_crit": [],
        "Lg_psi_crit": [],
        "h_all": [],
        "Lf_h_all": [],
        "vrel_sq2_all": [],
        "Lg_psi_all": [],
        "pair_mask": [],
        "d_all": [],
        "d_dot_all": [],
    }

    for step in history:
        dataset["q_arm"].append(step["q_arm"])
        dataset["dq_arm"].append(step["dq_arm"])
        dataset["target_prev"].append(step["target_prev"])
        dataset["target_next"].append(step["target_next"])
        dataset["ddq_nominal_arm"].append(step["ddq_nominal_arm"])
        dataset["ddq_safe_arm"].append(step["ddq_safe_arm"])
        dataset["qp_solved"].append(step["qp_solved"])
        dataset["min_h"].append(step["min_h"])
        dataset["min_psi"].append(step["min_psi"])
        dataset["min_dist"].append(step["min_dist"])
        dataset["gamma"].append(step["gamma"])
        dataset["beta"].append(step["beta"])
        dataset["d_margin"].append(step["d_margin"])
        dataset["time"].append(step["time"])
        dataset["n_crit"].append(step["n_crit"])
        dataset["d_crit"].append(step["d_crit"])
        dataset["d_dot_crit"].append(step["d_dot_crit"])
        dataset["tau_crit"].append(step["tau_crit"])
        dataset["h_crit"].append(step["h_crit"])
        dataset["Lf_h_crit"].append(step["Lf_h_crit"])
        dataset["vrel_sq2_crit"].append(step["vrel_sq2_crit"])
        dataset["Lg_psi_crit"].append(step["Lg_psi_crit"])
        dataset["h_all"].append(step["h_all"])
        dataset["Lf_h_all"].append(step["Lf_h_all"])
        dataset["vrel_sq2_all"].append(step["vrel_sq2_all"])
        dataset["Lg_psi_all"].append(step["Lg_psi_all"])
        dataset["pair_mask"].append(step["pair_mask"])
        dataset["d_all"].append(step["d_all"])
        dataset["d_dot_all"].append(step["d_dot_all"])

    for key in dataset:
        dataset[key] = np.array(dataset[key])

    return dataset


def save_dataset_npz(dataset, output_path):
    np.savez(output_path, **dataset)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fr3_rollout.py <scenario_yaml_path> [output_npz_path]")
        return

    scenario_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_npz = sys.argv[2]
    else:
        output_npz = None

    result = run_rollout_from_scenario(scenario_path)

    print("scenario =", result["scenario"])
    print("steps =", result["steps"])
    print("final_qp_solved =", result["final_qp_solved"])
    print("final_min_h =", result["final_min_h"])
    print("final_min_psi =", result["final_min_psi"])
    print("final_min_dist =", result["final_min_dist"])
    print("final_dist_to_goal =", result["final_dist_to_goal"])
    print("final_q_arm =", result["final_q_arm"])

    if output_npz is None:
        output_npz = result["scenario"] + "_offline_rollout.npz"

    save_dataset_npz(result["dataset"], output_npz)
    print("saved_dataset =", output_npz)


if __name__ == "__main__":
    main()