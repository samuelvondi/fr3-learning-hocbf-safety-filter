#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import pinocchio as pin

from fr3_geometry import CapsuleKinematics, compute_pair_geometry


def h_func(p_rel, robot_link_radius, obstacle_radius, d_margin=0.0):
    R_eff_sq = (robot_link_radius + obstacle_radius + d_margin) ** 2
    return np.dot(p_rel, p_rel) - R_eff_sq


def Lf_h(p_rel, v_rel):
    return 2.0 * np.dot(p_rel, v_rel)


def psi_func(h_val, Lf_h_val, gamma_param):
    return Lf_h_val + gamma_param * h_val


def Lf_psi(v_rel, Lf_h_val, gamma_param):
    term_vel_sq = 2.0 * np.dot(v_rel, v_rel)
    return term_vel_sq + gamma_param * Lf_h_val


def Lg_psi(J_p_robot_closest, p_rel, num_arm_joints):
    J_p_arm = J_p_robot_closest[:, :num_arm_joints]
    return 2.0 * np.dot(p_rel, J_p_arm)


def solve_hocbf_qp_standalone(
    model,
    data,
    num_arm_joints,
    dt_val,
    q_full_curr,
    dq_full_curr,
    ddq_nominal_arm_val,
    current_gamma_js_val,
    current_beta_js_val,
    d_margin,
    current_obstacles_list_sim,
    active_links_list,
    self_collision_link_pair,
    ddq_min_arm,
    ddq_max_arm,
    dq_min_arm,
    dq_max_arm,
    q_min_arm,
    q_max_arm,
):
    u_qp_js = cp.Variable(num_arm_joints)

    ddq_nominal_arm_np = np.array(ddq_nominal_arm_val, dtype=float).flatten()

    cost = cp.sum_squares(u_qp_js - ddq_nominal_arm_np)
    constraints_qp = []
    qp_failed_flag = False

    ddq_full_zeros = np.zeros(model.nv)
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, ddq_full_zeros)
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)

    # robot-obstacle constraints
    for link_info in active_links_list:
        start_frame_id = link_info['start_frame_id']
        end_frame_id = link_info['end_frame_id']
        link_radius_val = link_info['radius']

        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        for obs_idx, obs_item in enumerate(current_obstacles_list_sim):
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
                name=link_info['name'],
            )

            obs_capsule = CapsuleKinematics(
                p0=obs_item['pose_start'],
                p1=obs_item['pose_end'],
                radius=obs_item['radius'],
                linear_velocity=obs_item['velocity'],
                name=f"obs_{obs_idx}",
            )

            pair = compute_pair_geometry(
                robot_capsule,
                obs_capsule,
                dq=dq_full_curr,
                d_margin=d_margin,
            )

            p_rel = pair.p_rel
            h_val = pair.h

            if h_val < 0.07:
                J_C = pair.J_a
                v_rel = pair.v_rel

                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
                Lf_psi_val = Lf_psi(v_rel, Lf_h_val, current_gamma_js_val)
                Lg_psi_val_arm = Lg_psi(J_C, p_rel, num_arm_joints)

                constraints_qp.append(
                    Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val
                )

    # self-collision constraints
    for link1, link2 in self_collision_link_pair:
        link1_info = next((l for l in active_links_list if l['name'] == link1), None)
        link2_info = next((l for l in active_links_list if l['name'] == link2), None)

        if link1_info is None or link2_info is None:
            continue

        start_frame_id_1 = link1_info['start_frame_id']
        end_frame_id_1 = link1_info['end_frame_id']
        link_radius_val_1 = link1_info['radius']
        p1_1 = data.oMf[start_frame_id_1].translation
        p2_1 = data.oMf[end_frame_id_1].translation

        start_frame_id_2 = link2_info['start_frame_id']
        end_frame_id_2 = link2_info['end_frame_id']
        link_radius_val_2 = link2_info['radius']
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
            dq=dq_full_curr,
        )

        p_rel = pair.p_rel
        h_val = pair.h

        if h_val < 0.03:
            J_rel = pair.J_rel
            v_rel = pair.v_rel

            Lf_h_val = Lf_h(p_rel, v_rel)
            psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
            Lf_psi_val = Lf_psi(v_rel, Lf_h_val, current_gamma_js_val)
            Lg_psi_val_arm = Lg_psi(J_rel, p_rel, num_arm_joints)

            constraints_qp.append(
                Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val
            )

    # acceleration limits
    constraints_qp.append(u_qp_js >= ddq_min_arm)
    constraints_qp.append(u_qp_js <= ddq_max_arm)

    # velocity limits
    dq_current_arm_val = dq_full_curr[:num_arm_joints]
    constraints_qp.append(u_qp_js >= (dq_min_arm - dq_current_arm_val) / dt_val)
    constraints_qp.append(u_qp_js <= (dq_max_arm - dq_current_arm_val) / dt_val)

    # position limits
    q_current_arm_val = q_full_curr[:num_arm_joints]
    constraints_qp.append(
        u_qp_js >= 2.0 * (q_min_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val ** 2)
    )
    constraints_qp.append(
        u_qp_js <= 2.0 * (q_max_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val ** 2)
    )

    problem = cp.Problem(cp.Minimize(cost), constraints_qp)
    ddq_safe_arm_val = np.zeros(num_arm_joints)
    qp_solved_successfully = False

    try:
        problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=25000,
        )
    except cp.error.SolverError:
        try:
            problem.solve(verbose=False)
        except Exception:
            qp_failed_flag = True

    if not qp_failed_flag and (problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE):
        if u_qp_js.value is not None:
            ddq_safe_arm_val = u_qp_js.value
            qp_solved_successfully = True
        else:
            qp_failed_flag = True
    else:
        qp_failed_flag = True

    return np.array(ddq_safe_arm_val).flatten(), qp_solved_successfully