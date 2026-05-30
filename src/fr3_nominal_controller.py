#!/usr/bin/env python3

import numpy as np
import pinocchio as pin

NUM_ARM_JOINTS = 7

q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])


def nominal_controller_js_standalone(
    model,
    data,
    ee_frame_id,
    q_arm_curr,
    dq_arm_curr,
    target_ee_pos_cartesian,
    target,
):
    """
    Plain-Python version of the JS PD nominal controller from pd_nominal_controller.py.

    Inputs:
        model, data, ee_frame_id : Pinocchio objects / EE frame id
        q_arm_curr               : current 7D joint position
        dq_arm_curr              : current 7D joint velocity
        target_ee_pos_cartesian  : desired Cartesian EE position, shape (3,)
        target                   : previous smoothed target, shape (3,)

    Returns:
        ddq_nominal_arm          : nominal 7D joint acceleration
        target                   : updated smoothed target
    """

    # match the ROS implementation exactly
    Kp_cart = 500.0
    Kd_cart = 50.0
    alpha = 0.01
    lambda_damp = 0.01
    Kp_joint_limit = 200.0
    activation_threshold = 0.9

    q_arm_curr = np.asarray(q_arm_curr, dtype=float).reshape(NUM_ARM_JOINTS)
    dq_arm_curr = np.asarray(dq_arm_curr, dtype=float).reshape(NUM_ARM_JOINTS)
    target_ee_pos_cartesian = np.asarray(target_ee_pos_cartesian, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)

    # full Pinocchio state
    q_full_curr = np.zeros(model.nq)
    q_full_curr[:NUM_ARM_JOINTS] = q_arm_curr

    dq_full_curr = np.zeros(model.nv)
    dq_full_curr[:NUM_ARM_JOINTS] = dq_arm_curr

    # kinematics
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)

    # current EE pose and Jacobian
    current_ee_pos = data.oMf[ee_frame_id].translation
    J_ee_full = pin.getFrameJacobian(
        model, data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]

    # smooth target interpolation
    target = (1.0 - alpha) * target + alpha * target_ee_pos_cartesian

    # Cartesian errors
    error_pos_cart = target - current_ee_pos
    current_ee_vel_cart = J_ee_p_arm @ dq_arm_curr
    error_vel_cart = -current_ee_vel_cart

    # desired Cartesian force / acceleration-like command
    f_desired_cart = Kp_cart * error_pos_cart + Kd_cart * error_vel_cart

    # primary task
    J_pseudo_inv = np.linalg.pinv(J_ee_p_arm, rcond=lambda_damp)
    ddq_primary_task = J_pseudo_inv @ f_desired_cart

    # secondary task: joint limit avoidance
    q_mid = (q_max_arm + q_min_arm) / 2.0
    q_range = q_max_arm - q_min_arm
    ddq_secondary_task = np.zeros(NUM_ARM_JOINTS)

    for i in range(NUM_ARM_JOINTS):
        if abs(q_arm_curr[i] - q_mid[i]) > (q_range[i] * activation_threshold / 2.0):
            gradient = -2.0 * (q_arm_curr[i] - q_mid[i]) / (q_range[i] ** 2)
            ddq_secondary_task[i] = Kp_joint_limit * gradient

    # null-space projection
    I = np.identity(NUM_ARM_JOINTS)
    null_space_projector = I - (J_pseudo_inv @ J_ee_p_arm)
    ddq_secondary_projected = null_space_projector @ ddq_secondary_task

    # combine
    ddq_nominal_arm = ddq_primary_task + ddq_secondary_projected

    return ddq_nominal_arm, target