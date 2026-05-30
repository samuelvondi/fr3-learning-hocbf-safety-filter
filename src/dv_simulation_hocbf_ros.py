#!/usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pinocchio as pin
import os
import yaml
import pandas as pd
import time as timer
import json
from pathlib import Path

from fr3_rollout import compute_all_pair_terms
from fr3_qp_data import NUM_PAIR_ROWS
from fr3_compare_rollout_policies import (
    load_nn_policy,
    predict_p12_nn_current,
    minimal_g12_p12_current,
    min_g1_current,
    g12_values_hold,
)

# =============================================================================
# CONTROLLER SELECTION FLAG - CHANGE THIS TO SWITCH CONTROLLERS
# ======================================================================
USE_MPC_CONTROLLER = False  # Final ROS benchmark uses Davide's PD nominal path


# --- Pinocchio Model Setup ---
URDF_FILENAME = "fr3_robot.urdf"
package_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(package_directory, "include", "urdf", URDF_FILENAME)

if not os.path.exists(URDF_PATH):
    rclpy.logging.get_logger('cbf_controller_node').error(f"URDF file '{URDF_FILENAME}' not found at {URDF_PATH}.")
try:
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    print(f"Pinocchio model loaded from {URDF_PATH}: {model.name}")
except Exception as e:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error loading URDF with Pinocchio: {e}")
    exit()

NUM_ARM_JOINTS = 7
if model.nv < NUM_ARM_JOINTS:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error: Model nv ({model.nv}) is less than expected NUM_ARM_JOINTS ({NUM_ARM_JOINTS})")
    exit()

EE_FRAME_NAME = "fr3_hand_tcp"
try:
    EE_FRAME_ID = model.getFrameId(EE_FRAME_NAME)
except IndexError:
    rclpy.logging.get_logger('cbf_controller_node').error(f"Error: End-effector frame '{EE_FRAME_NAME}' not found in URDF.")
    rclpy.logging.get_logger('cbf_controller_node').error(f"Available frames: {[f.name for f in model.frames]}")
    exit()


# --- Define Control links on the Robot ---
links_def = [
    {'name': 'link2_base', 'start_frame_name': 'fr3_link2_offset1', 'end_frame_name': 'fr3_link2_offset2', 'radius': 0.055},
    {'name': 'link2', 'start_frame_name': 'fr3_link2', 'end_frame_name': 'fr3_link3', 'radius': 0.06},
    {'name': 'joint4',   'start_frame_name': 'fr3_link4', 'end_frame_name': 'fr3_link5_offset1', 'radius': 0.065},
    {'name': 'forearm1',   'start_frame_name': 'fr3_link5_offset2', 'end_frame_name': 'fr3_link5_offset3', 'radius': 0.035},
    {'name': 'forearm2',   'start_frame_name': 'fr3_link5_offset3', 'end_frame_name': 'fr3_link5', 'radius': 0.05},
    {'name': 'wrist',     'start_frame_name': 'fr3_link7_offset1', 'end_frame_name': 'fr3_hand',  'radius': 0.055},
    {'name': 'hand',     'start_frame_name': 'fr3_hand_offset1', 'end_frame_name': 'fr3_hand_offset2',  'radius': 0.03},
    {'name': 'end_effector',      'start_frame_name': EE_FRAME_NAME,  'end_frame_name': EE_FRAME_NAME, 'radius': 0.03},
]
active_links = []

self_collision_link_pair = [
    ('link2', 'hand'),
    ('link2', 'end_effector'),
    ('link2', 'forearm1')
]

rclpy.logging.get_logger('cbf_controller_node').info("Setting up control links (capsules)...")
for link_def in links_def:
    try:
        start_frame_id = model.getFrameId(link_def['start_frame_name'])
        end_frame_id = model.getFrameId(link_def['end_frame_name'])
        active_links.append({
            'name': link_def['name'],
            'start_frame_id': start_frame_id,
            'end_frame_id': end_frame_id,
            'radius': link_def['radius']
        })
    except IndexError as e:
        rclpy.logging.get_logger('cbf_controller_node').warn(
            f"  ! Warning: A frame for link '{link_def['name']}' not found in URDF: {e}. Skipping."
        )

if not active_links:
    rclpy.logging.get_logger('cbf_controller_node').error("No active links defined. Cannot run CBF. Exiting.")
    exit()

# Fixed row layout used by the offline-trained FR3 p12 NN.
MAX_OBSTACLE_SLOTS_FOR_NN = 5
NUM_ROBOT_OBSTACLE_ROWS_FOR_NN = len(active_links) * MAX_OBSTACLE_SLOTS_FOR_NN

ddq_max_scalar = 40.0  # rad/s^2
ddq_max_arm = np.full(NUM_ARM_JOINTS, ddq_max_scalar)
ddq_min_arm = np.full(NUM_ARM_JOINTS, -ddq_max_scalar)

# Joint velocity limits
dq_max_arm = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0]) # rad/s
dq_min_arm = - dq_max_arm # Symmetric limits

# Joint position limits
q_max_arm = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
q_min_arm = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])

# --- Constants for dynamic gamma/beta calculation ---
EPSILON_DENOMINATOR = 1e-6 # Avoid division by zero when h or psi are very close to zero
GAMMA_ADJUST_BUFFER = 0.5 # Small buffer to add to min_gamma_required to prevent numerical issues or oscillations
GAMMA_MAX_LIMIT = 200.0 # Upper limit for dynamically adjusted gamma_js (prevent it from exploding)
BETA_ADJUST_BUFFER = 0.5 # Small buffer to add to min_beta_required
BETA_MAX_LIMIT = 250.0 # Upper limit for dynamically adjusted beta_js

# --- Geometry Helper for Capsule-to-Capsule distance ---
def get_closest_points_between_segments(p1, p2, q1, q2):
    """
    Calculates the closest points between two line segments (p1, p2) and (q1, q2).


    Args:
        p1, p2 (np.array): Endpoints of the first segment.
        q1, q2 (np.array): Endpoints of the second segment.

    Returns:
        tuple: (c1, c2, s) where c1 is the closest point on segment 1, 
               c2 is the closest point on segment 2, and s is the interpolation
               factor for the first segment (the robot link).
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.dot(u, u)  # |u|^2
    b = np.dot(u, v)
    c = np.dot(v, v)  # |v|^2
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b  # D = |u|^2 * |v|^2 - |u.v|^2 = |u|^2 * |v|^2 * sin^2(theta)
    
    s_c, t_c = D, D  # Numerators for s and t
    s, t = 0.0, 0.0    # Final parameters

    # Compute the line parameters of the two closest points on the infinite lines
    # connecting vector of the closest points: dist = (p1 + s * u) - (q1 + t * v)
    # Both u and v must be parallel to the connecting vector dist * u = 0 and dist * v = 0
    # Which yelds the equations:
    # s = (b*e - c*d) / D
    # t = (a*e - b*d) / D
    if D < 1e-7:  # Lines are parallel
        s = np.clip(-d / a, 0.0, 1.0) if a > 1e-7 else 0.0
        t = (b * s + e) / c if c > 1e-7 else 0.0
    else:
        s_c = (b * e - c * d)
        t_c = (a * e - b * d)
        
        if s_c < 0.0:
            s_c = 0.0
            t_c = e
            t_c_denom = c
        elif s_c > D:
            s_c = D
            t_c = e + b
            t_c_denom = c
        else:
            t_c_denom = D
        
        s = s_c / D
        
        if t_c < 0.0:
            t = 0.0
            s_num = -d
            if s_num < 0.0:
                s = 0.0
            elif s_num > a:
                s = 1.0
            else:
                s = s_num / a if a > 1e-7 else 0.0
        elif t_c > t_c_denom:
            t = 1.0
            s_num = -d + b
            if s_num < 0.0:
                s = 0.0
            elif s_num > a:
                s = 1.0
            else:
                s = s_num / a if a > 1e-7 else 0.0
        else:
            t = t_c / t_c_denom

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    c1 = p1 + s * u
    c2 = q1 + t * v
    
    return c1, c2, s, t


# --- Joint Space HOCBF Functions ---
def h_func(p_rel, robot_link_radius, obstacle_radius, d_margin=0):
    """Barrier function h(x) for two capsules."""
    R_eff_sq = (robot_link_radius + obstacle_radius + d_margin)**2
    return np.dot(p_rel, p_rel) - R_eff_sq

def Lf_h(p_rel, v_rel):
    """Lie derivative of h along f (drift dynamics). v_rel = v_robot - v_obstacle."""
    return 2 * np.dot(p_rel, v_rel)

def psi_func(h_val, Lf_h_val, gamma_param):
    """Psi function for HOCBF."""
    return Lf_h_val + gamma_param * h_val

def Lf_psi(v_rel, Lf_h_val, gamma_param):
    """Lie derivative of psi along f."""
    term_vel_sq = 2 * np.dot(v_rel, v_rel)
    return term_vel_sq + gamma_param * Lf_h_val

def Lg_psi(J_p_robot_closest, p_rel):
    """Lie derivative of psi along g (actuation)."""
    J_p_arm = J_p_robot_closest[:, :NUM_ARM_JOINTS]
    return 2 * np.dot(p_rel, J_p_arm)


def dv_joint_limits_dict():
    """Joint-limit dictionary expected by the final FR3 NN/fallback helpers."""
    return {
        "ddq_min_arm": ddq_min_arm.copy(),
        "ddq_max_arm": ddq_max_arm.copy(),
        "dq_min_arm": dq_min_arm.copy(),
        "dq_max_arm": dq_max_arm.copy(),
        "q_min_arm": q_min_arm.copy(),
        "q_max_arm": q_max_arm.copy(),
    }


def dv_pair_scope_mask(pair_mask, pair_scope):
    """Restrict fixed 43-row pair masks to obstacle rows or all rows."""
    mask = np.asarray(pair_mask, dtype=float).reshape(NUM_PAIR_ROWS) > 0.5
    if pair_scope == "obstacle":
        scoped = np.zeros(NUM_PAIR_ROWS, dtype=bool)
        scoped[:NUM_ROBOT_OBSTACLE_ROWS_FOR_NN] = mask[:NUM_ROBOT_OBSTACLE_ROWS_FOR_NN]
        return scoped
    if pair_scope == "all":
        return mask
    raise ValueError(f"Unsupported pair scope: {pair_scope}")


def dv_min_g2_margin_from_terms(policy_terms, ddq_arm, p1, p2, pair_scope):
    """
    Signed active-row HOCBF G2 margin for a selected acceleration.

    This is a diagnostic/fallback monitor only. The actual ROS QP remains
    Davide's original cvxpy formulation in solve_hocbf_qp().
    """
    ddq = np.asarray(ddq_arm, dtype=float).reshape(NUM_ARM_JOINTS)
    if not np.all(np.isfinite(ddq)):
        return float("nan")

    active = dv_pair_scope_mask(policy_terms["pair_mask"], pair_scope)
    if not np.any(active):
        return float("nan")

    h_all = np.asarray(policy_terms["h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lf_h_all = np.asarray(policy_terms["Lf_h_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    vrel_sq2_all = np.asarray(policy_terms["vrel_sq2_all"], dtype=float).reshape(NUM_PAIR_ROWS)
    Lg_all = np.asarray(policy_terms["Lg_psi_all"], dtype=float).reshape(NUM_PAIR_ROWS, NUM_ARM_JOINTS)

    psi_all = Lf_h_all + float(p1) * h_all
    g2_all = (
        vrel_sq2_all
        + float(p1) * Lf_h_all
        + Lg_all @ ddq
        + float(p2) * psi_all
    )
    return float(np.min(g2_all[active]))


def dv_finite_or_none(value):
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def dv_series_stat(values, mode):
    x = np.asarray([float(v) for v in values if dv_finite_or_none(v) is not None], dtype=float)
    if x.size == 0:
        return None
    if mode == "min":
        return float(np.min(x))
    if mode == "max":
        return float(np.max(x))
    if mode == "mean":
        return float(np.mean(x))
    if mode == "p95":
        return float(np.percentile(x, 95))
    raise ValueError(mode)


# Helper to avoid code duplication between QP solver and dynamic param calculation
def get_point_kinematics(start_frame_id, end_frame_id, t, dq_full_curr):
    J_full_p1 = pin.getFrameJacobian(model, data, start_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    J_full_p2 = pin.getFrameJacobian(model, data, end_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    J_C = (1 - t) * J_full_p1 + t * J_full_p2

    v_C = J_C @ dq_full_curr
    return J_C, v_C

# --- Nominal Controller (Joint Space PD to Cartesian Goal) ---
def nominal_controller_pd(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, target):
    Kp_cart = 500.0 # Your current Kp
    Kd_cart = 50.0 # Your current Kd
    alpha = 0.01
    lambda_damp = 0.01 # Damping for pseudo-inverse
    
    Kp_joint_limit = 200.0  # Gain for the repulsive force
    activation_threshold = 0.9 # (e.g., 0.9 means start avoiding at 90% of the joint range)

    q_full_curr = np.zeros(model.nq)
    q_full_curr[:NUM_ARM_JOINTS] = q_arm_curr
    dq_full_curr = np.zeros(model.nv)
    dq_full_curr[:NUM_ARM_JOINTS] = dq_arm_curr

    # Perform kinematics once for nominal controller
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)
    
    current_ee_pos = data.oMf[EE_FRAME_ID].translation
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]

    target = target_ee_pos_cartesian
    target = (1-alpha)*target + alpha*target_ee_pos_cartesian
    error_pos_cart = target - current_ee_pos
    current_ee_vel_cart = J_ee_p_arm @ dq_arm_curr
    error_vel_cart = -current_ee_vel_cart

    f_desired_cart = Kp_cart * error_pos_cart + Kd_cart * error_vel_cart

    # Use np.linalg.pinv directly as it's more robust
    J_pseudo_inv = np.linalg.pinv(J_ee_p_arm, rcond=lambda_damp)
    
    ddq_primary_task = J_pseudo_inv @ f_desired_cart

    # Joint Limit Avoidance

    # Calculate the middle point and range for each joint
    q_mid = (q_max_arm + q_min_arm) / 2.0
    q_range = q_max_arm - q_min_arm
    
    ddq_secondary_task = np.zeros(NUM_ARM_JOINTS)
    
    for i in range(NUM_ARM_JOINTS):
        # Check if the joint is within the activation zone near its limits
        if abs(q_arm_curr[i] - q_mid[i]) / q_range[i] > (activation_threshold / 2.0):
            # This calculates a "repulsive" acceleration that pushes the joint back towards its center
            gradient = -2 * (q_arm_curr[i] - q_mid[i]) / (q_range[i]**2)
            ddq_secondary_task[i] = Kp_joint_limit * gradient

    # --- Null-Space Projection ---

    # 1. Calculate the null-space projector for the Jacobian
    I = np.identity(NUM_ARM_JOINTS)
    null_space_projector = I - J_pseudo_inv @ J_ee_p_arm
    
    # 2. Project the secondary task into the null space
    ddq_secondary_projected = null_space_projector @ ddq_secondary_task
        
    ddq_nominal_arm = ddq_primary_task + ddq_secondary_projected
    
    return ddq_nominal_arm, target

def joint_space_pd_controller(q_current, dq_current, q_target, dq_target):
    """
    A simple joint-space PD controller to track a target state.
    """
    Kp = 400.0
    Kd = 40.0

    error_pos = q_target - q_current
    error_vel = dq_target - dq_current

    # Calculate the desired joint acceleration
    ddq_nominal = Kp * error_pos + Kd * error_vel
    return ddq_nominal


# --- Nominal Controller (MPC to Cartesian Goal) ---
    
import casadi as ca
def nominal_controller_mpc(q_arm_curr, dq_arm_curr, target_ee_pos_cartesian, dt):
    """
    A simple kinematic MPC to generate a nominal joint acceleration command.
    This controller respects joint limits but is blind to obstacles.
    """
    # --- MPC Parameters (Tuned for stability and speed) ---
    N = 7          # Prediction horizon (longer horizon for smoother plans)
    Q_pos = 100.0  # Weight for end-effector position error (reduced for less aggression)
    R_accel = 0.001   # Weight for control effort (encourages smaller accelerations)
    W_jerk = 0.01    # Weight for jerk cost (encourages smooth accelerations)

    # --- Setup the Optimization Problem with CasADi ---
    opti = ca.Opti()

    # Decision variables (control inputs)
    ddq = opti.variable(NUM_ARM_JOINTS, N)

    # State variables (predicted trajectory)
    q = opti.variable(NUM_ARM_JOINTS, N + 1)
    dq = opti.variable(NUM_ARM_JOINTS, N + 1)

    # --- Cost Function ---
    cost = 0
    cost += R_accel * ca.sumsqr(ddq) # Penalize large control inputs

    # --- Constraints ---
    # Initial state constraint
    opti.subject_to(q[:, 0] == q_arm_curr)
    opti.subject_to(dq[:, 0] == dq_arm_curr)

    max_decel = np.abs(ddq_min_arm)
    stop_cost = 0
    jerk_cost = 0

    q_full_pin = np.zeros(model.nq)
    q_full_pin[:NUM_ARM_JOINTS] = q_arm_curr
    pin.forwardKinematics(model, data, q_full_pin)
    pin.computeJointJacobians(model, data, q_full_pin)
    pin.updateFramePlacements(model, data)
    
    J_ee_full = pin.getFrameJacobian(model, data, EE_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_ee_p_arm = J_ee_full[:3, :NUM_ARM_JOINTS]
    current_ee_pos = data.oMf[EE_FRAME_ID].translation

    # System dynamics and state/input constraints over the horizon
    for k in range(N):
        # Kinematic integration (Euler)
        q_next = q[:, k] + dq[:, k] * dt + 0.5 * ddq[:, k] * dt**2
        dq_next = dq[:, k] + ddq[:, k] * dt

        opti.subject_to(q[:, k + 1] == q_next)
        opti.subject_to(dq[:, k + 1] == dq_next)

        # CRITICAL: Enforce all joint limits on the predicted trajectory
        opti.subject_to(opti.bounded(q_min_arm, q[:, k + 1], q_max_arm))
        opti.subject_to(opti.bounded(dq_min_arm, dq[:, k + 1], dq_max_arm))
        # opti.subject_to(opti.bounded(ddq_min_arm, ddq[:, k], ddq_max_arm))

        # Stop Cost
        time_to_stop = (N - 1 - k) * dt
        dq_max_stoppable = max_decel * time_to_stop
        velocity_overshoot = ca.fmax(0, ca.fabs(dq[:, k+1]) - dq_max_stoppable)
        stop_cost += ca.sumsqr(velocity_overshoot)

        # Jerk Cost (applied to the decision variables)
        if k > 0:
            jerk_cost += ca.sumsqr(ddq[:, k] - ddq[:, k-1])

        # --- Terminal Cost (End-Effector Position) ---
        predicted_ee_pos = current_ee_pos + J_ee_p_arm @ (q[:, k] - q_arm_curr)
        error_pos_cart = predicted_ee_pos - target_ee_pos_cartesian
        cost += Q_pos**(k/2) * ca.sumsqr(error_pos_cart)

    cost += W_jerk * jerk_cost
    
    opti.minimize(cost)

    # --- Solve the QP ---
    p_opts = {"expand": True}
    # Use 'qpoases' for a massive speedup on this QP problem
    s_opts = {"print_level": 0, "sb": "yes"} 
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        # ddq_nominal_arm = sol.value(ddq[:, 0])
        planned_q = sol.value(q)
        planned_dq = sol.value(dq)
        
        return planned_q, planned_dq
    except RuntimeError:
        print("MPC Solver failed. Returning None.")
        return None, None
    

# --- HOCBF-QP Safety Filter ---
def solve_hocbf_qp(dt_val, q_full_curr, dq_full_curr, ddq_nominal_arm_val, current_gamma_js_val, current_beta_js_val, d_margin, current_obstacles_list_sim, active_links_list, node_logger):
    u_qp_js = cp.Variable(NUM_ARM_JOINTS)

    ddq_nominal_arm_np = np.array(ddq_nominal_arm_val).flatten()

    cost = cp.sum_squares(u_qp_js - ddq_nominal_arm_np)
    constraints_qp = []
    qp_failed_flag = False

    # Pre-calculate all necessary kinematics once per control loop for all CPs
    # This prevents redundant Pinocchio computations within the nested loops.
    ddq_full_zeros = np.zeros(model.nv)
    pin.forwardKinematics(model, data, q_full_curr, dq_full_curr, ddq_full_zeros)
    pin.computeJointJacobians(model, data, q_full_curr)
    pin.updateFramePlacements(model, data)

    # Iterate through each defined LINK and each obstacle
    for link_info in active_links_list:
        start_frame_id = link_info['start_frame_id']
        end_frame_id = link_info['end_frame_id']
        link_radius_val = link_info['radius']

        # Get 3D positions of the link's start and end points
        p1 = data.oMf[start_frame_id].translation
        p2 = data.oMf[end_frame_id].translation

        for obs_item in current_obstacles_list_sim:
            # --- 1. Find the closest points between the two capsule segments ---
            c_robot, c_obs, t1, t2 = get_closest_points_between_segments(
                p1, p2, obs_item['pose_start'], obs_item['pose_end']
            )
            
            p_rel = c_robot - c_obs
            h_val = h_func(p_rel, link_radius_val, obs_item['radius'], d_margin)
            if h_val < 0.07: # Only add constraints if h is small (indicating a risk of collision) [0.15m^2 * 3 = 0.0675]

                # --- 2. Calculate kinematics for the closest point on the ROBOT ---
                J_C, v_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t1, dq_full_curr
                )

                # --- 3. Calculate relative velocity and acceleration drift ---
                # Assuming obstacle velocity is constant, so acceleration is zero
                v_rel = v_C - obs_item['velocity']

                # --- 4. Use the new HOCBF functions for capsules ---
                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
                Lf_psi_val = Lf_psi(v_rel, Lf_h_val, current_gamma_js_val)
                Lg_psi_val_arm = Lg_psi(J_C, p_rel)

                constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Iterate through self-collision pairs
    for link1, link2 in self_collision_link_pair:
        link1_info = next((l for l in active_links_list if l['name'] == link1), None)
        start_frame_id_1 = link1_info['start_frame_id']
        end_frame_id_1 = link1_info['end_frame_id']
        link_radius_val_1 = link1_info['radius']
        p1_1 = data.oMf[start_frame_id_1].translation
        p2_1 = data.oMf[end_frame_id_1].translation

        link2_info = next((l for l in active_links_list if l['name'] == link2), None)
        start_frame_id_2 = link2_info['start_frame_id']
        end_frame_id_2 = link2_info['end_frame_id']
        link_radius_val_2 = link2_info['radius']
        p1_2 = data.oMf[start_frame_id_2].translation
        p2_2 = data.oMf[end_frame_id_2].translation

        c_link1, c_link2, t1, t2 = get_closest_points_between_segments(p1_1, p2_1, p1_2, p2_2)
        p_rel = c_link1 - c_link2
        h_val = h_func(p_rel, link_radius_val_1, link_radius_val_2)
        if h_val < 0.03:  # Only add constraints if h is small (indicating a risk of self-collision)
            # Calculate kinematics for the closest point on the first link
            J_C1, v_C1 = get_point_kinematics(
                start_frame_id_1, end_frame_id_1, t1, dq_full_curr
            )
            J_C2, v_C2 = get_point_kinematics(
                start_frame_id_2, end_frame_id_2, t2, dq_full_curr
            )
            # Calculate relative velocity and acceleration drift
            v_rel = v_C1 - v_C2
            J_rel = J_C1 - J_C2

            Lf_h_val = Lf_h(p_rel, v_rel)
            psi_val = psi_func(h_val, Lf_h_val, current_gamma_js_val)
            Lf_psi_val = Lf_psi(v_rel, Lf_h_val, current_gamma_js_val)
            Lg_psi_val_arm = Lg_psi(J_rel, p_rel)

            constraints_qp.append(Lg_psi_val_arm @ u_qp_js >= -Lf_psi_val - current_beta_js_val * psi_val)

    # Joint acceleration limits
    constraints_qp.append(u_qp_js >= ddq_min_arm)
    constraints_qp.append(u_qp_js <= ddq_max_arm)

    # Joint velocity limits (forward integration based on current velocity and desired acceleration)
    dq_current_arm_val = dq_full_curr[:NUM_ARM_JOINTS]
    # dq_min <= dq_current + ddq * dt
    constraints_qp.append(u_qp_js >= (dq_min_arm - dq_current_arm_val) / dt_val)
    constraints_qp.append(u_qp_js <= (dq_max_arm - dq_current_arm_val) / dt_val)

    # Joint position limits (forward integration twice)
    q_current_arm_val = q_full_curr[:NUM_ARM_JOINTS]
    # q_min <= q_current + dq_current * dt + 0.5 * ddq * dt^2
    constraints_qp.append(u_qp_js >= 2 * (q_min_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val**2))
    constraints_qp.append(u_qp_js <= 2 * (q_max_arm - q_current_arm_val - dq_current_arm_val * dt_val) / (dt_val**2))

    problem = cp.Problem(cp.Minimize(cost), constraints_qp)
    ddq_safe_arm_val = np.zeros(NUM_ARM_JOINTS)
    qp_solved_successfully = False 

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=25000)
    except cp.error.SolverError:
        node_logger.warn(f"OSQP SolverError. Trying default solver.")
        try:
            problem.solve(verbose=False) # Fallback to default solver
        except Exception as e_alt_solve:
            qp_failed_flag = True
            node_logger.warn("OSQP SolverError. Attempting emergency stop.") 


    # --- Check problem status and u_qp_js.value for definitive QP failure ---
    if not qp_failed_flag and (problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE):
        if u_qp_js.value is not None:
            ddq_safe_arm_val = u_qp_js.value
            qp_solved_successfully = True
        else: 
            qp_failed_flag = True # Solved, but no value (shouldn't happen with OPTIMAL status)
            node_logger.warn("QP solved, but u_qp_js.value is None. Emergency stop activated.")
    else: 
        # If not optimal or inaccurate, it means it failed or is infeasible.
        qp_failed_flag = True 
        node_logger.warn(f"QP status {problem.status}. Emergency stop activated.")
    
    return np.array(ddq_safe_arm_val).flatten(), qp_solved_successfully


class CbfControllerNode(Node):
    def __init__(self):
        super().__init__('cbf_controller_node')
        self.get_logger().info('CBF Controller Node has been started.')

        # Log which controller is being used
        controller_type = "MPC + PID Tracking" if USE_MPC_CONTROLLER else "Joint-Space PD"
        self.get_logger().info(f'Using {controller_type} controller')

        # --- Declare and Load Parameters ---
        self.declare_parameter('scenario_config_file', '')
        scenario_path = self.get_parameter('scenario_config_file').get_parameter_value().string_value

        # Final runtime controls. The default reproduces Davide's integrated ROS path.
        # PD-only is obtained with use_hocbf_filter:=false.
        self.declare_parameter('use_hocbf_filter', True)
        self.declare_parameter('p12_mode', 'davide_online')
        self.declare_parameter('nn_model_path', '')
        self.declare_parameter('nn_stats_path', '')
        self.declare_parameter('nn_hidden_dim', 256)
        self.declare_parameter('p1_floor', 1e-3)
        self.declare_parameter('p2_floor', 1e-3)
        self.declare_parameter('g12_fallback_pair_scope', 'all')
        self.declare_parameter('g12_fallback_tol', 1e-9)
        self.declare_parameter('g12_fallback_p1_floor', 1e-3)
        self.declare_parameter('g12_fallback_p2_floor', 1e-3)
        self.declare_parameter('result_dir', '')
        self.declare_parameter('save_plots', True)

        self.use_hocbf_filter = bool(self.get_parameter('use_hocbf_filter').get_parameter_value().bool_value)
        self.p12_mode = self.get_parameter('p12_mode').get_parameter_value().string_value.strip()
        self.nn_model_path = self.get_parameter('nn_model_path').get_parameter_value().string_value.strip()
        self.nn_stats_path = self.get_parameter('nn_stats_path').get_parameter_value().string_value.strip()
        self.nn_hidden_dim = int(self.get_parameter('nn_hidden_dim').get_parameter_value().integer_value)
        self.p1_floor = float(self.get_parameter('p1_floor').get_parameter_value().double_value)
        self.p2_floor = float(self.get_parameter('p2_floor').get_parameter_value().double_value)
        self.g12_fallback_pair_scope = self.get_parameter('g12_fallback_pair_scope').get_parameter_value().string_value.strip()
        self.g12_fallback_tol = float(self.get_parameter('g12_fallback_tol').get_parameter_value().double_value)
        self.g12_fallback_p1_floor = float(self.get_parameter('g12_fallback_p1_floor').get_parameter_value().double_value)
        self.g12_fallback_p2_floor = float(self.get_parameter('g12_fallback_p2_floor').get_parameter_value().double_value)
        self.result_dir = self.get_parameter('result_dir').get_parameter_value().string_value.strip()
        self.save_plots = bool(self.get_parameter('save_plots').get_parameter_value().bool_value)

        valid_p12_modes = {'davide_online', 'nn', 'nn_g12_fallback'}
        if self.p12_mode not in valid_p12_modes:
            raise ValueError(f"Unsupported p12_mode '{self.p12_mode}'. Valid: {sorted(valid_p12_modes)}")
        if self.g12_fallback_pair_scope not in {'obstacle', 'all'}:
            raise ValueError("g12_fallback_pair_scope must be 'obstacle' or 'all'.")
        self.joint_limits_for_nn = dv_joint_limits_dict()
        
        # Load parameters directly from the provided YAML file
        if not scenario_path or not os.path.exists(scenario_path):
            self.get_logger().fatal("HOCBF Node requires 'scenario_config_file', but it was not provided or file does not exist. Shutting down.")
            self.destroy_node()
            return
            
        with open(scenario_path, 'r') as file:
            config = yaml.safe_load(file)
            
        hocbf_params = config.get('hocbf_controller', {}).get('ros__parameters', {})
        obstacle_params_from_config = config.get('obstacles', [])
        
        self.goal_ee_pos = np.array(hocbf_params.get('goal_ee_pos', [0.3, 0.0, 0.5])) # Default value
        self.initial_gamma_js = float(hocbf_params.get('gamma_js', 2.0))
        self.initial_beta_js = float(hocbf_params.get('beta_js', 3.0))
        self.d_margin = float(hocbf_params.get('d_margin', 0.0))
        #self.d_margin = max(0.005, self.d_margin)
        self.output_basename = hocbf_params.get('output_data_basename', 'unnamed_scenario')
        self.initial_ee_pos = None

        # --- Load termination parameters ---
        self.goal_tolerance = float(hocbf_params.get('goal_tolerance_m', 0.02)) # Default 2cm
        self.goal_settle_time_s = float(hocbf_params.get('goal_settle_time_s', 2.0)) # Must be at goal for 2s
        self.max_sim_duration_s = float(hocbf_params.get('max_sim_duration_s', 60.0)) # Timeout after 1 minute
        self.start_time_ns = self.get_clock().now().nanoseconds
        self.at_goal_timer = 0.0 # Time we have been within goal tolerance
        self.is_shutdown_initiated = False # Flag to prevent multiple shutdowns
        self.final_run_status = "NONE" # Default to TIMEOUT

        self.get_logger().info(f"--- HOCBF Controller Config for '{self.output_basename}' ---")
        runtime_mode = 'pd_only' if not self.use_hocbf_filter else self.p12_mode
        self.get_logger().info(
            f"Runtime mode={runtime_mode}, save_plots={self.save_plots}, "
            f"result_dir='{self.result_dir or '<default plots/sim>'}'"
        )

        self.nn_model = None
        self.nn_feat_mean = None
        self.nn_feat_std = None
        if self.use_hocbf_filter and self.p12_mode in {'nn', 'nn_g12_fallback'}:
            if not self.nn_model_path:
                raise ValueError("NN runtime modes require nn_model_path.")
            stats_arg = self.nn_stats_path or None
            self.nn_model, self.nn_feat_mean, self.nn_feat_std = load_nn_policy(
                model_path=self.nn_model_path,
                stats_path=stats_arg,
                hidden_dim=self.nn_hidden_dim,
                p1_floor=self.p1_floor,
                p2_floor=self.p2_floor,
            )
            stats_label = self.nn_stats_path if self.nn_stats_path else '<checkpoint feat_mean/feat_std>'
            self.get_logger().info(
                f"Loaded NN p12 policy: model='{self.nn_model_path}', stats='{stats_label}'."
            )
        
        # Create the internal representation of obstacles
        self.current_obstacles_list_sim = []
        for obs in obstacle_params_from_config:
            self.current_obstacles_list_sim.append({
                'pose_start': np.array(obs['pose_start']['position']),
                'pose_end': np.array(obs['pose_start']['position']),
                'radius': float(obs['size']['radius']),
                'velocity': np.array(obs['velocity']['linear'])
            })
            # In case of sphere pose_end = pose_start
            if 'pose_end' in obs:
                self.current_obstacles_list_sim[-1]['pose_end'] = np.array(obs['pose_end']['position'])

        # Subscription to joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.joint_state_subscription

        # Publisher for joint position commands to the C++ controller
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_position_controller/external_commands',
            10
        )

        # Publisher for RViz markers
        self.marker_publisher_1 = self.create_publisher(
            Marker,
            '/visualization_marker_1', # Standard topic for RViz markers
            10
        )

        self.marker_publisher_2 = self.create_publisher(
            Marker,
            '/visualization_marker_2',
            10
        )

        self.marker_publisher_3 = self.create_publisher(
            Marker,
            '/visualization_marker_3',
            10
        )

        self.marker_publisher_4 = self.create_publisher(
            Marker,
            '/visualization_marker_4',
            10
        )

        self.bot_marker_publisher_1 = self.create_publisher(
            Marker,
            '/bot_marker_1',
            10
        )

        self.bot_marker_publisher_2 = self.create_publisher(
            Marker,
            '/bot_marker_2',
            10
        )

        self.bot_marker_publisher_3 = self.create_publisher(
            Marker,
            '/bot_marker_3',
            10
        )

        self.bot_marker_publisher_4 = self.create_publisher(
            Marker,
            '/bot_marker_4',
            10
        )

        self.bot_marker_publisher_5 = self.create_publisher(
            Marker,
            '/bot_marker_5',
            10
        )

        # --- Publisher for the EE Trajectory Marker ---
        self.trajectory_marker_publisher = self.create_publisher(
            Marker,
            '/ee_trajectory_marker',
            10
        )

        self.robot_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        self.num_arm_joints = len(self.robot_joint_names)

        # History for plotting ALL h and psi values (overall min)
        self.history_data_frames = []

        # Final ROS benchmark runtime: keep only scalar run statistics in-loop.
        # Per-step CSV histories are intentionally disabled to avoid perturbing
        # the wall-clock control loop.
        self.lightweight_run_stats = {
            'steps': 0,
            'last_time_s': None,
            'min_h': float('inf'),
            'min_dist': float('inf'),
            'min_psi': float('inf'),
            'qp_fail_count': 0,
            'fallback_count': 0,
            'initial_dist_to_goal': None,
            'final_dist_to_goal': None,
            'p1_sum': 0.0,
            'p2_sum': 0.0,
            'p_count': 0,
            'p1_max': -float('inf'),
            'p2_max': -float('inf'),
        }

        # --- List to store the history of EE positions ---
        self.ee_position_history = []

        # --- Dynamic CBF parameters (initially global values) ---
        self.current_gamma_js = self.initial_gamma_js
        self.current_beta_js = self.initial_beta_js

        self.current_joint_positions = np.zeros(self.num_arm_joints)
        self.current_joint_velocities = np.zeros(self.num_arm_joints)
        self.received_first_joint_state = False

        # Pinocchio full state for calculations
        self.q_full_pin = np.zeros(model.nq)
        self.dq_full_pin = np.zeros(model.nv)

        # MPC-specific state (only used if USE_MPC_CONTROLLER is True).
        # mpc_solve_time is kept defined in PD mode because it is logged per step.
        self.mpc_solve_time = 0.0
        if USE_MPC_CONTROLLER:
            self.mpc_planned_trajectory = None
            self.mpc_trajectory_start_time = 0.0
            self.mpc_frequency = 10  # Hz
            self.mpc_dt = 1.0 / self.mpc_frequency

        self.target = [0.307, 0.0, 0.487] # Initial target position for the end-effector in Cartesian space


        # --- Control Loop Frequencies ---
        self.mpc_frequency = 10  # Hz
        self.pid_frequency = 50  # Hz

        self.mpc_dt = 1.0 / self.mpc_frequency
        self.dt = 1.0 / self.pid_frequency

        # --- Timers for Each Loop ---
        # High-frequency loop for PID tracking and safety filter
        self.pd_timer = self.create_timer(self.dt, self.pid_and_safety_loop)

        # Only create MPC timer if using MPC controller
        if USE_MPC_CONTROLLER:
            self.mpc_timer = self.create_timer(self.mpc_dt, self.mpc_planning_loop)
            self.get_logger().info(f'MPC planning loop created at {self.mpc_frequency} Hz')

        # Timer for publishing markers (can be slower than control loop)
        self.marker_publish_timer = self.create_timer(0.5, self.publish_markers) # Publish markers every 0.5 seconds


    def joint_states_callback(self, msg):
        if not self.received_first_joint_state:
            # Ensure correct joint order for initial setup
            for i, name in enumerate(self.robot_joint_names):
                if name not in msg.name:
                    self.get_logger().error(f"Joint '{name}' not found in received joint states. Cannot proceed.")
                    self.destroy_node() # Self-destruct if essential joints are missing
                    return
            self.received_first_joint_state = True
            
        current_positions = []
        current_velocities = []

        # Map by name to ensure correct ordering, even if msg.name order changes
        for joint_name in self.robot_joint_names:
            try:
                idx = msg.name.index(joint_name)
                current_positions.append(msg.position[idx])
                current_velocities.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
            except ValueError:
                self.get_logger().warn(f"Joint '{joint_name}' not found in current JointState message. This should not happen if initial check passed.")
                return # Skip this update if any required joint is missing

        self.current_joint_positions = np.array(current_positions)
        self.current_joint_velocities = np.array(current_velocities)
        
        # Also update Pinocchio's full state based on actual robot state
        self.q_full_pin[:self.num_arm_joints] = self.current_joint_positions
        self.dq_full_pin[:self.num_arm_joints] = self.current_joint_velocities

    def mpc_planning_loop(self):
        """
        Runs the slow MPC to generate a future trajectory of joint states.
        """
        if not self.received_first_joint_state or self.is_shutdown_initiated:
            self.get_logger().info("Control loop skipped: No joint state received yet or shutdown initiated.")
            return

        q_arm_current = self.current_joint_positions
        dq_arm_current = self.current_joint_velocities

        start_mpc_solve_time = timer.time()

        # Solve the MPC to get a plan of future states (q and dq)
        planned_q, planned_dq = nominal_controller_mpc(
            q_arm_current,
            dq_arm_current,
            self.goal_ee_pos,
            self.mpc_dt
        )

        self.mpc_solve_time = timer.time() - start_mpc_solve_time

        # If the MPC solved successfully, store the trajectory
        if planned_q is not None and planned_dq is not None:
            self.mpc_planned_trajectory = {'q': planned_q, 'dq': planned_dq}
            # Record the time the trajectory was generated
            current_time_ns = self.get_clock().now().nanoseconds
            self.mpc_trajectory_start_time = (current_time_ns - self.start_time_ns) / 1e9

    def pid_and_safety_loop(self):
        if not self.received_first_joint_state or self.is_shutdown_initiated:
            return
        
        # Check if we need MPC trajectory and don't have it yet
        if USE_MPC_CONTROLLER and self.mpc_planned_trajectory is None:
            self.get_logger().info('Waiting for first MPC trajectory...', throttle_duration_sec=1)
            return
        
        for obs in self.current_obstacles_list_sim:
            obs['pose_start'] = obs['pose_start'] + obs['velocity'] * self.dt

        q_arm_current = self.current_joint_positions
        dq_arm_current = self.current_joint_velocities

        min_h_current = float('inf')
        min_psi_current = float('inf')
        min_dist_current = float('inf')

        # --- Track maximum required gamma for dynamic adjustment ---
        max_gamma_required_for_h = -float('inf') # Initialize with negative infinity
        max_beta_required_for_psi = -float('inf') 

        # Ensure Pinocchio data is fresh before calculating h/psi
        pin.forwardKinematics(model, data, self.q_full_pin, self.dq_full_pin, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        # --- Record EE position for trajectory ---
        current_ee_pos = data.oMf[EE_FRAME_ID].translation
        self.ee_position_history.append(Point(x=current_ee_pos[0], y=current_ee_pos[1], z=current_ee_pos[2]))


        # --- Calculate State-Dependent Kinematic Acceleration Bounds ---
        # These are the true acceleration limits for the next time step, considering
        # velocity and position constraints. This is the same logic used in the QP.
        
        # 1. Bounds from Velocity Limits
        ddq_max_from_vel = (dq_max_arm - dq_arm_current) / self.dt
        ddq_min_from_vel = (dq_min_arm - dq_arm_current) / self.dt

        # 2. Bounds from Position Limits
        ddq_max_from_pos = 2 * (q_max_arm - q_arm_current - dq_arm_current * self.dt) / (self.dt**2)
        ddq_min_from_pos = 2 * (q_min_arm - q_arm_current - dq_arm_current * self.dt) / (self.dt**2)

        # 3. Combine with actuator limits to find the most restrictive bounds
        # For the max bound, we take the minimum of all maximums.
        # For the min bound, we take the maximum of all minimums.
        ddq_max_kinematic = np.minimum(ddq_max_arm, np.minimum(ddq_max_from_vel, ddq_max_from_pos))
        ddq_min_kinematic = np.maximum(ddq_min_arm, np.maximum(ddq_min_from_vel, ddq_min_from_pos))
        

        # --- LOOP for h, psi, and dynamic gamma/beta calculation for LINKS ---
        current_h_psi_values = {}

        for link_info in active_links:
            start_frame_id = link_info['start_frame_id']
            end_frame_id = link_info['end_frame_id']
            link_radius_val = link_info['radius']

            p1 = data.oMf[start_frame_id].translation
            p2 = data.oMf[end_frame_id].translation

            for obs_idx, obs_item in enumerate(self.current_obstacles_list_sim):
                c_robot, c_obs, t1, t2 = get_closest_points_between_segments(
                    p1, p2, obs_item['pose_start'], obs_item['pose_end']
                )
                
                J_C, v_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t1, self.dq_full_pin
                )

                dist_centers = np.linalg.norm(c_robot - c_obs)
                dist_surfaces = dist_centers - (link_radius_val + obs_item['radius'])
                min_dist_current = min(min_dist_current, dist_surfaces)

                p_rel = c_robot - c_obs
                h_val = h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin)
                min_h_current = min(min_h_current, h_val)
                
                v_rel = v_C - obs_item['velocity']
                
                Lf_h_val = Lf_h(p_rel, v_rel)

                # Step 1: Calculate required gamma for this pair
                if h_val > EPSILON_DENOMINATOR:
                    gamma_needed_for_pair = -Lf_h_val / h_val
                    max_gamma_required_for_h = max(max_gamma_required_for_h, gamma_needed_for_pair)
                elif h_val < -EPSILON_DENOMINATOR:
                    max_gamma_required_for_h = GAMMA_MAX_LIMIT

        # --- Adjust gamma_js FIRST ---
        if self.use_hocbf_filter and self.p12_mode == 'davide_online' and max_gamma_required_for_h > -float('inf'):
            adjusted_gamma = max(0.0, max_gamma_required_for_h + GAMMA_ADJUST_BUFFER)
            self.current_gamma_js = max(self.initial_gamma_js, min(adjusted_gamma, GAMMA_MAX_LIMIT))
        # else: keep current_gamma_js

        # Now recalc beta with updated gamma
        for link_info in active_links:
            start_frame_id = link_info['start_frame_id']
            end_frame_id = link_info['end_frame_id']
            link_radius_val = link_info['radius']

            p1 = data.oMf[start_frame_id].translation
            p2 = data.oMf[end_frame_id].translation

            for obs_idx, obs_item in enumerate(self.current_obstacles_list_sim):
                c_robot, c_obs, t1, t2 = get_closest_points_between_segments(
                    p1, p2, obs_item['pose_start'], obs_item['pose_end']
                )

                J_C, v_C = get_point_kinematics(
                    link_info['start_frame_id'], link_info['end_frame_id'], t1, self.dq_full_pin
                )

                p_rel = c_robot - c_obs
                v_rel = v_C - obs_item['velocity']

                Lf_h_val = Lf_h(p_rel, v_rel)
                psi_val = psi_func(h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin), Lf_h_val, self.current_gamma_js)
                
                # --- 4. Update min_psi (using the dynamically adjusted gamma for THIS step) ---
                min_psi_current = min(min_psi_current, psi_val)
                
                Lf_psi_val = Lf_psi(v_rel, Lf_h_val, self.current_gamma_js)
                Lg_psi_val = Lg_psi(J_C, p_rel).flatten()
                
                sup_Lg_psi_u = np.sum(np.where(
                    Lg_psi_val >= 0, 
                    Lg_psi_val * ddq_max_kinematic, 
                    Lg_psi_val * ddq_min_kinematic))
                
                S_sup_val = Lf_psi_val + sup_Lg_psi_u
                
                if psi_val > EPSILON_DENOMINATOR:
                    beta_needed_for_pair = -S_sup_val / psi_val
                    max_beta_required_for_psi = max(max_beta_required_for_psi, beta_needed_for_pair)
                elif psi_val < -EPSILON_DENOMINATOR:
                    max_beta_required_for_psi = BETA_MAX_LIMIT

                key = f"{link_info['name']}_obs{obs_idx}"
                current_h_psi_values[f'h_{key}'] = h_func(p_rel, link_radius_val, obs_item['radius'], self.d_margin)
                current_h_psi_values[f'psi_{key}'] = psi_val

        # --- Adjust beta_js AFTER gamma ---
        if self.use_hocbf_filter and self.p12_mode == 'davide_online' and max_beta_required_for_psi > -float('inf'):
            adjusted_beta = max(0.0, max_beta_required_for_psi + BETA_ADJUST_BUFFER)
            self.current_beta_js = max(self.initial_beta_js, min(adjusted_beta, BETA_MAX_LIMIT))
        # else: keep current_beta_js


        # Record current time
        current_time_ns = self.get_clock().now().nanoseconds
        current_time_s = (current_time_ns - self.start_time_ns) / 1e9

        # --- Check for termination conditions at the end of the loop ---
        # Get current EE position from Pinocchio data
        current_ee_pos = data.oMf[EE_FRAME_ID].translation
        distance_to_goal = np.linalg.norm(current_ee_pos - self.goal_ee_pos)

        if self.initial_ee_pos is None:
            # Initialize the initial EE position on the first run
            self.initial_ee_pos = current_ee_pos.copy()

        # Condition 1: Goal Reached
        if distance_to_goal < self.goal_tolerance:
            self.at_goal_timer += self.dt
            if self.at_goal_timer >= self.goal_settle_time_s:
                self.get_logger().info(f"SUCCESS: Goal reached for {self.goal_settle_time_s}s. Simulation time {current_time_s:.2f}s.")
                self.final_run_status = "SUCCESS" # Set success status
                self.is_shutdown_initiated = True
                self.destroy_node() # This will trigger the main `finally` block
                return
        else:
            self.at_goal_timer = 0.0 # Reset timer if we move out of tolerance

        # Condition 2: Timeout
        if current_time_s > self.max_sim_duration_s:
            self.get_logger().warn(f"TIMEOUT: Simulation exceeded {self.max_sim_duration_s}s. Shutting down.")
            if np.linalg.norm(current_ee_pos - self.initial_ee_pos) < 0.01: # e.g., less than 1cm of movement
                self.get_logger().error("CONTROLLER FAILED: Robot did not move from initial position.")
                self.final_run_status = "FAILED_NO_MOVEMENT"
            else:
                self.final_run_status = "TIMEOUT"
            self.is_shutdown_initiated = True
            self.destroy_node()
            return

        
        # =================================================================
        # CONTROLLER SELECTION - Generate nominal acceleration
        # =================================================================
        if USE_MPC_CONTROLLER:
            # --- MPC + PID Tracking Controller ---
            # 1. Determine the current time relative to the MPC plan's start time
            time_since_plan_start = current_time_s - self.mpc_trajectory_start_time

            # 2. Find the correct setpoint from the MPC's trajectory
            target_index = int(np.floor(time_since_plan_start / self.mpc_dt)) + 1
            
            # Ensure the index is within the bounds of the trajectory
            num_steps_in_plan = self.mpc_planned_trajectory['q'].shape[1]
            if target_index >= num_steps_in_plan:
                target_index = num_steps_in_plan - 1

            q_target = self.mpc_planned_trajectory['q'][:, target_index]
            dq_target = self.mpc_planned_trajectory['dq'][:, target_index]
            
            # 3. Use the PID controller to track the MPC trajectory
            ddq_nominal_arm_cmd = joint_space_pd_controller(
                q_arm_current, dq_arm_current, q_target, dq_target
            )
        else:
            # --- Joint-Space PD Controller ---
            # Direct operational-space control to Cartesian goal
            ddq_nominal_arm_cmd, self.target = nominal_controller_pd(
                q_arm_current, dq_arm_current, self.goal_ee_pos, self.target
            )

        # =================================================================
        # Final runtime mode: PD-only, Davide online, NN, or NN+G12 fallback
        # =================================================================
        fallback_used = False
        p1_for_summary = None
        p2_for_summary = None

        ddq_safe_arm_cmd = np.asarray(
            ddq_nominal_arm_cmd, dtype=float
        ).reshape(NUM_ARM_JOINTS).copy()
        qp_solved = True

        if not self.use_hocbf_filter:
            # Pure nominal PD baseline. No QP is solved.
            qp_solved = True

        elif self.p12_mode == 'davide_online':
            # Davide's original online gamma/beta path.
            ddq_safe_arm_cmd, qp_solved = solve_hocbf_qp(
                self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal_arm_cmd,
                self.current_gamma_js, self.current_beta_js, self.d_margin,
                self.current_obstacles_list_sim, active_links, self.get_logger()
            )
            p1_for_summary = float(self.current_gamma_js)
            p2_for_summary = float(self.current_beta_js)

        else:
            # Fixed pair terms are required by the trained NN features and by
            # the G1/G2 fallback monitor.
            policy_terms = compute_all_pair_terms(
                model=model,
                data=data,
                q_full=self.q_full_pin,
                dq_full=self.dq_full_pin,
                obstacles=self.current_obstacles_list_sim,
                active_links=active_links,
                self_collision_link_pair=self_collision_link_pair,
                d_margin=self.d_margin,
            )

            selected_p1, selected_p2 = predict_p12_nn_current(
                model=self.nn_model,
                feat_mean=self.nn_feat_mean,
                feat_std=self.nn_feat_std,
                q_arm=q_arm_current,
                dq_arm=dq_arm_current,
                ddq_nominal_arm=ddq_nominal_arm_cmd,
                joint_limits=self.joint_limits_for_nn,
                h_all=policy_terms['h_all'],
                Lf_h_all=policy_terms['Lf_h_all'],
            )

            if self.p12_mode == 'nn':
                self.current_gamma_js = float(selected_p1)
                self.current_beta_js = float(selected_p2)
                ddq_safe_arm_cmd, qp_solved = solve_hocbf_qp(
                    self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal_arm_cmd,
                    self.current_gamma_js, self.current_beta_js, self.d_margin,
                    self.current_obstacles_list_sim, active_links, self.get_logger()
                )

            elif self.p12_mode == 'nn_g12_fallback':
                # Exact existing fallback logic, kept lean:
                # 1) Solve candidate NN QP.
                # 2) Reuse it if the candidate satisfies the G1/G2 monitor.
                # 3) Solve a fallback QP only when the monitor rejects the NN candidate.
                self.current_gamma_js = float(selected_p1)
                self.current_beta_js = float(selected_p2)

                min_g1_candidate = min_g1_current(
                    all_pair_terms=policy_terms,
                    p1=selected_p1,
                    pair_scope=self.g12_fallback_pair_scope,
                )

                ddq_candidate, candidate_qp_solved = solve_hocbf_qp(
                    self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal_arm_cmd,
                    self.current_gamma_js, self.current_beta_js, self.d_margin,
                    self.current_obstacles_list_sim, active_links, self.get_logger()
                )

                min_g2_candidate = float('nan')
                if candidate_qp_solved:
                    min_g2_candidate = dv_min_g2_margin_from_terms(
                        policy_terms=policy_terms,
                        ddq_arm=ddq_candidate,
                        p1=selected_p1,
                        p2=selected_p2,
                        pair_scope=self.g12_fallback_pair_scope,
                    )

                nn_candidate_holds = bool(candidate_qp_solved) and g12_values_hold(
                    min_g1=min_g1_candidate,
                    min_g2=min_g2_candidate,
                    tol=self.g12_fallback_tol,
                )

                if nn_candidate_holds:
                    ddq_safe_arm_cmd = np.asarray(
                        ddq_candidate, dtype=float
                    ).reshape(NUM_ARM_JOINTS)
                    qp_solved = True
                else:
                    fallback_used = True
                    selected_p1, selected_p2, _, _ = minimal_g12_p12_current(
                        q_arm=q_arm_current,
                        dq_arm=dq_arm_current,
                        all_pair_terms=policy_terms,
                        limits=self.joint_limits_for_nn,
                        dt=self.dt,
                        p1_floor=self.g12_fallback_p1_floor,
                        p2_floor=self.g12_fallback_p2_floor,
                        pair_scope=self.g12_fallback_pair_scope,
                    )
                    self.current_gamma_js = float(selected_p1)
                    self.current_beta_js = float(selected_p2)
                    ddq_safe_arm_cmd, qp_solved = solve_hocbf_qp(
                        self.dt, self.q_full_pin, self.dq_full_pin, ddq_nominal_arm_cmd,
                        self.current_gamma_js, self.current_beta_js, self.d_margin,
                        self.current_obstacles_list_sim, active_links, self.get_logger()
                    )
            else:
                raise RuntimeError(f'Unhandled p12_mode: {self.p12_mode}')

            p1_for_summary = float(selected_p1)
            p2_for_summary = float(selected_p2)

        if not qp_solved:
            # Preserve Davide's original emergency-hold behavior after an infeasible QP.
            next_dq_arm = dq_arm_current
            next_q_arm = q_arm_current
            self.get_logger().warn(f"QP infeasible, using current positions: {np.round(q_arm_current, 3)}")
        else:
            next_dq_arm = dq_arm_current + ddq_safe_arm_cmd * self.dt
            next_q_arm = q_arm_current + dq_arm_current * self.dt + 0.5 * ddq_safe_arm_cmd * self.dt**2

        self.update_lightweight_run_stats(
            current_time_s=current_time_s,
            min_h_current=min_h_current,
            min_dist_current=min_dist_current,
            min_psi_current=min_psi_current,
            qp_solved=qp_solved,
            fallback_used=fallback_used,
            distance_to_goal=distance_to_goal,
            p1_value=p1_for_summary,
            p2_value=p2_for_summary,
        )

        self.publish_joint_commands(next_q_arm)

    def publish_joint_commands(self, positions_array):
        msg = Float64MultiArray()
        msg.data = positions_array.tolist()
        self.joint_command_publisher.publish(msg)

    # --- Marker Publishing Functions ---
    def create_sphere_marker(self, marker_id, position, radius, r, g, b, a, frame_id="world"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0 # No rotation for a sphere
        marker.scale.x = float(radius * 2) # Diameter
        marker.scale.y = float(radius * 2)
        marker.scale.z = float(radius * 2)
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg() # Automatically disappear if not updated

        return marker
    
    def create_cylinder_marker(self, marker_id, p1, p2, radius, r, g, b, a, frame_id="world"):
        """Creates a CYLINDER marker for a link between two points."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz_links"
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position is the midpoint of the link
        midpoint = (p1 + p2) / 2
        marker.pose.position.x = float(midpoint[0])
        marker.pose.position.y = float(midpoint[1])
        marker.pose.position.z = float(midpoint[2])

        # Scale is diameter (x,y) and length (z)
        link_vec = p2 - p1
        length = np.linalg.norm(link_vec)
        marker.scale.x = float(radius * 2)
        marker.scale.y = float(radius * 2)
        marker.scale.z = float(length)

        # Orientation: Calculate quaternion to align cylinder's Z-axis with the link vector
        if length > 1e-6:
            z_axis = np.array([0.0, 0.0, 1.0])
            axis = np.cross(z_axis, link_vec)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis /= axis_norm
                angle = np.arccos(np.dot(z_axis, link_vec) / length)
                q = pin.Quaternion(np.cos(angle / 2.0),
                                   axis[0] * np.sin(angle / 2.0),
                                   axis[1] * np.sin(angle / 2.0),
                                   axis[2] * np.sin(angle / 2.0))
                marker.pose.orientation.w = q.w
                marker.pose.orientation.x = q.x
                marker.pose.orientation.y = q.y
                marker.pose.orientation.z = q.z
            else: # Aligned with Z-axis
                marker.pose.orientation.w = 1.0
        else: # Zero length
            marker.pose.orientation.w = 1.0

        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()
        return marker

    def create_goal_marker(self, marker_id, position, frame_id="world"):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cbf_viz"
        marker.id = marker_id
        marker.type = Marker.CUBE # Using a cube for goal
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8 # Green, semi-transparent
        marker.lifetime = rclpy.duration.Duration(seconds=0.7).to_msg()

        return marker

    def publish_markers(self):
        # Publish obstacle markers
        # Marker IDs for obstacles start from 0
        for i, obs in enumerate(self.current_obstacles_list_sim):
            marker_obs_cap1 = self.create_sphere_marker(
                marker_id=i,
                position=obs['pose_start'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7 # Orangered, semi-transparent
            )
            marker_obs_cap2 = self.create_sphere_marker(
                marker_id=i+len(self.current_obstacles_list_sim),
                position=obs['pose_end'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7
            )
            marker_obs_cyl = self.create_cylinder_marker(
                marker_id=i+len(self.current_obstacles_list_sim)*2,
                p1=obs['pose_start'],
                p2=obs['pose_end'],
                radius=obs['radius'],
                r=1.0, g=0.5, b=0.0, a=0.7
            )
            self.marker_publisher_2.publish(marker_obs_cap1)
            if not np.array_equal(obs['pose_start'], obs['pose_end']):
                self.marker_publisher_3.publish(marker_obs_cap2)
                self.marker_publisher_4.publish(marker_obs_cyl)

        # Publish goal position marker
        # Marker ID for goal is after all obstacles
        marker_goal = self.create_goal_marker(
            marker_id=len(self.current_obstacles_list_sim)*3,
            position=self.goal_ee_pos
        )
        self.marker_publisher_1.publish(marker_goal)
        
        base_marker_id_link = len(self.current_obstacles_list_sim)*3 + 1 

        pin.forwardKinematics(model, data, self.q_full_pin, self.dq_full_pin, np.zeros(model.nv))
        pin.updateFramePlacements(model, data)

        bot_number_marker = 0
        for i, link_info in enumerate(active_links):
            p1 = data.oMf[link_info['start_frame_id']].translation
            p2 = data.oMf[link_info['end_frame_id']].translation
            
            marker_link = self.create_cylinder_marker(
                marker_id=base_marker_id_link + i,
                p1=p1,
                p2=p2,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6 # Blue, semi-transparent
            )
            marker_cap_1 = self.create_sphere_marker(
                marker_id=base_marker_id_link + len(active_links) + i, # Unique ID for each marker
                position=p1,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6 # Blue, semi-transparent for robot parts
            )
            marker_cap_2 = self.create_sphere_marker(
                marker_id=base_marker_id_link + 2*len(active_links) + i, # Unique ID for each marker
                position=p2,
                radius=link_info['radius'],
                r=0.2, g=0.2, b=0.8, a=0.6 # Blue, semi-transparent for robot parts
            )

            # Distribute markers among publishers (they don't plot well more than 5 markers at once)
            if bot_number_marker < 5:
                self.bot_marker_publisher_1.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 5:
                        self.bot_marker_publisher_1.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_2.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 5:
                        self.bot_marker_publisher_1.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_2.publish(marker_link)
                        bot_number_marker += 1
            elif bot_number_marker < 10:
                self.bot_marker_publisher_2.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 10:
                        self.bot_marker_publisher_2.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_3.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 10:
                        self.bot_marker_publisher_2.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_3.publish(marker_link)
                        bot_number_marker += 1
            elif bot_number_marker < 15:
                self.bot_marker_publisher_3.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 15:
                        self.bot_marker_publisher_3.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_4.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 15:
                        self.bot_marker_publisher_3.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_4.publish(marker_link)
                        bot_number_marker += 1
            elif bot_number_marker < 20:
                self.bot_marker_publisher_4.publish(marker_cap_1)
                bot_number_marker += 1
                if p1.tolist() != p2.tolist():
                    if bot_number_marker < 20:
                        self.bot_marker_publisher_4.publish(marker_cap_2)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_5.publish(marker_cap_2)
                        bot_number_marker += 1
                    if bot_number_marker < 20:
                        self.bot_marker_publisher_4.publish(marker_link)
                        bot_number_marker += 1
                    else:
                        self.bot_marker_publisher_5.publish(marker_link)
                        bot_number_marker += 1
            else:
                self.bot_marker_publisher_5.publish(marker_cap_1)
                if p1.tolist() != p2.tolist():
                    self.bot_marker_publisher_5.publish(marker_cap_2)
                    self.bot_marker_publisher_5.publish(marker_link)

        # --- Publish the End-Effector Trajectory Marker ---
        if self.ee_position_history:
            traj_marker = Marker()
            traj_marker.header.frame_id = "world"
            traj_marker.header.stamp = self.get_clock().now().to_msg()
            traj_marker.ns = "ee_trajectory"
            traj_marker.id = 0
            traj_marker.type = Marker.LINE_STRIP
            traj_marker.action = Marker.ADD

            # Set the points for the line strip
            traj_marker.points = self.ee_position_history

            # Line strip settings
            traj_marker.scale.x = 0.01  # Line width
            traj_marker.color.r = 0.0
            traj_marker.color.g = 1.0
            traj_marker.color.b = 0.8  # Cyan
            traj_marker.color.a = 1.0
            
            # This marker should persist and not auto-delete
            traj_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg() 

            self.trajectory_marker_publisher.publish(traj_marker)


    def update_lightweight_run_stats(
        self,
        current_time_s,
        min_h_current,
        min_dist_current,
        min_psi_current,
        qp_solved,
        fallback_used,
        distance_to_goal,
        p1_value,
        p2_value,
    ):
        stats = self.lightweight_run_stats
        stats['steps'] += 1
        stats['last_time_s'] = float(current_time_s)

        if np.isfinite(min_h_current):
            stats['min_h'] = min(stats['min_h'], float(min_h_current))
        if np.isfinite(min_dist_current):
            stats['min_dist'] = min(stats['min_dist'], float(min_dist_current))
        if np.isfinite(min_psi_current):
            stats['min_psi'] = min(stats['min_psi'], float(min_psi_current))

        if not qp_solved:
            stats['qp_fail_count'] += 1
        if fallback_used:
            stats['fallback_count'] += 1

        if distance_to_goal is not None and np.isfinite(distance_to_goal):
            distance_to_goal = float(distance_to_goal)
            if stats['initial_dist_to_goal'] is None:
                stats['initial_dist_to_goal'] = distance_to_goal
            stats['final_dist_to_goal'] = distance_to_goal

        if p1_value is not None and p2_value is not None:
            if np.isfinite(p1_value) and np.isfinite(p2_value):
                p1_value = float(p1_value)
                p2_value = float(p2_value)
                stats['p1_sum'] += p1_value
                stats['p2_sum'] += p2_value
                stats['p_count'] += 1
                stats['p1_max'] = max(stats['p1_max'], p1_value)
                stats['p2_max'] = max(stats['p2_max'], p2_value)

    @staticmethod
    def lightweight_finite_or_none(value):
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        return value if np.isfinite(value) else None

    def save_run_summary_json(self, output_dir, df=None):
        """Write one compact per-run JSON summary without reading per-step CSV history."""
        stats = self.lightweight_run_stats
        runtime_mode = 'pd_only' if not self.use_hocbf_filter else self.p12_mode

        initial_dist = self.lightweight_finite_or_none(stats['initial_dist_to_goal'])
        final_dist = self.lightweight_finite_or_none(stats['final_dist_to_goal'])
        progress_score = None
        if initial_dist is not None and final_dist is not None and abs(initial_dist) > 1e-12:
            progress_score = float(1.0 - final_dist / initial_dist)

        p_count = int(stats['p_count'])
        p1_mean = float(stats['p1_sum'] / p_count) if p_count > 0 else None
        p2_mean = float(stats['p2_sum'] / p_count) if p_count > 0 else None
        p1_max = self.lightweight_finite_or_none(stats['p1_max']) if p_count > 0 else None
        p2_max = self.lightweight_finite_or_none(stats['p2_max']) if p_count > 0 else None

        success = bool(self.final_run_status == 'SUCCESS')
        task_completion_time_s = (
            self.lightweight_finite_or_none(stats['last_time_s']) if success else None
        )

        summary = {
            'scenario': self.output_basename,
            'mode': runtime_mode,
            'runtime_mode': runtime_mode,
            'final_run_status': self.final_run_status,
            'success': success,
            'steps': int(stats['steps']),
            'task_completion_time_s': task_completion_time_s,
            'initial_dist_to_goal': initial_dist,
            'final_dist_to_goal': final_dist,
            'progress_score': progress_score,
            'min_h': self.lightweight_finite_or_none(stats['min_h']),
            'min_dist': self.lightweight_finite_or_none(stats['min_dist']),
            'min_psi': self.lightweight_finite_or_none(stats['min_psi']),
            'qp_fail_count': int(stats['qp_fail_count']),
            'fallback_count': int(stats['fallback_count']),
            'fallback_rate': float(stats['fallback_count'] / max(1, int(stats['steps']))),
            'p1_mean': p1_mean,
            'p1_max': p1_max,
            'p2_mean': p2_mean,
            'p2_max': p2_max,
        }

        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, 'run_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.get_logger().info(f"Saved lightweight run summary JSON to {summary_path}")


    def save_and_plot_results(self):
        """Final benchmark output: write only the compact run_summary.json."""
        if getattr(self, 'result_dir', None):
            output_pack = self.result_dir
        else:
            output_pack = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "sim")
            if getattr(self, 'result_subdir', ''):
                output_pack = os.path.join(output_pack, self.result_subdir)

        os.makedirs(output_pack, exist_ok=True)
        output_dir = os.path.join(output_pack, self.output_basename)
        os.makedirs(output_dir, exist_ok=True)
        self.save_run_summary_json(output_dir, None)


def main(args=None):
    rclpy.init(args=args)
    node = CbfControllerNode()
    try:
        # Loop and process callbacks as long as the ROS context is alive and our node hasn't signaled for shutdown.
        while rclpy.ok() and not node.is_shutdown_initiated:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # This block now runs upon clean shutdown from destroy_node() OR KeyboardInterrupt
        node.save_and_plot_results()
        if not node.is_shutdown_initiated:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()