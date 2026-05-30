# Copyright (c) 2024 Franka Robotics GmbH
#
# Final integrated ROS/Gazebo launch for the FR3 HOCBF runtime benchmark.
# The controller executable is Davide's integrated ROS path with surgically
# added FR3 NN-p12 and NN+G12-fallback runtime modes.

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def get_robot_description(context: LaunchContext, arm_id, load_gripper, franka_hand, use_sim_time):
    arm_id_str = context.perform_substitution(arm_id)
    load_gripper_str = context.perform_substitution(load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)

    print("\n[DEBUG] Loading robot configuration:")
    print(f"- arm_id: {arm_id_str}")
    print(f"- load_gripper: {load_gripper_str}")
    print(f"- franka_hand: {franka_hand_str}")

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        arm_id_str,
        arm_id_str + '.urdf.xacro',
    )
    print(f"[DEBUG] URDF file path: {franka_xacro_file}")
    if not os.path.exists(franka_xacro_file):
        raise FileNotFoundError(f"URDF file not found: {franka_xacro_file}")

    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'arm_id': arm_id_str,
            'hand': load_gripper_str,
            'ros2_control': 'true',
            'gazebo': 'true',
            'ee_id': franka_hand_str,
        },
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            robot_description,
            {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
        ],
    )
    return [robot_state_publisher]


def generate_launch_description():
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false', description='Whether to run Gazebo in headless mode.'
    )
    scenario_config_file_arg = DeclareLaunchArgument(
        'scenario_config_file', description='Full path to the scenario YAML file.'
    )
    # Wall-clock timers are required for this runtime stack; the ROS control loop
    # otherwise waits for a /clock topic that is not guaranteed in this setup.
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use Gazebo simulation time.'
    )

    use_hocbf_filter_arg = DeclareLaunchArgument(
        'use_hocbf_filter', default_value='true', description='false gives the PD-only baseline.'
    )
    p12_mode_arg = DeclareLaunchArgument(
        'p12_mode', default_value='davide_online', description='davide_online, nn, or nn_g12_fallback.'
    )
    nn_model_path_arg = DeclareLaunchArgument(
        'nn_model_path', default_value='', description='Absolute path to the trained NN checkpoint.'
    )
    nn_stats_path_arg = DeclareLaunchArgument(
        'nn_stats_path', default_value='', description='Optional stats NPZ path; may be empty when checkpoint stores feat_mean/feat_std.'
    )
    nn_hidden_dim_arg = DeclareLaunchArgument('nn_hidden_dim', default_value='256')
    p1_floor_arg = DeclareLaunchArgument('p1_floor', default_value='0.001')
    p2_floor_arg = DeclareLaunchArgument('p2_floor', default_value='0.001')
    g12_fallback_pair_scope_arg = DeclareLaunchArgument('g12_fallback_pair_scope', default_value='all')
    g12_fallback_tol_arg = DeclareLaunchArgument('g12_fallback_tol', default_value='1e-9')
    g12_fallback_p1_floor_arg = DeclareLaunchArgument('g12_fallback_p1_floor', default_value='0.001')
    g12_fallback_p2_floor_arg = DeclareLaunchArgument('g12_fallback_p2_floor', default_value='0.001')
    result_dir_arg = DeclareLaunchArgument(
        'result_dir', default_value='', description='Optional absolute output folder for run_data.csv/run_summary.json/plots.'
    )
    save_plots_arg = DeclareLaunchArgument(
        'save_plots', default_value='true', description='Whether to generate Davide per-run plots at shutdown.'
    )

    load_gripper_arg = DeclareLaunchArgument('load_gripper', default_value='false')
    franka_hand_arg = DeclareLaunchArgument('franka_hand', default_value='franka_hand')
    arm_id_arg = DeclareLaunchArgument('arm_id', default_value='fr3')

    headless = LaunchConfiguration('headless')
    scenario_config_file = LaunchConfiguration('scenario_config_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    load_gripper = LaunchConfiguration('load_gripper')
    franka_hand = LaunchConfiguration('franka_hand')
    arm_id = LaunchConfiguration('arm_id')

    robot_state_publisher = OpaqueFunction(
        function=get_robot_description,
        args=[arm_id, load_gripper, franka_hand, use_sim_time],
    )

    gz_args_expression = PythonExpression([
        '"-r -s empty.sdf" if "', headless, '".lower() == "true" else "-r empty.sdf"'
    ])
    gazebo_empty_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': gz_args_expression}.items(),
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description'],
        output='screen',
    )

    # Kept because Davide's original integrated launch starts it; ros2_control
    # still provides the joint states used by the controller.
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
        output='screen',
    )

    rviz_file = os.path.join(get_package_share_directory('cbf_safety_filter'), 'rviz', 'cbf_debug.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['--display-config', rviz_file, '-f', 'world'],
        condition=UnlessCondition(headless),
        parameters=[{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}],
        output='screen',
    )

    controller_config = os.path.join(
        get_package_share_directory('cbf_safety_filter'), 'config', 'cbf_controllers.yaml'
    )
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--param-file', controller_config,
        ],
        output='screen',
    )
    joint_position_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_position_controller',
            '--controller-manager', '/controller_manager',
            '--controller-type', 'cbf_safety_filter/JointPositionController',
            '--param-file', controller_config,
        ],
        output='screen',
    )

    simulation_hocbf = Node(
        package='cbf_safety_filter',
        executable='dv_simulation_hocbf_ros.py',
        name='simulation_HOCBF',
        output='screen',
        parameters=[{
            'scenario_config_file': scenario_config_file,
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'use_hocbf_filter': ParameterValue(LaunchConfiguration('use_hocbf_filter'), value_type=bool),
            'p12_mode': LaunchConfiguration('p12_mode'),
            'nn_model_path': LaunchConfiguration('nn_model_path'),
            'nn_stats_path': LaunchConfiguration('nn_stats_path'),
            'nn_hidden_dim': ParameterValue(LaunchConfiguration('nn_hidden_dim'), value_type=int),
            'p1_floor': ParameterValue(LaunchConfiguration('p1_floor'), value_type=float),
            'p2_floor': ParameterValue(LaunchConfiguration('p2_floor'), value_type=float),
            'g12_fallback_pair_scope': LaunchConfiguration('g12_fallback_pair_scope'),
            'g12_fallback_tol': ParameterValue(LaunchConfiguration('g12_fallback_tol'), value_type=float),
            'g12_fallback_p1_floor': ParameterValue(LaunchConfiguration('g12_fallback_p1_floor'), value_type=float),
            'g12_fallback_p2_floor': ParameterValue(LaunchConfiguration('g12_fallback_p2_floor'), value_type=float),
            'result_dir': LaunchConfiguration('result_dir'),
            'save_plots': ParameterValue(LaunchConfiguration('save_plots'), value_type=bool),
        }],
        on_exit=Shutdown(),
    )

    spawn_jsb_after_robot = RegisterEventHandler(
        OnProcessExit(target_action=spawn, on_exit=[joint_state_broadcaster_spawner])
    )
    spawn_jpc_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=joint_state_broadcaster_spawner, on_exit=[joint_position_controller_spawner])
    )
    start_hocbf_after_jpc = RegisterEventHandler(
        OnProcessExit(target_action=joint_position_controller_spawner, on_exit=[simulation_hocbf])
    )

    return LaunchDescription([
        headless_arg,
        scenario_config_file_arg,
        use_sim_time_arg,
        use_hocbf_filter_arg,
        p12_mode_arg,
        nn_model_path_arg,
        nn_stats_path_arg,
        nn_hidden_dim_arg,
        p1_floor_arg,
        p2_floor_arg,
        g12_fallback_pair_scope_arg,
        g12_fallback_tol_arg,
        g12_fallback_p1_floor_arg,
        g12_fallback_p2_floor_arg,
        result_dir_arg,
        save_plots_arg,
        load_gripper_arg,
        franka_hand_arg,
        arm_id_arg,
        gazebo_empty_world,
        robot_state_publisher,
        spawn,
        joint_state_publisher,
        rviz,
        spawn_jsb_after_robot,
        spawn_jpc_after_jsb,
        start_hocbf_after_jpc,
    ])
