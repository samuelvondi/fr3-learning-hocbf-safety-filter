# Copyright (c) 2024 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def get_robot_description(context: LaunchContext, arm_id, load_gripper, franka_hand, use_sim_time):
    print(f"\n[DEBUG] Loading robot configuration:")
    print(f"- arm_id: {context.perform_substitution(arm_id)}")
    print(f"- load_gripper: {context.perform_substitution(load_gripper)}")
    print(f"- franka_hand: {context.perform_substitution(franka_hand)}")

    arm_id_str = context.perform_substitution(arm_id)
    load_gripper_str = context.perform_substitution(load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        arm_id_str,
        arm_id_str + '.urdf.xacro'
    )

    print(f"[DEBUG] URDF file path: {franka_xacro_file}")
    if not os.path.exists(franka_xacro_file):
        print(f"[ERROR] URDF file not found at {franka_xacro_file}")
        raise FileNotFoundError(f"URDF file not found: {franka_xacro_file}")

    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'arm_id': arm_id_str,
            'hand': load_gripper_str,
            'ros2_control': 'true',
            'gazebo': 'true',
            'ee_id': franka_hand_str,
        }
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    return [robot_state_publisher]


def generate_launch_description():
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Whether to run Gazebo in headless mode (no GUI).'
    )
    headless = LaunchConfiguration('headless')

    scenario_config_file_arg = DeclareLaunchArgument(
        'scenario_config_file',
        description='Full path to the scenario YAML file'
    )
    scenario_config_file = LaunchConfiguration('scenario_config_file')

    use_hocbf_filter_arg = DeclareLaunchArgument(
        'use_hocbf_filter',
        default_value='true',
        description='Enable the HOCBF safety filter. Set false for PD-only ROS baseline.'
    )
    use_hocbf_filter = LaunchConfiguration('use_hocbf_filter')

    p12_mode_arg = DeclareLaunchArgument(
        'p12_mode',
        default_value='davide_online',
        description='p12 policy: davide_online, nn, or nn_g12_fallback.'
    )
    p12_mode = LaunchConfiguration('p12_mode')

    nn_model_path_arg = DeclareLaunchArgument(
        'nn_model_path',
        default_value='',
        description='Absolute path to trained p12 NN checkpoint (.pt).'
    )
    nn_model_path = LaunchConfiguration('nn_model_path')

    nn_stats_path_arg = DeclareLaunchArgument(
        'nn_stats_path',
        default_value='',
        description='Absolute path to NN feature stats (.npz).'
    )
    nn_stats_path = LaunchConfiguration('nn_stats_path')

    result_subdir_arg = DeclareLaunchArgument(
        'result_subdir',
        default_value='',
        description='Optional subdirectory under plots/sim for mode-separated batch outputs.'
    )
    result_subdir = LaunchConfiguration('result_subdir')

    online_pair_scope_arg = DeclareLaunchArgument(
        'online_pair_scope',
        default_value='obstacle',
        description='Davide online pair scope: obstacle or all.'
    )
    online_pair_scope = LaunchConfiguration('online_pair_scope')

    g12_fallback_pair_scope_arg = DeclareLaunchArgument(
        'g12_fallback_pair_scope',
        default_value='all',
        description='NN G12 fallback pair scope: obstacle or all.'
    )
    g12_fallback_pair_scope = LaunchConfiguration('g12_fallback_pair_scope')

    g12_fallback_tol_arg = DeclareLaunchArgument(
        'g12_fallback_tol',
        default_value='1e-9',
        description='Tolerance for G1/G2 monitor before triggering fallback.'
    )
    g12_fallback_tol = LaunchConfiguration('g12_fallback_tol')

    p1_max_arg = DeclareLaunchArgument(
        'p1_max',
        default_value='200.0',
        description='Maximum p1/gamma for online/fallback runtime.'
    )
    p1_max = LaunchConfiguration('p1_max')

    p2_max_arg = DeclareLaunchArgument(
        'p2_max',
        default_value='250.0',
        description='Maximum p2/beta for online/fallback runtime.'
    )
    p2_max = LaunchConfiguration('p2_max')

    load_gripper_name = 'load_gripper'
    franka_hand_name = 'franka_hand'
    arm_id_name = 'arm_id'

    load_gripper = LaunchConfiguration(load_gripper_name)
    franka_hand = LaunchConfiguration(franka_hand_name)
    arm_id = LaunchConfiguration(arm_id_name)

    load_gripper_launch_argument = DeclareLaunchArgument(
        load_gripper_name,
        default_value='false',
        description='true/false for activating the gripper'
    )
    franka_hand_launch_argument = DeclareLaunchArgument(
        franka_hand_name,
        default_value='franka_hand',
        description='Default value: franka_hand'
    )
    arm_id_launch_argument = DeclareLaunchArgument(
        arm_id_name,
        default_value='fr3',
        description='Available values: fr3, fp3 and fer'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    robot_state_publisher = OpaqueFunction(
        function=get_robot_description,
        args=[arm_id, load_gripper, franka_hand, use_sim_time]
    )

    gz_args_expression = PythonExpression([
        '"-r -s empty.sdf" if "', headless, '".lower() == "true" else "-r empty.sdf"'
    ])

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gazebo_empty_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': gz_args_expression}.items(),
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description'],
        output='screen',
    )

    rviz_file = os.path.join(
        get_package_share_directory('cbf_safety_filter'),
        'rviz',
        'cbf_debug.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['--display-config', rviz_file, '-f', 'world'],
        condition=UnlessCondition(headless),
        output='screen',
    )

    controller_config = os.path.join(
        get_package_share_directory('cbf_safety_filter'),
        'config',
        'cbf_controllers.yaml'
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

    hocbf_controller_node = Node(
        package='cbf_safety_filter',
        executable='simulation_HOCBF.py',
        name='simulation_HOCBF',
        output='screen',
        parameters=[{
            'scenario_config_file': scenario_config_file,
            'use_hocbf_filter': ParameterValue(use_hocbf_filter, value_type=bool),
            'p12_mode': p12_mode,
            'nn_model_path': nn_model_path,
            'nn_stats_path': nn_stats_path,
            'result_subdir': result_subdir,
            'online_pair_scope': online_pair_scope,
            'g12_fallback_pair_scope': g12_fallback_pair_scope,
            'g12_fallback_tol': ParameterValue(g12_fallback_tol, value_type=float),
            'p1_max': ParameterValue(p1_max, value_type=float),
            'p2_max': ParameterValue(p2_max, value_type=float),
        }]
    )

    franka_description_share = get_package_share_directory('franka_description')
    franka_description_parent = os.path.dirname(franka_description_share)

    set_env_vars_resources_gz = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        franka_description_parent
    )

    set_env_vars_resources_ign = AppendEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        franka_description_parent
    )


    return LaunchDescription([
        headless_arg,
        scenario_config_file_arg,
        use_hocbf_filter_arg,
        p12_mode_arg,
        nn_model_path_arg,
        nn_stats_path_arg,
        result_subdir_arg,
        online_pair_scope_arg,
        g12_fallback_pair_scope_arg,
        g12_fallback_tol_arg,
        p1_max_arg,
        p2_max_arg,
        load_gripper_launch_argument,
        franka_hand_launch_argument,
        arm_id_launch_argument,
        use_sim_time_arg,

        set_env_vars_resources_gz,
        set_env_vars_resources_ign,

        gazebo_empty_world,
        robot_state_publisher,
        rviz,
        spawn,

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[joint_position_controller_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_position_controller_spawner,
                on_exit=[hocbf_controller_node]
            )
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'source_list': ['joint_states'], 'rate': 30}
            ],
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=hocbf_controller_node,
                on_exit=[Shutdown(reason='Controller node exited')],
            )
        ),

    ])
