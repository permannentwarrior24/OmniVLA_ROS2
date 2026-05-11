import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    package_name = 'mpc_planner'
    urdf_name = 'fishbot_gazebo.urdf'

    pkg_share = FindPackageShare(package=package_name).find(package_name)
    urdf_model_path = os.path.join(pkg_share, 'urdf', urdf_name)
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'default.rviz')
    goal_file_path = os.path.join(pkg_share, 'goal', 'goal.txt')
    
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='warn',
        description='Logger level for all nodes')

    log_level = LaunchConfiguration('log_level')
    
    is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='true',
        description='Whether to include simulator and goal sender nodes'
    )
    is_sim = LaunchConfiguration('is_sim')
    
    
    default_params_file = '/home/apollo/disk/ros2/src/mpc_planner/config/mpc_params.yaml'

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to mpc params yaml'
    )
    params_file = LaunchConfiguration('params_file')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        arguments=[urdf_model_path, '--ros-args', '--log-level', log_level],
        output='screen')

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level])

    simulator_node = Node(
        package='mpc_planner',
        executable='simple_simulator',
        name='simple_simulator',
        output='screen',
        parameters=[{
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'publish_rate': 50.0,
            'cmd_timeout': 0.5
        }],
        arguments=['--ros-args', '--log-level', log_level],
        condition=IfCondition(is_sim)
    )

    mpc_controller_node = Node(
        package='mpc_planner',
        executable='mpc_controller',
        name='mpc_controller',
        output='screen',
        parameters=[params_file],
        arguments=['--ros-args', '--log-level', log_level],
    )

    goal_sender_node = Node(
        package='mpc_planner',
        executable='goal_sender',
        name='goal_sender',
        output='screen',
        parameters=[{'goal_file': goal_file_path}],
        arguments=['--ros-args', '--log-level', log_level],
        condition=IfCondition(is_sim)
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path, '--ros-args', '--log-level', log_level],
    )

    ld = LaunchDescription()
    ld.add_action(log_level_arg)
    ld.add_action(params_file_arg)
    ld.add_action(is_sim_arg)
    
    ld.add_action(joint_state_publisher)
    ld.add_action(robot_state_publisher)
    ld.add_action(mpc_controller_node)
    ld.add_action(rviz_node)
    
    ld.add_action(simulator_node)
    # ld.add_action(goal_sender_node)
    return ld