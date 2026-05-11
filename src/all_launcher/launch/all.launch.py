from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

is_sim = False

def generate_launch_description():
    vllm_and_camera_log_level_arg = DeclareLaunchArgument('vllm_and_camera_log_level', default_value='info')
    hunter_log_level_arg = DeclareLaunchArgument('hunter_log_level', default_value='error')

    vllm_and_camera_log_level = LaunchConfiguration('vllm_and_camera_log_level')
    hunter_log_level = LaunchConfiguration('hunter_log_level')

    vllm_and_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('car'), 'launch', 'car_launch.py')
        ),
        launch_arguments={'log_level': vllm_and_camera_log_level}.items()
    )

    # MPC planner 已移除：客户端直接通过 PD+卡尔曼滤波控制 /cmd_vel

    hunter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('hunter_base'), 'launch', 'hunter_base.launch.py')
        ),
        launch_arguments={'log_level': hunter_log_level}.items()
    )

    ld = LaunchDescription()

    ld.add_action(vllm_and_camera_log_level_arg)
    ld.add_action(vllm_and_camera)

    if not is_sim:
        ld.add_action(hunter_log_level_arg)
        ld.add_action(hunter)

    return ld