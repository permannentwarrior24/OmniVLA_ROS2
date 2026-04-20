from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch_ros.actions import Node
from enum import Enum

# 图片目录
PIC_DIR = "/home/apollo/disk/ros2/src/car/pic/9"
# 话题
PIC_TOPIC = "/car/pic"
PROCESS_PIC_TOPIC = "/car/process_pic"
COMMD_TOPIC = "/goal_point"
PROMPT_TOPIC = "/car/prompt"

class PicModeType(Enum):
    LOCAL = "local"
    CAMERA_SINGAL = "camera_signal"
    CAMERA_DUAL = "camera_dual"

MODE=PicModeType.CAMERA_DUAL
# 发布频率(fps)
FPS = 1
# 模型API
API_URL = "http://localhost:8003/v1/chat/completions"
# 图片压缩质量 (1-100, 越低压缩率越高)
COMPRESSION_QUALITY = 30
IMG_WIDTH=1280
IMG_HIGHT=960
# 模型输出最大token数
MAX_TOKENS = 100

class ModelType(Enum):
    QWEN = "qwen"
    OMNI = "omni"
    OMNI_CLIENT = "omni_client"  # 新增：远程客户端模式

MODEL_TYPE = ModelType.OMNI_CLIENT

# 服务器地址（OmniVLA API）
SERVER_URL = "http://localhost:8000"

TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 5
ENABLE_THINKING = "True"

# prompt
# TEXT_1="""
# You are an autonomous driving planner,has extensive driving experience.
# Coordinate system: X-axis is lateral, Y-axis is longitudinal.
# The ego vehicle is at (0,0), units are meters.
# Based on the provided front-view image and driving context, plan future waypoints at 0.5-second intervals for the next 3 seconds.

# Here is the front-view image from the car:
# """

# TEXT_2="""

# Traffic rules:
# - Avoid collision with other objects.
# - Always drive on drivable regions.
# - Avoid occupied regions.
# - Pay attention to the direction of the vehicle in front and go around it accordingly
# - Turn the steering wheel a little more when detouring

# Please plan future waypoints at 0.5-second intervals for the next 3 seconds.
# """

TEXT_1="""
tell what you see
"""
TEXT_2="""
tell what you see
"""

def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Logger level for all nodes')

    log_level = LaunchConfiguration('log_level')
    
    image_publisher = Node(
        package='car',
        executable='image_publisher',
        name='image_publisher',
        parameters=[{
            'pic_topic': PIC_TOPIC,
            'fps': FPS,
            'pic_dir': PIC_DIR,
            'mode': MODE.value,
            'camera_device':  [
                '/dev/video0', 
                '/dev/video1',
            ],
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )
    
    recv_prompt = Node(
        package='car',
        executable='recv_prompt',
        name='recv_prompt',
        parameters=[{
            'prompt_topic': PROMPT_TOPIC,
            'image_topic': PROCESS_PIC_TOPIC,
            'text_topic': "/car/model_text",
            'waypoint_topic': COMMD_TOPIC,
            'http_host': "0.0.0.0",
            'http_port': 8787
        }],
        arguments=['--ros-args', '--log-level', log_level],
    )
    
    if MODEL_TYPE == ModelType.QWEN:
        model_name = 'vllm_ask'
        model_node = Node(
            package='car',
            executable= model_name,
            name= model_name,
            parameters=[{
                'prompt_topic': PROMPT_TOPIC,
                'pic_topic': PIC_TOPIC,
                'process_pic_topic': PROCESS_PIC_TOPIC,
                'commd_topic': COMMD_TOPIC,
                'api_url': API_URL,
                'compression_quality': COMPRESSION_QUALITY,
                'img_width': IMG_WIDTH,
                'img_hight': IMG_HIGHT,
                'max_tokens': MAX_TOKENS,
                'text_1': TEXT_1,
                'text_2': TEXT_2,
                'temperature': TEMPERATURE,
                'top_p': TOP_P,
                'top_k': TOP_K,
                "enable_thinking": ENABLE_THINKING,
            }],
            arguments=['--ros-args', '--log-level', log_level],
        )
    elif MODEL_TYPE == ModelType.OMNI:
        '''
        MODEL_PATH = "/home/apollo/disk/ros2/src/car/car/omnivla-edge/omnivla-edge.pth"
        LANGUAGE_PROMPT = "stop"
        GOAL_LAT = 0.0
        GOAL_LON = 0.0
        GOAL_COMPASS = 0.0
        GOAL_IMAGE_PATH = ""

        model_name = 'omnivla_node'
        model_node = Node(
            package='car',
            executable= model_name,
            name= model_name,
            parameters=[{
                'prompt_topic': PROMPT_TOPIC,
                'pic_topic': PIC_TOPIC,
                'process_pic_topic': PROCESS_PIC_TOPIC,
                'commd_topic': COMMD_TOPIC,
                'api_url': API_URL,
                'compression_quality': COMPRESSION_QUALITY,
                'img_width': IMG_HIGHT,
                'img_hight': IMG_WIDTH,
                'max_tokens': MAX_TOKENS,
                'text_1': TEXT_1,
                'text_2': TEXT_2,
                'temperature': TEMPERATURE,
                'top_p': TOP_P,
                'top_k': TOP_K,
                "enable_thinking": ENABLE_THINKING,

                'model_path': MODEL_PATH,
                'language_prompt': LANGUAGE_PROMPT,
                'goal_lat': GOAL_LAT,
                'goal_lon': GOAL_LON,
                'goal_compass': GOAL_COMPASS,
                'goal_image_path': GOAL_IMAGE_PATH
            }],
            arguments=['--ros-args', '--log-level', log_level],
        )
        '''
    elif MODEL_TYPE == ModelType.OMNI_CLIENT:
        # 远程客户端模式
        model_name = 'omnivla_client'
        model_node = Node(
            package='car',
            executable=model_name,
            name=model_name,
            parameters=[{
                'prompt_topic': PROMPT_TOPIC,
                'pic_topic': PIC_TOPIC,
                'process_pic_topic': PROCESS_PIC_TOPIC,
                'commd_topic': COMMD_TOPIC,
                'server_url': SERVER_URL,
                'request_timeout': 30.0,
                'compression_quality': COMPRESSION_QUALITY,
                'img_width': IMG_WIDTH,
                'img_height': IMG_HIGHT,
                'metric_waypoint_spacing': 0.2,
                'retry_count': 3,
                'retry_delay': 0.5,
                'goal_lat': 0.0,
                'goal_lon': 0.0,
                'goal_compass': 0.0,
                # 卡尔曼滤波器参数
                'high_freq_rate': 20.0,
                'max_linear_accel': 0.3,
                'max_angular_accel': 0.5,
                'enable_kalman_filter': True,
                # 置信度感知速度调节参数
                'enable_confidence_control': True,
                'high_conf_threshold': 0.1,  # rad/s
                'low_conf_threshold': 0.3,   # rad/s
                'low_conf_max_count': 3,
            }],
            arguments=['--ros-args', '--log-level', log_level],
        )
    else:
        raise ValueError(f"未知的模型类型: {MODEL_TYPE}")
    
    
    ld = LaunchDescription()
    ld.add_action(log_level_arg)
    ld.add_action(model_node)
    ld.add_action(recv_prompt)
    ld.add_action(image_publisher)
    return ld
