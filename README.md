# OmniVLA ROS2 集成系统

基于 ROS2 Humble 的无人车控制系统，集成 OmniVLA 模型推理、深度感知、运动学滤波和置信度感知速度调节功能。

## 项目概述

本项目是毕业设计"基于多模态大语言模型的无人车轨迹预测方法"的 ROS2 集成模块，运行在边缘计算设备（NVIDIA Jetson AGX Orin）上，负责：

1. **图像采集与发布**：从摄像头获取图像并发布到 ROS2 话题
2. **深度图处理**：分析深度图生成障碍物描述（创新点一）
3. **模型推理调用**：调用云端 OmniVLA API 获取轨迹预测
4. **运动学滤波**：平滑速度指令，消除顿挫感（创新点二）
5. **置信度感知控制**：根据预测一致性调整速度（创新点三）
6. **运动控制**：将处理后的指令发送给底盘控制器

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        云端服务器 (RTX 4080)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              OmniVLA API Server (端口 8000)                 │  │
│  │         接收图像+指令 → 返回路径点+速度                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Jetson AGX Orin (边缘端)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ 图像发布节点   │  │ 深度处理节点   │  │ OmniVLA客户端 │             │
│  │image_publisher│  │depth_processor│  │omnivla_client│             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                     │
│         ▼                 ▼                 ▼                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    ROS2 话题系统                            │  │
│  │  /car/pic  /car/depth_description  /car/prompt  /cmd_vel │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              MPC 控制器 + Hunter 底盘驱动                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
OmniVLA_ROS2/
├── src/
│   ├── car/                          # 核心功能包
│   │   ├── car/
│   │   │   ├── image_publisher_node.py   # 图像发布节点
│   │   │   ├── depth_processor_node.py   # 深度图处理节点（创新点一）
│   │   │   ├── omnivla_client_node.py    # OmniVLA 客户端节点
│   │   │   ├── recv_prompt.py            # Prompt 接收与 HTTP API
│   │   │   ├── vllm_ask_node.py          # VLM 推理节点（备选）
│   │   │   └── omnivla_vllm_ask_node.py  # OmniVLA 本地推理节点
│   │   ├── launch/
│   │   │   └── car_launch.py             # 单包启动文件
│   │   ├── test/
│   │   │   ├── test_kalman_filter.py     # 卡尔曼滤波器测试
│   │   │   └── test_integration_confidence.py  # 置信度控制测试
│   │   └── setup.py
│   ├── mpc_planner/                  # MPC 控制器包
│   │   ├── mpc_planner/
│   │   │   ├── mpc_controller.py         # MPC 控制器
│   │   │   ├── mpc_core.py               # MPC 核心算法
│   │   │   ├── goal_sender.py            # 目标发送器
│   │   │   └── simple_simulator.py       # 简单仿真器
│   │   └── launch/
│   │       └── gzaebo.launch.py          # MPC 启动文件
│   ├── all_launcher/                 # 总启动器包
│   │   └── launch/
│   │       └── all.launch.py             # 全系统启动文件
│   └── hunter_base/                  # Hunter 底盘驱动（外部依赖）
├── bar.sh                            # 快捷命令脚本
└── pyproject.toml
```

## 核心节点详解

### 1. 图像发布节点 (`image_publisher_node.py`)

**功能**：从摄像头或本地图像目录获取图像，发布到 `/car/pic` 话题。

**运行模式**：
- `local`：从本地目录循环发布图像（调试用）
- `camera`：RGB-D 深度相机（ORBBEC Astra Pro，红外结构光）实时采集

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `local` | 运行模式 |
| `pic_topic` | `/car/pic` | 图像发布话题 |
| `fps` | `30` | 发布帧率 |
| `pic_dir` | - | 本地图像目录 |

### 2. 深度处理节点 (`depth_processor_node.py`) - 创新点一

**功能**：订阅深度图，分析左/前/右三个区域的障碍物距离，生成自然语言描述。

**处理流程**：
```
深度图 → 区域划分 → 有效值过滤 → 5%分位数计算 → 障碍物判断 → 自然语言描述
```

**区域划分**：
- 仅分析图像下半部分（上半部分为远景/天花板）
- 按列三等分：左侧、正前方、右侧

**障碍物检测**：
- 有效深度范围：[0.3m, 5.0m]
- 使用 5% 分位数作为鲁棒的最近距离估计
- 障碍物阈值：1.5m

**输出示例**：
```
左侧通道畅通，最近物体在2.3米外；正前方1.2米处有障碍物；右侧通道畅通，最近物体在3.1米外；建议左转避障
```

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `depth_topic` | `/camera/depth/image_raw` | 深度图话题 |
| `output_topic` | `/car/depth_description` | 描述输出话题 |
| `obstacle_threshold` | `1.5` | 障碍物距离阈值（米） |
| `valid_depth_min` | `0.3` | 有效深度最小值（米） |
| `valid_depth_max` | `5.0` | 有效深度最大值（米） |
| `publish_rate` | `10.0` | 发布频率（Hz） |

### 3. OmniVLA 客户端节点 (`omnivla_client_node.py`)

**功能**：订阅图像和 Prompt，调用云端 OmniVLA API，处理返回结果并发布控制指令。

**核心功能**：

#### 3.1 深度描述注入（创新点一）
```python
# 拼接深度描述到 prompt
effective_prompt = f"{language_prompt}。[环境感知] {latest_depth_description}"
```

#### 3.2 运动学卡尔曼滤波器（创新点二）
```python
class KinematicKalmanFilter:
    """面向大模型高延迟推理的运动学卡尔曼滤波器"""

    # 状态向量: [v, ω, a_v, a_ω]
    # 状态转移矩阵: 匀加速运动模型
    # 观测矩阵: 只观测速度，不观测加速度
```

**滤波流程**：
1. 高频定时器（20Hz）执行卡尔曼预测步
2. 收到模型输出时执行卡尔曼更新步
3. 加速度限幅：`|Δv| < a_max * dt`, `|Δω| < ω_max * dt`

#### 3.3 置信度感知速度调节（创新点三）
```python
class ConfidenceAwareSpeedController:
    """通过分析 N=8 步动作序列的角速度一致性来评估模型预测置信度"""

    # 置信度判断逻辑：
    # - σ_ω < 0.1 rad/s → 高置信度，保持原速
    # - 0.1 < σ_ω < 0.3 rad/s → 中等置信度，按比例降速
    # - σ_ω > 0.3 rad/s → 低置信度，降速50%
    # - 连续3帧低置信度 → 紧急制动，降速70%
```

**关键参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `server_url` | `http://localhost:8000` | OmniVLA 服务器地址 |
| `request_timeout` | `30.0` | 请求超时（秒） |
| `compression_quality` | `75` | JPEG 压缩质量 |
| `img_width` / `img_height` | `640` / `480` | 图像尺寸 |
| `high_freq_rate` | `20.0` | 卡尔曼滤波高频（Hz） |
| `max_linear_accel` | `0.3` | 最大线加速度（m/s²） |
| `max_angular_accel` | `0.5` | 最大角加速度（rad/s²） |
| `high_conf_threshold` | `0.1` | 高置信度阈值（rad/s） |
| `low_conf_threshold` | `0.3` | 低置信度阈值（rad/s） |
| `low_conf_max_count` | `3` | 紧急制动帧数 |

### 4. Prompt 接收节点 (`recv_prompt.py`)

**功能**：提供 HTTP API 接口，接收前端发送的 Prompt，转发到 ROS2 话题。

**HTTP API**：
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/state` | GET | 获取系统状态 |
| `/api/prompt` | POST | 提交 Prompt |
| `/api/prompt_mode` | GET/POST | 获取/设置下发模式 |
| `/api/raw_camera` | GET | 获取原始图像 |

**下发模式**：
- `single`：单次下发，等待完成后才能发送新 Prompt
- `repeat_1hz`：1Hz 重复下发，新 Prompt 可打断旧 Prompt

## 话题系统

| 话题名 | 消息类型 | 发布者 | 订阅者 | 说明 |
|--------|----------|--------|--------|------|
| `/car/pic` | `sensor_msgs/Image` | image_publisher | omnivla_client | 原始图像 |
| `/car/process_pic` | `sensor_msgs/Image` | omnivla_client | recv_prompt | 处理后图像 |
| `/car/prompt` | `std_msgs/String` | recv_prompt | omnivla_client | Prompt 指令 |
| `/car/depth_description` | `std_msgs/String` | depth_processor | omnivla_client | 深度描述 |
| `/car/model_text` | `std_msgs/String` | omnivla_client | recv_prompt | 模型输出文本 |
| `/car/prompt_complete` | `std_msgs/String` | omnivla_client | recv_prompt | Prompt 完成信号 |
| `/car/model_ready` | `std_msgs/Bool` | omnivla_client | image_publisher | 模型就绪信号 |
| `/goal_point` | `geometry_msgs/PoseArray` | omnivla_client | mpc_planner | 路径点 |
| `/cmd_vel` | `geometry_msgs/Twist` | omnivla_client | hunter_base | 速度指令 |

## 环境配置

### 1. 基础依赖

```bash
# 安装 ROS2 Humble（如果未安装）
# 参考：https://docs.ros.org/en/humble/Installation.html

# 安装 Python 依赖
pip install casadi requests opencv-python

# 安装 ROS2 包
sudo apt install ros-humble-osqp-vendor

# 克隆 Hunter 驱动
cd ~/ros2_ws/src
git clone https://github.com/agilexrobotics/ugv_sdk.git
git clone https://github.com/agilexrobotics/hunter_ros2.git

# 编译
cd ~/ros2_ws
colcon build
```

### 2. 摄像头配置

```bash
# RGB-D 深度相机（ORBBEC Astra Pro，红外结构光）
ros2 launch astra_camera astra_pro.launch.xml \
  uvc_vendor_id:=0x2bc5 \
  uvc_product_id:=0x050f \
  serial_number:=ACR874300E4
```

### 3. CAN 总线配置（Hunter 底盘）

```bash
# 编译 gs_usb 驱动
ros2/tools/jetson-gs_usb-kernel-builder.sh

# 配置 CAN 端口
# 修改 ros2/src/ugv_sdk/scripts 中的端口号为 can0

# 测试
candump can0
```

## 使用方法

### 1. 启动完整系统

```bash
# 编译工作空间
cd ~/ros2_ws
colcon build

# 启动所有节点
ros2 launch all_launcher all.launch.py
```

### 2. 单独启动节点

```bash
# 启动图像发布节点
ros2 run car image_publisher --ros-args -p mode:=camera_dual

# 启动深度处理节点
ros2 run car depth_processor

# 启动 OmniVLA 客户端节点
ros2 run car omnivla_client --ros-args -p server_url:=http://云端IP:8000

# 启动 Prompt 接收节点
ros2 run car recv_prompt --ros-args -p http_port:=8787
```

### 3. 停止所有节点

```bash
pkill -f car
pkill -f all_launcher
pkill -f mpc_planner
pkill -f hunter
pkill -f astra_camera
```

## 启动配置

### `car_launch.py` 配置项

```python
# 图像模式
MODE = PicModeType.CAMERA_DUAL  # LOCAL, CAMERA_DUAL

# 图像参数
FPS = 1
COMPRESSION_QUALITY = 30
IMG_WIDTH = 1280
IMG_HIGHT = 960

# 模型类型
MODEL_TYPE = ModelType.OMNI_CLIENT  # QWEN, OMNI, OMNI_CLIENT

# 服务器地址
SERVER_URL = "http://localhost:8000"
```

## 测试

```bash
# 卡尔曼滤波器单元测试
ros2 run car test_kalman_filter

# 置信度控制集成测试
ros2 run car test_integration_confidence
```

## 故障排查

### 1. 无法连接到 OmniVLA 服务器

```bash
# 检查服务器状态
curl http://云端IP:8000/api/health

# 检查网络连通性
ping 云端IP
```

### 2. 摄像头无法打开

```bash
# 检查设备
ls -la /dev/video*

# 检查权限
sudo usermod -aG video $USER
```

### 3. Hunter 底盘无响应

```bash
# 检查 CAN 状态
ip link show can0

# 重启 CAN
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 500000
```

## 相关项目

- [OmniVLA_Interference](../OmniVLA_Interference/) - OmniVLA 推理服务
- [OnmiVLA_ROS2_WebFronted](../OnmiVLA_ROS2_WebFronted/) - Web 控制台

## 硬件依赖

| 设备 | 型号 | 用途 |
|------|------|------|
| 无人车 | 松灵 HUNTER SE | 移动平台 |
| 摄像头 | ORBBEC Astra Pro | RGB-D 深度相机 |
| 边缘计算 | NVIDIA Jetson AGX Orin | 本地处理 |
| 云端服务器 | vGPU-RTX4080 (32G) | 模型推理 |
