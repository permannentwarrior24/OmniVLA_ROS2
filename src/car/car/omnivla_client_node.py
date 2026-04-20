#!/usr/bin/env python3
"""
OmniVLA 客户端 ROS 节点
订阅图像话题，调用远程 OmniVLA 服务器 API 进行推理，发布路径点 (PoseArray)

此节点用于 Jetson 端，通过网络连接到远程服务器上的 OmniVLA 模型
"""
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sys
import os
import time
import math
import json
import base64
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point, Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import requests


class KinematicKalmanFilter:
    """面向大模型高延迟推理的运动学卡尔曼滤波器"""

    def __init__(self, dt_high_freq: float = 0.05):
        """
        Args:
            dt_high_freq: 高频发布周期 (秒)，默认0.05对应20Hz
        """
        self.dt = dt_high_freq

        # 状态向量: [v, ω, a_v, a_ω]
        self.x = np.zeros(4)

        # 状态转移矩阵 (匀加速运动模型)
        self.F = np.array([
            [1, 0, self.dt, 0],           # v += a_v * dt
            [0, 1, 0, self.dt],           # ω += a_ω * dt
            [0, 0, 1, 0],                 # a_v 保持（后续会被限幅）
            [0, 0, 0, 1],                 # a_ω 保持
        ])

        # 观测矩阵: 只观测速度，不观测加速度
        self.H = np.array([
            [1, 0, 0, 0],  # 观测 v
            [0, 1, 0, 0],  # 观测 ω
        ])

        # 过程噪声协方差 Q (模型不确定性)
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1])

        # 观测噪声协方差 R (大模型输出噪声)
        self.R = np.diag([0.05, 0.05])

        # 状态协方差矩阵 P
        self.P = np.diag([1.0, 1.0, 1.0, 1.0])

    def predict(self) -> np.ndarray:
        """预测步：高频定时器调用"""
        # 状态预测
        self.x = self.F @ self.x

        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, z: np.ndarray):
        """更新步：收到大模型输出时调用

        Args:
            z: 观测向量 [v_mllm, ω_mllm]
        """
        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新
        y = z - self.H @ self.x  # 观测残差
        self.x = self.x + K @ y

        # 协方差更新
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        # 加速度衰减（若无新观测）
        self.x[2] *= 0.9
        self.x[3] *= 0.9


class OmniVLAClientNode(Node):
    """OmniVLA 客户端节点 - 通过 HTTP 调用远程服务器"""

    def __init__(self):
        super().__init__('omnivla_client_node')

        # 声明参数
        self.declare_parameter('pic_topic', '/car/pic')
        self.declare_parameter('process_pic_topic', '/car/process_pic')
        self.declare_parameter('commd_topic', '/goal_point')
        self.declare_parameter('prompt_topic', '/car/prompt')
        self.declare_parameter('server_url', 'http://localhost:8000')
        self.declare_parameter('request_timeout', 30.0)
        self.declare_parameter('compression_quality', 75)
        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('metric_waypoint_spacing', 0.2)
        self.declare_parameter('retry_count', 3)
        self.declare_parameter('retry_delay', 0.5)

        # 目标参数（发送到服务器）
        self.declare_parameter('goal_lat', 0.0)
        self.declare_parameter('goal_lon', 0.0)
        self.declare_parameter('goal_compass', 0.0)

        # 卡尔曼滤波器参数
        self.declare_parameter('high_freq_rate', 20.0)  # Hz
        self.declare_parameter('max_linear_accel', 0.3)  # m/s²
        self.declare_parameter('max_angular_accel', 0.5)  # rad/s²
        self.declare_parameter('enable_kalman_filter', True)

        # 获取参数值
        self.pic_topic = self.get_parameter('pic_topic').get_parameter_value().string_value
        self.process_pic_topic = self.get_parameter('process_pic_topic').get_parameter_value().string_value
        self.commd_topic = self.get_parameter('commd_topic').get_parameter_value().string_value
        self.prompt_topic = self.get_parameter('prompt_topic').get_parameter_value().string_value
        self.server_url = self.get_parameter('server_url').get_parameter_value().string_value
        self.request_timeout = self.get_parameter('request_timeout').get_parameter_value().double_value
        self.compression_quality = self.get_parameter('compression_quality').get_parameter_value().integer_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('img_height').get_parameter_value().integer_value
        self.metric_waypoint_spacing = self.get_parameter('metric_waypoint_spacing').get_parameter_value().double_value
        self.retry_count = self.get_parameter('retry_count').get_parameter_value().integer_value
        self.retry_delay = self.get_parameter('retry_delay').get_parameter_value().double_value

        self.goal_lat = self.get_parameter('goal_lat').get_parameter_value().double_value
        self.goal_lon = self.get_parameter('goal_lon').get_parameter_value().double_value
        self.goal_compass = self.get_parameter('goal_compass').get_parameter_value().double_value

        # 卡尔曼滤波器参数获取
        self.high_freq_rate = self.get_parameter('high_freq_rate').get_parameter_value().double_value
        self.max_linear_accel = self.get_parameter('max_linear_accel').get_parameter_value().double_value
        self.max_angular_accel = self.get_parameter('max_angular_accel').get_parameter_value().double_value
        self.enable_kalman_filter = self.get_parameter('enable_kalman_filter').get_parameter_value().bool_value

        self.get_logger().info(f"服务器地址: {self.server_url}")
        # self.get_logger().info(f"请求超时: {self.request_timeout}s")

        # 初始化状态
        self.language_prompt = "stop"
        self.active_prompt = False
        self.current_prompt_uuid: Optional[str] = None

        # 性能统计
        self.total_duration = 0.0
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

        # 卡尔曼滤波器初始化
        if self.enable_kalman_filter:
            dt_high_freq = 1.0 / self.high_freq_rate
            self.kf = KinematicKalmanFilter(dt_high_freq=dt_high_freq)
            self.v_prev = 0.0
            self.w_prev = 0.0
            self.get_logger().info(
                f"卡尔曼滤波器已启用: {self.high_freq_rate}Hz, "
                f"线加速度限制={self.max_linear_accel}m/s², "
                f"角加速度限制={self.max_angular_accel}rad/s²"
            )

        # 创建发布者和订阅者
        self.waypoints_publisher = self.create_publisher(PoseArray, self.commd_topic, 10)
        self.processed_image_publisher = self.create_publisher(Image, self.process_pic_topic, 10)
        self.prompt_complete_pub = self.create_publisher(String, "/car/prompt_complete", 10)

        # cmd_vel 发布者（卡尔曼滤波器高频输出）
        if self.enable_kalman_filter:
            self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
            dt_high_freq = 1.0 / self.high_freq_rate
            self.high_freq_timer = self.create_timer(
                dt_high_freq,
                self.high_freq_timer_callback
            )

        # 订阅图像话题
        self.image_subscription = self.create_subscription(
            Image,
            self.pic_topic,
            self.image_callback,
            10
        )

        # 订阅 prompt 话题
        self.prompt_sub = self.create_subscription(
            String,
            self.prompt_topic,
            self.prompt_callback,
            10
        )

        self.bridge = CvBridge()

        # 检查服务器连接
        self.check_server_health()

        self.get_logger().info("OmniVLA 客户端节点初始化完成")
        self.say_model_ready()

    def say_model_ready(self):
        """通知启动就绪"""
        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.model_ready_pub = self.create_publisher(Bool, "/car/model_ready", ready_qos)
        msg = Bool()
        msg.data = True
        self.model_ready_pub.publish(msg)

    def check_server_health(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(
                f"{self.server_url}/api/health",
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                model_loaded = data.get("model_loaded", False)
                device = data.get("device", "unknown")
                self.get_logger().info(f"服务器连接成功, 模型已加载: {model_loaded}, 设备: {device}")
            else:
                self.get_logger().warn(f"服务器返回非200状态: {response.status_code}")
                self.get_logger().warn(f"服务器返回非200状态: {response}")
        except Exception as e:
            self.get_logger().warn(f"无法连接到服务器: {e}")

    def prompt_callback(self, msg: String):
        """处理接收到的 prompt 消息"""
        try:
            data = json.loads(msg.data)
            uuid = data.get("uuid")
            text = data.get("text")
            if not uuid or not text:
                self.get_logger().warn(f"解析 prompt 消息失败 data: {data}")
                return
        except Exception as e:
            self.get_logger().warn(f"解析 prompt 消息失败: {e}")
            return

        text = text.strip()
        if not text:
            return

        # 如果当前有活跃的 prompt，发布中断信号
        if self.active_prompt and self.current_prompt_uuid:
            interrupt_msg = String()
            interrupt_data = json.dumps({"uuid": self.current_prompt_uuid, "status": "interrupted"})
            interrupt_msg.data = interrupt_data
            self.prompt_complete_pub.publish(interrupt_msg)
            self.get_logger().info(f"prompt {self.current_prompt_uuid} 被打断")

        # 更新 prompt 和 UUID
        self.current_prompt_uuid = uuid
        self.language_prompt = text
        self.active_prompt = True
        self.get_logger().info(f"新 prompt {uuid}: {text}")

    def high_freq_timer_callback(self):
        """高频定时器回调：执行卡尔曼预测步并发布平滑速度"""
        if not self.enable_kalman_filter:
            return

        # 1. 卡尔曼预测步
        x_predict = self.kf.predict()
        v_raw = x_predict[0]
        w_raw = x_predict[1]

        # 2. 严格的运动学限幅（物理底线）
        dt = 1.0 / self.high_freq_rate
        v_out = np.clip(
            v_raw,
            self.v_prev - self.max_linear_accel * dt,
            self.v_prev + self.max_linear_accel * dt
        )
        w_out = np.clip(
            w_raw,
            self.w_prev - self.max_angular_accel * dt,
            self.w_prev + self.max_angular_accel * dt
        )

        # 3. 发布给底盘
        twist_msg = Twist()
        twist_msg.linear.x = float(v_out)
        twist_msg.angular.z = float(w_out)
        self.cmd_vel_publisher.publish(twist_msg)

        # 4. 更新前一时刻状态
        self.v_prev = v_out
        self.w_prev = w_out

    def image_callback(self, msg):
        """处理接收到的图像消息"""
        # 检查是否有活跃的 prompt
        if not self.active_prompt or not self.current_prompt_uuid:
            self.get_logger().debug("没有活跃的 prompt，跳过推理")
            return

        current_uuid = self.current_prompt_uuid
        start_time = time.time()

        try:
            # 1. 转换 ROS 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 调整图像大小
            if self.img_width > 0 and self.img_height > 0:
                cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))

            # 2. 编码为 base64
            image_base64 = self.encode_image_to_base64(cv_image)

            # 3. 调用远程服务器
            result = self.send_inference_request(image_base64, current_uuid)

            # 4. 处理结果
            if result and result.get("status") == "success":
                waypoints = result.get("waypoints", [])
                linear_vel = result.get("linear_vel", 0.0)
                angular_vel = result.get("angular_vel", 0.0)
                inference_time = result.get("inference_time", 0.0)

                # 【关键改动】将大模型输出作为观测值，触发卡尔曼更新步
                if self.enable_kalman_filter:
                    z = np.array([linear_vel, angular_vel])
                    self.kf.update(z)

                # 发布路径点
                if waypoints:
                    self.publish_waypoints(waypoints)

                # 发布处理后的图像
                self.publish_processed_image(cv_image)

                # 检查 prompt 是否已被中断
                if current_uuid == self.current_prompt_uuid:
                    complete_msg = String()
                    complete_data = json.dumps({"uuid": current_uuid, "status": "completed"})
                    complete_msg.data = complete_data
                    self.prompt_complete_pub.publish(complete_msg)
                    self.active_prompt = False
                    self.current_prompt_uuid = None

                # 性能统计
                duration = time.time() - start_time
                self.total_duration += duration
                self.request_count += 1
                self.success_count += 1
                avg_duration = self.total_duration / self.request_count

                frame_id = msg.header.frame_id if msg.header.frame_id else "N/A"

                self.get_logger().info(
                    f"响应 {frame_id}, 总耗时: {duration:.2f}s, 推理: {inference_time:.2f}s, "
                    f"平均: {avg_duration:.2f}s, 成功: {self.success_count}/{self.request_count}\n"
                    f"线速度: {linear_vel:.3f} m/s, 角速度: {angular_vel:.3f} rad/s"
                )
            else:
                self.error_count += 1
                error_msg = result.get("message", "unknown error") if result else "no response"
                self.get_logger().error(f"推理失败: {error_msg}")
                self.active_prompt = False
                self.current_prompt_uuid = None

        except Exception as e:
            self.error_count += 1
            self.get_logger().error(f"处理图像出错: {str(e)}")
            self.active_prompt = False
            self.current_prompt_uuid = None

    def encode_image_to_base64(self, cv_image) -> str:
        """将图像编码为 base64"""
        success, buffer = cv2.imencode(
            '.jpg',
            cv_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality]
        )
        if not success:
            self.get_logger().error("图像编码失败")
            return ""

        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"

    def send_inference_request(self, image_base64: str, uuid: str) -> Optional[dict]:
        """发送推理请求到远程服务器"""
        url = f"{self.server_url}/api/inference"

        payload = {
            "image_base64": image_base64,
            "prompt": self.language_prompt,
            "uuid": uuid,
            "goal_lat": self.goal_lat,
            "goal_lon": self.goal_lon,
            "goal_compass": self.goal_compass
        }

        headers = {"Content-Type": "application/json"}

        # 重试机制
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.request_timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    self.get_logger().warn(
                        f"请求失败 (尝试 {attempt + 1}/{self.retry_count}): "
                        f"状态码 {response.status_code}"
                    )

            except requests.exceptions.Timeout:
                self.get_logger().warn(
                    f"请求超时 (尝试 {attempt + 1}/{self.retry_count})"
                )
            except requests.exceptions.ConnectionError as e:
                self.get_logger().warn(
                    f"连接错误 (尝试 {attempt + 1}/{self.retry_count}): {e}"
                )
            except Exception as e:
                self.get_logger().error(f"请求异常: {e}")
                break

            # 重试前等待
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)

        return None

    def publish_waypoints(self, waypoints: list):
        """发布路径点"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"

        for i, wp in enumerate(waypoints):
            # waypoints 格式: [dx, dy, hx, hy]
            if len(wp) >= 2:
                dx_real = wp[0] * self.metric_waypoint_spacing
                dy_real = wp[1] * self.metric_waypoint_spacing

                pose = Pose()
                pose.position.x = float(dx_real)
                pose.position.y = float(dy_real)
                pose.position.z = 0.0
                pose.orientation.w = 1.0

                pose_array.poses.append(pose)

        self.waypoints_publisher.publish(pose_array)

    def publish_processed_image(self, cv_image):
        """发布处理后的图像"""
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "omnivla_processed"
            self.processed_image_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().warn(f"发布处理图像失败: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()