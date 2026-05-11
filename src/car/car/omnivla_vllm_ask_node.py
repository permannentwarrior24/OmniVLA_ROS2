#!/usr/bin/env python3
"""
OmniVLA ROS 节点
订阅图像话题，调用 OmniVLA 模型进行推理，发布路径点 (PoseArray)
"""
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sys
import os
import time
import math
from pathlib import Path
import json
import time
from typing import Optional
# 添加 OmniVLA 推理代码路径
omni_path = "/home/apollo/disk/git/omni/OmniVLA/inference"
if omni_path not in sys.path:
    sys.path.insert(0, omni_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
import clip
import utm

from utils_policy import load_model, transform_images_PIL_mask, transform_images_map

class OmniVLANode(Node):
    def __init__(self):
        super().__init__('omnivla_node')
        
        # 声明参数        
        self.declare_parameter('pic_topic', '')
        self.declare_parameter('process_pic_topic', '')
        self.declare_parameter('commd_topic', '')
        self.declare_parameter('api_url', '')
        self.declare_parameter('compression_quality', 0)
        self.declare_parameter('img_width', 0)
        self.declare_parameter('img_hight', 0)
        self.declare_parameter('max_tokens', 0)
        self.declare_parameter('text_1', '')
        self.declare_parameter('text_2', '')
        self.declare_parameter('temperature', 0.0)
        self.declare_parameter('top_p', 1.0)
        self.declare_parameter('top_k', 1)
        self.declare_parameter('enable_thinking', 'False')
        self.declare_parameter('prompt_topic', '/car/prompt')
        
        # OmniVLA 模型参数
        self.declare_parameter('model_path', '')
        self.declare_parameter('language_prompt', 'stop and dont move')
        self.declare_parameter('goal_lat', 0.0)
        self.declare_parameter('goal_lon', 0.0)
        self.declare_parameter('goal_compass', 0.0)
        self.declare_parameter('goal_image_path', '')
        
        # 图像处理参数
        self.declare_parameter('image_size', 96)  # 用于模型输入的大小
        self.declare_parameter('clip_size', 224)  # CLIP 输入大小
        
        # 获取参数值
        self.pic_topic = self.get_parameter('pic_topic').get_parameter_value().string_value
        self.process_pic_topic = self.get_parameter('process_pic_topic').get_parameter_value().string_value
        self.commd_topic = self.get_parameter('commd_topic').get_parameter_value().string_value
        self.api_url = self.get_parameter('api_url').get_parameter_value().string_value
        self.compression_quality = self.get_parameter('compression_quality').get_parameter_value().integer_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_hight = self.get_parameter('img_hight').get_parameter_value().integer_value
        self.max_tokens = self.get_parameter('max_tokens').get_parameter_value().integer_value
        self.text_1 = self.get_parameter('text_1').get_parameter_value().string_value
        self.text_2 = self.get_parameter('text_2').get_parameter_value().string_value
        self.enable_thinking = self.get_parameter('enable_thinking').get_parameter_value().string_value
        self.prompt_topic = self.get_parameter('prompt_topic').get_parameter_value().string_value

        self.temperature = self.get_parameter('temperature').get_parameter_value().double_value
        self.top_p = self.get_parameter('top_p').get_parameter_value().double_value
        self.top_k = self.get_parameter('top_k').get_parameter_value().integer_value
        
        self.get_logger().warn(f"temperature={self.temperature}, top_p={self.top_k}, top_p={self.top_p}")
        
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.language_prompt = self.get_parameter('language_prompt').get_parameter_value().string_value
        self.goal_lat = self.get_parameter('goal_lat').get_parameter_value().double_value
        self.goal_lon = self.get_parameter('goal_lon').get_parameter_value().double_value
        self.goal_compass = self.get_parameter('goal_compass').get_parameter_value().double_value
        self.goal_image_path = self.get_parameter('goal_image_path').get_parameter_value().string_value
        
        self.image_size = self.get_parameter('image_size').get_parameter_value().integer_value
        self.clip_size = self.get_parameter('clip_size').get_parameter_value().integer_value
        
        # 初始化变量
        self.imgsize = (self.image_size, self.image_size)
        self.imgsize_clip = (self.clip_size, self.clip_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"使用设备: {self.device}")
        self.prompt_sub = self.create_subscription(
            String,
            self.prompt_topic,
            self.prompt_callback,
            10
        )
        
        # 模态配置（根据您的需求调整）
        self.pose_goal = False
        self.satellite = False
        self.image_goal = False
        self.lan_prompt = True
        
        # 模型参数
        self.metric_waypoint_spacing = 0.2
        self.thres_dist = 30.0
        
        # 性能统计
        self.total_duration = 0.0
        self.request_count = 0
        
        # 创建发布者和订阅者（使用正确的参数名称）
        self.waypoints_publisher = self.create_publisher(PoseArray, self.commd_topic, 10)
        self.processed_image_publisher = self.create_publisher(Image, self.process_pic_topic, 10)
        self.image_subscription = self.create_subscription(
            Image,
            self.pic_topic,
            self.image_callback,
            10
        )
        self.prompt_complete_pub = self.create_publisher(String, "/car/prompt_complete", 10)
        self.active_prompt = False  # 当前是否有活跃的 prompt 需要处理
        self.current_prompt_uuid: Optional[str] = None  # 当前处理的 UUID
        
        
        self.bridge = CvBridge()
        
        # 加载模型
        self.load_omnivla_model()
        
        self.get_logger().info("OmniVLA 节点初始化完成")
        self.say_model_ready()
        
    def prompt_callback(self, msg: String):
        try:
            # 尝试解析 JSON 格式的消息
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
    
    # 通知启动就绪
    def say_model_ready(self):
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
    
    def load_omnivla_model(self):
        """加载 OmniVLA 模型和相关组件"""
        try:
            # 创建默认目标图像（黑色）
            self.goal_image_PIL = PILImage.new("RGB", self.imgsize, color=(0, 0, 0))
            self.get_logger().info(f"目标图像不存在: {self.goal_image_path}，使用默认黑色图像")
            
            # 计算目标 UTM 坐标
            self.goal_utm = utm.from_latlon(self.goal_lat, self.goal_lon)
            self.goal_compass_rad = -float(self.goal_compass) / 180.0 * math.pi
            
            # 模型配置（与 run_omnivla_edge_new.py 保持一致）
            self.model_params = {
                "model_type": "omnivla-edge",
                "len_traj_pred": 8,
                "learn_angle": True,
                "context_size": 5,
                "obs_encoder": "efficientnet-b0",
                "encoding_size": 256,
                "obs_encoding_size": 1024,
                "goal_encoding_size": 1024,
                "late_fusion": False,
                "mha_num_attention_heads": 4,
                "mha_num_attention_layers": 4,
                "mha_ff_dim_factor": 4,
                "clip_type": "ViT-B/32"
            }
            
            # 加载模型
            self.get_logger().info(f"加载模型从: {self.model_path}")
            if not os.path.exists(self.model_path):
                self.get_logger().error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.model, self.text_encoder, self.preprocess = load_model(
                self.model_path,
                self.model_params,
                self.device
            )
            
            # 设置模型为评估模式
            self.model = self.model.to(self.device).eval()
            self.text_encoder = self.text_encoder.to(self.device).eval()
            
            # 初始化掩码（无掩码情况）
            self.mask_360_pil_96 = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)
            self.mask_360_pil_224 = np.ones((self.clip_size, self.clip_size, 3), dtype=np.float32)
            
            self.get_logger().info("OmniVLA 模型加载成功")
            
        except Exception as e:
            self.get_logger().error(f"加载模型失败: {str(e)}")
            raise
    
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        # 检查是否有活跃的 prompt，没有则直接返回
        if not self.active_prompt or not self.current_prompt_uuid:
            self.get_logger().debug("没有活跃的 prompt，跳过推理")
            return
        
        # 记录当前 prompt UUID，用于检查是否被中断
        current_uuid = self.current_prompt_uuid
        start_time = time.time()
        
        try:
            # 1. 转换 ROS 图像为 PIL 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 调整图像大小以匹配参数
            if self.img_width > 0 and self.img_hight > 0:
                cv_image = cv2.resize(cv_image, (self.img_width, self.img_hight))
            
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # 2. 运行 OmniVLA 推理
            waypoints, linear_vel, angular_vel = self.run_omnivla_inference(pil_image)
            
            # 3. 发布路径点
            if waypoints is not None:
                self.publish_waypoints(waypoints)
            
            # 4. 发布处理后的图像（可选）
            self.publish_processed_image(cv_image)
            
            # 检查 prompt 是否已被中断
            if current_uuid == self.current_prompt_uuid:
                # 仍然是同一个 prompt，发布完成信号
                complete_msg = String()
                complete_data = json.dumps({"uuid": current_uuid, "status": "completed"})
                complete_msg.data = complete_data
                self.prompt_complete_pub.publish(complete_msg)
                self.active_prompt = False
                self.current_prompt_uuid = None
                self.get_logger().debug(f"prompt {current_uuid} 处理完成，发布完成信号")
            else:
                # prompt 已被中断，不发布完成信号
                self.get_logger().debug(f"prompt {current_uuid} 已被中断，跳过完成信号")
            
            # 性能统计
            duration = time.time() - start_time
            self.total_duration += duration
            self.request_count += 1
            avg_duration = self.total_duration / self.request_count
            
            frame_id = msg.header.frame_id if msg.header.frame_id else "N/A"
            
            self.get_logger().info(
                f"响应{frame_id}, 用时: {duration:.2f}s, 平均: {avg_duration:.2f}s, "
                f"线速度: {linear_vel:.3f} m/s, 角速度: {angular_vel:.3f} rad/s \n"
                f"路径点: \n{waypoints}"
            )
            
        except Exception as e:
            self.get_logger().error(f"推理过程中出错: {str(e)}")
            # 出错时也视为当前 prompt 处理结束
            self.active_prompt = False
            self.current_prompt_uuid = None
    
    def run_omnivla_inference(self, current_image_pil):
        """运行 OmniVLA 推理"""
        try:
            # 调整图像大小
            current_image_pil_96 = current_image_pil.resize(self.imgsize)
            current_image_pil_224 = current_image_pil.resize(self.imgsize_clip)
            
            # 创建上下文队列（使用当前图像填充）
            context_queue = [current_image_pil_96] * 6
            
            # 转换观察图像
            obs_images = transform_images_PIL_mask(context_queue, self.mask_360_pil_96)
            obs_images = torch.split(obs_images.to(self.device), 3, dim=1)
            obs_image_cur = obs_images[-1].to(self.device)
            obs_images = torch.cat(obs_images, dim=1).to(self.device)
            
            # 当前大图像（用于 CLIP）
            cur_large_img = transform_images_PIL_mask(current_image_pil_224, self.mask_360_pil_224).to(self.device)
            
            # 创建卫星图像占位符（黑色）
            satellite_cur = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
            satellite_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
            current_map_image = transform_images_map(satellite_cur)
            goal_map_image = transform_images_map(satellite_goal)
            map_images = torch.cat((current_map_image.to(self.device), goal_map_image.to(self.device), obs_image_cur), axis=1)
            
            # 语言指令编码
            obj_inst_lan = clip.tokenize(self.language_prompt, truncate=True).to(self.device)
            
            # 目标图像编码
            goal_image = transform_images_PIL_mask(self.goal_image_PIL, self.mask_360_pil_96).to(self.device)
            
            # 计算目标姿态（使用固定值或从参数计算）
            # 这里使用固定值作为示例，您可能需要根据实际 GPS 数据计算
            goal_pose_torch = torch.from_numpy(np.array([
                1.0 / self.metric_waypoint_spacing,
                -10.0 / self.metric_waypoint_spacing,
                np.cos(-90.0 / 180.0 * math.pi),
                np.sin(-90.0 / 180.0 * math.pi)
            ])).unsqueeze(0).float().to(self.device)
            
            # 创建批次数据
            batch = {
                "obs_images": obs_images,
                "goal_pose_torch": goal_pose_torch,
                "map_images": map_images,
                "goal_image": goal_image,
                "obj_inst_lan": obj_inst_lan,
                "cur_large_img": cur_large_img,
            }
            
            # 运行前向传播
            with torch.no_grad():
                # 编码语言特征
                feat_text_lan = self.text_encoder.encode_text(batch["obj_inst_lan"])
                
                # 确定模态 ID
                modality_id = self.get_modality_id()
                modality_id_select = torch.tensor([modality_id]).to(self.device)
                
                # 模型推理
                bimg, _, _, _ = batch["goal_image"].size()
                predicted_actions, distances, mask_number = self.model(
                    batch["obs_images"].repeat(bimg, 1, 1, 1),
                    batch["goal_pose_torch"].repeat(bimg, 1),
                    batch["map_images"].repeat(bimg, 1, 1, 1),
                    batch["goal_image"],
                    modality_id_select.repeat(bimg),
                    feat_text_lan.repeat(bimg, 1),
                    batch["cur_large_img"].repeat(bimg, 1, 1, 1),
                )
            
            # 提取路径点
            waypoints = predicted_actions.float().cpu().numpy()[0]  # 形状: (8, 4)
            
            # 计算速度（使用 PD 控制器，与原始代码相同）
            waypoint_select = 4
            chosen_waypoint = waypoints[waypoint_select].copy()
            chosen_waypoint[:2] *= self.metric_waypoint_spacing
            dx, dy, hx, hy = chosen_waypoint
            
            # PD 控制器计算速度
            linear_vel, angular_vel = self.calculate_velocities(dx, dy, hx, hy)
            
            return waypoints, linear_vel, angular_vel
            
        except Exception as e:
            self.get_logger().error(f"推理过程中出错: {str(e)}")
            raise
    
    def get_modality_id(self):
        """根据配置确定模态 ID"""
        if self.pose_goal and self.satellite and self.image_goal and not self.lan_prompt:
            return 3
        elif not self.pose_goal and self.satellite and not self.image_goal and not self.lan_prompt:
            return 0
        elif self.pose_goal and not self.satellite and not self.image_goal and not self.lan_prompt:
            return 4
        elif self.pose_goal and self.satellite and not self.image_goal and not self.lan_prompt:
            return 1
        elif not self.pose_goal and self.satellite and self.image_goal and not self.lan_prompt:
            return 2
        elif self.pose_goal and not self.satellite and self.image_goal and not self.lan_prompt:
            return 5
        elif not self.pose_goal and not self.satellite and self.image_goal and not self.lan_prompt:
            return 6
        elif not self.pose_goal and not self.satellite and not self.image_goal and self.lan_prompt:
            return 7
        elif self.pose_goal and not self.satellite and not self.image_goal and self.lan_prompt:
            return 8
        elif not self.pose_goal and not self.satellite and self.image_goal and self.lan_prompt:
            return 9
        else:
            return 7  # 默认使用语言提示模态
    
    def calculate_velocities(self, dx, dy, hx, hy):
        """PD 控制器计算线速度和角速度（与原始代码相同）"""
        EPS = 1e-8
        DT = 1 / 3
        
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * self.clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * math.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT
        
        # 限制速度
        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)
        
        # 进一步限制（与原始代码相同）
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        
        return linear_vel_value_limit, angular_vel_value_limit
    
    def clip_angle(self, angle):
        """限制角度在 [-π, π] 范围内"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def publish_waypoints(self, waypoints):
        self.get_logger().debug(f"=== 发布路径点 ===")
        for i, wp in enumerate(waypoints):
            dx, dy, _, _ = wp
            dx_real = dx * self.metric_waypoint_spacing
            dy_real = dy * self.metric_waypoint_spacing
            self.get_logger().debug(
                f"发布点[{i}]: x={dx_real:.3f}m, y={dy_real:.3f}m"
            )
        
        """发布路径点作为 PoseArray"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_link"  # 相对于机器人基座标系
        
        # 提取前两个维度作为位置（dx, dy）
        for i in range(len(waypoints)):
            dx, dy, _, _ = waypoints[i]
            
            # 转换为真实距离
            dx_real = dx * self.metric_waypoint_spacing
            dy_real = dy * self.metric_waypoint_spacing
            
            pose = Pose()
            pose.position.x = float(dx_real)
            pose.position.y = float(dy_real)
            pose.position.z = 0.0
            
            # 方向设置为单位四元数
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            
            pose_array.poses.append(pose)
        
        self.waypoints_publisher.publish(pose_array)
    
    def publish_processed_image(self, cv_image):
        """发布处理后的图像（可选）"""
        try:
            # 可以在这里添加图像处理逻辑
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "omnivla_processed"
            self.processed_image_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().warn(f"发布处理图像失败: {str(e)}")
        

def main(args=None):
    rclpy.init(args=args)
    omnivla_node = OmniVLANode()
    try:
        rclpy.spin(omnivla_node)
    except KeyboardInterrupt:
        pass
    finally:
        omnivla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()