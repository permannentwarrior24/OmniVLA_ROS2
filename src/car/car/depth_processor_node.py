#!/usr/bin/env python3
"""
深度图处理节点
订阅 Astra 深度图，分析区域距离，生成自然语言描述
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import json


class DepthProcessorNode(Node):
    def __init__(self):
        super().__init__('depth_processor_node')

        # 参数声明
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('output_topic', '/car/depth_description')
        self.declare_parameter('obstacle_threshold', 1.5)  # 障碍物距离阈值（米）
        self.declare_parameter('valid_depth_min', 0.3)     # 有效深度最小值（米）
        self.declare_parameter('valid_depth_max', 5.0)     # 有效深度最大值（米）
        self.declare_parameter('publish_rate', 10.0)       # 发布频率（Hz）

        # 获取参数
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').get_parameter_value().double_value
        self.valid_depth_min = self.get_parameter('valid_depth_min').get_parameter_value().double_value
        self.valid_depth_max = self.get_parameter('valid_depth_max').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        # 订阅深度图
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        # 发布深度描述
        self.description_pub = self.create_publisher(String, self.output_topic, 10)

        # 定时发布（使用最新深度数据）
        self.latest_depth_description = ""
        self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_description)

        self.bridge = CvBridge()
        self.get_logger().info(f"深度处理节点初始化完成，订阅: {self.depth_topic}")

    def depth_callback(self, msg: Image):
        """处理深度图消息"""
        try:
            # 转换为 numpy 数组（Astra 深度图通常是 16-bit 单通道，单位毫米）
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # 转换为米（Astra 默认单位是毫米）
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0

            # 分析区域并生成描述
            description = self.analyze_depth(depth_image)
            self.latest_depth_description = description

        except Exception as e:
            self.get_logger().error(f"深度图处理失败: {e}")

    def analyze_depth(self, depth_image: np.ndarray) -> str:
        """分析深度图，返回自然语言描述"""
        h, w = depth_image.shape

        # 仅分析下半部分（上半部分为远景/天花板，对导航无意义）
        lower_half = depth_image[h//2:, :]

        # 划分三个区域：左、中、右
        regions = {
            "左侧": lower_half[:, :w//3],
            "正前方": lower_half[:, w//3:2*w//3],
            "右侧": lower_half[:, 2*w//3:],
        }

        region_info = {}
        for name, region in regions.items():
            # 过滤无效值（太近或太远）
            valid = region[(region > self.valid_depth_min) & (region < self.valid_depth_max)]

            if len(valid) > 10:  # 至少 10 个有效像素
                # 使用 5% 分位数，比 min 更鲁棒
                min_dist = float(np.percentile(valid, 5))
                mean_dist = float(np.mean(valid))
                has_obstacle = min_dist < self.obstacle_threshold
            else:
                # 无有效数据，标记为未知
                min_dist = float('inf')
                mean_dist = float('inf')
                has_obstacle = False

            region_info[name] = {
                "最近距离": min_dist,
                "平均距离": mean_dist,
                "有障碍物": has_obstacle
            }

        return self.depth_to_language(region_info)

    def depth_to_language(self, region_info: dict) -> str:
        """将区域信息转换为自然语言描述"""
        parts = []

        for name, info in region_info.items():
            dist = info["最近距离"]
            if dist == float('inf'):
                parts.append(f"{name}深度数据无效")
            elif info["有障碍物"]:
                parts.append(f"{name}{dist:.1f}米处有障碍物")
            else:
                parts.append(f"{name}通道畅通，最近物体在{dist:.1f}米外")

        # 通行建议
        front_info = region_info.get("正前方", {})
        if front_info.get("有障碍物"):
            left_dist = region_info.get("左侧", {}).get("最近距离", float('inf'))
            right_dist = region_info.get("右侧", {}).get("最近距离", float('inf'))

            if left_dist > right_dist:
                parts.append("建议左转避障")
            elif right_dist > left_dist:
                parts.append("建议右转避障")
            else:
                parts.append("前方受阻，两侧均有障碍")

        return "；".join(parts)

    def publish_description(self):
        """定时发布深度描述"""
        if self.latest_depth_description:
            msg = String()
            msg.data = self.latest_depth_description
            self.description_pub.publish(msg)
            self.get_logger().debug(f"发布深度描述: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()