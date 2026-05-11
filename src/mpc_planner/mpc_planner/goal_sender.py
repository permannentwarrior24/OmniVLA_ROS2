from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from nav_msgs.msg import Odometry
import os
import time
import math
import numpy as np

# 默认路径
DEFAULT_GOAL_FILE = "/home/apollo/disk/ros2/src/mpc/goal/goal.txt"
class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')
        self.declare_parameter('goal_file', DEFAULT_GOAL_FILE)
        self.goal_file = self.get_parameter('goal_file').get_parameter_value().string_value

        self.publisher_ = self.create_publisher(PoseArray, '/goal_point', 10)
        self.global_path_pub = self.create_publisher(Path, '/global_goals_path', 10)
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info(f'Goal Sender Started. Reading from: {self.goal_file}')

        self.current_pose = None

        self.global_goals = []
        self.next_goal_idx = 0

        self.batch_size = 6
        self.reach_tol = 0.2

        self.published_batch = []

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.goals_loaded = False

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        _, _, yaw = self.euler_from_quaternion(ori)
        self.current_pose = np.array([pos.x, pos.y, yaw])
    
    def euler_from_quaternion(self, q):
        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(t3, t4)
        return 0.0, 0.0, yaw

    def publish_global_path(self):
        if not self.global_goals:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        for (x, y) in self.global_goals:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.global_path_pub.publish(path)

    def load_goals_from_file(self):
        goals = []
        if not os.path.exists(self.goal_file):
            self.get_logger().error(f"Goal file not found: {self.goal_file}")
            return goals

        with open(self.goal_file, "r") as f:
            for line in f:
                if not line or line.strip() == "" or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        goals.append((x, y))
                    except ValueError:
                        continue
        return goals

    def convert_global_batch_to_relative(self, start_idx):
        if self.current_pose is None:
            return None
        end_idx = min(start_idx + self.batch_size, len(self.global_goals))
        batch_global = self.global_goals[start_idx:end_idx]

        x_car, y_car, theta_car = self.current_pose
        cos_theta = np.cos(theta_car)
        sin_theta = np.sin(theta_car)

        relative_batch = []
        for xg, yg in batch_global:
            dx = xg - x_car
            dy = yg - y_car
            x_rel = dx * cos_theta + dy * sin_theta
            y_rel = -dx * sin_theta + dy * cos_theta
            relative_batch.append((x_rel, y_rel))
        return relative_batch

    def publish_current_batch(self):
        # 计算并发布从 self.next_goal_idx 开始的最多 batch_size 个相对目标
        if not self.global_goals or self.next_goal_idx >= len(self.global_goals):
            self.get_logger().info("No remaining goals to publish.")
            self.published_batch = []
            empty = PoseArray()
            empty.header.stamp = self.get_clock().now().to_msg()
            empty.header.frame_id = "base_footprint"
            self.publisher_.publish(empty)
            return

        rel_batch = self.convert_global_batch_to_relative(self.next_goal_idx)
        if rel_batch is None:
            self.get_logger().info("Waiting for odometry to compute relative batch.")
            return

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "base_footprint"
        for (x_rel, y_rel) in rel_batch:
            p = Pose()
            p.position.x = x_rel
            p.position.y = y_rel
            p.position.z = 0.0
            p.orientation.w = 1.0
            pose_array.poses.append(p)

        self.publisher_.publish(pose_array)
        self.published_batch = rel_batch
        # self.get_logger().info(f"Published batch starting idx={self.next_goal_idx} size={len(rel_batch)}")

    def timer_callback(self):
        if not self.goals_loaded:
            self.global_goals = self.load_goals_from_file()
            self.goals_loaded = True
            if not self.global_goals:
                self.get_logger().error("No global goals loaded from file; nothing to do.")
                return
        self.publish_global_path()
        
        if self.publisher_.get_subscription_count() == 0:
            return

        if self.current_pose is None:
            return

        # 若还未发布过批次，先发布第一批
        if not self.published_batch:
            self.publish_current_batch()
            return

        # 检查是否到达当前指向的全局目标（索引 self.next_goal_idx）
        if self.next_goal_idx < len(self.global_goals):
            goal_global = self.global_goals[self.next_goal_idx]
            dx = goal_global[0] - self.current_pose[0]
            dy = goal_global[1] - self.current_pose[1]
            dist = math.hypot(dx, dy)
            if dist <= self.reach_tol:
                self.get_logger().info(f"Reached goal idx={self.next_goal_idx} global={goal_global} dist={dist:.3f}")
                # 前进到下一个目标并立即发布新的一批（从新的 next_goal_idx）
                self.next_goal_idx += 1
            self.publish_current_batch()

def main(args=None):
    rclpy.init(args=args)
    node = GoalSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()