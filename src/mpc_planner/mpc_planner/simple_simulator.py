import math
from typing import List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class SimpleSimulator(Node):
    """Tiny kinematic simulator for RViz-only workflows."""

    def __init__(self) -> None:
        super().__init__('simple_simulator')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('initial_pose', [1.0, 0.0, 1.57])
        self.declare_parameter('enable_tf', True)

        init_pose: List[float] = list(self.get_parameter('initial_pose').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.publish_tf = bool(self.get_parameter('enable_tf').value)

        self.state = [init_pose[0], init_pose[1], init_pose[2]]
        self.last_cmd = (0.0, 0.0)
        self.last_cmd_time = self.get_clock().now()
        self.last_update_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, '/odom', 20)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 20)

        period = 1.0 / float(self.get_parameter('publish_rate').value)
        self.create_timer(period, self.timer_callback)
        self.get_logger().info(
            f'Simulator online: publishing {self.odom_frame}->{self.base_frame} at {1.0/period:.1f} Hz'
        )

    def cmd_callback(self, msg: Twist) -> None:
        self.last_cmd = (float(msg.linear.x), float(msg.angular.z))
        self.last_cmd_time = self.get_clock().now()

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        v, w = self.last_cmd
        if (now - self.last_cmd_time).nanoseconds / 1e9 > self.cmd_timeout:
            v = 0.0
            w = 0.0

        # Simple unicycle integration
        self.state[0] += v * math.cos(self.state[2]) * dt
        self.state[1] += v * math.sin(self.state[2]) * dt
        self.state[2] += w * dt
        self.state[2] = math.atan2(math.sin(self.state[2]), math.cos(self.state[2]))

        self.last_update_time = now
        self.publish_odom(now, v, w)

    def publish_odom(self, stamp, v: float, w: float) -> None:
        qz = math.sin(self.state[2] * 0.5)
        qw = math.cos(self.state[2] * 0.5)

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)

        if not self.publish_tf:
            return

        tf_msg = TransformStamped()
        tf_msg.header.stamp = odom.header.stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = self.state[0]
        tf_msg.transform.translation.y = self.state[1]
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()