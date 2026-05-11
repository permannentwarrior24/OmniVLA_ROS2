#!/usr/bin/env python3
"""
置信度感知速度调节器 - 集成测试
模拟服务器响应，验证置信度控制与卡尔曼滤波器的集成效果
"""
import numpy as np
import math
from typing import List, Tuple


def clip_angle(angle: float) -> float:
    """限制角度在 [-π, π] 范围内"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class ConfidenceAwareSpeedController:
    """置信度感知速度调节器"""

    def __init__(self, metric_waypoint_spacing: float = 0.2):
        self.metric_waypoint_spacing = metric_waypoint_spacing

        # 置信度阈值
        self.high_conf_threshold = 0.1  # rad/s
        self.low_conf_threshold = 0.3   # rad/s

        # 连续低置信度计数
        self.low_confidence_count = 0
        self.low_confidence_max_count = 3

        # 历史记录
        self.history_sigma_omega = []
        self.history_max_len = 5

    def calculate_waypoint_angular_vel(self, waypoint: List[float]) -> float:
        """计算单个 waypoint 的角速度"""
        dx, dy, hx, hy = waypoint
        dx_real = dx * self.metric_waypoint_spacing
        dy_real = dy * self.metric_waypoint_spacing

        DT = 1 / 3
        EPS = 1e-8

        if np.abs(dx_real) < EPS and np.abs(dy_real) < EPS:
            return 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx_real) < EPS:
            return 1.0 * np.sign(dy_real) * np.pi / (2 * DT)
        else:
            return np.arctan(dy_real / dx_real) / DT

    def compute_confidence(self, waypoints: List[List[float]]) -> float:
        """计算角速度序列的标准差"""
        if len(waypoints) < 2:
            return 0.0

        angular_vels = [self.calculate_waypoint_angular_vel(wp) for wp in waypoints]
        sigma_omega = np.std(angular_vels)

        self.history_sigma_omega.append(sigma_omega)
        if len(self.history_sigma_omega) > self.history_max_len:
            self.history_sigma_omega.pop(0)

        return sigma_omega

    def adjust_speed(self, linear_vel: float, waypoints: List[List[float]]) -> float:
        """根据置信度调整线速度"""
        if len(waypoints) < 2 or linear_vel <= 0:
            return linear_vel

        sigma_omega = self.compute_confidence(waypoints)

        # 高置信度
        if sigma_omega < self.high_conf_threshold:
            self.low_confidence_count = 0
            return linear_vel

        # 低置信度
        if sigma_omega > self.low_conf_threshold:
            self.low_confidence_count += 1
            if self.low_confidence_count >= self.low_confidence_max_count:
                return linear_vel * 0.3
            return linear_vel * 0.5

        # 中等置信度
        scale = min(1.0, self.low_conf_threshold / sigma_omega)
        return linear_vel * scale


class KinematicKalmanFilter:
    """卡尔曼滤波器（简化版，用于集成测试）"""

    def __init__(self, dt_high_freq: float = 0.05):
        self.dt = dt_high_freq
        self.x = np.zeros(4)  # [v, ω, a_v, a_ω]

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.Q = np.diag([0.01, 0.01, 0.1, 0.1])
        self.R = np.diag([0.05, 0.05])
        self.P = np.diag([1.0, 1.0, 1.0, 1.0])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.x[2] *= 0.9
        self.x[3] *= 0.9


def simulate_inference_response(scenario: str) -> Tuple[List[List[float]], float, float]:
    """模拟不同场景的推理响应"""

    if scenario == "straight":
        # 直行：高置信度
        waypoints = [[1, 0, 1, 0] for _ in range(8)]
        linear_vel = 0.3
        angular_vel = 0.0

    elif scenario == "smooth_turn":
        # 平滑转弯：中等置信度
        waypoints = [
            [1, 0.05, 1, 0],
            [1, 0.08, 0.99, 0.01],
            [1, 0.1, 0.98, 0.02],
            [1, 0.12, 0.97, 0.03],
            [1, 0.14, 0.96, 0.04],
            [1, 0.16, 0.95, 0.05],
            [1, 0.18, 0.94, 0.06],
            [1, 0.2, 0.93, 0.07],
        ]
        linear_vel = 0.25
        angular_vel = 0.1

    elif scenario == "conflicting":
        # 方向矛盾：低置信度
        waypoints = [
            [1, 0.3, 0.9, 0.1],
            [1, -0.3, 0.9, -0.1],
            [1, 0.4, 0.85, 0.15],
            [1, -0.4, 0.85, -0.15],
            [1, 0.5, 0.8, 0.2],
            [1, -0.5, 0.8, -0.2],
            [1, 0.6, 0.7, 0.3],
            [1, -0.6, 0.7, -0.3],
        ]
        linear_vel = 0.3
        angular_vel = 0.0

    elif scenario == "emergency":
        # 连续低置信度场景（更极端）
        waypoints = [
            [1, 0.5, 0.8, 0.2],
            [1, -0.5, 0.8, -0.2],
        ] * 4
        linear_vel = 0.3
        angular_vel = 0.0

    else:
        raise ValueError(f"未知场景: {scenario}")

    return waypoints, linear_vel, angular_vel


def run_integration_test():
    """运行集成测试"""
    ctrl = ConfidenceAwareSpeedController()
    kf = KinematicKalmanFilter(dt_high_freq=0.05)

    scenarios = ["straight", "smooth_turn", "conflicting", "emergency"]

    print("=" * 60)
    print("置信度感知速度调节 - 集成测试")
    print("=" * 60)

    for scenario in scenarios:
        print(f"\n{'='*20} 场景: {scenario} {'='*20}")

        waypoints, linear_vel, angular_vel = simulate_inference_response(scenario)

        # 置信度计算
        sigma = ctrl.compute_confidence(waypoints)
        print(f"角速度标准差: {sigma:.3f} rad/s")

        # 置信度判断
        if sigma < ctrl.high_conf_threshold:
            conf_level = "高置信度 (保持原速)"
        elif sigma > ctrl.low_conf_threshold:
            conf_level = "低置信度 (降速50%)"
        else:
            conf_level = "中等置信度 (按比例降速)"
        print(f"置信度级别: {conf_level}")

        # 速度调节
        adjusted_vel = ctrl.adjust_speed(linear_vel, waypoints)
        print(f"原始速度: {linear_vel:.3f} m/s")
        print(f"调节后速度: {adjusted_vel:.3f} m/s")
        if linear_vel > 0:
            reduction = (1 - adjusted_vel/linear_vel) * 100
            print(f"降速比例: {reduction:.1f}%")

        # 连续低置信度计数
        print(f"连续低置信计数: {ctrl.low_confidence_count}/{ctrl.low_confidence_max_count}")

        # 卡尔曼更新
        kf.update(np.array([adjusted_vel, angular_vel]))

        # 模拟高频发布（5步）
        print("高频发布模拟:")
        for i in range(5):
            x_pred = kf.predict()
            print(f"  步{i}: v={x_pred[0]:.3f} m/s, ω={x_pred[1]:.3f} rad/s")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def run_emergency_brake_sequence():
    """测试连续低置信度触发紧急制动的完整序列"""
    print("\n" + "=" * 60)
    print("紧急制动序列测试 - 连续3帧低置信度")
    print("=" * 60)

    ctrl = ConfidenceAwareSpeedController()
    kf = KinematicKalmanFilter(dt_high_freq=0.05)

    waypoints_conflicting = [
        [1, 0.5, 0.8, 0.2],
        [1, -0.5, 0.8, -0.2],
    ] * 4

    linear_vel = 0.3
    angular_vel = 0.0

    for frame in range(5):
        print(f"\n--- 帧 {frame + 1} ---")

        adjusted_vel = ctrl.adjust_speed(linear_vel, waypoints_conflicting)
        sigma = ctrl.history_sigma_omega[-1] if ctrl.history_sigma_omega else 0

        print(f"σ_ω = {sigma:.3f} rad/s")
        print(f"连续低置信计数: {ctrl.low_confidence_count}/{ctrl.low_confidence_max_count}")
        print(f"调节后速度: {adjusted_vel:.3f} m/s")

        # 卡尔曼更新
        kf.update(np.array([adjusted_vel, angular_vel]))

        # 高频发布
        x_pred = kf.predict()
        print(f"卡尔曼输出: v={x_pred[0]:.3f}, ω={x_pred[1]:.3f}")

        if ctrl.low_confidence_count >= ctrl.low_confidence_max_count:
            print("*** 紧急制动已触发！降速70% ***")


if __name__ == "__main__":
    run_integration_test()
    run_emergency_brake_sequence()