#!/usr/bin/env python3
"""
置信度感知速度调节器 - 单元测试
测试覆盖：角速度计算、置信度计算、速度调节、边界条件
"""
import pytest
import numpy as np
import math
from typing import List


def clip_angle(angle: float) -> float:
    """限制角度在 [-π, π] 围围内"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class ConfidenceAwareSpeedController:
    """置信度感知速度调节器 - 测试版本"""

    def __init__(self, metric_waypoint_spacing: float = 0.2):
        self.metric_waypoint_spacing = metric_waypoint_spacing

        # 置信度阈值
        self.high_conf_threshold = 0.1  # rad/s
        self.low_conf_threshold = 0.3   # rad/s

        # 连续低置信度计数（用于紧急制动）
        self.low_confidence_count = 0
        self.low_confidence_max_count = 3  # 连续3帧低置信度触发减速

        # 历史角速度标准差记录
        self.history_sigma_omega = []
        self.history_max_len = 5

    def calculate_waypoint_angular_vel(self, waypoint: List[float]) -> float:
        """计算单个 waypoint 的角速度

        waypoint: [dx, dy, hx, hy]
        """
        dx, dy, hx, hy = waypoint
        dx_real = dx * self.metric_waypoint_spacing
        dy_real = dy * self.metric_waypoint_spacing

        # 使用与服务器相同的计算逻辑
        DT = 1 / 3
        EPS = 1e-8

        if np.abs(dx_real) < EPS and np.abs(dy_real) < EPS:
            # 纯转向
            return 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx_real) < EPS:
            return 1.0 * np.sign(dy_real) * np.pi / (2 * DT)
        else:
            return np.arctan(dy_real / dx_real) / DT

    def compute_confidence(self, waypoints: List[List[float]]) -> float:
        """计算角速度序列的标准差作为置信度分数

        Args:
            waypoints: 8个路径点列表

        Returns:
            sigma_omega: 角速度标准差 (rad/s)
        """
        if len(waypoints) < 2:
            return 0.0

        angular_vels = [self.calculate_waypoint_angular_vel(wp) for wp in waypoints]
        sigma_omega = np.std(angular_vels)

        # 记录历史
        self.history_sigma_omega.append(sigma_omega)
        if len(self.history_sigma_omega) > self.history_max_len:
            self.history_sigma_omega.pop(0)

        return sigma_omega

    def adjust_speed(self, linear_vel: float, waypoints: List[List[float]]) -> float:
        """根据置信度调整线速度

        Args:
            linear_vel: 原始线速度
            waypoints: 8个路径点

        Returns:
            adjusted_vel: 调整后的线速度
        """
        # 边界处理
        if len(waypoints) < 2:
            return linear_vel

        if linear_vel <= 0:
            return linear_vel

        sigma_omega = self.compute_confidence(waypoints)

        # 高置信度：保持原速
        if sigma_omega < self.high_conf_threshold:
            self.low_confidence_count = 0
            return linear_vel

        # 低置信度：降速
        if sigma_omega > self.low_conf_threshold:
            self.low_confidence_count += 1

            # 连续低置信度触发紧急制动
            if self.low_confidence_count >= self.low_confidence_max_count:
                return linear_vel * 0.3  # 降速70%

            return linear_vel * 0.5  # 降速50%

        # 中等置信度：按比例降速
        scale = min(1.0, self.low_conf_threshold / sigma_omega)
        return linear_vel * scale


class TestConfidenceController:

    def setup_method(self):
        self.ctrl = ConfidenceAwareSpeedController(metric_waypoint_spacing=0.2)

    # ========== 角速度计算测试 ==========

    def test_pure_rotation_waypoint(self):
        """测试纯转向 waypoint（dx=0, dy=0）的角速度计算"""
        # waypoint [0, 0, 1, 0] → 方向向量指向 x+
        wp = [0, 0, 1, 0]
        omega = self.ctrl.calculate_waypoint_angular_vel(wp)
        # arctan2(0, 1) = 0
        assert np.abs(omega) < 0.1

    def test_forward_waypoint(self):
        """测试前进 waypoint 的角速度计算"""
        # waypoint [1, 0, cos(0), sin(0)] → 直行
        wp = [1, 0, 1, 0]
        omega = self.ctrl.calculate_waypoint_angular_vel(wp)
        # dy/dx = 0，角速度应为 0
        assert np.abs(omega) < 0.1

    def test_left_turn_waypoint(self):
        """测试左转 waypoint 的角速度计算"""
        # waypoint [1, 0.5, cos(θ), sin(θ)] → 左转
        wp = [1, 0.5, 0.9, 0.1]  # 有正向横向偏移
        omega = self.ctrl.calculate_waypoint_angular_vel(wp)
        # arctan(0.5/1) / DT > 0
        assert omega > 0

    def test_right_turn_waypoint(self):
        """测试右转 waypoint 的角速度计算"""
        wp = [1, -0.5, 0.9, -0.1]  # 负向横向偏移
        omega = self.ctrl.calculate_waypoint_angular_vel(wp)
        assert omega < 0

    # ========== 置信度计算测试 ==========

    def test_high_confidence_uniform_waypoints(self):
        """测试高置信度场景：8个waypoints方向一致"""
        # 所有 waypoints 都是直行方向
        waypoints = [[1, 0, 1, 0] for _ in range(8)]
        sigma = self.ctrl.compute_confidence(waypoints)
        # 标准差应接近 0
        assert sigma < 0.1

    def test_low_confidence_conflicting_waypoints(self):
        """测试低置信度场景：waypoints方向矛盾"""
        # 前4个左转，后4个右转
        waypoints = [
            [1, 0.3, 0.9, 0.1],  # 左转
            [1, 0.4, 0.85, 0.15],
            [1, 0.5, 0.8, 0.2],
            [1, 0.6, 0.7, 0.3],
            [1, -0.3, 0.9, -0.1],  # 右转
            [1, -0.4, 0.85, -0.15],
            [1, -0.5, 0.8, -0.2],
            [1, -0.6, 0.7, -0.3],
        ]
        sigma = self.ctrl.compute_confidence(waypoints)
        # 标准差应较大
        assert sigma > 0.2

    def test_medium_confidence(self):
        """测试中等置信度场景"""
        # 逐渐转向的 waypoints（较平滑）
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
        sigma = self.ctrl.compute_confidence(waypoints)
        # 标准差应在中等范围
        assert 0.05 < sigma < 0.3

    # ========== 速度调节测试 ==========

    def test_high_conf_keep_speed(self):
        """测试高置信度保持原速"""
        waypoints = [[1, 0, 1, 0] for _ in range(8)]  # 一致
        linear_vel = 0.3
        adjusted = self.ctrl.adjust_speed(linear_vel, waypoints)
        # 应保持原速
        assert adjusted == linear_vel

    def test_low_conf_reduce_speed_50(self):
        """测试低置信度降速50%"""
        # 矛盾 waypoints
        waypoints = [
            [1, 0.5, 0.8, 0.2],  # 左转
            [1, -0.5, 0.8, -0.2],  # 右转
        ] * 4
        linear_vel = 0.3
        adjusted = self.ctrl.adjust_speed(linear_vel, waypoints)
        # 应降速50%
        assert adjusted == linear_vel * 0.5

    def test_consecutive_low_conf_emergency_brake(self):
        """测试连续低置信度触发紧急制动"""
        waypoints_conflicting = [
            [1, 0.5, 0.8, 0.2],
            [1, -0.5, 0.8, -0.2],
        ] * 4

        # 第1帧
        self.ctrl.adjust_speed(0.3, waypoints_conflicting)
        assert self.ctrl.low_confidence_count == 1

        # 第2帧
        self.ctrl.adjust_speed(0.3, waypoints_conflicting)
        assert self.ctrl.low_confidence_count == 2

        # 第3帧：触发紧急制动
        adjusted = self.ctrl.adjust_speed(0.3, waypoints_conflicting)
        assert adjusted == 0.3 * 0.3  # 降速70%
        assert self.ctrl.low_confidence_count == 3

    def test_confidence_reset_after_high_conf(self):
        """测试高置信度后计数器重置"""
        waypoints_low = [[1, 0.5, 0.8, 0.2], [1, -0.5, 0.8, -0.2]] * 4
        waypoints_high = [[1, 0, 1, 0] for _ in range(8)]

        # 先低置信度
        self.ctrl.adjust_speed(0.3, waypoints_low)
        self.ctrl.adjust_speed(0.3, waypoints_low)
        assert self.ctrl.low_confidence_count == 2

        # 突然高置信度
        self.ctrl.adjust_speed(0.3, waypoints_high)
        assert self.ctrl.low_confidence_count == 0

    # ========== 边界测试 ==========

    def test_empty_waypoints(self):
        """测试空 waypoints 边界"""
        adjusted = self.ctrl.adjust_speed(0.3, [])
        # 应返回原速
        assert adjusted == 0.3

    def test_single_waypoint(self):
        """测试单个 waypoint 边界"""
        adjusted = self.ctrl.adjust_speed(0.3, [[1, 0, 1, 0]])
        # 单点无法计算标准差，应返回原速
        assert adjusted == 0.3

    def test_zero_linear_vel(self):
        """测试零速度边界"""
        waypoints = [[1, 0.5, 0.8, 0.2], [1, -0.5, 0.8, -0.2]] * 4
        adjusted = self.ctrl.adjust_speed(0.0, waypoints)
        assert adjusted == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])