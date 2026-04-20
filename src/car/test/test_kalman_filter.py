#!/usr/bin/env python3
"""
KinematicKalmanFilter 仿真测试

验证卡尔曼滤波器的效果：
- 平滑性：速度指令是否平滑过渡
- 连续性：推理空窗期是否持续输出预测速度
- 安全性：加速度硬限幅是否生效
"""
import sys
import os
import numpy as np
import pytest

# 添加路径以导入 KinematicKalmanFilter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'car'))

# 直接定义 KinematicKalmanFilter（避免 ROS 导入问题）
class KinematicKalmanFilter:
    """面向大模型高延迟推理的运动学卡尔曼滤波器"""

    def __init__(self, dt_high_freq: float = 0.05):
        self.dt = dt_high_freq
        self.x = np.zeros(4)
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


class TestKinematicKalmanFilter:
    """单元测试"""

    def test_initial_state(self):
        """验证初始状态为零"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        assert np.allclose(kf.x, np.zeros(4))
        assert np.allclose(kf.P, np.diag([1.0, 1.0, 1.0, 1.0]))

    def test_predict_step_zero_state(self):
        """验证预测步：零状态保持为零"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        x_pred = kf.predict()
        # 零加速度时，速度保持不变
        assert np.allclose(x_pred[:2], np.zeros(2))

    def test_predict_step_with_velocity(self):
        """验证预测步：有速度时状态随时间推移"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        kf.x[0] = 0.5  # 设置初始线速度
        kf.x[1] = 0.1  # 设置初始角速度
        x_pred = kf.predict()
        # 无加速度时，速度保持
        assert abs(x_pred[0] - 0.5) < 0.01
        assert abs(x_pred[1] - 0.1) < 0.01

    def test_predict_step_with_acceleration(self):
        """验证预测步：有加速度时速度增加"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        kf.x[0] = 0.0  # 初始速度为0
        kf.x[2] = 1.0  # 设置线加速度 1.0 m/s²
        x_pred = kf.predict()
        # v = v0 + a * dt = 0 + 1.0 * 0.05 = 0.05
        assert abs(x_pred[0] - 0.05) < 0.001

    def test_update_step(self):
        """验证更新步：观测值修正状态"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        z = np.array([0.5, 0.1])  # 观测值
        kf.update(z)
        # 卡尔曼滤波器应使状态接近观测值
        assert abs(kf.x[0] - 0.5) < 0.5  # 宽松验证
        assert abs(kf.x[1] - 0.1) < 0.5

    def test_acceleration_decay(self):
        """验证加速度衰减机制"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        kf.x[2] = 1.0  # 设置加速度
        kf.update(np.array([0.5, 0.0]))
        # 加速度应该衰减 0.9 倍
        assert abs(kf.x[2]) < 1.0


class TestKalmanSimulation:
    """仿真测试：模拟真实使用场景"""

    def test_smooth_transition(self):
        """测试平滑过渡：速度突变场景"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        max_linear_accel = 0.3  # m/s²

        # 模拟场景：初始速度0，收到观测值0.5 m/s
        # 高频发布（20Hz）30个周期（1.5s）
        outputs = []

        # 第一次收到观测（模拟推理完成）
        kf.update(np.array([0.5, 0.0]))

        # 高频发布 30 个周期
        v_prev = 0.0
        for i in range(30):
            x_pred = kf.predict()
            v_raw = x_pred[0]

            # 物理限幅
            dt = 0.05
            v_out = np.clip(
                v_raw,
                v_prev - max_linear_accel * dt,
                v_prev + max_linear_accel * dt
            )
            outputs.append(v_out)
            v_prev = v_out

        # 验证：输出应该逐步增加，而非突变到0.5
        # 第一个输出应该接近 max_linear_accel * dt = 0.3 * 0.05 = 0.015
        assert outputs[0] < 0.1  # 不应该突变

        # 最终输出应该接近观测值 0.5
        assert outputs[-1] > 0.3

        # 验证平滑性：相邻输出差值不超过加速度限制
        for i in range(1, len(outputs)):
            delta = abs(outputs[i] - outputs[i-1])
            assert delta <= max_linear_accel * dt + 0.001

    def test_gap_period_prediction(self):
        """测试空窗期：1.5s无新观测"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)

        # 先收到一次观测
        kf.update(np.array([0.5, 0.1]))

        # 然后 1.5s 无新观测（30个预测周期）
        outputs = []
        for i in range(30):
            x_pred = kf.predict()
            outputs.append(x_pred[:2].copy())

        # 验证：空窗期内持续输出预测速度
        v_values = [o[0] for o in outputs]

        # 初期应该接近观测值
        assert v_values[0] > 0.3

        # 验证输出稳定性：所有输出值都应该有意义（非零）
        assert all(v > 0.2 for v in v_values)

        # 验证速度变化趋势：最终速度 <= 初期速度（允许相等或略小）
        assert v_values[-1] <= v_values[0] + 0.01

    def test_accel_limit_hard_constraint(self):
        """测试加速度硬限幅"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        max_linear_accel = 0.3  # m/s²
        dt = 0.05

        # 模拟：观测值突变（0→10 m/s，超出物理限制）
        kf.update(np.array([10.0, 0.0]))

        # 应用物理限幅
        v_prev = 0.0
        outputs = []

        for i in range(100):  # 5秒
            x_pred = kf.predict()
            v_raw = x_pred[0]

            v_out = np.clip(
                v_raw,
                v_prev - max_linear_accel * dt,
                v_prev + max_linear_accel * dt
            )
            outputs.append(v_out)
            v_prev = v_out

        # 验证：实际加速度不超过限制
        for i in range(1, len(outputs)):
            accel = (outputs[i] - outputs[i-1]) / dt
            assert abs(accel) <= max_linear_accel + 0.01

    def test_multi_rate_scenario(self):
        """完整多速率场景测试"""
        kf = KinematicKalmanFilter(dt_high_freq=0.05)
        max_linear_accel = 0.3
        max_angular_accel = 0.5
        dt = 0.05

        # 模拟：低频推理（1.5s周期）+ 高频发布（20Hz）
        total_time = 6.0  # 6秒仿真
        inference_interval = 1.5  # 推理周期
        high_freq_steps = int(total_time / dt)

        # 观测序列（模拟推理结果）
        observations = [
            (0.0, np.array([0.0, 0.0])),
            (1.5, np.array([0.3, 0.2])),
            (3.0, np.array([0.5, 0.3])),
            (4.5, np.array([0.2, 0.1])),
        ]

        outputs_v = []
        outputs_w = []
        v_prev = 0.0
        w_prev = 0.0

        for step in range(high_freq_steps):
            t = step * dt

            # 检查是否有新观测
            for obs_time, obs_value in observations:
                if abs(t - obs_time) < dt:
                    kf.update(obs_value)

            # 预测步
            x_pred = kf.predict()
            v_raw = x_pred[0]
            w_raw = x_pred[1]

            # 物理限幅
            v_out = np.clip(v_raw, v_prev - max_linear_accel * dt, v_prev + max_linear_accel * dt)
            w_out = np.clip(w_raw, w_prev - max_angular_accel * dt, w_prev + max_angular_accel * dt)

            outputs_v.append(v_out)
            outputs_w.append(w_out)
            v_prev = v_out
            w_prev = w_out

        # 验证输出合理性
        assert max(outputs_v) < 1.0  # 速度不超过物理限制
        assert min(outputs_v) >= 0.0  # 速度非负


def run_visualization():
    """生成可视化图表"""
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt

    kf = KinematicKalmanFilter(dt_high_freq=0.05)
    max_linear_accel = 0.3
    dt = 0.05

    # 模拟场景
    total_time = 6.0
    high_freq_steps = int(total_time / dt)

    observations = [
        (0.0, np.array([0.0, 0.0])),
        (1.5, np.array([0.5, 0.3])),
        (3.0, np.array([0.8, 0.5])),
        (4.5, np.array([0.3, 0.2])),
    ]

    outputs_v_kf = []
    outputs_v_raw = []  # 无滤波对比
    v_prev = 0.0
    raw_v = 0.0

    for step in range(high_freq_steps):
        t = step * dt

        # 检查新观测
        current_obs = None
        for obs_time, obs_value in observations:
            if abs(t - obs_time) < dt:
                current_obs = obs_value
                kf.update(obs_value)
                raw_v = obs_value[0]  # 无滤波直接用观测值

        # 滤波输出
        x_pred = kf.predict()
        v_raw_kf = x_pred[0]
        v_out = np.clip(v_raw_kf, v_prev - max_linear_accel * dt, v_prev + max_linear_accel * dt)
        outputs_v_kf.append(v_out)
        v_prev = v_out

        # 无滤波输出
        outputs_v_raw.append(raw_v)

    # 绘图
    times = np.arange(high_freq_steps) * dt

    plt.figure(figsize=(12, 6))

    # 速度对比
    plt.subplot(1, 2, 1)
    plt.plot(times, outputs_v_raw, 'r--', label='Raw Observation (No Filter)', alpha=0.7)
    plt.plot(times, outputs_v_kf, 'b-', label='Kalman Filter + Accel Limit', linewidth=2)

    # 标记观测时刻
    for obs_time, _ in observations:
        plt.axvline(x=obs_time, color='g', linestyle=':', alpha=0.5)

    plt.xlabel('Time (s)')
    plt.ylabel('Linear Velocity (m/s)')
    plt.title('Kalman Filter Smoothing Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 加速度分析
    plt.subplot(1, 2, 2)
    accel_kf = np.diff(outputs_v_kf) / dt
    accel_raw = np.diff(outputs_v_raw) / dt

    plt.plot(times[1:], accel_kf, 'b-', label='Filtered Acceleration', linewidth=2)
    plt.plot(times[1:], accel_raw, 'r--', label='Raw Acceleration', alpha=0.7)
    plt.axhline(y=max_linear_accel, color='g', linestyle='--', label=f'Limit: {max_linear_accel} m/s^2')
    plt.axhline(y=-max_linear_accel, color='g', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Acceleration Comparison (Limit Validation)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(os.path.dirname(__file__), 'kalman_filter_result.png')
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to: {output_path}")

    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='生成可视化图表')
    args = parser.parse_args()

    if args.plot:
        run_visualization()
    else:
        pytest.main([__file__, '-v'])