from typing import Tuple
import math
import casadi as ca
import numpy as np
import scipy.sparse as spa
import osqp
import math

class LinearMPCConfig:
    def __init__(self, T, N, Q, R, u_lower, u_upper):
        self.T = T
        self.N = N
        self.Q = Q
        self.R = R
        self.u_lower = u_lower
        self.u_upper = u_upper
        
    def __str__(self):
        return (f"T: {self.T}, N: {self.N}, Q: {np.diag(self.Q)}, R: {np.diag(self.R)}, "
                f"u_lower: {self.u_lower}, u_upper: {self.u_upper}")

class MPC:
    def __init__(self, config):
        self.config = config
        
    def solve(self, current_pose, goal_point, logger) -> Tuple[float, float]:
        raise NotImplementedError

class LinearMPC(MPC):
    def __init__(self, config):
        super().__init__(config)
        self.prev_u = np.array([0.0, 0.0])

    def __mpc_controller(self, x_curr, u_prev, ref, dt, horizon):
        """
        修正版：在当前工作点 (Operating Point) 进行线性化
        x_curr: [x, y, theta]
        u_prev: [v, w]
        ref: [xr, yr, thetar, vr, wr]
        """
        x_dim = 3
        u_dim = 2
        
        # 提取参考轨迹 (Goal)
        x_ref_traj = ref[:3]  # 目标位置
        u_ref_traj = ref[3:]  # 目标速度
        
        # --- 核心修正 1: 线性化点选取 (Linearization Point) ---
        # 使用机器人当前真实的航向角 theta 和上一次的速度 v 来计算 Jacobian
        # 这确保了 B 矩阵反映真实的物理运动方向
        v_op = u_prev[0]      # Operating Point Velocity
        theta_op = x_curr[2]  # Operating Point Theta
        
        # 为了防止 v=0 时产生奇异或无梯度，给一个小量或使用参考速度
        if abs(v_op) < 0.1:
            v_op = 0.1 

        # --- 核心修正 2: 泰勒展开 (Taylor Expansion) ---
        # 非线性模型 f(x,u):
        # x' = x + v*cos(theta)*dt
        # y' = y + v*sin(theta)*dt
        # th' = th + w*dt
        #
        # 线性化: x_{k+1} = A x_k + B u_k + c
        # 其中 c = f(x_op, u_op) - A*x_op - B*u_op
        
        # 1. 计算 A 矩阵 (df/dx) | op
        A = np.eye(x_dim)
        A[0, 2] = -v_op * np.sin(theta_op) * dt
        A[1, 2] =  v_op * np.cos(theta_op) * dt
        
        # 2. 计算 B 矩阵 (df/du) | op
        B = np.zeros((x_dim, u_dim))
        B[0, 0] = np.cos(theta_op) * dt
        B[1, 0] = np.sin(theta_op) * dt
        B[2, 1] = dt

        # 3. 计算常数项 c (Affine term)
        # f(x_op, u_op)
        f_x = x_curr[0] + v_op * np.cos(theta_op) * dt
        f_y = x_curr[1] + v_op * np.sin(theta_op) * dt
        f_theta = x_curr[2] + u_prev[1] * dt
        f_val = np.array([f_x, f_y, f_theta])
        
        # c = f_val - A @ x_curr - B @ u_prev
        c = f_val - A @ x_curr - B @ u_prev
        
        # --- 构造预测矩阵 (Prediction Matrices) ---
        # x = M_A * x0 + M_B * u + M_C
        M_A = np.zeros((x_dim * horizon, x_dim))
        M_B = np.zeros((x_dim * horizon, u_dim * horizon))
        M_C = np.zeros((x_dim * horizon)) # 扁平化，方便后续加法

        curr_A = np.eye(x_dim)
        curr_C = np.zeros(x_dim)
        
        for i in range(horizon):
            # M_A
            curr_A = A @ curr_A
            M_A[i*x_dim : (i+1)*x_dim, :] = curr_A
            
            # M_C (积累误差项)
            curr_C = A @ curr_C + c
            M_C[i*x_dim : (i+1)*x_dim] = curr_C
            
            # M_B
            for j in range(i + 1):
                # A^(i-j) * B
                if i == j:
                    term = B
                else:
                    term = np.linalg.matrix_power(A, i - j) @ B
                
                M_B[i*x_dim : (i+1)*x_dim, j*u_dim : (j+1)*u_dim] = term

        # --- QP 构建 ---
        # Cost: (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)
        # 展开后: 1/2 u^T P u + q^T u
        
        # 构造参考向量 (Reference Vector over Horizon)
        # 简单起见，假设参考点在整个 Horizon 内是不动的 (Goal Regulation)
        ref_traj = np.tile(x_ref_traj, horizon)
        
        Q_bar = spa.block_diag([self.config.Q] * horizon, format='csc')
        R_bar = spa.block_diag([self.config.R] * horizon, format='csc')
        
        # P = 2 * (B^T Q B + R)
        P = 2 * (M_B.T @ Q_bar @ M_B + R_bar)
        
        # q = 2 * B^T Q (M_A x0 + M_C - x_ref)
        # 注意: 我们这里直接优化全量 u，所以 R 的惩罚项需要考虑 u_ref
        # (u - u_ref)^T R (u - u_ref) -> u^T R u - 2 u_ref^T R u
        
        u_ref_vec = np.tile(u_ref_traj, horizon)
        predicted_path_deviation = M_A @ x_curr + M_C - ref_traj
        
        q = 2 * (M_B.T @ Q_bar @ predicted_path_deviation - R_bar @ u_ref_vec)

        # --- 约束 ---
        l = np.tile(self.config.u_lower, horizon)
        u = np.tile(self.config.u_upper, horizon)

        # --- 求解 ---
        solver = osqp.OSQP()
        P_sparse = spa.csc_matrix(P)
        A_constraint = spa.eye(u_dim * horizon, format='csc')
        
        solver.setup(P=P_sparse, q=q, A=A_constraint, l=l, u=u, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
        res = solver.solve()

        if res.info.status != 'solved':
            return False, None

        # 直接返回第一个控制量
        u_optimal = res.x[:u_dim]
        return True, u_optimal

    def solve(self, current_pose, goal_point, logger):
        x, y, theta = current_pose
        dx = goal_point[0] - x
        dy = goal_point[1] - y
        
        # 动态计算参考朝向
        theta_ref = math.atan2(dy, dx)
        
        # # 参考输入：靠近目标时减速，避免震荡
        # dist = math.hypot(dx, dy)
        # v_ref = 0.5 if dist > 0.5 else 0.2  # 简单的减速逻辑
        # w_ref = 0.0
        
        v_ref = 0.0
        w_ref = 0.0

        reference = np.array([goal_point[0], goal_point[1], theta_ref, v_ref, w_ref])

        success, u_cmd = self.__mpc_controller(
            x_curr=current_pose,
            u_prev=self.prev_u, 
            ref=reference,
            dt=self.config.T,
            horizon=int(self.config.N)
        )

        if not success or u_cmd is None:
            logger.error("MPC Solver failed!")
            return 0.0, 0.0
        
        return float(u_cmd[0]), float(u_cmd[1])

class NonlinearMPC(MPC):
    def __init__(self, config):
        super().__init__(config)
        self.T = config.T
        self.N = int(config.N)
        
        # 建立 NMPC 求解器
        self.solver = self._build_solver()
        self.u0 = np.zeros((self.N, 2)) # 初始猜测控制量
        self.x0 = np.zeros((self.N + 1, 3)) # 初始猜测状态量

    def _build_solver(self):
        # 1. 定义状态变量和控制变量 (使用 SX)
        x = ca.SX.sym('x') 
        y = ca.SX.sym('y') 
        theta = ca.SX.sym('theta') 
        states = ca.vertcat(x, y, theta)
        n_states = states.size1()

        v = ca.SX.sym('v') 
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        n_controls = controls.size1()

        # 2. 运动学模型 (UnICYCLE Model)
        rhs = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega
        )
        f = ca.Function('f', [states, controls], [rhs])

        # 3. 优化变量
        U = ca.SX.sym('U', n_controls, self.N)
        P = ca.SX.sym('P', n_states + n_states) # 参数: [Current State, Reference State]
        
        X = ca.SX.sym('X', n_states, self.N + 1)

        # 4. 构建代价函数与约束
        obj = 0
        g = []  # 约束向量

        # 初始状态约束
        st = X[:, 0]
        g.append(st - P[:3]) 

        # 权重矩阵 - 使用 SX 确保类型一致
        # CasADi 的 SX 可以从 numpy 数组创建
        Q = ca.SX(self.config.Q)
        R = ca.SX(self.config.R)

        # 提取目标状态
        target_state = P[3:6]

        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            
            # --- Cost Function ---
            st_err = st - target_state
            
            # 使用矩阵乘法: err^T * Q * err
            # CasADi 的 SX 支持 @ 运算符 (Python 3.5+)
            obj = obj + (st_err.T @ Q @ st_err) + (con.T @ R @ con)

            # --- System Dynamics Constraints ---
            st_next = X[:, k+1]
            f_value = f(st, con)
            st_next_euler = st + f_value * self.T
            g.append(st_next - st_next_euler)

        # 5. 求解器配置
        opt_variables = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )
        
        nlp_prob = {
            'f': obj,
            'x': opt_variables,
            'g': ca.vertcat(*g),
            'p': P
        }

        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }
        
        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def solve(self, current_pose, goal_point, logger) -> Tuple[float, float]:
        x, y, theta = current_pose
        dx = goal_point[0] - x
        dy = goal_point[1] - y
        
        # 1. 计算目标朝向与归一化
        theta_ref = math.atan2(dy, dx)
        diff = theta_ref - theta
        
        # 角度归一化，确保 NMPC 走最短旋转路径
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        theta_ref_adjusted = theta + diff

        target_state = np.array([goal_point[0], goal_point[1], theta_ref_adjusted])
        
        # 2. 构造参数 P
        p_val = np.concatenate((current_pose, target_state))
        
        # 3. 变量边界
        lbg = 0.0
        ubg = 0.0
        
        # 状态边界 (无限制)
        lbx = -ca.inf * np.ones((self.N + 1) * 3)
        ubx =  ca.inf * np.ones((self.N + 1) * 3)
        
        # 控制边界
        lbu = np.tile(self.config.u_lower, self.N)
        ubu = np.tile(self.config.u_upper, self.N)
        
        lb_var = np.concatenate((lbx, lbu))
        ub_var = np.concatenate((ubx, ubu))
        
        # 4. 初始猜测 (Warm Start)
        # 将上一帧的解平移作为初值
        pass # 暂时省略复杂的 Warm Start，直接用 0 或当前状态

        x0_guess = np.tile(current_pose, (self.N + 1, 1)).flatten()
        u0_guess = np.zeros(self.N * 2)
        x_init = np.concatenate((x0_guess, u0_guess))

        # 5. 求解
        try:
            res = self.solver(x0=x_init, lbx=lb_var, ubx=ub_var, lbg=lbg, ubg=ubg, p=p_val)
            
            # 提取结果
            opt_var = res['x'].full().flatten()
            # X_opt = opt_var[:(self.N + 1) * 3].reshape(self.N + 1, 3)
            U_opt = opt_var[(self.N + 1) * 3:].reshape(self.N, 2)
            
            v_cmd = float(U_opt[0, 0])
            w_cmd = float(U_opt[0, 1])
            
            return v_cmd, w_cmd

        except Exception as e:
            logger.error(f"NMPC Solver failed: {e}")
            return 0.0, 0.0

