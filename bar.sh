#!/usr/bin/env bash
# set -euo pipefail

root_path="/home/apollo/disk/ros2"
TARGET_ENV="omnivla"
CAN_IFACE="can2"
CAN_BRINGUP_SCRIPT="$root_path/src/ugv_sdk/scripts/bringup_can2usb_500k.bash"
CAN_CHECK_TIMEOUT="1"

clear
cd "$root_path"

init_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
      . "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
      . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
      echo "[ERROR] 找不到 conda.sh，请先安装/配置 conda"
      echo "可先执行一次: conda init bash"
      exit 1
    fi
  else
    eval "$(conda shell.bash hook)" 2>/dev/null || true
  fi
}

ensure_conda_env() {
  if [ "${CONDA_DEFAULT_ENV:-}" != "$TARGET_ENV" ]; then
    echo "当前 conda 环境: ${CONDA_DEFAULT_ENV:-<none>}，切换到 $TARGET_ENV"
    conda activate "$TARGET_ENV" || {
      echo "[ERROR] 激活 conda 环境失败: $TARGET_ENV"
      exit 1
    }
  fi
  echo "已使用 conda 环境: $CONDA_DEFAULT_ENV"
}

ensure_gs_usb() {
  if ! lsmod | grep -q "gs_usb"; then
    echo "加载 gs_usb 模块..."
    sudo modprobe gs_usb
  else
    echo "gs_usb 模块已经加载。"
  fi
}

ensure_can_interface() {
  ensure_gs_usb
  if ! ip link show "$CAN_IFACE" 2>/dev/null | grep -q "state UP"; then
    echo "启动 $CAN_IFACE 接口..."
    if [ -f "$CAN_BRINGUP_SCRIPT" ]; then
      bash "$CAN_BRINGUP_SCRIPT"
    else
      echo "[ERROR] 找不到 CAN 启动脚本: $CAN_BRINGUP_SCRIPT"
      exit 1
    fi
  else
    echo "$CAN_IFACE 接口已经启动。"
  fi
}

check_can_data() {
  local can_frame

  if ! command -v candump >/dev/null 2>&1; then
    echo "[ERROR] 未找到 candump 命令，请先安装 can-utils"
    sudo ip link set "$CAN_IFACE" down
    exit 1
  fi

  echo "检测 $CAN_IFACE 是否有数据（超时 ${CAN_CHECK_TIMEOUT}s）..."
  can_frame=$(timeout "$CAN_CHECK_TIMEOUT" candump "$CAN_IFACE" 2>/dev/null | head -n 1 || true)

  if [ -z "$can_frame" ]; then
    echo "[ERROR] $CAN_IFACE 在 ${CAN_CHECK_TIMEOUT}s 内无数据，关闭接口并退出。"
    sudo ip link set "$CAN_IFACE" down
    exit 1
  fi

  echo "$CAN_IFACE 检测到数据: $can_frame"
}

clean_build_artifacts() {
  rm -rf ./build/all_launcher ./build/car ./build/mpc_planner
  rm -rf ./install/all_launcher ./install/car ./install/mpc_planner
}

build_workspace() {
  colcon build --packages-select car mpc_planner all_launcher
}

launch_system() {
  source "$root_path/install/setup.bash"
  ros2 launch all_launcher all.launch.py
}

start_ssh_tunnel_terminal() {
  local ssh_cmd="ssh -p 28853 -N -L 8000:localhost:8000 root@connect.cqa1.seetacloud.com"

  # 必须有图形会话才能开“新终端窗口”
  if [ -z "${DISPLAY:-}" ]; then
    echo "[ERROR] 未检测到图形桌面（DISPLAY 为空），无法打开新终端窗口。"
    echo "可手动在另一个终端执行: $ssh_cmd"
    exit 1
  fi

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -lc "$ssh_cmd; exec bash" &
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "$ssh_cmd; exec bash" &
  elif command -v xterm >/dev/null 2>&1; then
    xterm -e bash -lc "$ssh_cmd; exec bash" &
  else
    echo "[ERROR] 找不到可用终端程序（gnome-terminal/x-terminal-emulator/xterm）"
    echo "请手动在另一个终端执行: $ssh_cmd"
    exit 1
  fi

  echo "已打开新终端启动 SSH 隧道，请在新终端输入密码。"
}

main() {
  init_conda
  ensure_conda_env
  ensure_can_interface
  check_can_data
  clean_build_artifacts
  build_workspace
  launch_system
}

main "$@"