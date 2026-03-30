# base

cd /home/apollo/disk/ros2/src
pip install casadi
sudo apt install ros-humble-osqp-vendor
git clone https://github.com/agilexrobotics/ugv_sdk.git
git clone https://github.com/agilexrobotics/hunter_ros2.git
cd ..
colcon build

# 有关摄像头

## 单目摄像头
sudo apt install v4l-utils
v4l2-ctl --list-devices    
command -v v4l2-ctl && ls -1 /dev/video*
nvgstcapture-1.0 --camsrc=0 --cap-dev-node=0

    cap-dev-node为dev编号

## 双目摄像头
ros2 launch astra_camera astra_pro.launch.xml uvc_vendor_id:=0x2bc5 uvc_product_id:=0x050f serial_number:=ACR874300E4

# gs_usb 驱动
ros2/tools/jetson-gs_usb-kernel-builder.sh 这个脚本编译并安装gs_usb驱动
orin的usb转can的端口号是can0, ros2/src/ugv_sdk/scripts里面要改一下
candump can2

# 停止包中的所有节点
pkill -f car &&
pkill -f all_launcher &&
pkill -f mpc_planner &&
pkill -f hunter &&
ros2 node list

# omni依赖问题
pip install -e .
conda activate omnivla


