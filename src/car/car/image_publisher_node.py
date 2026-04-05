import signal
from rclpy.qos import qos_profile_sensor_data
import subprocess
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        # 模式：local 或 camera
        self.declare_parameter('mode', 'local')
        self.declare_parameter('pic_topic', '/car/pic')
        self.declare_parameter('fps', 30)
        self.declare_parameter('pic_dir', '/home/apollo/disk/ros2/src/car/pic/0')
        self.declare_parameter('camera_device', ['/dev/video114514'])
        self.declare_parameter('frame_width', 0)
        self.declare_parameter('frame_height', 0)

        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.pic_topic = self.get_parameter('pic_topic').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.pic_dir = self.get_parameter('pic_dir').get_parameter_value().string_value
        camera_devices_param = self.get_parameter('camera_device').get_parameter_value().string_array_value
        self.camera_devices = list(camera_devices_param) if camera_devices_param else ['/dev/video0']
        self.camera_device = ''  # 实际选中的设备
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value

        self.publisher_ = self.create_publisher(Image, self.pic_topic, 10)
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.astra_proc = None
        self.latest_camera_msg = None
        self.camera_sub = None

        self.bridge = CvBridge()
        self.cv_images = []
        self.image_files = []
        self.current_image_index = 0
        self.cap = None  # camera capture handle

        if self.mode == 'local':
            self.load_images()
            self.get_logger().info(f'本地图片模式: 目录 {self.pic_dir}, {len(self.cv_images)} 张, {self.fps} FPS')
        elif self.mode == 'camera_signal':
            self.init_camera_singal()
            self.get_logger().info(f'摄像头模式: 设备 {self.camera_device}, {self.fps} FPS')
        elif self.mode == 'camera_dual':
            self.init_camera_dual()
            self.get_logger().info(f'双摄像头模式: 设备 {self.camera_device}, {self.fps} FPS')
        else:
            self.get_logger().error(f'未知的模式: {self.mode}')

        self.model_ready = False
        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_sub = self.create_subscription(
            Bool, "/car/model_ready", self._on_model_ready, ready_qos
        )
    
    def _on_model_ready(self, msg: Bool):
        self.model_ready = bool(msg.data)
        if self.model_ready:
            self.get_logger().info("收到模型就绪信号，开始发布图像")

    def init_camera_singal(self):
        self.cap = None
        for device in self.camera_devices:
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                self.cap = cap
                self.camera_device = device
                self.get_logger().info(f'已选择可用摄像头: {self.camera_device}')
                break
            cap.release()

        if self.cap is None:
            self.get_logger().error(f'无法打开任何摄像头设备: {self.camera_devices}')
            rclpy.shutdown()
            os._exit(1)
            return
        
    def init_camera_dual(self):
        # 启动 astra_camera
        cmd = (
            "ros2 launch astra_camera astra_pro.launch.xml "
            "uvc_vendor_id:=0x2bc5 uvc_product_id:=0x050f serial_number:=ACR874300E4"
        )
        self.astra_proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            preexec_fn=os.setsid,
        )
        self.get_logger().info("已启动 astra_camera 子进程")

        # 订阅 astra 彩色图像
        self.camera_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._on_dual_camera_image,
            qos_profile_sensor_data,
        )
        
    def _on_dual_camera_image(self, msg: Image):
        self.latest_camera_msg = msg

    def load_images(self):
        """从 self.pic_dir 加载所有图片文件。"""
        if not os.path.isdir(self.pic_dir):
            self.get_logger().error(f"图片目录不存在: {self.pic_dir}")
            return
        
        image_files = sorted([f for f in os.listdir(self.pic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            self.get_logger().error(f"在目录 {self.pic_dir} 中没有找到图片文件。")
            return
        self.get_logger().info(f"目录 {self.pic_dir}下共有{len(image_files)} 张图片，正在加载...")
        self.image_files = image_files

        for image_file in image_files:
            image_path = os.path.join(self.pic_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                self.cv_images.append(image)
            else:
                self.get_logger().warn(f"无法读取图片: {image_path}")
        
        if not self.cv_images:
            self.get_logger().error("没有成功加载任何图片。")
            
    def publish_frame(self, cv_image, frame_id):
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = frame_id
        self.publisher_.publish(ros_image)

    def timer_callback(self):
        if not self.model_ready:
            return
        
        if self.mode == 'local':
            if not self.cv_images:
                self.get_logger().warn('没有可发布的图片，请检查图片目录。', throttle_duration_sec=5)
                return
            if self.current_image_index >= len(self.cv_images):
                return
            cv_image = self.cv_images[self.current_image_index]
            image_file = self.image_files[self.current_image_index]
            self.publish_frame(cv_image, image_file)
            self.current_image_index += 1
            if self.current_image_index >= len(self.cv_images):
                reload = True
                if reload:
                    self.get_logger().info(f'已发布完所有图片共{len(self.cv_images)}张，重新发布。')
                    self.current_image_index = 0
                else:
                    self.get_logger().info(f'已发布完所有图片共{len(self.cv_images)}张，停止发布。')
                    self.timer.cancel()
        elif self.mode == 'camera_signal':
            if self.cap is None or not self.cap.isOpened():
                self.get_logger().warn('摄像头未打开，无法发布。', throttle_duration_sec=5)
                return
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('读取摄像头帧失败。', throttle_duration_sec=5)
                return
            self.publish_frame(frame, 'camera_frame')
        elif self.mode == 'camera_dual':
            if self.latest_camera_msg is None:
                self.get_logger().warn('尚未收到 /camera/color/image_raw', throttle_duration_sec=5)
                return
            msg = self.latest_camera_msg
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(msg)
                
    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.astra_proc is not None and self.astra_proc.poll() is None:
            os.killpg(os.getpgid(self.astra_proc.pid), signal.SIGTERM)
        super().destroy_node()        

def main(args=None):
    rclpy.init(args=args)
    image_publisher_node = ImagePublisherNode()
    rclpy.spin(image_publisher_node)
    image_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()