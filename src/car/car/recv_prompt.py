import base64
import json
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import rclpy
import uvicorn
from cv_bridge import CvBridge
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from geometry_msgs.msg import PoseArray
from pydantic import BaseModel
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

# 更新 PromptIn 模型
class PromptIn(BaseModel):
    text: str
    timestamp: Optional[float] = None  # 前端提供的时间戳（可选）


class PromptModeIn(BaseModel):
    mode: str

class RecvPromptNode(Node):
    def __init__(self):
        super().__init__("recv_prompt")

        self.declare_parameter("prompt_topic", "/car/prompt")
        self.declare_parameter("image_topic", "/car/process_pic")
        self.declare_parameter("raw_image_topic", "/car/pic")
        self.declare_parameter("camera_mode", "")
        self.declare_parameter("text_topic", "/car/model_text")
        self.declare_parameter("waypoint_topic", "/goal_point")
        self.declare_parameter("prompt_complete_topic", "/car/prompt_complete")  # 新增
        self.declare_parameter("prompt_dispatch_mode", "repeat_1hz")
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 8787)

        self.prompt_topic = self.get_parameter("prompt_topic").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.raw_image_topic = self.get_parameter("raw_image_topic").get_parameter_value().string_value
        self.camera_mode = self.get_parameter("camera_mode").get_parameter_value().string_value
        self.text_topic = self.get_parameter("text_topic").get_parameter_value().string_value
        self.waypoint_topic = self.get_parameter("waypoint_topic").get_parameter_value().string_value
        self.prompt_complete_topic = self.get_parameter("prompt_complete_topic").get_parameter_value().string_value  # 新增
        raw_dispatch_mode = (
            self.get_parameter("prompt_dispatch_mode").get_parameter_value().string_value
        )
        self.http_host = self.get_parameter("http_host").get_parameter_value().string_value
        self.http_port = self.get_parameter("http_port").get_parameter_value().integer_value

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        camera_mode_norm = (self.camera_mode or "").strip().lower()
        if camera_mode_norm == "camera_dual" and self.raw_image_topic == "/car/pic":
            self.raw_image_topic = "/camera/color/image_raw"
        raw_qos = qos_profile_sensor_data if camera_mode_norm == "camera_dual" else qos

        self.prompt_pub = self.create_publisher(String, self.prompt_topic, qos)
        self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.create_subscription(Image, self.raw_image_topic, self.on_raw_image, raw_qos)
        self.create_subscription(String, self.text_topic, self.on_text, qos)
        self.create_subscription(PoseArray, self.waypoint_topic, self.on_waypoints, qos)
        self.create_subscription(String, self.prompt_complete_topic, self.on_prompt_complete, qos)

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._pending_prompts: List[Dict[str, Any]] = []  # 改为存储字典
        self._latest_image_b64 = ""
        self._latest_frame_id = ""
        self._latest_raw_image_b64 = ""
        self._latest_raw_frame_id = ""
        self._latest_text = ""
        self._latest_waypoints: List[Dict[str, float]] = []
        self._current_prompt_processing = False
        self._current_prompt_uuid: Optional[str] = None  # 新增：当前处理的 UUID
        self._current_prompt_text: Optional[str] = None
        self._repeat_prompt_uuid: Optional[str] = None
        self._repeat_prompt_text: Optional[str] = None
        self._last_prompt_complete_time = None
        self._dispatch_mode = self._normalize_dispatch_mode(raw_dispatch_mode)
        self._last_prompt_publish_time = 0.0

        if self._dispatch_mode is None:
            self._dispatch_mode = "repeat_1hz"
            self.get_logger().warn(
                f"无效 prompt_dispatch_mode={raw_dispatch_mode}，回退到 repeat_1hz"
            )

        self.create_timer(0.05, self.flush_prompt_queue)

        self._start_http_server()
        self.get_logger().info(
            f"HTTP ready at http://{self.http_host}:{self.http_port}, prompt_topic={self.prompt_topic}, dispatch_mode={self._dispatch_mode}"
        )
        self.get_logger().info(
            f"raw_image_topic={self.raw_image_topic}, camera_mode={camera_mode_norm}"
        )

    @staticmethod
    def _normalize_dispatch_mode(mode: str) -> Optional[str]:
        value = (mode or "").strip().lower()
        if value in {"single", "once", "normal", "legacy", "default"}:
            return "single"
        if value in {"repeat_1hz", "repeat", "1hz", "loop"}:
            return "repeat_1hz"
        return None

    def _start_http_server(self):
        app = FastAPI(title="RecvPromptNode API")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/api/health")
        def health():
            return {"ok": True}

        @app.post("/api/prompt")
        def post_prompt(body: PromptIn):
            text = (body.text or "").strip()
            if not text:
                return {"ok": False, "error": "empty prompt"}
            
            # 生成或使用前端提供的时间戳作为 UUID
            if body.timestamp is not None:
                uuid = str(float(body.timestamp))
            else:
                uuid = str(time.time_ns() / 1e9)  # 纳秒转秒
            
            with self._lock:
                self._pending_prompts.append({"uuid": uuid, "text": text})
            
            return {"ok": True, "queued": text, "uuid": uuid}

        @app.get("/api/state")
        def get_state():
            with self._lock:
                return {
                    "ok": True,
                    "text": self._latest_text,
                    "frame_id": self._latest_frame_id,
                    "image_jpeg_b64": self._latest_image_b64,
                    "raw_frame_id": self._latest_raw_frame_id,
                    "raw_image_topic": self.raw_image_topic,
                    "raw_image_jpeg_b64": self._latest_raw_image_b64,
                    "waypoints": self._latest_waypoints,
                    "prompt_processing": self._current_prompt_processing,
                    "prompt_uuid": self._current_prompt_uuid,  # 新增
                    "prompt_complete_time": self._last_prompt_complete_time,
                    "pending_prompts_count": len(self._pending_prompts),
                    "prompt_dispatch_mode": self._dispatch_mode,
                }

        @app.get("/api/raw_camera")
        def get_raw_camera():
            with self._lock:
                return {
                    "ok": True,
                    "frame_id": self._latest_raw_frame_id,
                    "image_jpeg_b64": self._latest_raw_image_b64,
                }

        # 可选：专门的状态端点
        @app.get("/api/prompt_status")
        def get_prompt_status():
            with self._lock:
                return {
                    "ok": True,
                    "processing": self._current_prompt_processing,
                    "complete_time": self._last_prompt_complete_time,
                    "pending_count": len(self._pending_prompts),
                    "mode": self._dispatch_mode,
                }

        @app.get("/api/prompt_mode")
        def get_prompt_mode():
            with self._lock:
                return {
                    "ok": True,
                    "mode": self._dispatch_mode,
                    "supported_modes": ["single", "repeat_1hz"],
                }

        @app.post("/api/prompt_mode")
        def set_prompt_mode(body: PromptModeIn):
            mode = self._normalize_dispatch_mode(body.mode)
            if mode is None:
                return {
                    "ok": False,
                    "error": "invalid mode",
                    "supported_modes": ["single", "repeat_1hz"],
                }

            with self._lock:
                self._dispatch_mode = mode
                if mode == "single" and not self._current_prompt_processing:
                    self._current_prompt_uuid = None
                    self._current_prompt_text = None

            self.get_logger().info(f"切换 prompt 下发模式: {mode}")
            return {"ok": True, "mode": mode}

        def run_server():
            uvicorn.run(app, host=self.http_host, port=self.http_port, log_level="warning")

        t = threading.Thread(target=run_server, daemon=True)
        t.start()

    def flush_prompt_queue(self):
        msg_data = None
        text = None
        uuid = None

        with self._lock:
            now = time.monotonic()

            should_take_new_prompt = False
            if self._pending_prompts:
                if self._dispatch_mode == "repeat_1hz":
                    should_take_new_prompt = True
                elif not self._current_prompt_processing:
                    should_take_new_prompt = True

            if should_take_new_prompt:
                prompt = self._pending_prompts.pop(0)
                self._current_prompt_uuid = prompt["uuid"]
                self._current_prompt_text = prompt["text"]
                self._current_prompt_processing = True
                self._repeat_prompt_uuid = prompt["uuid"]
                self._repeat_prompt_text = prompt["text"]
                text = prompt["text"]
                uuid = prompt["uuid"]
                msg_data = json.dumps({"uuid": uuid, "text": text})
                self._last_prompt_publish_time = now
            elif (
                self._dispatch_mode == "repeat_1hz"
                and self._repeat_prompt_uuid is not None
                and self._repeat_prompt_text is not None
                and (now - self._last_prompt_publish_time) >= 1.0
            ):
                uuid = self._repeat_prompt_uuid
                text = self._repeat_prompt_text
                msg_data = json.dumps({"uuid": uuid, "text": text})
                self._last_prompt_publish_time = now

        if msg_data is None:
            return

        # 发布 JSON 格式的消息
        msg = String()
        msg.data = msg_data
        self.prompt_pub.publish(msg)
        self.get_logger().info(f"发布 prompt: {text}, uuid: {uuid}")
        
    def on_prompt_complete(self, msg: String):
        """处理 prompt 完成信号"""
        try:
            data = json.loads(msg.data)
            uuid = data.get("uuid")
            status = data.get("status")  # "completed" 或 "interrupted"
            if not uuid or not status:
                return
            
            with self._lock:
                # 检查是否与当前正在处理的 prompt 匹配
                if uuid == self._current_prompt_uuid:
                    if status == "completed":
                        self._last_prompt_complete_time = self.get_clock().now().nanoseconds / 1e9
                        self._current_prompt_processing = False
                        if self._dispatch_mode == "single":
                            self._current_prompt_uuid = None
                            self._current_prompt_text = None
                        self.get_logger().info(f"prompt {uuid} 完成")
                    elif status == "interrupted":
                        # 被中断，标记为未处理
                        self._current_prompt_processing = False
                        if self._dispatch_mode == "single":
                            self._current_prompt_uuid = None
                            self._current_prompt_text = None
                        self.get_logger().info(f"prompt {uuid} 被中断")
                else:
                    # UUID 不匹配，可能是一个旧的被中断的 prompt
                    self.get_logger().debug(f"收到不匹配的 UUID 完成信号: {uuid}")
        except Exception as e:
            self.get_logger().warn(f"解析完成信号失败: {e}")

    def on_text(self, msg: String):
        with self._lock:
            self._latest_text = msg.data

    def on_waypoints(self, msg: PoseArray):
        points = []
        for p in msg.poses:
            points.append({"x": p.position.x, "y": p.position.y, "z": p.position.z})
        with self._lock:
            self._latest_waypoints = points

    def on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, buf = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return
            b64 = base64.b64encode(buf).decode("utf-8")
            with self._lock:
                self._latest_image_b64 = b64
                self._latest_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"on_image error: {e}")

    def on_raw_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, buf = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return
            b64 = base64.b64encode(buf).decode("utf-8")
            with self._lock:
                self._latest_raw_image_b64 = b64
                self._latest_raw_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"on_raw_image error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RecvPromptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()