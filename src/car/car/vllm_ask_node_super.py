import json 
from geometry_msgs.msg import Point, PoseArray, Pose
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np
import base64
import os
import time
from datetime import datetime
import re
from openai import OpenAI

class VllmAskNodeSuper(Node):
    def __init__(self):
        super().__init__('vllm_ask_node_super')
        
        self.total_duration = 0.0
        self.request_count = 0
        
                
        self.declare_parameter('pic_topic', '')
        self.declare_parameter('process_pic_topic', '')
        self.declare_parameter('commd_topic', '')
        self.declare_parameter('api_url', '')
        self.declare_parameter('compression_quality', 0)
        self.declare_parameter('img_width', 0)
        self.declare_parameter('img_hight', 0)
        self.declare_parameter('max_tokens', 0)
        self.declare_parameter('text_1', '')
        self.declare_parameter('text_2', '')
        self.declare_parameter('temperature', 0.0)
        self.declare_parameter('top_p', 1.0)
        self.declare_parameter('top_k', 1)
        self.declare_parameter('enable_thinking', 'False')

        # 获取参数并赋值为实例变量
        self.pic_topic = self.get_parameter('pic_topic').get_parameter_value().string_value
        self.process_pic_topic = self.get_parameter('process_pic_topic').get_parameter_value().string_value
        self.commd_topic = self.get_parameter('commd_topic').get_parameter_value().string_value
        self.api_url = self.get_parameter('api_url').get_parameter_value().string_value
        self.compression_quality = self.get_parameter('compression_quality').get_parameter_value().integer_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_hight = self.get_parameter('img_hight').get_parameter_value().integer_value
        self.max_tokens = self.get_parameter('max_tokens').get_parameter_value().integer_value
        self.text_1 = self.get_parameter('text_1').get_parameter_value().string_value
        self.text_2 = self.get_parameter('text_2').get_parameter_value().string_value
        self.enable_thinking = self.get_parameter('enable_thinking').get_parameter_value().string_value

        self.temperature = self.get_parameter('temperature').get_parameter_value().double_value
        self.top_p = self.get_parameter('top_p').get_parameter_value().double_value
        self.top_k = self.get_parameter('top_k').get_parameter_value().integer_value
        
        self.get_logger().warn(f"temperature={self.temperature}, top_p={self.top_k}, top_p={self.top_p}")
        
        self.subscription = self.create_subscription(
            Image,
            self.pic_topic,
            self.image_callback,
            100)
        self.point_publisher_ = self.create_publisher(PoseArray, self.commd_topic, 10)
        self.process_img_publisher_ = self.create_publisher(Image, self.process_pic_topic, 10)
        self.bridge = CvBridge()
        
    def image_callback(self, msg):
        imgBase64, imgName = self.__process_image(msg)
        self.__showImg(imgBase64, imgName)
        
        start_time = time.time()
        
        respone = self.__send_sequential_request(imgBase64, imgName)
        if respone is None:
            self.get_logger().warn(f"--> 响应 for {imgName} 失败")
            return

        duration = time.time() - start_time
        self.total_duration += duration
        self.request_count += 1
        avg_duration = self.total_duration / self.request_count
        self.get_logger().info(f"--> 响应 for {imgName} ({duration:.2f}s) 平均: {avg_duration:.2f}s: \n{respone}")
        
        points = self.__parse_point_from_response(respone)
        if points is not None and len(points) > 0:
            self.__publish_points(points)
        
    def __showImg(self, imgBase64, imgName):
        # 解码 base64 为 numpy 数组
        nparr = np.frombuffer(base64.b64decode(imgBase64.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 转换并发布图片
        ros_image = self.bridge.cv2_to_imgmsg(img, "bgr8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = imgName
        self.process_img_publisher_.publish(ros_image)

    def __process_image(self, msg):
        # 1. 解析图片
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_hight))
        # 2. 压缩为JPEG
        success, buffer = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality])
        if not success:
            self.get_logger().error("图片压缩失败")
            return None, None
        # 3. 编码为base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        base64_image = f"data:image/jpeg;base64,{jpg_as_text}"
        # 4. 获取文件名
        image_name = msg.header.frame_id if msg.header.frame_id else ""
        return base64_image, image_name

    def __publish_points(self, points):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map" 

        for p in points:
            pose = Pose()
            pose.position = p
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
            
        self.point_publisher_.publish(pose_array)   
        
    def __send_sequential_request(self, imgBase64, imgName):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "/app/model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.text_1
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": imgBase64
                            }
                        },
                        {
                            "type": "text",
                            "text": self.text_2
                        }
                    ]
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_thinking": True,
            # "extra_body": {
            #     "enable_thinking": True
            # },
            "logprobs":True,
            "top_logprobs": 1,
        }
        # "enable_thinking": self.enable_thinking,

        try:        
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()            
            response_data = response.json()
            self.get_logger().info(f"响应 for {imgName}:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
            assistant_message = response_data['choices'][0]['message']['content']

            return assistant_message
        except requests.exceptions.RequestException as e:
            self.get_logger().info(f"  -> 请求失败 for {imgName}: {e}")
            
    def __parse_point_from_response(self, response_text):
        matches = re.findall(
            r'\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)',
            response_text
        )

        if not matches:
            self.get_logger().warn("模型输出未找到坐标，默认丢弃")
            return None
        
        points = []
        for x_str, y_str in matches:
            p = Point()
            # p.x = float(x_str)
            # p.y = float(y_str)
            
            # 模型生成的是左手坐标系, ros2里用的是右手坐标系, 换一下xy位置
            p.x = float(y_str)
            p.y = -float(x_str)
            p.z = 0.0
            points.append(p)
        return points

def main(args=None):
    rclpy.init(args=args)
    vllm_ask_node = VllmAskNode()
    rclpy.spin(vllm_ask_node)
    vllm_ask_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



