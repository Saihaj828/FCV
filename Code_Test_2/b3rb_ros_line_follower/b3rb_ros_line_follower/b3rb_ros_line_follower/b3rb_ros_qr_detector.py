# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

# Import PyZbar for faster and more reliable decoding; fallback to OpenCV if missing
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False


class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic
    ONLY IF that specific payload has never been published before during this session.
    """
    def __init__(self):
        super().__init__('qr_detector')

        # Permanent set to track all unique QR codes published during this run
        self.published_qrs = set()

        self.prev_qr = None

        # OpenCV detector instance reused across callbacks
        self.cv_qr_detector = cv2.QRCodeDetector()

        # Subscription for camera images
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        if PYZBAR_AVAILABLE:
            self.get_logger().info("QR Detector Node started with PyZbar + OpenCV fallback.")
        else:
            self.get_logger().warn("PyZbar not found (install via 'pip install pyzbar'). Using OpenCV detector only.")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().error("Failed to decode compressed image message.")
            return

        # Reads frame every single time
        qr_data = self.detect_qr_code(image)

        # Check if code was found AND hasn't been published previously in this session
        if qr_data and qr_data != self.prev_qr:
            self.published_qrs.add(qr_data)  # Lock it in so it's never published again
            self.prev_qr = qr_data
            
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"Published NEW QR Data: {qr_data}")

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image using PyZbar (primary) 
        and OpenCV (fallback). Pre-processing converts to grayscale for higher contrast.
        """
        # 1. Image Pre-processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Try PyZbar first
        if PYZBAR_AVAILABLE:
            try:
                decoded_objects = pyzbar.decode(gray)
                for obj in decoded_objects:
                    payload = obj.data.decode('utf-8').strip()
                    if payload:
                        return payload
            except Exception as e:
                self.get_logger().debug(f"PyZbar detection error: {e}")

        # 3. Fallback to OpenCV Built-in QR Detector
        try:
            data, bbox, _ = self.cv_qr_detector.detectAndDecode(gray)
            if bbox is not None and data:
                return data.strip()
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR Detection error: {e}")

        return None


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()