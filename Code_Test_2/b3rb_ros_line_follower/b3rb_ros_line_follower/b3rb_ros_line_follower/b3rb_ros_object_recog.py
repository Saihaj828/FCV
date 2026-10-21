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
import os
from ament_index_python.packages import get_package_share_directory
import json

# HINT: TensorFlow/Keras can be heavy and might not be installed by default.
# We wrap the import in a try-except block so the node runs even if TensorFlow is missing.
# Install it using: pip install tensorflow
tf = None

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign boards.
    It publishes the detected sign type/labels on the `/sign_board_detection` topic.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        self.previous_mapping = {}
        self.board_currently_in_view = False
        self.letter_templates = self.load_templates("letters")
        self.arrow_templates = self.load_templates("arrows")

        # Attempt to load the pre-trained Keras model (model.h5) located in the same directory.
        self.model = None
        if tf is not None:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, 'model.h5')
                if os.path.exists(model_path):
                    self.model = tf.keras.models.load_model(model_path)
                    self.get_logger().info(f"Loaded Keras model from {model_path}")
                else:
                    self.get_logger().warn(f"Model file not found at {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load Keras model: {e}")
        else:
            self.get_logger().warn("TensorFlow is not installed. Running in CV/Placeholder mode.")

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        sign_detected = self.classify_sign(image)
        
        # RESET mapping when board leaves view so the NEXT board is detected freshly
        if sign_detected is None:
            if self.board_currently_in_view:
                # self.get_logger().info("Board left view. Ready for next board.")
                self.board_currently_in_view = False
                self.previous_mapping = {}
            # cv2.waitKey(1)
            return

        # Publish and print ONLY ONCE when a new valid board is processed
        if sign_detected != self.previous_mapping:
            self.previous_mapping = sign_detected
            msg = String()
            msg.data = json.dumps(sign_detected)
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"--> PUBLISHED SIGN BOARD: {sign_detected}")

        # cv2.waitKey(1)

    def classify_sign(self, image):
        board = self.detect_board(image)
        # if board is not None:
        #     cv2.imshow("Board", board)
        #     cv2.waitKey(1)
            
        if board is None:
            # Removed frame-by-frame log spam
            return None

        # Process panels ONLY ONCE per newly seen board
        if not self.board_currently_in_view:
            # self.get_logger().info("==========================================")
            # self.get_logger().info("New Board Detected! Processing Panels...")
            # self.get_logger().info("==========================================")
            self.board_currently_in_view = True
            
            panels = self.extract_panels(board)
            mapping = self.build_mapping(panels)
            if len(mapping) == 0:
                return None
            return mapping
        else:
            # Return current mapping silently while staying in front of the same board
            return self.previous_mapping

    def load_templates(self, template_type):
        """
        Loads all templates from disk into memory.
        Args:
            template_type: "letters" or "arrows"
        Returns:
            dict: {template_name: template_image}
        """
        package_path = get_package_share_directory("b3rb_ros_line_follower")
        template_dir = os.path.join(
            package_path,
            "resource",
            "templates",
            template_type
        )
        templates = {}
        for filename in os.listdir(template_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(template_dir, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            _, image = cv2.threshold(
                image,
                150,
                255,
                cv2.THRESH_BINARY
            )
            image = self.crop_to_symbol(image)
            image = cv2.resize(image, (64, 64))
            if image is None:
                continue
            templates[os.path.splitext(filename)[0]] = image
        return templates

    def detect_board(self, image):
        # Create a copy of the live image to draw debug bounding boxes on
        # debug_image = image.copy()

        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Threshold for green board
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Show the binary mask window to see what green pixels are detected
        # cv2.imshow("Debug - Green Mask", mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            # cv2.imshow("Debug - Detected Board Contour", debug_image)
            # cv2.waitKey(1)
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Target aspect ratio: 195 (width) / 45 (height) = 4.333
        target_ratio = 195.0 / 45.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                continue
                
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # --- DRAW ALL CANDIDATE CONTOURS IN BLUE ---
            # cv2.drawContours(debug_image, [contour], -1, (255, 0, 0), 2)

            # 1. Clean 4-corner convex polygon check
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = self.order_points(approx.reshape(4, 2))
                
                (tl, tr, br, bl) = corners
                width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
                height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
                
                if height > 0:
                    aspect_ratio = width / height
                    if 1.8 <= aspect_ratio <= 5.0:
                        # --- DRAW MATCHED ACCEPTED BOARD IN GREEN & RED CORNERS ---
                        # cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 3)
                        # for point in corners:
                        #     cv2.circle(debug_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                        
                        # cv2.imshow("Debug - Detected Board Contour", debug_image)
                        # cv2.waitKey(1)

                        board = self.rectify_board(image, corners)
                        return board

            # 2. Fallback: Merged shape handling preserving perspective tilt
            pts = contour.reshape(-1, 2)
            min_y = np.min(pts[:, 1])
            
            # Extract upper band of the contour to find top corners
            top_points = pts[pts[:, 1] <= min_y + 25]
            
            if len(top_points) > 0:
                # Top-Left is point furthest left-and-top, Top-Right is furthest right-and-top
                tl = top_points[np.argmin(top_points[:, 0] + top_points[:, 1])].astype("float32")
                tr = top_points[np.argmax(top_points[:, 0] - top_points[:, 1])].astype("float32")
                
                top_width = np.linalg.norm(tr - tl)
                
                if top_width > 50:
                    # Apply a 0.92 height factor so the bottom edge doesn't include wooden post/ground artifacts
                    calculated_height = (top_width / target_ratio) * 0.92
                    
                    # Compute perpendicular vector to the top edge (tr - tl)
                    edge_vec = tr - tl
                    normal_vec = np.array([-edge_vec[1], edge_vec[0]], dtype="float32")
                    
                    # Ensure normal vector points downward
                    if normal_vec[1] < 0:
                        normal_vec = -normal_vec
                    
                    # Normalize vector and scale by calculated height
                    normal_vec = (normal_vec / np.linalg.norm(normal_vec)) * calculated_height
                    
                    # Project downward to form proper perspective bottom corners
                    bl = tl + normal_vec
                    br = tr + normal_vec
                    
                    estimated_corners = np.array([tl, tr, br, bl], dtype="float32")
                    ordered_corners = self.order_points(estimated_corners)
                    
                    # --- DRAW FALLBACK ESTIMATED BOX IN YELLOW WITH CORNER DOTS ---
                    # pts_draw = ordered_corners.astype(int)
                    # cv2.polylines(debug_image, [pts_draw], True, (0, 255, 255), 3)
                    # for point in ordered_corners:
                    #     cv2.circle(debug_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

                    # cv2.imshow("Debug - Detected Board Contour", debug_image)
                    # cv2.waitKey(1)

                    return self.rectify_board(image, ordered_corners)

        # Show debug window even if no contour met the criteria
        # cv2.imshow("Debug - Detected Board Contour", debug_image)
        # cv2.waitKey(1)
        return None
            
    def build_mapping(self, panels):
        mapping = {}

        for i, panel in enumerate(panels):
            # self.get_logger().info(f"--- Processing Panel {i} ---")

            letter = self.recognize_letter(panel)
            arrow = self.recognize_arrow(panel)

            # self.get_logger().info(
            #     f"Panel {i} Result: Letter={letter}, Arrow={arrow}"
            # )

            if letter is not None and arrow is not None:
                mapping[letter] = arrow

            # cv2.waitKey(1)

        # STRICT CHECK: Ensure ALL 6 panel mappings are detected
        if len(mapping) < 6:
            # self.get_logger().warn(
            #     f"Incomplete board detected ({len(mapping)}/6 panels recognized). Rejecting."
            # )
            return {}

        # self.get_logger().info(f"Final Mapping Result: {mapping}")
        return mapping

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # Top-left
        rect[2] = pts[np.argmax(s)]   # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect
            
    def rectify_board(self, image, corners):
        (tl, tr, br, bl) = corners
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    def extract_panels(self, board):
        if board is None:
            return []
        height, width = board.shape[:2]
        num_panels = 6
        panel_width = width // num_panels
        panels = []
        for i in range(num_panels):
            x_start = i * panel_width
            if i == num_panels - 1:
                x_end = width
            else:
                x_end = (i + 1) * panel_width
            panel = board[:, x_start:x_end]
            panels.append(panel)
        return panels
    
    def crop_to_symbol(self, binary_img):
        contours, _ = cv2.findContours(
            binary_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return binary_img
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = binary_img[y:y+h, x:x+w]
        return cropped
    
    def match_template(self, roi, template_type):
        if roi is None or roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return None, -1.0
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_gray = cv2.threshold(
            roi_gray,
            150,
            255,
            cv2.THRESH_BINARY
        )

        roi_gray = self.crop_to_symbol(roi_gray)
        roi_gray = cv2.resize(roi_gray, (64, 64))

        templates = self.letter_templates if template_type == "letters" else self.arrow_templates

        best_match = None
        best_score = float("-inf")

        for name, template in templates.items():
            result = cv2.matchTemplate(
                roi_gray,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            _, score, _, _ = cv2.minMaxLoc(result)

            if score > best_score:
                best_score = score
                best_match = name

        return best_match, best_score

    def recognize_letter(self, panel):
        letter_roi = panel[:panel.shape[0]//2]
        letter, score = self.match_template(letter_roi, "letters")
        if score < 0.4:
            return None
        return letter

    def recognize_arrow(self, panel):
        arrow_roi = panel[panel.shape[0]//2:]
        arrow, score = self.match_template(arrow_roi, "arrows")
        if score < 0.4:
            return None
        return arrow.upper()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()