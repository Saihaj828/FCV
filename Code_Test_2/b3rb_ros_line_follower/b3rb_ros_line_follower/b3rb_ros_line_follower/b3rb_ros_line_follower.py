# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication
from nav_msgs.msg import Odometry

import json

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.1
SPEED_MAX = 3.0
TURN_MIN = -1.0
TURN_MAX = 1.0

TARGET_STRAIGHT_SPEED = 3.0
TARGET_TURN_SPEED = 0.6
ZONE_SPEED = 1.0
PARK_SPEED = 0.4
PARK_TURN = 1.0
JUNCTION_TURN_ERROR = -0.6
TURN_ERROR = 0.47
OBSTACLE_SPEED = 1.0
ZONE_BUILDING_THRESHOLD = 0.8
PARK_HERE_THRESHOLD = 4.2
PARK_HERE_THRESHOLD_2 = 4.2
PARK_THERE_THRESHOLD = 2.0

MIN_CLUSTER_PARKING = 2

STOP_CENTER_ANGLE = 90

TURN_COUNTER_THRESHOLD = 60

UT_DISTANCE_THRESHOLD = 30.0

GO_STRAIGHT = "GO_STRAIGHT"
GO_LEFT = "GO_LEFT"
GO_RIGHT = "GO_RIGHT"

STRAIGHT = "STRAIGHT"
RIGHT = "RIGHT"
LEFT = "LEFT"

SIGN_RESET_TIMEOUT = 9.5

QR_TO_KEY = {
    "PATIENT_1": "A",
    "PATIENT_2": "B",
    "PATIENT_3": "C",
    "HOSPITAL_1": "X",
    "HOSPITAL_2": "Y",
    "HOSPITAL_3": "Z",
}

ENABLE_UTURN = False

# CONFIGURATION:
# The buggy is driven in manual mode by publishing standard controller Joy messages to /cerebri/in/joy.
# The layout is: msg.axes = [0.0, speed, 0.0, turn]
# - speed: positive for forward, negative for reverse. Range: [-1.0, 1.0]
# - turn: positive for left steer, negative for right steer. Range: [-1.0, 1.0]
# msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] (Keep buttons set to this pattern for manual override mode)

SERVER_WAIT_TIMEOUT_SEC = 5.0

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    By default, it publishes a safe drive-straight command on a timer loop.
    Implement logic inside the callbacks to steer, dodge obstacles, detect destinations,
    communicate with the server, and park.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        
        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # 6. Position
        self.subscription_odom = self.create_subscription(
            Odometry,
            "/cerebri/out/odometry",
            self.odom_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        
        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------
        
        # Default controls: drive straight slowly
        self.target_speed = 0.0
        self.target_turn = 0.0
        self.no_lane_counter = 0

        # State variables (You can add your own state flags / state machines here)
        self.obstacle_in_front = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = "A"
        self.mission_completed = False
        self.motion_state = GO_STRAIGHT
        self.sign_board_data = None
        self.next_turn = STRAIGHT
        self.sign_reset_timer = None
        self.waiting_for_server = False
        self.entering_zone = False
        self.parking = False
        self.side = None
        self.scan = None
        self.comm_timer = None

        self.parking_speed = False

        self.uid_curr = 0

        self.turn_counter = 0

        self.patients_completed = 0

        self.initial_x = 0.0
        self.initial_y = 0.0

        self.buggy_x = 0.0
        self.buggy_y = 0.0
        self.buggy_yaw = 0.0

        self.uturn_active = False
        self.uturn_state = 0

        self.uturn_start_yaw = 0.0
        self.uturn_start_x = 0.0
        self.uturn_start_y = 0.0

        self.prev_destination = None

        self.total_distance = 0.0
        self.prev_odom_x = None
        self.prev_odom_y = None

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.logger = self.get_logger()

        self.get_logger().info("Line Follower controller initialized. Safe Drive-Straight Mode active.")


    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        
        self.target_turn = max(min(self.target_turn, TURN_MAX), TURN_MIN)  # Clamp to [-1, 1]
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ Callback Implementations ------------------
        
    def edge_vectors_callback(self, message):
        """
        Receives lane boundaries from the camera vector extractor.
        
        GUIDELINE (Lane Following):
        - `message.vector_count` contains the number of active bounds seen (0, 1, or 2).
        - `message.vector_1` and `message.vector_2` contain the points defining the bounds.
        - You need to write logic to compute the centerline deviation and adjust `self.target_turn`.
        - E.g., if only one line is seen, steer away from it to keep distance; if two lines are seen,
          calculate the midpoint relative to the image width and steer to center the buggy.
        """
        # HINTS:
        # width = message.image_width
        # half_width = width / 2.0
        # For now, we do not modify self.target_turn so the buggy continues straight.
        # pass

        if self.obstacle_in_front or self.parking:
            return

        if self.uturn_active:
            self.u_turn()
            if self.uturn_state != 1:
                return
        
        if self.waiting_for_server :
            self.target_speed = 0.0
            self.target_turn = 0.0
            return

        if self.entering_zone:

            left_found, left_dist, _ = self.find_wall(
                center_angle_deg=STOP_CENTER_ANGLE
            )

            right_found, right_dist, _ = self.find_wall(
                center_angle_deg=-STOP_CENTER_ANGLE
            )

            if right_found:
                self.side = "right"
                self.reached_zone("right")
                return
            
            elif left_found:
                self.side = "left"
                self.reached_zone("left")
                return

        self.edge_vectors_curr = message


        vector_count = message.vector_count
        
        if vector_count == 2:                

            self.no_lane_counter = 0
            left_vector = message.vector_1
            right_vector = message.vector_2

            # Bottom points
            left_bottom = left_vector[0] if left_vector[0].y > left_vector[1].y else left_vector[1]
            right_bottom = right_vector[0] if right_vector[0].y > right_vector[1].y else right_vector[1]

            # Difference in bottom y coordinates
            # +ve -> right boundary lower than left -> buggy turned right
            # -ve -> left boundary lower than right -> buggy turned left
            error = right_bottom.y - left_bottom.y

            # Normalize
            error /= message.image_height

            if (self.entering_zone and self.patients_completed == 2 and (self.current_destination in ("X","Y","Z"))) or self.parking_speed:
                self.target_speed = ZONE_SPEED
            else:
                if self.turn_counter == 0:
                        self.target_speed = TARGET_STRAIGHT_SPEED
                else:
                    self.turn_counter -= 1
                    self.target_speed = TARGET_TURN_SPEED

                # speed = TARGET_STRAIGHT_SPEED - abs(error)*5
                # self.target_speed = max(min(speed, 2.0), 0.8)  # Clamp to [0.8, 2]

            if abs(error) < 0.05:
                self.motion_state = GO_STRAIGHT
            elif error > 0:
                self.motion_state = GO_RIGHT
            else:
                self.motion_state = GO_LEFT

            self.target_turn = error

            # self.get_logger().info(
            #     f"Error={error:.2f}"
            # )

        elif vector_count == 1:

            self.turn_counter = TURN_COUNTER_THRESHOLD

            self.no_lane_counter = 0
            detected_vector = message.vector_1
            # finding bottom point of the vector
            if detected_vector[0].y > detected_vector[1].y:
                bottom_point = detected_vector[0]
            else:
                bottom_point = detected_vector[1]
            image_center = message.image_width / 2.0
            error = (bottom_point.x - image_center)
            error /= (message.image_width / 2.0)

            if error > 0:
                self.target_turn = TURN_ERROR
            else :
                self.target_turn = -TURN_ERROR


            self.target_speed = TARGET_TURN_SPEED
            # turn = 0.6 * error
            # turn = max(min(turn, 1.0), -1.0)

            if bottom_point.x < image_center:
                self.motion_state = GO_RIGHT
            else:
                self.motion_state = GO_LEFT
            # self.get_logger().info(
            #     f"Single Lane Detected. Bottom={bottom_point.x:.1f} Error={error:.2f} State={self.motion_state}"
            # )

        else:

            # self.logger.info("Junction Detected")
            self.total_distance = 0.0

            self.target_speed = TARGET_TURN_SPEED

            if self.next_turn == STRAIGHT:

                self.target_turn = 0.0
                self.logger.info("Going Straight")

            elif self.next_turn == RIGHT:

                self.target_turn = JUNCTION_TURN_ERROR
                self.logger.info("Taking Right")

            else:

                self.target_turn = -JUNCTION_TURN_ERROR
                self.logger.info("Taking Left")
                

    def reached_zone(self, side):

        self.waiting_for_server = True
        self.entering_zone = False

        self.logger.info(f"Reached Zone. Wall on {side}")

        self.send_server_update(self.curr_location)
        
        if self.curr_location in ("X","Y","Z"):

            self.patients_completed += 1

    def lidar_callback(self, message):
        """
        Receives LIDAR range measurements.
        
        GUIDELINE (Obstacle Avoidance & Building Range):
        - `message.ranges` is an array of distances in meters around the buggy.
        - The laser scans cover 360 degrees. Find which indices correspond to the front of the buggy.
        - If a range value in the front sector is below a threshold (e.g. 0.8m), flag an obstacle.
        - Write obstacle avoidance maneuvers (e.g. stop, steer left/right around the block, and merge back).
        - Use LIDAR side-ranges to verify distance to building/QR signs before patient pickup/hospital drop actions.
        """
        # HINTS:
        # num_readings = len(message.ranges)
        # front_sector = message.ranges[int(num_readings * 7/18): int(num_readings * 11/18)]
        # min_front_dist = min(front_sector)

        self.scan = message

        if self.parking or self.parking_speed or self.uturn_active:
            return

        ranges = message.ranges
        num_readings = len(ranges)
        if num_readings == 0:
            self.get_logger().debug("[LIDAR] Received empty LaserScan reading.")
            return

        MAX_TURN_ERROR = 1.0

        normalized = self.front_obstacle_direction(message)

        if normalized:

            if(self.is_obstacle_in_front(message)):

                self.obstacle_in_front = True

                self.target_speed = OBSTACLE_SPEED
                self.turn_counter =  TURN_COUNTER_THRESHOLD

                if normalized < 0:
                    self.target_turn = MAX_TURN_ERROR
                else:
                    self.target_turn = -MAX_TURN_ERROR

                # self.logger.info(
                #     f"offset={normalized:.2f} | "
                #     f"target_turn={self.target_turn}"
                # )

            else:
                self.obstacle_in_front = False
        else:
            self.obstacle_in_front = False

    def is_obstacle_in_front(self, message):

        ranges = list(message.ranges)
        n = len(ranges)

        # Parameters (tune these)
        MAX_DETECT_DIST = 1.15      # Ignore anything farther away
        FRONT_HALF_ANGLE = 15      # Look ±45° ahead

        # Assuming front of buggy is at 180°
        front_idx = n // 2
        angle_per_index = 360.0 / n
        half_width = int(FRONT_HALF_ANGLE / angle_per_index)

        start = front_idx - half_width
        end = front_idx + half_width

        indices = []

        for i in range(start, end):

            d = ranges[i]

            if math.isnan(d) or math.isinf(d):
                continue

            if d < MAX_DETECT_DIST:
                indices.append(i)

        return len(indices) != 0

    def front_obstacle_direction(self, message):

        ranges = list(message.ranges)
        n = len(ranges)

        # Parameters (tune these)
        MAX_DETECT_DIST = 1.15      # Ignore anything farther away
        FRONT_HALF_ANGLE = 45      # Look ±45° ahead

        # Assuming front of buggy is at 180°
        front_idx = n // 2
        angle_per_index = 360.0 / n
        half_width = int(FRONT_HALF_ANGLE / angle_per_index)

        start = front_idx - half_width
        end = front_idx + half_width

        indices = []

        for i in range(start, end):

            d = ranges[i]

            if math.isnan(d) or math.isinf(d):
                continue

            if d < MAX_DETECT_DIST:
                indices.append(i)

        if len(indices):

            self.obstacle_in_front = True

            centroid = sum(indices) / len(indices)

            normalized = (centroid - front_idx) / half_width

            return normalized

        else:

            return None

    def server_communication_callback(self, message):
        """
        Receives coordination commands from the server.
        
        GUIDELINE (Server Communication):
        - Check if the message is destined for the Buggy (`message.dest == 1`).
		- Do not forget to check for ACK messages from server
        - The server communicates mission info in the `message.msg` payload string.
        - Parse server instructions (e.g., patient pickup, target hospitals).
        - Call `self.send_server_update` to report your status when you reach a checkpoint.
        """
        if message.dest == 1:
            self.get_logger().info(f"Received Server Message: {message.msg}")
            if message.ack == 1:
                self.get_logger().info(f"Received Ack")
                try:
                    self.comm_timer.cancel()
                except:
                    self.logger.warn("Comm timer error")
            else:
                self.prev_destination = self.current_destination
                self.current_destination = message.msg

                uturn = self.need_uturn()
                if uturn:
                    self.start_uturn()

                self.waiting_for_server = False

                self.uid_curr = message.uid
                self.send_server_update("")

                if message.msg == "OK" and self.patients_completed == 3:
                    
                    if self.side == "right":
                        self.park("left")
                    else:
                        self.park("right")

    def send_server_update(self, text_msg):
        """Sends status messages to the server. (Do not forget to send ACK messages to server)"""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = self.uid_curr     # Replace with a rolling message ID/counter
        self.uid_curr += 1

        if text_msg == "":
            server_msg.ack = 1
        else:
            server_msg.ack = 0


        server_msg.msg = text_msg
        
        self.publisher_server.publish(server_msg)
        if server_msg.ack == 0:
            self.comm_timer = self.create_timer(1, lambda: self.publisher_server.publish(server_msg))
        

        self.logger.info(f"Sent Server Message: {text_msg}")


    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.
        
        GUIDELINE (Patient/Hospital Identification):
        - Parse the decoded string payload in `message.data` (e.g. "PATIENT_A", "HOSPITAL_B").
        - If it matches your target destination, stop the vehicle close to the building (verify range using LIDAR),
          perform the action (pick patient / drop patient), and communicate the arrival to the server.
        """

        qr_data = message.data
        self.get_logger().info(f"Heard QR code: {message.data}")

        curr_location_name = qr_data[6:-1]

        self.curr_location = QR_TO_KEY.get(curr_location_name)

        self.logger.info(f"LOC : {self.curr_location}")

        if self.curr_location == self.current_destination:

            self.logger.info("Entering Zone")

            self.entering_zone = True

    def sign_board_callback(self, message):
        """
        Receives traffic sign boards.
        
        GUIDELINE (Sign Board Routing):
        - Use the detected signs to choose the quickest route at intersections.
        """
        self.get_logger().info(f"Heard Sign Board: {message.data}")

        self.sign_mapping = json.loads(message.data)

        key = self.current_destination

        self.next_turn = self.sign_mapping.get(key, STRAIGHT)

        if self.sign_reset_timer is not None:
            self.sign_reset_timer.cancel()
            self.sign_reset_timer = None

        self.sign_reset_timer = self.create_timer(SIGN_RESET_TIMEOUT, self.reset_sign_turn)

    def reset_sign_turn(self):
        """Resets next_turn back to STRAIGHT after SIGN_RESET_TIMEOUT seconds."""
        self.next_turn = STRAIGHT
        if self.sign_reset_timer is not None:
            self.sign_reset_timer.cancel()
            self.sign_reset_timer = None
        self.get_logger().info("Sign turn reset to STRAIGHT after timeout.")

    def find_wall(self,
                center_angle_deg,
                sector_width_deg=20,
                max_range=3.0,
                min_cluster_size=20,
                gap_thresh=0.20):
        """
        Returns:
            found      : bool
            distance   : average distance of wall
            cluster    : list of indices belonging to wall
        """

        scan = self.scan

        ranges = scan.ranges
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        center = math.radians(center_angle_deg)
        half = math.radians(sector_width_deg / 2)

        start = int((center - half - angle_min) / angle_increment)
        end   = int((center + half - angle_min) / angle_increment)

        start = max(0, start)
        end = min(len(ranges) - 1, end)

        clusters = []
        current = []

        for i in range(start, end + 1):

            d = ranges[i]

            if not math.isfinite(d) or d > max_range:
                if len(current):
                    clusters.append(current)
                    current = []
                continue

            if not current:
                current.append((i, d))
            else:
                if abs(d - current[-1][1]) < gap_thresh:
                    current.append((i, d))
                else:
                    clusters.append(current)
                    current = [(i, d)]

        if current:
            clusters.append(current)

        clusters = [c for c in clusters if len(c) >= min_cluster_size]

        if not clusters:
            return False, float("inf"), []

        wall = max(clusters, key=len)

        avg_dist = sum(d for _, d in wall) / len(wall)

        return True, avg_dist, wall

    def park(self, side):

        self.parking = True
        self.parking_speed = True

        self.logger.info(f"Parking on {side} side")

        is_parking_side = self.detect_parking_side(side)

        self.initial_x = self.buggy_x
        self.initial_y = self.buggy_y

        if(is_parking_side):
            self.logger.info("Parking is Here")
            self.park_timer = self.create_timer(0.1, lambda: self.park_here(side, PARK_HERE_THRESHOLD))
        else:
            self.logger.info("Parking is there")
            self.parking = False
            self.park_timer = self.create_timer(0.1, lambda: self.park_there(side))

    def park_here(self, side, threshold):        

        if math.hypot(self.buggy_x - self.initial_x, self.buggy_y - self.initial_y) >= threshold:

            self.target_speed = 0.0
            self.target_turn = 0.0
            self.send_server_update("PARKED")
            self.park_timer.cancel()
            self.logger.info("PARKED")

        else:

            self.target_speed = PARK_SPEED
            if side == "left":
                self.target_turn = PARK_TURN
            else:
                self.target_turn = -PARK_TURN

    def park_there(self, side):

        if math.hypot(self.buggy_x - self.initial_x, self.buggy_y - self.initial_y) >= PARK_THERE_THRESHOLD:

            self.parking = True
            self.park_timer.cancel()
            self.initial_x = self.buggy_x
            self.initial_y = self.buggy_y
            self.park_timer = self.create_timer(0.1, lambda: self.park_here(side, PARK_HERE_THRESHOLD_2))

    def detect_parking_side(self, side):
            """
            Returns True if the parking slots are detected beside the buggy.
    
            scan : sensor_msgs.msg.LaserScan
            """
    
            scan = self.scan
    
            ranges = scan.ranges
            angle_min = scan.angle_min
            angle_inc = scan.angle_increment
    
            MAX_DIST = 3.0          # Ignore anything farther
            CLUSTER_GAP = 0.20      # New cluster if distance jumps by >20 cm
            MIN_POINTS = 1          # Ignore tiny noise
            MAX_POINTS = 5         # Wall clusters are much larger
    
            # --------- Sector to inspect ----------
            if side == "right":
                # Example: right side = -90° to -20° 
                START = math.radians(-90)
                END   = math.radians(-45)
            else:
                START = math.radians(45)
                END   = math.radians(90)
    
            start_idx = max(0, int((START - angle_min) / angle_inc))
            end_idx   = min(len(ranges)-1, int((END - angle_min) / angle_inc))
    
            clusters = []
    
            current = []
    
            prev_range = None
    
            for i in range(start_idx, end_idx + 1):
    
                r = ranges[i]
    
                if math.isnan(r) or math.isinf(r):
                    if current:
                        clusters.append(current)
                        current = []
                    prev_range = None
                    continue
    
                if r > MAX_DIST:
                    if current:
                        clusters.append(current)
                        current = []
                    prev_range = None
                    continue
    
                if prev_range is None:
                    current.append(r)
    
                elif abs(r - prev_range) < CLUSTER_GAP:
                    current.append(r)
    
                else:
                    clusters.append(current)
                    current = [r]
    
                prev_range = r
    
            if current:
                clusters.append(current)
    
            # Count only small clusters (cones)
            cone_clusters = 0
    
            for c in clusters:
                if MIN_POINTS <= len(c) <= MAX_POINTS:
                    cone_clusters += 1
    
            print("Total clusters :", len(clusters))
            print("Cone clusters  :", cone_clusters)
    
            # Tune this threshold in simulation
            return cone_clusters >= MIN_CLUSTER_PARKING

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Accumulate actual distance travelled
        if self.prev_odom_x is not None:
            dx = x - self.prev_odom_x
            dy = y - self.prev_odom_y

            self.total_distance += (dx**2 + dy**2) ** 0.5

        self.prev_odom_x = x
        self.prev_odom_y = y

        # Current pose
        self.buggy_x = x
        self.buggy_y = y
        self.buggy_yaw = self.yaw_from_quaternion(
            msg.pose.pose.orientation
        )

    def start_uturn(self):
        self.uturn_active = True
        self.uturn_state = 1

        self.uturn_start_yaw = self.buggy_yaw
        self.uturn_start_x = self.buggy_x
        self.uturn_start_y = self.buggy_y

    def u_turn(self):

        if not self.uturn_active:
            return

        UT_FOW_ANGLE = 27
        UT_INITIAL_DIST = 1.8
        UT_SPEED = 0.5
        UT_REV_ANGLE = 80

        current_yaw = self.buggy_yaw

        # --------------------------------------------------
        # STATE 1: Move straight forward for 1m
        # --------------------------------------------------
        if self.uturn_state == 1:

            dx = self.buggy_x - self.uturn_start_x
            dy = self.buggy_y - self.uturn_start_y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > UT_INITIAL_DIST:
                # Reset starting position for the next state
                self.uturn_start_x = self.buggy_x
                self.uturn_start_y = self.buggy_y
                self.uturn_state = 2
            else:
                self.logger.info(f"Initial Dist : {distance}")

        # --------------------------------------------------
        # STATE 2: Move forward while turning left
        # --------------------------------------------------
        elif self.uturn_state == 2:

            self.target_speed = UT_SPEED
            self.target_turn = 1.0       # left steering

            # dx = self.buggy_x - self.uturn_start_x
            # dy = self.buggy_y - self.uturn_start_y
            # distance = math.sqrt(dx*dx + dy*dy)

            # if distance > UT_FOW_DIST:
            #     self.uturn_state = 3
            # else:
            #     self.logger.info(f"Dist : {distance}")

            yaw_change = abs(
                self.angle_diff(current_yaw, self.uturn_start_yaw)
            )

            if yaw_change > math.radians(UT_FOW_ANGLE):
                self.uturn_state = 3
            

        # --------------------------------------------------
        # STATE 3: Reverse while turning right
        # --------------------------------------------------
        elif self.uturn_state == 3:

            self.target_speed = -UT_SPEED
            self.target_turn = -1.0      # opposite steering

            yaw_change = abs(
                self.angle_diff(current_yaw, self.uturn_start_yaw)
            )

            if yaw_change > math.radians(UT_REV_ANGLE):
                self.uturn_state = 4

        # --------------------------------------------------
        # STATE 4: Move forward and finish the U-turn
        # --------------------------------------------------
        elif self.uturn_state == 4:

            self.target_speed = UT_SPEED
            self.target_turn = 1.0

            yaw_change = abs(
                self.angle_diff(current_yaw, self.uturn_start_yaw)
            )

            # 180 degrees reached
            if yaw_change > math.radians(160):

                self.target_speed = 0.0
                self.target_turn = 0.0

                self.uturn_active = False
                self.uturn_state = 0

    def need_uturn(self):

        if not ENABLE_UTURN:
            return False

        if self.total_distance > UT_DISTANCE_THRESHOLD:
            self.logger.warn(f"Travelled too much : {self.total_distance}")
            return False

        prev_turn = self.next_turn
        curr_turn = self.sign_mapping.get(self.current_destination)

        return prev_turn != curr_turn

    def yaw_from_quaternion(self, q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def angle_diff(self, a, b):
        """Return a-b normalized to [-pi, pi]."""
        return (a - b + math.pi) % (2 * math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
