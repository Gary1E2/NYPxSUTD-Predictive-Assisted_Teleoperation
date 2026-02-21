#!/usr/bin/env python3

"""
aeb_node.py used as a fall back reactive safety measure.

Automatically brakes using immediate sensor readings to avoid collision/crashes.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

# ==============================
# QUALITY OF SERVICE POLICIES
# ==============================
qos_profile_sub = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

qos_profile_pub = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ==============================
# CONSTANTS
# ==============================
R_MIN = 0.05
R_MAX = 10.0
FRONT_ANGLE = 0.0
LEFT_ANGLE = 0.261799  # +15 deg
RIGHT_ANGLE = -0.261799  # -15 deg
TTC_THRESH_FRONT = 1
TTC_THRESH_SIDE = 0.8

# ==============================
# NODE CLASS
# ==============================
class AebNode(Node):
    def __init__(self) -> None:
        super().__init__("aeb_node")
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None

        # ========== SUBS & PUBS ==========
        self.scan_sub = self.create_subscription(
            LaserScan, 
            "/scan_filtered", 
            self.scan_callback, 
            qos_profile_sub
        )

        self.odom_sub = self.create_subscription(
            Odometry, 
            "/vehicle_state", 
            self.odom_callback, 
            qos_profile_sub
        )

        self.plan_sub = self.create_subscription(
            AckermannDriveStamped, 
            "/drive_mux", 
            self.plan_callback, 
            qos_profile_sub
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, 
            "/drive_safe", 
            qos_profile_pub
        )


    def scan_callback(self, msg: LaserScan) -> None:
        """Receives and stores LiDAR scan values."""
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry) -> None:
        """Receives and stores vehicle odometry."""
        self.latest_odom = msg

    def _beam_index(self, scan: LaserScan, angle: float) -> Optional[int]:
        """Computes LiDAR scan index correspondng to a given angle."""
        idx = round((angle - scan.angle_min) / scan.angle_increment)
        if 0 <= idx < len(scan.ranges):
            return idx
        return None

    def _beam_ttc(self, scan: LaserScan, angle: float, v: float) -> Optional[float]:
        """
        Estimates time to collision along LiDAR scan ray.

        Inputs:
            scan:
                Incoming LiDAR scans
            angle:
                LiDAR scan angle from vehicle heading (rads)
            v:
                Current car speed
        Outputs:
            (float value):
                Estimated TTC in seconds
            (None value):
                TTC not computed
        """
        idx = self._beam_index(scan, angle)

        if idx is None:
            return None
        
        r = scan.ranges[idx]

        if r <= R_MIN:
            return 0.0
        
        if r >= R_MAX or math.isinf(r) or math.isnan(r):
            return None
        
        v_eff = max(v, 0.0)
        v_closing = max(v_eff * math.cos(angle), 0.1)

        return r / v_closing

    def plan_callback(self, msg: AckermannDriveStamped) -> None:
        """
        Autonomous emergency braking for unsafe motion along 3 LiDAR scan paths.

        Inputs:
            msg:
                Driving command from upstream controller.
        """
        # If lack data, pass through unchanged
        if self.latest_scan is None or self.latest_odom is None:
            self._publish_drive_safe(msg.header, msg.drive, msg.drive.speed)
            return

        scan = self.latest_scan
        v = max(self.latest_odom.twist.twist.linear.x, 0.0)

        ttc_front = self._beam_ttc(scan, FRONT_ANGLE, v)
        ttc_left = self._beam_ttc(scan, LEFT_ANGLE, v)
        ttc_right = self._beam_ttc(scan, RIGHT_ANGLE, v)

        unsafe = (
            (ttc_front is not None and ttc_front < TTC_THRESH_FRONT)
            or (ttc_left is not None and ttc_left < TTC_THRESH_SIDE)
            or (ttc_right is not None and ttc_right < TTC_THRESH_SIDE)
        )

        if not unsafe:
            self._publish_drive_safe(msg.header, msg.drive, msg.drive.speed)
            return

        # Emergency brake
        self._publish_drive_safe(msg.header, msg.drive, 0.0)

    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_drive_safe(self, header,
        drive_in: AckermannDriveStamped,
        speed: float
    ) -> None:
        """Publishes a safety-checked driving command."""

        out_msg = AckermannDriveStamped()
        out_msg.header = header
        out_msg.drive.steering_angle = drive_in.steering_angle
        out_msg.drive.steering_angle_velocity = drive_in.steering_angle_velocity
        out_msg.drive.speed = speed
        out_msg.drive.acceleration = drive_in.acceleration
        out_msg.drive.jerk = drive_in.jerk

        self.drive_pub.publish(out_msg)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = AebNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
