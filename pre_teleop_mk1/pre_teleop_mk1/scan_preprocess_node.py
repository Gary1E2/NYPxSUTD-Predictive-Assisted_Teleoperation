#!/usr/bin/env python3

"""
scan_preprocess_node.py used to clamp and downsample raw LiDAR scans for downstream calculations.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# ==============================
# QUALITY OF SERVICE POLICIES
# ============================== 
# using best effort QOS with a small history to tolerate sensor dropouts.
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# ==============================
# CONSTANTS
# ==============================
R_MIN = 0.05
R_MAX = 8.0
STRIDE = 2

# ==============================
# NODE CLASS
# ==============================
class ScanPreprocessNode(Node):
    def __init__(self):
        super().__init__('scan_preprocess_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
       
        self.scan_pub = self.create_publisher(
            LaserScan,
            '/scan_filtered',
            qos_profile
        )

    def scan_callback(self, msg) -> None:
        """
        Publishes clamped and downsampled LiDAR scans as LaserScan message.
        
        Inputs:
            msg (LaserScan):
                Raw LiDAR scans containing values and intensities (optional) in sensor frame.
        """
        # Clamp extremely large and small LiDAR values
        processed_ranges = []
        for r in msg.ranges:
            if not math.isfinite(r):
                processed_ranges.append(R_MAX)
            else:
                processed_ranges.append(min(max(r, R_MIN), R_MAX))

        # Downsample LiDAR values used for downstream
        downsampled_ranges = processed_ranges[::STRIDE]
        output_count = len(downsampled_ranges)

        # Update angle size between LiDAR rays
        angle_increment_out = msg.angle_increment * STRIDE
        angle_max_out = msg.angle_min + (output_count - 1) * angle_increment_out if output_count > 0 else msg.angle_min

        # Downsample LiDAR intensities to match values
        if len(msg.intensities) == len(msg.ranges):
            downsampled_intensities = list(msg.intensities)[::STRIDE]
        else:
            downsampled_intensities = []

        self._publish_scan(
            msg.header, msg.angle_min, angle_max_out, angle_increment_out,
            msg.time_increment * STRIDE, msg.scan_time, downsampled_ranges, downsampled_intensities
        )
    
    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_scan(self, header,
        angle_min: float,
        angle_max: float,
        angle_increment: float,
        time_increment: float,
        scan_time: float,
        ranges: list,
        intensities: list,
    ) -> None:
        """Creates and publishes a valid, processed LaserScan message."""

        msg = LaserScan()
        msg.header = header
        msg.angle_min = angle_min
        msg.angle_max = angle_max
        msg.angle_increment = angle_increment
        msg.time_increment = time_increment
        msg.scan_time = scan_time
        msg.range_min = R_MIN
        msg.range_max = R_MAX
        msg.ranges = ranges
        msg.intensities = intensities

        self.scan_pub.publish(msg)


def main():
    rclpy.init()
    node = ScanPreprocessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
