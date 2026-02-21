#!/usr/bin/env python3

"""
scan_preprocess_node.py used to clamp and downsample raw LiDAR scans for downstream calculations.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


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

        self.get_logger().info(f"Initialized and running.")


        # ========== OPTIONAL LIDAR VISUALIZATION ==========
        self.declare_parameter("enable_lidar_viz", False)
        self.declare_parameter("lidar_viz_stride", 1)        # extra stride for visualization only
        self.declare_parameter("lidar_viz_max_range", R_MAX)
        self.declare_parameter("lidar_viz_lifetime_s", 0.10)

        self.enable_lidar_viz = bool(self.get_parameter("enable_lidar_viz").value)
        self.lidar_viz_stride = int(self.get_parameter("lidar_viz_stride").value)
        self.lidar_viz_max_range = float(self.get_parameter("lidar_viz_max_range").value)
        self.lidar_viz_lifetime_s = float(self.get_parameter("lidar_viz_lifetime_s").value)


        # ========== SCANS & PUBS ==========
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

        self.lidar_marker_pub = self.create_publisher(
            Marker,
            "/scan_filtered_viz",
            qos_profile
        )

    # ==============================
    # LiDAR SCAN PROCESSING
    # ==============================
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
                # ========== OPTIONAL LIDAR VISUALIZATION ==========
        self._publish_lidar_marker(
            msg.angle_min,
            angle_increment_out,
            downsampled_ranges
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


    # ========== OPTIONAL LIDAR VISUALIZATION ==========
    def _publish_lidar_marker(
        self,
        angle_min: float,
        angle_increment: float,
        ranges: list,
    ) -> None:
        """
        Visualizes LiDAR scan rays as a Marker LINE_LIST.

        Each LiDAR ray is rendered as a short line segment from the origin
        to the measured obstacle point in the vehicle frame.

        Inputs:
            angle_min:
                Starting angle of the scan (rad)
            angle_increment:
                Angular resolution between rays (rad)
            ranges:
                List of LiDAR ranges after preprocessing
        """

        stride = max(1, int(self.lidar_viz_stride))
        max_range = self.lidar_viz_max_range

        marker = Marker()
        marker.header.frame_id = "ego_racecar/base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lidar_scan"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # line width
        marker.scale.x = 0.01

        # color (cyan)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9

        # lifetime
        marker.lifetime.sec = int(self.lidar_viz_lifetime_s)
        marker.lifetime.nanosec = int(
            (self.lidar_viz_lifetime_s - int(self.lidar_viz_lifetime_s)) * 1e9
        )

        for i, r in enumerate(ranges):
            if (i % stride) != 0:
                continue

            r = min(r, max_range)
            theta = angle_min + i * angle_increment

            # ray endpoint
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            # origin to endpoint
            marker.points.append(Point(x=0.0, y=0.0, z=0.0))
            marker.points.append(Point(x=x, y=y, z=0.0))

        self.lidar_marker_pub.publish(marker)


def main():
    rclpy.init()
    node = ScanPreprocessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
