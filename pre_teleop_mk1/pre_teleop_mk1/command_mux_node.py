#!/usr/bin/env python3

"""
command_mux_node.py used to smooth and ensure feasibility of control actions from MPPI planner
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
# VARIABLES
# ==============================
DELTA_MAX = 0.418          # rad
DELTA_DOT_MAX = 3.0        # rad/s
V_MIN = 0.0                # m/s
V_MAX = 8.0                # m/s
V_DOT_MAX = 5.0            # m/s^2
DT_NOMINAL = 0.05          # s
DT_MIN = 0.02              # s
DT_MAX = 0.10              # s

# ==============================
# NODE CLASS
# ==============================
class CommandMuxNode(Node):
    def __init__(self) -> None:
        super().__init__("command_mux_node")

        self.cmd_sub = self.create_subscription(
            AckermannDriveStamped,
            "/drive_plan",
            self.cmd_callback,
            qos_profile_sub
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped,
            "/drive_mux",
            qos_profile_pub
        )

        self.prev_output: AckermannDriveStamped | None = None
    
    # ==============================
    # COMMAND MUX
    # ==============================
    def cmd_callback(self, msg: AckermannDriveStamped) -> None:
        """
        Process and enforces feasibility of incoming planned drive commands.

        Inputs:
            msg:
                Planned Ackermann drive command from upstream planner.
        """
        drive_in = msg.drive

        # Hard clamp incoming values
        delta_clamped = self._clamp(drive_in.steering_angle, -DELTA_MAX, DELTA_MAX)
        v_clamped = self._clamp(drive_in.speed, V_MIN, V_MAX)

        # First message initializes previous output after clamping
        if self.prev_output is None:
            self._publish_cmd(msg.header, delta_clamped, v_clamped, msg.drive)
            return

        prev = self.prev_output
        dt = self._compute_dt(msg.header, prev.header)

        # Rate limit relative to previous OUTPUT
        delta_limit = DELTA_DOT_MAX * dt
        v_limit = V_DOT_MAX * dt

        delta_out = self._clamp(delta_clamped,
                          prev.drive.steering_angle - delta_limit,
                          prev.drive.steering_angle + delta_limit)
        v_out = self._clamp(v_clamped,
                      prev.drive.speed - v_limit,
                      prev.drive.speed + v_limit)

        # Publish result (preserve header)
        self._publish_cmd(msg.header, delta_out, v_out, msg.drive)

    # ==============================
    # UTILITIES
    # ==============================
    def _compute_dt(self, current_header, prev_header) -> float:
        """
        Estimates and clamps dt between consecutive command messages.
        Prevent instability due to sensor jitter or delayed messages.

        Inputs:
            current_header:
                Header of current command message
            prev_header:
                Header of previous command message
        Outputs:
            dt (float):
                Time step in seconds used for command mux.
        """
        try:
            t_curr = Time.from_msg(current_header.stamp)
            t_prev = Time.from_msg(prev_header.stamp)
            dt = (t_curr - t_prev).nanoseconds * 1e-9
            if not math.isfinite(dt) or dt <= 0.0:
                dt = DT_NOMINAL
        except Exception:
            dt = DT_NOMINAL
        return self._clamp(dt, DT_MIN, DT_MAX)

    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_cmd(self, header,
        delta: float,
        speed: float,
        drive_in: AckermannDriveStamped
    ) -> None:
        """Creates and publishes a valid, processed AckermannDriveStamped message."""
        out_msg = AckermannDriveStamped()
        out_msg.header = header
        out_msg.drive.steering_angle = delta
        out_msg.drive.steering_angle_velocity = drive_in.steering_angle_velocity
        out_msg.drive.speed = speed
        out_msg.drive.acceleration = drive_in.acceleration
        out_msg.drive.jerk = drive_in.jerk

        self.cmd_pub.publish(out_msg)

        # Store as previous output
        self.prev_output = out_msg

    def _clamp(self, val: float, low: float, high: float) -> float:
        """Value clamping"""
        return min(max(val, low), high)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandMuxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
