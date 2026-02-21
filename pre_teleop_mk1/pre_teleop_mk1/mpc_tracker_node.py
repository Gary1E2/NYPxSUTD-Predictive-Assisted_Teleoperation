#!/usr/bin/env python3

"""
mpc_tracker_node.py used to track upstream drive commands.

This is a *tracking MPC* only (no obstacle logic).
"""

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

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
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ==============================
# NODE CLASS
# ==============================
class TrackingMpcNode(Node):
    def __init__(self) -> None:
        super().__init__("mpc_tracker_node")

        # ========== VARIABLES ==========

        # Timings
        self.dt = 0.05  # must match your stack timing
        self.N = 12     # MPC horizon steps (0.6s at 20Hz)

        # Actuator/state limits (Tunable)
        self.delta_max = 0.418
        self.delta_dot_max = 3.0     # rad/s (steering rate limit)
        self.a_max = 6.0             # m/s^2 (accel magnitude limit)
        self.v_min = 0.0
        self.v_max = 8.0

        # ========== MPC weights ==========
        # (track MPPI's immediate command)

        # State error weights (v, delta)
        self.Q = np.diag([8.0, 7.0])

        # Input weights (a, delta_dot)
        self.R = np.diag([0.8, 0.6])

        # Terminal weight
        self.Qf = np.diag([10.0, 12.0])

        # Internal state
        self.v_meas: float = 0.0
        self.have_odom = False

        # NOTE: Not common to measure steering angle from Odometry in F1TENTH.
        # Bounded estimate from own commands
        self.delta_est: float = 0.0

        # Reference from /drive_mux
        self.delta_ref: float = 0.0
        self.v_ref: float = 0.0
        self.have_ref = False

        # Hold last output for a short time if no reference comes
        self.last_cmd: Tuple[float, float] = (0.0, 0.0)

        # ========== SUBS & PUBS ==========
        self.ref_sub = self.create_subscription(
            AckermannDriveStamped, 
            "/drive_mux", 
            self.ref_callback, 
            qos_profile_sub
        )
        self.odom_sub = self.create_subscription(
            Odometry, 
            "/vehicle_state", 
            self.odom_callback, 
            qos_profile_sub
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped, 
            "/drive_mpc", 
            qos_profile_pub
        )

        self.create_timer(self.dt, self._timer_cb)

        # Status log
        self.get_logger().info("mpc_tracker_node started: tracking /drive_mux")

    # ==============================
    # MPC TRACKER
    # ==============================
    def _timer_cb(self) -> None:
        """
        MPC tracks reference command from upstream.

        Applies optional steering deadband and actuator lag modeling.
        Publishes the final steering and speed commands.
        """
        if not self.have_ref:
            return

        # Reference command from mux (already clamped + rate-limited)
        delta_ref = self.delta_ref
        v_ref = self.v_ref

        # Optional: small deadband
        if abs(delta_ref) < 0.01:
            delta_ref = 0.0

        # Optional: emulate steering actuator lag (1st-order), NOT shaping
        # tau: smaller = follows faster (less smoothing). start 0.05–0.10
        tau = 0.07
        alpha = self.dt / max(self.dt + tau, 1e-6)

        # delta_est acts as "actuator state"
        self.delta_est = (1.0 - alpha) * self.delta_est + alpha * delta_ref
        self.delta_est = self._clamp(self.delta_est, -self.delta_max, self.delta_max)

        # Speed: pass through reference (mux already rate-limited)
        v_cmd = self._clamp(v_ref, self.v_min, self.v_max)

        self._publish_mpc(self.delta_est, v_cmd)

    # ==============================
    # UTITLIES & CALLBACKS
    # ==============================
    def _clamp(self, x: float, lo: float, hi: float) -> float:
        """Value clamping"""
        return max(lo, min(hi, x))

    def ref_callback(self, msg: AckermannDriveStamped) -> None:
        """Receives and stores reference steering and speed commands."""
        self.delta_ref = float(msg.drive.steering_angle)
        self.v_ref = float(msg.drive.speed)
        self.have_ref = True

    def odom_callback(self, msg: Odometry) -> None:
        """Receives vehicle odometry and updates measured speed and odometry availability status."""
        self.v_meas = float(msg.twist.twist.linear.x)
        self.have_odom = True

    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_mpc(self, delta: float, v: float) -> None:
        """Publish the mpc tracker driving command"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ego_racecar/base_link"
        msg.drive.steering_angle = float(delta)
        msg.drive.speed = float(v)
        self.cmd_pub.publish(msg)
        self.last_cmd = (delta, v)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackingMpcNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
