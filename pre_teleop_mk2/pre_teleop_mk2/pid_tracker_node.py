#!/usr/bin/env python3

"""
pid_tracker_node.py used to track upstream drive commands.

Tracks /drive_clamp by smoothly updating an internal output state and publishing to /drive.

Important: PID output is treated as a *correction* applied to the current output,
not as the final command itself. This prevents "inventing" speeds (e.g., saturating to v_max).

UNUSED: No longer used but keep for future reference.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
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
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ==============================
# NODE CLASS
# ==============================
class PidTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("pid_tracker_node")

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        # PID gains (tunable)
        self.Kp_steer = 16
        self.Ki_steer = 1
        self.Kd_steer = 0.12

        self.Kp_speed = 1.5
        self.Ki_speed = 0.0
        self.Kd_speed = 0.05

        # control limits
        self.max_steer = 0.418
        self.v_min = 0.0
        self.v_max = 8.0

        # optional rate limits for smoothing
        self.max_steer_rate = 1.0
        self.max_accel = 6.0

        # integrator state + limits
        self.int_steer = 0.0
        self.int_speed = 0.0
        self.int_limit_steer = 1.4
        self.int_limit_speed = 1.5

        # previous errors
        self.prev_err_steer = 0.0
        self.prev_err_speed = 0.0

        # last desired commands (for integrator reset)
        self.last_desired_steer = 0.0
        self.last_desired_speed = 0.0

        # current output state (what we command)
        self.curr_steer_cmd = 0.0
        self.curr_speed_cmd = 0.0

        # latest incoming command
        self.latest_cmd: Optional[AckermannDriveStamped] = None

        self.dt = 0.05  # control timestep (s)


        # ========== SUBS & PUBS ==========
        self.cmd_sub = self.create_subscription(
            AckermannDriveStamped,
            "/drive_clamp",
            self.cmd_callback,
            qos_profile_sub,
        )
        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            qos_profile_pub,
        )

        self.create_timer(self.dt, self.timer_callback)
        
    # ==============================
    # PID TRACKER
    # ==============================
    def timer_callback(self) -> None:
        """
        Periodic control loop that tracks latest desired drive command.

        Behaviour:
        - Reads most recent desired steering and speed commands
        - Applies PID corrections relative to current output state
        - Enforces rate limits and saturation constraints
        - Publishes a smooth AckermannDriveStamped command
        """
        # if no command yet, hold zero.
        if self.latest_cmd is None:
            desired_steer = 0.0
            desired_speed = 0.0
        else:
            desired_steer = float(self.latest_cmd.drive.steering_angle)
            desired_speed = float(self.latest_cmd.drive.speed)

        self._reset_integrators_if_needed(desired_steer, desired_speed)


        # ========== STEERING PID ==========
        err_steer = desired_steer - self.curr_steer_cmd
        if abs(err_steer) < 0.01:
            err_steer = 0.0

        d_err_steer = (err_steer - self.prev_err_steer) / self.dt

        # propose integrator update
        new_int_steer = self.int_steer + err_steer * self.dt
        new_int_steer = self._clamp(new_int_steer, -self.int_limit_steer, self.int_limit_steer)

        u_steer = (
            self.Kp_steer * err_steer
            + self.Ki_steer * new_int_steer
            + self.Kd_steer * d_err_steer
        )

        # apply correction to output state (rate-limited)
        steer_delta = self._clamp(u_steer, -self.max_steer_rate, self.max_steer_rate) * self.dt
        new_steer_cmd = self.curr_steer_cmd + steer_delta
        new_steer_cmd = self._clamp(new_steer_cmd, -self.max_steer, self.max_steer)

        # anti-windup: only accept integrator update if not saturating "into" clamp
        if not self._is_saturating_into(new_steer_cmd, self.curr_steer_cmd, -self.max_steer, self.max_steer):
            self.int_steer = new_int_steer


        # ========== SPEED PID ==========
        err_speed = desired_speed - self.curr_speed_cmd
        d_err_speed = (err_speed - self.prev_err_speed) / self.dt

        new_int_speed = self.int_speed + err_speed * self.dt
        new_int_speed = self._clamp(new_int_speed, -self.int_limit_speed, self.int_limit_speed)

        u_speed = (
            self.Kp_speed * err_speed
            + self.Ki_speed * new_int_speed
            + self.Kd_speed * d_err_speed
        )

        # apply correction to output state (accel-limited)
        accel = self._clamp(u_speed, -self.max_accel, self.max_accel)
        new_speed_cmd = self.curr_speed_cmd + accel * self.dt
        new_speed_cmd = self._clamp(new_speed_cmd, self.v_min, self.v_max)

        # anti-windup: only accept integrator update if not saturating "into" clamp
        if not self._is_saturating_into(new_speed_cmd, self.curr_speed_cmd, self.v_min, self.v_max):
            self.int_speed = new_int_speed

        out = AckermannDriveStamped()
        out.header.frame_id = "ego_racecar/base_link"
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = float(new_steer_cmd)
        out.drive.speed = float(new_speed_cmd)
        self.cmd_pub.publish(out)

        # update state
        self.curr_steer_cmd = new_steer_cmd
        self.curr_speed_cmd = new_speed_cmd
        self.prev_err_steer = err_steer
        self.prev_err_speed = err_speed
        
    # ==============================
    # UTILITIES & CALLBACKS
    # ==============================

    # ========== UTILITIES ==========
    def _reset_integrators_if_needed(self, desired_steer: float, desired_speed: float) -> None:
        """
        resets:
        - steering integrator if command crosses through ~0 or sign flips meaningfully.
        - speed integrator if stopping or reversing intent (mostly stop case for F1TENTH)
        """
        if (abs(desired_steer) < 1e-3) or (desired_steer * self.last_desired_steer < 0.0):
            self.int_steer = 0.0

        if (abs(desired_speed) < 1e-3) or (desired_speed * self.last_desired_speed < 0.0):
            self.int_speed = 0.0

        self.last_desired_steer = desired_steer
        self.last_desired_speed = desired_speed


    def _clamp(self, val: float, v_min: float, v_max: float) -> float:
        """value clamping"""
        return max(v_min, min(v_max, val))


    def _is_saturating_into(self, new_cmd: float, old_cmd: float, v_min: float, v_max: float) -> bool:
        """Saturation consideration: triggered if at limit and still pushing towards it"""
        if new_cmd <= v_min + 1e-9 and new_cmd < old_cmd:
            return True
        if new_cmd >= v_max - 1e-9 and new_cmd > old_cmd:
            return True
        return False

    # ========== CALLBACKS ==========
    def cmd_callback(self, msg: AckermannDriveStamped) -> None:
        """Receives and stores the latest desired drive command."""
        self.latest_cmd = msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PidTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
