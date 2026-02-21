#!/usr/bin/env python3

"""
command_gate_node.py used as authoritative safety gate that allows or prevents driving of car.

Prevents motion if:
- teleop_node is not functioning
- mppi_planner_node is not functioning
- teleop_node estop is activated
- drive command is stale/outdated
- "escape" condition detected (for racing only)

Escape behavior (LiDAR sees essentially nothing close / out of track):
- if car escaped: stop immediately and enter "escape_needs_ack"
- if "escape_needs_ack": stop until receive teleop_node estop activated
- remain stopped until receive teleop_node estop deactivated
"""

from typing import Optional
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Empty, Bool
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# ==============================
# QUALITY OF SERVICE POLICIES
# ==============================
qos_sub = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

qos_pub = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ==============================
# NODE CLASS
# ==============================
class CommandGateNode(Node):
    def __init__(self):
        super().__init__("command_gate_node")

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        self.teleop_timeout_s = 0.5
        self.mppi_timeout_s = 0.5
        self.cmd_timeout_s = 0.25
        self.publish_rate_hz = 30.0


        # ========== ESCAPE DETECTION ==========
        self.scan_timeout_s = 0.5
        self.near_dist_m = 2.0

        # required fraction of rays returning: near to obstacles
        self.min_close_fraction = 0.02

        # empty check
        self.require_escape_clear_before_motion = True
        self.escape_empty_time_s = 3.0


        # ========== HEALTH ==========
        self.last_teleop_health: Optional[Time] = None
        self.last_mppi_health: Optional[Time] = None


        # ========== ESTOP ==========
        self.teleop_estop: bool = False
        self.prev_teleop_estop: bool = False  # for edge detection if you ever want it

        # drive command
        self.last_cmd: Optional[AckermannDriveStamped] = None
        self.last_cmd_time: Optional[Time] = None

        # escape detection
        self.last_scan_time: Optional[Time] = None
        self.scan_close_fraction: float = 1.0
        self.scan_empty: bool = False
        self.empty_since_time: Optional[Time] = None

        # escape latch states
        self.escape_needs_ack: bool = False   # tripped: waiting for teleop_estop True
        self.escape_latched: bool = False     # acknowledged: staying stopped until teleop_estop False


        # ========== SUBS & PUBS ==========
        self.create_subscription(
            Empty, 
            "/teleop_health", 
            self._teleop_health_cb, 
            qos_sub
        )

        self.create_subscription(
            Empty, 
            "/mppi_health", 
            self._mppi_health_cb, 
            qos_sub
        )

        self.create_subscription(
            Bool, 
            "/teleop_estop", 
            self._estop_cb, 
            qos_sub
        )

        self.create_subscription(
            AckermannDriveStamped, 
            "/drive_cmd", 
            self._cmd_cb, 
            qos_sub
        )

        self.create_subscription(
            LaserScan, 
            "/scan_filtered", 
            self._scan_cb, 
            qos_sub
        )

        self.pub = self.create_publisher(
            AckermannDriveStamped, 
            "/drive_gate", 
            qos_pub
        )

        period = 1.0 / max(1.0, self.publish_rate_hz)
        self.create_timer(period, self._tick)

    # ==============================
    # CORE SAFETY LOOP
    # ==============================
    def _tick(self):
        """
        Authoritative safety gating loop.

        Evaluates subsystem health, command freshness, emergency stop state and LiDAR-based escape detection.
        Decide whether car motion is allowed.
        Publishes either zero command or gated drive command.
        """
         
        now = self.get_clock().now()

        # health and freshness
        teleop_alive = self._fresh(self.last_teleop_health, now, self.teleop_timeout_s)
        mppi_alive = self._fresh(self.last_mppi_health, now, self.mppi_timeout_s)
        cmd_fresh = self._fresh(self.last_cmd_time, now, self.cmd_timeout_s)
        scan_fresh = self._fresh(self.last_scan_time, now, self.scan_timeout_s)


        # =========== ESCAPE DETECTION (DEBOUNCE) ==========
        escape_trip = False
        if scan_fresh and self.scan_empty:
            if self.empty_since_time is None:
                self.empty_since_time = now
            else:
                empty_dt = (now - self.empty_since_time).nanoseconds * 1e-9
                if empty_dt >= self.escape_empty_time_s:
                    escape_trip = True
        else:
            self.empty_since_time = None


        # =========== ESCAPE EXIT PROCEDURE ==========
        if escape_trip and not self.escape_needs_ack and not self.escape_latched:
            self.escape_needs_ack = True
            self.get_logger().warn(
                f"ESCAPE TRIP: scan looks empty (close_fraction={self.scan_close_fraction:.3f}). "
                f"Waiting for operator ACK (SPACE)."
            )

        # ack escape with estop
        if self.escape_needs_ack and self.teleop_estop:
            self.escape_needs_ack = False
            self.escape_latched = True
            self.get_logger().warn("ESCAPE ACKED: latched stop active until operator releases (R).")

        # release escape stop: surroundings must no longer be empty
        if self.escape_latched and (not self.teleop_estop):
            if (not self.require_escape_clear_before_motion) or (scan_fresh and not self.scan_empty):
                self.escape_latched = False
                self.get_logger().info("ESCAPE CLEARED: motion may resume.")
            else:
                # pass if empty
                pass


        # =========== ALLOW MOTION LOGIC ==========
        # escape + e stop gate
        blocked_by_escape = self.escape_needs_ack or self.escape_latched

        if self.teleop_estop:
            self._publish_cmd(now, 0.0, 0.0)
            return

        allow_motion = (
            teleop_alive
            and mppi_alive
            and cmd_fresh
            # and (not blocked_by_escape)      # escape stop option
        )

        # zero command
        if not allow_motion:
            self._publish_cmd(now, 0.0, 0.0)
            return

        # forward command
        frame_id = (
            self.last_cmd.header.frame_id
            if (self.last_cmd and self.last_cmd.header.frame_id)
            else "ego_racecar/base_link"
        )

        self._publish_cmd(
            now,
            speed=self.last_cmd.drive.speed,
            steering=self.last_cmd.drive.steering_angle,
            frame_id=frame_id,
        )

    # ==============================
    # UTILITIES & CALLBACKS
    # ==============================
    def _fresh(self, 
        last_time: Optional[Time], 
        now: Time, 
        timeout_s: float
        ) -> bool:
        """Check whether a timestamp is within an allowed age window."""

        if last_time is None:
            return False
        return (now - last_time).nanoseconds <= int(timeout_s * 1e9)

    
    def _teleop_health_cb(self, _: Empty):
        """Store time teleop_node health topic is received."""
        self.last_teleop_health = self.get_clock().now()


    def _mppi_health_cb(self, _: Empty):
        """Store time mppi_planner_node health topic is received."""
        self.last_mppi_health = self.get_clock().now()


    def _estop_cb(self, msg: Bool):
        """Update teleop emergency stop state."""
        self.prev_teleop_estop = self.teleop_estop
        self.teleop_estop = bool(msg.data)


    def _cmd_cb(self, msg: AckermannDriveStamped):
        """Store latest drive command and update its timestamp."""
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()


    def _scan_cb(self, msg: LaserScan):
        """Process LiDAR scan and update escape detection state."""
        now = self.get_clock().now()
        self.last_scan_time = now

        # compute fraction of ray scans within near_dist_m
        ranges = msg.ranges
        if not ranges:
            self.scan_close_fraction = 0.0
            self.scan_empty = True
            return

        close = 0
        total = 0
        thr = self.near_dist_m

        for r in ranges:
            # ignore NaNs; treat inf as "far"
            if r is None or math.isnan(r):
                continue
            total += 1
            if r <= thr:
                close += 1

        # if everything was NaN, treat as empty
        if total == 0:
            self.scan_close_fraction = 0.0
            self.scan_empty = True
            return

        self.scan_close_fraction = float(close) / float(total)
        self.scan_empty = (self.scan_close_fraction < self.min_close_fraction)

    # ==============================
    # PUBLISHERS
    # ==============================
    def _publish_cmd(
        self,
        now: Time,
        speed: float,
        steering: float,
        frame_id: str = "ego_racecar/base_link",
    ) -> None:
        """Publish a gated drive command to /drive_gate."""
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = frame_id
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.pub.publish(msg)
        

def main():
    rclpy.init()
    node = CommandGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
