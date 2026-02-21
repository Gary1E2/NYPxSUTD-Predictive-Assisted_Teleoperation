#!/usr/bin/env python3

"""
teleop_node.py used for joystick-based teleoperation using /joy

- publishes /teleop_health for downstream health checks
- publishes /teleop_estop for downstream command gate
- publishes /teleop_mppi_enable for downstream forced fully autonomous mode (DEMONSTRATION PURPOSES ONLY)

Logitech Wireless Gamepad F710 mapping (X mode):
- Left stick vertical  -> speed
- Right stick horizontal -> steering
- B button -> emergency stop
- A button -> emergency stop reset
- RB button -> full autonomous mode
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Joy
from std_msgs.msg import Empty, Bool
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
class TeleopAckermann(Node):
    def __init__(self):
        super().__init__('teleop_node')

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        self.max_speed = 3.0
        self.max_steer = 0.42

        # Logitech Wireless Gamepad F710 bindings (X mode)
        self.axis_speed = 1         # left stick vertical
        self.axis_steer = 3         # right stick horizontal
        self.btn_estop = 1          # B
        self.btn_reset = 0          # A
        self.btn_mppi_enable = 5    # RB (X mode)

        self.last_joy_time = None
        self.joy_timeout_s = 0.5

        self.estop_active = False


        # ========== SUBS & PUBS ==========
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive_teleop',
            10
        )

        self.health_pub = self.create_publisher(
            Empty,
            '/teleop_health',
            10
        )

        self.estop_pub = self.create_publisher(
            Bool,
            '/teleop_estop',
            10
        )

        self.mppi_enable_pub = self.create_publisher(
            Bool,
            "/teleop_mppi_enable",
            10
        )

        self.create_timer(0.1, self._health_tick)

        # NOTE: CONTROL BINDINGS INFO
        self.get_logger().info(
            "Joystick teleop ready:\n"
            " Left stick vertical  -> speed\n"
            " Right stick horizontal -> steering\n"
            " B -> E-STOP\n"
            " A -> E-STOP reset"
        )

    # ==============================
    # JOYSTICK COMMAND
    # ==============================
    def joy_callback(self, msg: Joy):
        """
        Processes incoming joystick (/joy) messages and generates teleoperation commands.

        - Handles emergency stop activation and reset
        - Handles full autonomous mode activation
        - Converts joystick axes to speed and steering commands

        Emergency stop has priority and suppresses all drive commands
        while active.

        Inputs:
            msg:
                Joy message containing joystick axes and button states.
        """
        self.last_joy_time = self.get_clock().now()

        if len(msg.axes) <= max(self.axis_speed, self.axis_steer):
            self.get_logger().warn("Joy message missing expected axes")
            return

        if len(msg.buttons) > self.btn_estop and msg.buttons[self.btn_estop]:
            if not self.estop_active:
                self.estop_active = True
                self.estop_pub.publish(Bool(data=True))
                self.get_logger().warn("E-STOP ACTIVATED")
            return

        if len(msg.buttons) > self.btn_reset and msg.buttons[self.btn_reset]:
            if self.estop_active:
                self.estop_active = False
                self.estop_pub.publish(Bool(data=False))
                self.get_logger().info("E-STOP RELEASED")

        if self.estop_active:
            return

        mppi_enable = (
            len(msg.buttons) > self.btn_mppi_enable
            and msg.buttons[self.btn_mppi_enable]
        )

        self.mppi_enable_pub.publish(Bool(data=bool(mppi_enable)))

        # get drive commands
        speed_cmd = msg.axes[self.axis_speed] * self.max_speed
        steer_cmd = msg.axes[self.axis_steer] * self.max_steer

        # NOTE: DEBUG LOGS: joystick axes command & published driving commands
        # if self.get_clock().now().nanoseconds % 500_000_000 < 20_000_000:
        #     self.get_logger().info(
        #         f"[JOY] axes[1]={msg.axes[1]:+.2f}, axes[3]={msg.axes[3]:+.2f} | "
        #         f"[CMD] v={speed_cmd:+.2f} m/s, delta={steer_cmd:+.2f} rad"
        #     )

        self._publish_drive_cmd(speed_cmd, steer_cmd)
    

    # ==============================
    # UTILITIES
    # ==============================
    def _health_tick(self):
        """Periodic health heartbeat publisher for teleoperation."""
        if self.last_joy_time is None:
            return

        now = self.get_clock().now()
        if (now - self.last_joy_time).nanoseconds <= int(self.joy_timeout_s * 1e9):
            self.health_pub.publish(Empty())


    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_drive_cmd(self, speed: float, steering: float) -> None:
        """Publishes teleop drive command."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)

        self.cmd_pub.publish(msg)
        

def main():
    rclpy.init()
    node = TeleopAckermann()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
