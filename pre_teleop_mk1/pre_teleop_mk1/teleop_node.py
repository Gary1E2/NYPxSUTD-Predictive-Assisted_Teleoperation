#!/usr/bin/env python3

"""
teleop_node.py used for joystick-based teleoperation using /joy

Logitech F710 mapping (X mode):
- Left stick vertical  -> speed
- Right stick horizontal -> steering
- B button -> emergency stop
- A button -> emergency stop reset
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import Empty, Bool
from ackermann_msgs.msg import AckermannDriveStamped

# ==============================
# NODE CLASS
# ==============================
class TeleopAckermann(Node):
    def __init__(self):
        super().__init__('teleop_node')

        # ========== VARIABLES ==========
        self.max_speed = 3.0        # m/s
        self.max_steer = 0.42       # rad (~24 deg)

        # Logitech Wireless Gamepad F710 bindings (X mode)
        self.axis_speed = 1         # left stick vertical
        self.axis_steer = 3         # right stick horizontal
        self.btn_estop = 1          # B
        self.btn_reset = 0          # A

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

        self.create_timer(0.1, lambda: self.health_pub.publish(Empty()))

        self.get_logger().info(
            "Joystick teleop ready:\n"
            " Left stick vertical  -> speed\n"
            " Right stick horizontal -> steering\n"
            " B -> E-STOP\n"
            " A -> E-STOP reset"
        )

    # ==============================
    # JOYSTICK COMMAND LOOP
    # ==============================
    def joy_callback(self, msg: Joy):
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

        # get drive commands
        speed_cmd = msg.axes[self.axis_speed] * self.max_speed
        steer_cmd = msg.axes[self.axis_steer] * self.max_steer

        # debug log
        if self.get_clock().now().nanoseconds % 500_000_000 < 20_000_000:
            self.get_logger().info(
                f"[JOY] axes[1]={msg.axes[1]:+.2f}, axes[3]={msg.axes[3]:+.2f} | "
                f"[CMD] v={speed_cmd:+.2f} m/s, delta={steer_cmd:+.2f} rad"
            )

        self._publish_drive_cmd(speed_cmd, steer_cmd)

    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_drive_cmd(self, speed: float, steering: float) -> None:
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
