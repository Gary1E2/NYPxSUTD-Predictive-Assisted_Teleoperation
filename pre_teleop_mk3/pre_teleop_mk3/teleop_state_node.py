#!/usr/bin/env python3

"""
teleop_state_node.py used to predict the teleoperation state and trajectory
"""

import math
import random
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

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
class TeleopStateNode(Node):
    """
    Predict teleop motion with a single constant-command rollout (no control authority).
    """

    def __init__(self) -> None:
        super().__init__('teleop_state_node')

        self.get_logger().info(f"Initialized and running.")


        # ========== MPPI MATCHING VARIABLES ==========
        self.dt = 0.05
        self.wheelbase_L = 0.33
        self.collision_radius = 0.22
        self.max_input_age_s = 0.25


        # ========== STATE ==========
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.current_speed: Optional[float] = None
        self.last_odom_time: Optional[Time] = None

        self.last_cmd: Optional[AckermannDriveStamped] = None
        self.last_cmd_time: Optional[Time] = None

        self.scan_ranges: Optional[List[float]] = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.last_scan_time: Optional[Time] = None


        # ========== BUFFER ==========
        self.d_safe = 0.8


        # ========== SUBS & PUBS ==========
        self.create_subscription(
            AckermannDriveStamped, 
            '/drive_teleop', 
            self.teleop_cmd_callback, 
            qos_profile_sub
        )

        self.create_subscription(
            Odometry, 
            '/ego_racecar/odom', 
            self.odom_callback, 
            qos_profile_sub
        )

        self.create_subscription(
            LaserScan, 
            '/scan_filtered', 
            self.scan_callback, 
            qos_profile_sub
        )

        self.pred_pub = self.create_publisher(
            DiagnosticArray, 
            '/teleop_pred', 
            qos_profile_pub
        )

        self.traj_pub = self.create_publisher(
            Marker, 
            '/teleop_trajectory', 
            qos_profile_pub
        )

        self.create_timer(self.dt, self._predict_timer)

    # ==============================
    # TELEOP PREDICTION LOOP
    # ==============================
    def _predict_timer(self) -> None:
        """
        Main teleop predition loop executed periodically.

        Predicts a future teleop trajectory using constant command rollout.
        Publishes conservative outputs when required inputs are stale.
        """
        # ========== INIT/REFRESH ==========
        now = self.get_clock().now()
        if self._inputs_stale(now):
            self._publish_conservative(now)
            return

        assert self.current_pose is not None
        assert self.current_speed is not None
        assert self.last_cmd is not None

        v_cmd = float(self.last_cmd.drive.speed)
        delta_cmd = float(self.last_cmd.drive.steering_angle)

        horizon_steps = self._compute_horizon_steps(self.current_speed)

        x0, y0, yaw0 = self.current_pose

        # NOTE: ASSUMPTIONS:
        # constant-command: hold teleop input fixed over the horizon.
        # constant-speed: uses a single v across all steps (no acceleration model).
        v = float(self.current_speed)

        if abs(v) < 0.05:
            v = 0.05

        traj_points: List[Tuple[float, float]] = [(0.0, 0.0)]

        # clearance tracking
        min_clearance = math.inf
        clear_now = math.inf
        clear_min_future = math.inf

        collision_predicted = False
        collision_time = math.inf

        x = x0
        y = y0
        yaw = yaw0

        clear_now = self._min_clearance_disc_ray(0.0, 0.0, 0.0)
        min_clearance = clear_now

        # NOTE: DEBUG LOGS: estimated teleop velocity, steering and steps
        # self.get_logger().info(
        #     f"teleop rollout: v={v:.3f}, delta={delta_cmd:.3f}, steps={horizon_steps}"
        # )

        for step_idx in range(horizon_steps):
            # kinematic bicycle model
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += (v / self.wheelbase_L) * math.tan(delta_cmd) * self.dt

            x_rel, y_rel, yaw_rel = self._to_base_link(x, y, yaw, x0, y0, yaw0)
            traj_points.append((x_rel, y_rel))

            # clearance check
            clearance = self._min_clearance_disc_ray(x_rel, y_rel, yaw_rel)

            # global minimum clearance (legacy / confidence use)
            if clearance < min_clearance:
                min_clearance = clearance

            # future-only minimum clearance (exclude step 0)
            if step_idx >= 0:
                if clearance < clear_min_future:
                    clear_min_future = clearance

            # collision check
            if self._collides_disc_ray(x_rel, y_rel, yaw_rel):
                collision_predicted = True
                collision_time = (step_idx + 1) * self.dt
                break

        # positive: moving away from obstacles
        # negative: moving deeper into danger
        clear_delta = clear_min_future - clear_now
        buffer_violation = (clear_now < self.d_safe)

        # trajectory-based TTC
        # if no collision within horizon, set TTC to horizon time (finite) for robustness.
        if math.isfinite(collision_time):
            ttc_min = collision_time
        else:
            ttc_min = horizon_steps * self.dt

        self._publish_pred(
            now,
            min_clearance,
            ttc_min,
            collision_predicted,
            collision_time,
            clear_now,
            clear_min_future,
            clear_delta,
            buffer_violation,
        )

        self._publish_trajectory(now, traj_points, collision_predicted, collision_time)

    # ==============================
    # UTITLIES & CALLBACKS
    # ==============================

    # ========== RAY LOGIC ==========
    def _ray_range(self, theta: float) -> float:
        """
        Gets LiDAR scan at given angle.

        Inputs:
            theta:
                Angle from vehicle frame
        Outputs:
            r:
                Distance to nearest obstacle along the scan line
        """
        if self.scan_ranges is None or self.scan_angle_inc == 0.0:
            return math.inf

        idx = int(math.floor((theta - self.scan_angle_min) / self.scan_angle_inc))
        if idx < 0:
            idx = 0
        elif idx >= len(self.scan_ranges):
            idx = len(self.scan_ranges) - 1

        r = self.scan_ranges[idx]
        return r if math.isfinite(r) else math.inf


    def _ray_range_window(self, theta: float, dtheta: float = 0.03) -> float:
        """
        Gets max LiDAR value from a small angular window

        Inputs:
            theta:
                Angle from the vehicle frame
            dtheta:
                Angular offset for window sampling
        Outputs:
            best:
                Max distance to nearest obstacle found from the window
        """
        best = math.inf
        for a in (-dtheta, 0.0, dtheta):
            r = self._ray_range(theta + a)
            if r < best:
                best = r
        return best


    # ========== COLLISION & CLEARANCE ==========
    def _collides_disc_ray(self, x: float, y: float, yaw: float) -> bool:
        """
        Collision check with disc approximation and LiDAR scans

        Inputs:
            x:
                Car's x position in local frame
            y:
                Car's y position in local frame
            yaw:
                Car's heading.
        Outputs:
            (bool value):
                True: Collision
                False: No collision
        """
        r = self.collision_radius
        offsets = [(r, 0.0), (0.0, 0.0), (0.0, r), (0.0, -r)]

        for ox, oy in offsets:
            wx = x + ox * math.cos(yaw) - oy * math.sin(yaw)
            wy = y + ox * math.sin(yaw) + oy * math.cos(yaw)

            theta = math.atan2(wy, wx)
            dist = math.hypot(wx, wy)

            r_obs = self._ray_range_window(theta)
            if dist >= r_obs - 0.05:
                return True

        return False


    def _min_clearance_disc_ray(self, x: float, y: float, yaw: float) -> float:
        """
        Computes minimum obstacle clearance using disc-based ray checks.

        Inputs:
            x:
                Car's x position in local frame
            y:
                Car's y position in local frame
            yaw:
                Car's heading angle
        Outputs:
            min_clearance:
                Minimum clearance distance from nearest obstacle
        """
        r = self.collision_radius
        offsets = [(0.0, 0.0), (0.0, r), (0.0, -r)]
        min_clearance = math.inf

        for ox, oy in offsets:
            wx = x + ox * math.cos(yaw) - oy * math.sin(yaw)
            wy = y + ox * math.sin(yaw) + oy * math.cos(yaw)

            theta = math.atan2(wy, wx)
            dist = math.hypot(wx, wy)

            r_obs = self._ray_range_window(theta)
            clearance = r_obs - 0.05 - dist
            if clearance < min_clearance:
                min_clearance = clearance

        return min_clearance


    # ========== ADDITIONAL ==========
    def _yaw_from_quat(self, q) -> float:
        """Get yaw angle from quaternion"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


    def _wrap_angle(self, a: float) -> float:
        """Wraps angle to range [-pi, pi]"""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


    def _to_base_link(self,
        x: float, 
        y: float, 
        yaw: float, 
        x0: float, 
        y0: float, 
        yaw0: float
    ) -> Tuple[float, float, float]:
        """
        Transforms world frame pose to vehicle base_link frame.

        Inputs:
            x, y, yaw:
                World-frame pose
            x0, y0, yaw0:
                Reference origin pose
        Outputs:
            (x_rel, y_rel, yaw_rel):
                Pose expressed in base_link frame
        """
        dx = x - x0
        dy = y - y0
        cos_y = math.cos(yaw0)
        sin_y = math.sin(yaw0)
        x_rel = cos_y * dx + sin_y * dy
        y_rel = -sin_y * dx + cos_y * dy
        yaw_rel = self._wrap_angle(yaw - yaw0)
        return x_rel, y_rel, yaw_rel


    def _compute_horizon_steps(self, v: float) -> int:
        """
        Computes MPPI planning horizon dynamically based on speed

        Inputs:
            v:
                Current car speed
        Outputs:
            (int value):
                Number of horizon steps
        """
        horizon_time = min(3.0, 1.5 + 0.15 * v)
        return int(math.ceil(horizon_time / self.dt))


    def _inputs_stale(self, now: Time) -> bool:
        """Check if inputs are missing or outdated"""
        max_age_ns = int(self.max_input_age_s * 1e9)
        if self.current_pose is None or self.current_speed is None:
            return True
        if self.last_cmd is None or self.last_cmd_time is None:
            return True
        if self.scan_ranges is None or self.last_scan_time is None:
            return True
        if self.last_odom_time is None:
            return True

        if (now - self.last_cmd_time).nanoseconds > max_age_ns:
            return True
        if (now - self.last_odom_time).nanoseconds > max_age_ns:
            return True
        if (now - self.last_scan_time).nanoseconds > max_age_ns:
            return True

        return False

    
    # ========== CALLBACKS ==========
    def teleop_cmd_callback(self, msg: AckermannDriveStamped) -> None:
        """Receives latest teleoperation command and records timestamp."""
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()


    def odom_callback(self, msg: Odometry) -> None:
        """Receives odometry and updates vehicle pose, speed, and timestamp."""
        pos = msg.pose.pose.position
        yaw = self._yaw_from_quat(msg.pose.pose.orientation)
        self.current_pose = (pos.x, pos.y, yaw)
        self.current_speed = msg.twist.twist.linear.x
        self.last_odom_time = Time.from_msg(msg.header.stamp)


    def scan_callback(self, msg: LaserScan) -> None:
        """Receives filtered LiDAR scan and stores values, angle info and timestamp."""
        self.scan_ranges = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment
        self.last_scan_time = Time.from_msg(msg.header.stamp)


    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_pred(
        self,
        now: Time,
        min_clearance: float,
        ttc_min: float,
        collision_predicted: bool,
        collision_time: float,
        clear_now: float,
        clear_min_future: float,
        clear_delta: float,
        buffer_violation: bool,
    ) -> None:
        """
        Publishes teleop safety prediction diagnostics.

        Includes minimum clearance, TTC, collision flags, and teleop confidence.
        """
        
        msg = DiagnosticArray()
        msg.header.stamp = now.to_msg()

        status = DiagnosticStatus()
        status.name = "teleop_pred"
        status.level = DiagnosticStatus.OK if not collision_predicted else DiagnosticStatus.WARN
        status.message = "teleop prediction"

        # ========== TELEOP CONFIDENCE ==========
        d_safe = 0.8  # based on track width

        if collision_predicted:
            teleop_confidence = 0.0
        else:
            teleop_confidence = min(1.0, max(0.0, min_clearance / d_safe))

        # ========== TELEOP ASSIST LEVEL ==========
        # assist_level indicates how strongly MPPI should preserve teleop intent.
        # 0.0 = teleop is safe, no assistance needed
        # 1.0 = teleop is unsafe, full assistance required
        if collision_predicted:
            assist_level = 1.0
        else:
            # Ramp up assistance as clearance drops below d_safe
            assist_level = max(0.0, min(1.0, 1.0 - (min_clearance / d_safe)))

        status.values = [
            KeyValue(key="min_clearance", value=f"{min_clearance:.6f}"),
            KeyValue(key="ttc_min", value=f"{ttc_min:.6f}"),
            KeyValue(key="collision_predicted", value=str(bool(collision_predicted))),
            KeyValue(key="collision_time", value=f"{collision_time:.6f}"),
            KeyValue(key="teleop_confidence", value=f"{teleop_confidence:.3f}"),
            KeyValue(key="assist_level", value=f"{assist_level:.3f}"),
            KeyValue(key="d_safe", value=f"{self.d_safe:.6f}"),
            KeyValue(key="clear_now", value=f"{clear_now:.6f}"),
            KeyValue(key="clear_min_future", value=f"{clear_min_future:.6f}"),
            KeyValue(key="clear_delta", value=f"{clear_delta:.6f}"),
            KeyValue(key="buffer_violation", value=str(bool(buffer_violation))),
        ]

        # NOTE: DEBUG LOGS: assist level, confidence, minimum clearance, collision status
        if random.random() < 0.05:  # ~5% of cycles
            self.get_logger().info(
                f"[TELEOP PRED] "
                f"assist={assist_level:.2f} "
                f"conf={teleop_confidence:.2f} "
                f"min_clear={min_clearance:.2f} "
                f"collision={collision_predicted}"
            )

        msg.status.append(status)
        self.pred_pub.publish(msg)


    def _publish_trajectory(
        self,
        now: Time,
        traj_points: List[Tuple[float, float]],
        collision_predicted: bool,
        collision_time: float,
    ) -> None:
        """Publishes predicted teleop trajectory for visualization."""

        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = now.to_msg()

        marker.ns = "teleop_reference"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # lifetime enforces freshness (MPPI can ignore expired refs)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = int(2 * self.dt * 1e9)

        marker.scale.x = 0.03

        # encode safety confidence via color
        if collision_predicted:
            marker.color.r = 1.0   # red = unsafe intent
            marker.color.g = 0.3
            marker.color.a = 0.8
        else:
            marker.color.g = 1.0   # green = safe intent
            marker.color.a = 1.0

        for x, y in traj_points:
            marker.points.append(Point(x=x, y=y, z=0.0))

        self.traj_pub.publish(marker)


    def _publish_conservative(self, now: Time) -> None:
        """
        Publishes a conservative fallback prediction and minimal trajectory.

        Used when inputs are missing or stale.
        """
        self._publish_pred(now, 0.0, 0.0, True, 0.0, 0.0, 0.0, 0.0, True)
        self._publish_trajectory(
            now,
            [(0.0, 0.0)],
            collision_predicted=True,
            collision_time=0.0,
        )
        

def main():
    rclpy.init()
    node = TeleopStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
