#!/usr/bin/env python3

"""
state_estimator_ekf_node.py used to estimate the state of the vehicle for downstream trajectory planning.
"""

import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu
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
class StateEstimatorEkfNode(Node):
    def __init__(self) -> None:
        super().__init__('state_estimator_ekf_node')

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        # tunable scalars for the extended Kalman Filter
        self.q_v = 0.5
        self.r_v = 0.5
        self.q_xy = 0.1
        self.q_yaw = 0.05
        self.r_xy = 0.2
        self.r_yaw = 0.1

        # state: [x, y, yaw, v] and covariance (P)
        self.x_est = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float)

        # timings
        self.min_dt = 0.02
        self.max_dt = 0.10
        self.default_dt = 0.05

        self.last_odom_time: Optional[Time] = None
        self.last_odom_yaw: Optional[float] = None
        self.last_odom_v: Optional[float] = None

        # timer gate: once odom measurements arrive, timer-driven updates stop
        self.received_odom = False

        self.last_update_time: Optional[Time] = None


        # ========== SUBS & PUBS ==========
        # NOTE: FUTURE ENHANCEMENT: f1tenth gym ros simulation does not publish /imu (hardware might)
        self.imu_sub = None

        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            qos_profile_sub,
        )

        self.vehicle_state_pub = self.create_publisher(
            Odometry, 
            '/vehicle_state', 
            qos_profile_pub
        )

        # timer ensures prediction-only updates still occur when no measurements arrive
        self.create_timer(self.default_dt, self.timer_callback)

    # ==============================
    # EKF STEP
    # ==============================
    def _ekf_step(self, current_time: Time, odom_msg: Optional[Odometry], yaw_rate: float, a_long: float) -> None:
        """
        Extended Kalman Filter step for vehicle state estimation

        Inputs:
            current_time:
                Time raw Odometry msg was received
            odom_msg:
                Raw odometry information containing pose + twist (may be None for prediction-only)
            yaw_rate:
                Synthetic IMU yaw rate derived from odom (rad/s)
            a_long:
                Synthetic IMU longitudinal acceleration derived from odom (m/s^2)
        """
        dt = self._compute_dt_seconds(current_time)


        # ========== PREDICTION STEP ==========
        x = float(self.x_est[0, 0])
        y = float(self.x_est[1, 0])
        yaw = float(self.x_est[2, 0])
        v = float(self.x_est[3, 0])

        x_pred = x + v * math.cos(yaw) * dt
        y_pred = y + v * math.sin(yaw) * dt
        yaw_pred = self._wrap_angle(yaw + yaw_rate * dt)
        v_pred = v + a_long * dt

        self.x_est[0, 0] = x_pred
        self.x_est[1, 0] = y_pred
        self.x_est[2, 0] = yaw_pred
        self.x_est[3, 0] = v_pred

        # jacobian F = d f / d x for [x, y, yaw, v]
        F = np.array(
            [
                [1.0, 0.0, -v * math.sin(yaw) * dt, math.cos(yaw) * dt],
                [0.0, 1.0, v * math.cos(yaw) * dt, math.sin(yaw) * dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # process noise Q
        Q = np.diag([self.q_xy, self.q_xy, self.q_yaw, self.q_v]).astype(float)

        self.P = F @ self.P @ F.T + Q


        # ========== UPDATE STEP (ODOM) ==========
        if odom_msg is not None:
            # extract measurement: z = [x, y, yaw, v]
            ox = float(odom_msg.pose.pose.position.x)
            oy = float(odom_msg.pose.pose.position.y)
            oyaw = float(self._quat_to_yaw(odom_msg.pose.pose.orientation))
            ov = float(odom_msg.twist.twist.linear.x)

            if math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oyaw) and math.isfinite(ov):
                z = np.array([[ox], [oy], [oyaw], [ov]], dtype=float)

                # measurement model: z = H x + noise, H = I
                H = np.eye(4, dtype=float)

                # measurement noise R
                R = np.diag([self.r_xy, self.r_xy, self.r_yaw, self.r_v]).astype(float)

                # innovation with yaw wrapped
                y_innov = z - (H @ self.x_est)
                y_innov[2, 0] = self._wrap_angle(float(y_innov[2, 0]))

                S = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S)

                self.x_est = self.x_est + K @ y_innov
                self.x_est[2, 0] = self._wrap_angle(float(self.x_est[2, 0]))

                I = np.eye(4, dtype=float)
                self.P = (I - K @ H) @ self.P

        # NOTE: DEBUG LOGS: time & estimated x, y, steering, velocity, steering rate, along?
        # self.get_logger().info(
        #     f"[EKF] dt={dt:.3f} | "
        #     f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}, v={v:.2f} | "
        #     f"yaw_rate={yaw_rate:.2f}, a_long={a_long:.2f}"
        # )

        self._publish_state(current_time)
    
    # ==============================
    # UTILITIES & CALLBACKS
    # ==============================
    def _wrap_angle(self, a: float) -> float:
        """Wraps angle to range [-pi, pi]"""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    
    def _quat_to_yaw(self, q) -> float:
        # geometry_msgs/Quaternion: x,y,z,w
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


    def _compute_dt_seconds(self, current_time: Time) -> float:
        """
        Estimates and clamps dt to predefined limits.
        Prevent instability due to sensor jitter or delayed messages.

        Inputs:
            current_time:
                Timestamp of current EKF update.

        Outputs:
            dt (float):
                Time step in seconds used for the EKF step.
        """
        if self.last_update_time is None:
            dt = self.default_dt
        else:
            delta = (current_time - self.last_update_time).nanoseconds * 1e-9
            dt = delta if delta > 0.0 else self.default_dt

        dt = max(self.min_dt, min(self.max_dt, dt))
        self.last_update_time = current_time
        return dt


    def imu_callback(self, msg: Imu) -> None:
        """
        FUTURE ENHANCEMENT: Callback function for IMU.
        """
        # Explicitly ignore IMU yaw/orientation
        _ = msg


    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback function that validates Odometry measurements before EKF updates.
        Invalid measurements -> set as None and perform prediction-only update.
        
        Inputs:
            msg:
                Raw odometry information containing measured velocity value
        """
        v_measured = msg.twist.twist.linear.x
        current_time = Time.from_msg(msg.header.stamp)

        # synthetic IMU derived from odom deltas
        yaw_now = self._quat_to_yaw(msg.pose.pose.orientation)

        yaw_rate = 0.0
        a_long = 0.0

        if self.last_odom_time is not None:
            dt_odom = (current_time - self.last_odom_time).nanoseconds * 1e-9
            if dt_odom > 1e-6:
                if self.last_odom_yaw is not None:
                    dyaw = self._wrap_angle(yaw_now - self.last_odom_yaw)
                    yaw_rate = dyaw / dt_odom
                if self.last_odom_v is not None and math.isfinite(v_measured):
                    a_long = (v_measured - self.last_odom_v) / dt_odom

        self.last_odom_time = current_time
        self.last_odom_yaw = yaw_now
        self.last_odom_v = v_measured if math.isfinite(v_measured) else self.last_odom_v

        if not math.isfinite(v_measured):
            self._ekf_step(current_time, None, yaw_rate, a_long)
            return

        self.received_odom = True
        self._ekf_step(current_time, msg, yaw_rate, a_long)


    def timer_callback(self) -> None:
        """
        Callback function that updates EKF even without odometry measurements.
        Skips EKF update when odometry measurements are received.
        """
        # keep publishing even without measurements
        current_time = self.get_clock().now()
        self._ekf_step(current_time, None, 0.0, 0.0)
        
    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_state(self, stamp: Time) -> None:
        """Creates and publishes a valid, processed Odometry message."""
        msg = Odometry()
        msg.header.frame_id = "ego_racecar/base_link"
        msg.header.stamp = stamp.to_msg()

        x = float(self.x_est[0, 0])
        y = float(self.x_est[1, 0])
        yaw = float(self.x_est[2, 0])

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        # yaw-only quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(yaw * 0.5)

        v = float(self.x_est[3, 0])

        msg.twist.twist.linear.x = v
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0

        self.vehicle_state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateEstimatorEkfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
