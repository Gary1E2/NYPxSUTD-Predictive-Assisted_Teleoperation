#!/usr/bin/env python3

"""
state_estimator_ekf_node.py used to estimate the state of the vehicle for downstream trajectory planning.

Maintains a forward-speed EKF estimate.
Pose is clamped to zero; no yaw integration, lateral velocity, or TF usage.
"""

import math
from typing import Optional

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

        # ========== VARIABLES ==========
        # Tunable scalars for the 1D EKF
        self.q_v = float(self.declare_parameter('q_v', 0.5).value)
        self.r_v = float(self.declare_parameter('r_v', 0.5).value)

        # State: velocity estimate (v) and covariance (P)
        self.v_est = 0.0
        self.P = 1.0

        # Timing helpers
        self.min_dt = 0.02
        self.max_dt = 0.10
        self.default_dt = 0.05
        self.last_update_time: Optional[Time] = None

        # Timer gate: once odom measurements arrive, timer-driven updates stop
        self.received_odom = False

        # ========== SUBS & PUBS ==========
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            qos_profile_sub,
        )

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

        # Timer ensures prediction-only updates still occur when no measurements arrive
        self.create_timer(self.default_dt, self.timer_callback)

    # ==============================
    # EKF STEP
    # ==============================
    def _ekf_step(self, current_time: Time, v_measured: Optional[float]) -> None:
        """
        EKF
        
        Inputs:
            current_time:
                Time raw Odometry msg was received 
            v_measured:
                Measured velocity of the car.
        """
        dt = self._compute_dt_seconds(current_time)
        v_prev = self.v_est

        ### FUTURE ENHANCEMENT ###
        # NOTE: Process model excludes measurements: acceleration assumed zero 
        a = 0.0
        x_pred = v_prev + a * dt

        ### FUTURE ENHANCEMENT ###
        # F = [1], so P_pred = P + Q
        P_pred = self.P + self.q_v   

        if v_measured is not None:
            # Measurement model: H = [1], R = [r_v].
            y = v_measured - x_pred
            S = P_pred + self.r_v
            K = P_pred / S if S > 0.0 else 0.0
            self.v_est = x_pred + K * y
            self.P = (1.0 - K) * P_pred
        else:
            # No measurement -> publish prediction-only estimate.
            self.v_est = x_pred
            self.P = P_pred

        self._publish_state(current_time)
    
    # ==============================
    # UTILITIES & CALLBACKS
    # ==============================
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
        UNUSED: Callback function for IMU #Future Enhancement
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
        # Only forward speed measurement is permitted; pose is never used ### FUTURE ENHANCEMENT ###
        v_measured = msg.twist.twist.linear.x

        if not math.isfinite(v_measured):
            # Invalid measurement > prediction-only update with measurement skipped.
            current_time = Time.from_msg(msg.header.stamp)
            self._ekf_step(current_time, None)
            return

        self.received_odom = True
        current_time = Time.from_msg(msg.header.stamp)
        self._ekf_step(current_time, v_measured)

    def timer_callback(self) -> None:
        """
        Callback function that updates EKF even without odometry measurements.
        Skips EKF update when odometry measurements are received.
        """
        # Skip timer updates once odom measurements are flowing to avoid double updates
        if self.received_odom:
            return
        
        # Keep publishing even without measurements
        current_time = self.get_clock().now()
        self._ekf_step(current_time, None)

    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_state(self, stamp: Time) -> None:
        """Creates and publishes a valid, processed Odometry message."""
        msg = Odometry()
        msg.header.frame_id = "ego_racecar/base_link"
        msg.header.stamp = stamp.to_msg()

        # Pose is fixed at the origin with no rotation; pose covariance unused.
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        # Only forward velocity is estimated/published; no lateral/yaw content. ### FUTURE ENHANCEMENT ###
        msg.twist.twist.linear.x = float(self.v_est)
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0

        self.vehicle_state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateEstimatorEkfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
