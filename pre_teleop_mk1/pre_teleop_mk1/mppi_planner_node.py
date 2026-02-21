#!/usr/bin/env python3

"""
mppi_planner_node.py used to plan, simulate and evaluate trajectories before publishing control actions.

Uses:
- ray-based collision and clearance check from /scan_filtered topic.
- simulated teleoperation trajectory from /teleop_pred topic.
"""

import math
import random
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from std_msgs.msg import Empty, Bool
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from diagnostic_msgs.msg import DiagnosticArray

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
class MppiPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('mppi_planner_node')

        # ========== VARIABLES ==========
        # MPPI parameters
        self.K = 16
        self.dt = 0.05
        self.lambd = 0.65

        self.sigma_delta = 0.015
        self.sigma_v = 0.2

        # Noise temporal correlation (higher = smoother exploration)
        self.rho_delta = 0.8  # 0.6–0.9
        self.rho_v = 0.65     # 0.4–0.8

        # Steering and velocity limits
        self.delta_max = 0.418
        self.v_min = 0.0
        self.v_max = 8.0

        self.wheelbase_L = 0.33
        self.k_v = 2.0

        self.plan_period = 0.05

        # Post-update smoothing (reduces trajectory wiggle without "clamping")
        self.u_smooth_alpha = 0.25  # 0.15–0.35

        # Cost weights
        self.collision_penalty = 2e6
        self.w_center = 1.0
        self.w_delta = 1.0
        self.w_ddelta = 10.0
        self.w_progress = 15.0
        self.w_speed_risk = 6.0

        # speed command smoothness
        self.w_dv = 2.0  # 1–4

        self.w_heading = 4.0
        self.w_turn = 16.0

        # Collision constraint
        self.collision_radius = 0.22

        # Filters
        self.y_center_filt = 0.0
        self.psi_filt = 0.0
        self.alpha_center = 0.15
        self.alpha_heading = 0.3

        # Vehicle state
        self.current_speed = 0.0
        self.last_odom_time: Optional[Time] = None

        self.u_bar: List[Tuple[float, float]] = []

        # LiDAR scan cache
        self.scan_ranges: Optional[List[float]] = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0

        self.turn_strength_now = 0.0

        self.last_delta_cmd = 0.0
        self.last_speed_cmd = 0.0

        # Teleop state reference
        self.teleop_traj: Optional[List[Tuple[float, float]]] = None
        self.teleop_last_time: Optional[Time] = None

        self.teleop_timeout_s = 0.3   # ignore stale teleop predictions

        self.teleop_confidence = 0.0

        # Teleop-following cost params
        self.w_teleop = 4.0           # 2–6
        self.teleop_deadzone = 0.25   # meters

        # ========== SUBS & PUBS ==========
        self.create_subscription(
            Odometry, 
            '/vehicle_state', 
            self.odom_callback, 
            qos_profile_sub
        )

        self.create_subscription(
            LaserScan, 
            '/scan_filtered', 
            self.scan_callback, 
            qos_profile_sub
        )
        self.create_subscription(
            Marker,
            '/teleop_trajectory',
            self.teleop_traj_callback,
            qos_profile_sub
        )  
        self.create_subscription(
            DiagnosticArray,
            '/teleop_pred',
            self.teleop_pred_callback,
            qos_profile_sub
        )

        self.plan_pub = self.create_publisher(
            AckermannDriveStamped, 
            '/drive_plan', 
            qos_profile_pub
        )

        self.best_traj_pub = self.create_publisher(
            Marker, 
            '/planned_trajectory',
            qos_profile_pub
        )

        self.exec_traj_pub = self.create_publisher(
            Marker, 
            '/executed_trajectory', 
            qos_profile_pub
        )
        self.health_pub = self.create_publisher(
            Empty, 
            "/mppi_health", 
            10
        )
        
        self.create_timer(0.1, lambda: self.health_pub.publish(Empty()))
        self.create_timer(self.plan_period, self._planner_timer)

    # ==============================
    # PLANNER LOOP
    # ==============================
    def _planner_timer(self) -> None:
        """
        Main planner loop executed periodically.

        Runs MPPI planner when all required sensor data is available.
        Publishes first control command and best trajectory marker.
        """

        # Sensor check
        if self.scan_ranges is None or self.last_odom_time is None:
            return

        # Turn strength check
        self.turn_strength_now = self._estimate_turn_strength_now()
        plan = self._run_mppi_plan(self.current_speed, self.last_delta_cmd, self.last_speed_cmd)

        # Absolute safe 0, 0 command if no trajectories possible
        if plan is None:
            self._publish_drive_command(0.0, 0.0)
            return

        controls, best_traj = plan
        delta_cmd, speed_cmd = controls[0]

        self.last_delta_cmd = float(delta_cmd)
        self.last_speed_cmd = float(speed_cmd)

        self._publish_drive_command(delta_cmd, speed_cmd)
        self._publish_best_marker(best_traj)

    # ==============================
    # MPPI PLANNER
    # ==============================
    def _run_mppi_plan(self, current_speed: float, delta0: float, v0_cmd: float):
        """
        Executes MPPI optimization loop.

        Samples control noise, rolling out trajectories and evaluating their costs before updating best control sequence.

        Inputs:
            current_speed:
                Current forward speed of car
            delta0:
                Previous executed steering command
            v0_cmd:
                Previous executed speed command
        Outputs:
            self.u_bar:
                Updated control sequence
            trajectories[best_idx]:
                Best trajectory to visualize
        """
        # ========== INIT/REFRESH ==========
        T = self._compute_horizon_steps(current_speed)
        self._warm_start_controls(T)

        trajectories = []
        costs = []
        epsilons = []
        best_cost = math.inf
        best_idx = -1

        # NOTE: K == amount of trajectories/rollouts to evaluate
        for k in range(self.K):
            # ========== INIT/REFRESH ==========
            # NOTE: Limited vehicle state needs future enhancement
            state = (0.0, 0.0, 0.0, current_speed)

            # Anchor rate penalties to what we actually executed last cycle
            delta_prev = float(delta0)
            v_prev_cmd = float(v0_cmd)

            traj = [(0.0, 0.0)]
            cost = 0.0
            eps_k = []
            valid = True

            eps_d_prev = 0.0
            eps_v_prev = 0.0
            rho_d = self.rho_delta
            rho_v = self.rho_v
            rho_d = self._clamp(rho_d, 0.0, 0.95)
            rho_v = self._clamp(rho_v, 0.0, 0.95)
            s_d = math.sqrt(max(1e-6, 1.0 - rho_d * rho_d))
            s_v = math.sqrt(max(1e-6, 1.0 - rho_v * rho_v))

            for t in range(T):
                sigma_d = self.sigma_delta * (1.0 + 4.0 * self.turn_strength_now)
                w_d = random.gauss(0.0, sigma_d)
                w_v = random.gauss(0.0, self.sigma_v)

                eps_d = rho_d * eps_d_prev + s_d * w_d
                eps_v = rho_v * eps_v_prev + s_v * w_v

                eps_d_prev = eps_d
                eps_v_prev = eps_v

                eps_k.append((eps_d, eps_v))

                d_nom, v_nom = self.u_bar[t]
                delta = self._clamp(d_nom + eps_d, -self.delta_max, self.delta_max)
                v_cmd = self._clamp(v_nom + eps_v, self.v_min, self.v_max)

                # Evaluate rollout/trajectory costs
                state, step_cost, collided, delta_prev, v_prev_cmd = self._rollout_step(
                    state, delta, v_cmd, delta_prev, v_prev_cmd
                )


                traj.append((state[0], state[1]))
                cost += step_cost

                if collided:
                    valid = False
                    break

            # NOTE: MPPI invariant: epsilon sequence must be length T
            if len(eps_k) < T:
                eps_k.extend([(0.0, 0.0)] * (T - len(eps_k)))

            trajectories.append(traj if valid else [])
            costs.append(cost if valid else math.inf)
            epsilons.append(eps_k)

            # Get best cost
            if valid and cost < best_cost:
                best_cost = cost
                best_idx = k

        if best_idx < 0:
            # Recovery fallback: small forward exploration with slight steering noise
            v_seed = max(0.5, 0.6 * self.current_speed)
            self.u_bar = [
                (random.uniform(-0.05, 0.05), v_seed)
                for _ in range(T)
            ]
            return self.u_bar, [(0.0, 0.0), (0.2, 0.0)]


        S_min = min(costs)
        inv_lambda = 1.0 / self.lambd

        weights = []
        for c in costs:
            if c < math.inf:
                x = -inv_lambda * (c - S_min)

                # prevent numerical under and overflow
                x = max(-60.0, min(0.0, x))  # exp(-60) ≈ 8e-27

                weights.append(math.exp(x))
            else:
                weights.append(0.0)

        w_sum = sum(weights) + 1e-9
        weights = [w / w_sum for w in weights]


        du_seq = [0.0] * T
        dv_seq = [0.0] * T

        for t in range(T):
            du = dv = 0.0
            for k in range(self.K):
                du += weights[k] * epsilons[k][t][0]
                dv += weights[k] * epsilons[k][t][1]
            du_seq[t] = du
            dv_seq[t] = dv

        # Smooth the update itself (prevents injecting wiggle)
        du_seq, dv_seq = self._smooth_update_seq(du_seq, dv_seq)

        for t in range(T):
            d, v = self.u_bar[t]
            self.u_bar[t] = (
                self._clamp(d + du_seq[t], -self.delta_max, self.delta_max),
                self._clamp(v + dv_seq[t], self.v_min, self.v_max),
            )

        self._smooth_u_bar(T)

        return self.u_bar, trajectories[best_idx]

    # ==============================
    # MPPI ROLLOUT STEP
    # ==============================
    def _rollout_step(self, state, delta, v_cmd, delta_prev, v_prev_cmd):
        """
        Simulates single rollout step of the vehicle model.

        Applies kinematic bicycle vehicle dynamics, evaluating collision and cost.
        Returns updated state and step cost.

        Inputs:
            state:
                Current simulated car state. (x, y, yaw, v)
            delta:
                Steering command for this step
            v_cmd:
                Speed command for this step
            delta_prev:
                Steering command for previous step
            v_prev_cmd:
                Speed command for previous step

        Outputs:
            (x, y, yaw, v):
                Next car state
            total_cost:
                Step cost
            (bool value):
                True: Collision
                False: No collision
            delta:
                Steering command applied for this step
            v_cmd:
                Speed command applied for this step
        """
        # ========== INIT/REFRESH ==========
        x, y, yaw, v = state

        v = self._clamp(v + self.k_v * (v_cmd - v) * self.dt, self.v_min, self.v_max)

        x0 = x
        
        # Kinematic bicycle model
        x += v * math.cos(yaw) * self.dt
        y += v * math.sin(yaw) * self.dt
        yaw += (v / self.wheelbase_L) * math.tan(delta) * self.dt

        if self._collides_disc_ray(x, y, yaw):
            return (x, y, yaw, v), self.collision_penalty, True, delta, v_cmd

        # ========== TURN INTENT ==========
        left = max(self._ray_range(yaw + math.pi/2 + a) for a in (0.0, 0.12, -0.12))
        right = max(self._ray_range(yaw - math.pi/2 + a) for a in (0.0, 0.12, -0.12))
        turn_intent = abs(left - right)
        ts_raw = min(1.0, turn_intent / 1.0)

        # Hard gate based on measurements
        if ts_raw < 0.35:          # MEASURED: mild turns (0.1–0.2)
            turn_strength = 0.0
        elif ts_raw > 0.75:        # MEASURED: sharp corner (~1.0)
            turn_strength = 1.0
        else:
            turn_strength = (ts_raw - 0.35) / (0.75 - 0.35)

        # Debug logging
        # if random.random() < 0.002:
        #     self.get_logger().info(f"turn_strength={turn_strength:.2f} left={left:.2f} right={right:.2f} yaw={yaw:.2f}")

        # ========== FORWARD RISK ==========
        lookahead = self._clamp(1.2 + 0.35 * v, 0.8, 3.5)
        free_frac = self._forward_clearance_fraction(yaw, lookahead)
        risk = (1.0 - free_frac)

        risk_scale = (1.0 - 0.85 * turn_strength)
        speed_risk_cost = risk_scale * self.w_speed_risk * v**2 * risk

        # ========== PROGRESS REWARD ==========
        progress_reward = -self.w_progress * max(0.0, x - x0)

        # ========== CORRIDOR CENTER COST ==========
        # corridor geometry
        d_safe = 0.55  # meters; tune 0.45–0.70 depending on track width
        w_side = 18.0  # start 8–20

        def barrier(d):
            return max(0.0, d_safe - d) ** 3   # cubic wall repulsion
        side_barrier = w_side * (barrier(left) + barrier(right))

        # corridor center proxy: positive = closer to right wall
        center_offset = (right - left) / max(left + right, 1e-3)
        w_center_corridor = 4.0   # start 3–6
        center_corridor_cost = w_center_corridor * (center_offset ** 2)

        # ==========  HEADING TURN COST ==========
        corridor_yaw = math.atan2(
            self._ray_range(yaw + 0.4) - self._ray_range(yaw - 0.4),
            2.0
        )
        yaw_error = self._wrap_angle(yaw - corridor_yaw)

        heading_turn_cost = 5.0 * turn_strength * (yaw_error ** 2)

        # ========== OUTERWALL APPROACH ==========
        # (pre-rotation only)
        outer_side = (right - left)
        pre_rotate = abs(yaw) < 0.15
        w_outer = 2.0
        outer_wall_bias = (
            w_outer * turn_strength * (1.0 if pre_rotate else 0.0) * (-outer_side)
        )

        # ========== APEX FACING REWARD ==========
        apex_dir = math.atan2(left - right, 1.2)
        yaw_error_apex = self._wrap_angle(yaw - apex_dir)
        w_apex = 6.0
        apex_heading_cost = w_apex * turn_strength * (yaw_error_apex ** 2)

        # ========== CENTER GATING DURING TURNS ==========
        center_corridor_cost *= (1.0 - 0.8 * turn_strength)
        side_barrier *= (1.0 - 0.6 * turn_strength)

        center_corridor_cost *= (1.0 - turn_strength)
        side_barrier *= (1.0 - 0.85 * turn_strength)

        # ========== CONTROL SMOOTHNESS COST ==========
        steering_cost = self.w_delta * delta ** 2

        w_ddelta_eff = self.w_ddelta * (1.0 - 0.7 * turn_strength)
        steering_rate_cost = w_ddelta_eff * (delta - delta_prev) ** 2

        w_vtrack = 0.3  # small; start 0.2–0.6

        speed_track_cost = w_vtrack * ((v_cmd - v) ** 2)
        accel_cmd_cost = self.w_dv * ((v_cmd - v_prev_cmd) ** 2)

        # ========== TELEOP PATH SIMILARITY (soft) ==========
        teleop_cost = 0.0

        if self.teleop_traj is not None and self.teleop_last_time is not None:
            # Drop stale teleop predictions
            age = (self.get_clock().now() - self.teleop_last_time).nanoseconds * 1e-9
            if age <= self.teleop_timeout_s:
                # Distance to teleop reference
                d_lat = self._teleop_lateral_distance(x, y)

                # Risk-adaptive weighting (safety dominates)
                w_eff = self.w_teleop * self.teleop_confidence * (1.0 - risk) ** 2

                # Deadzone prevents fighting obstacle avoidance
                if d_lat > self.teleop_deadzone:
                    teleop_cost = w_eff * (d_lat - self.teleop_deadzone) ** 2

        # ========== TOTAL COST ==========
        total_cost = (
            speed_risk_cost
            + progress_reward
            + side_barrier
            + center_corridor_cost
            + heading_turn_cost
            + outer_wall_bias
            + apex_heading_cost
            + steering_cost
            + steering_rate_cost
            + speed_track_cost
            + accel_cmd_cost
            + teleop_cost
        )

        return (x, y, yaw, v), total_cost, False, delta, v_cmd
    
    # ==============================
    # UTILITIES & CALLBACKS
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

    def _ray_range_window(self, theta, dtheta=0.03):
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
        best = 0.0
        for a in (-dtheta, 0.0, dtheta):
            r = self._ray_range(theta + a)
            if r > best:
                best = r
        return best

    # ========== COLLISION & CLEARANCE ==========
    def _collides_disc_ray(self, x, y, yaw) -> bool:
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
        offsets = [(0.0, 0.0), (0.0, r), (0.0, -r)]

        for ox, oy in offsets:
            wx = x + ox * math.cos(yaw) - oy * math.sin(yaw)
            wy = y + ox * math.sin(yaw) + oy * math.cos(yaw)

            theta = math.atan2(wy, wx)
            dist = math.hypot(wx, wy)

            r_obs = self._ray_range_window(theta)
            if dist >= r_obs - 0.05:
                return True


        return False

    def _forward_clearance_fraction(self, yaw, lookahead_m) -> float:
        """
        Estimates fraction of forward LiDAR scans that is free space

        Inputs:
            yaw:
                Car's heading
            lookahead_m:
                Lookahead distance for car
        Outputs:
            (float value):
                Fraction of scans indicating free space.
        """
        angles = [-0.15, -0.07, 0.0, 0.07, 0.15]
        free = 0
        for a in angles:
            if self._ray_range(yaw + a) > lookahead_m:
                free += 1
        return free / len(angles)
    
    # ========== TURN STRENGTH LOGIC ==========
    def _estimate_turn_strength_now(self) -> float:
        """
        Estimate turn sharpness from LiDAR side disparity (edges/corners)
        Uses side disparity at +/- 90 degrees, robust to mild turns.

        Outputs:
            turn_strength:
                Normalized value [0.0, 1.0] to represent turn strength
        """
        # If scan not ready, do nothing special
        if self.scan_ranges is None or self.scan_angle_inc == 0.0:
            return 0.0

        # In vehicle frame: left is +pi/2, right is -pi/2
        left = max(self._ray_range(math.pi / 2 + a) for a in (0.0, 0.12, -0.12))
        right = max(self._ray_range(-math.pi / 2 + a) for a in (0.0, 0.12, -0.12))

        # Raw sharpness proxy (same scaling you already use)
        ts_raw = min(1.0, abs(left - right) / 1.0)

        # HARD GATE based on your measurements:
        # mild turns: 0.1–0.2 => should become 0
        # sharp corners: ~1.0 => should become 1
        if ts_raw < 0.35:
            return 0.0
        if ts_raw > 0.75:
            return 1.0

        # Smooth ramp in between
        return (ts_raw - 0.35) / (0.75 - 0.35)
    
    # ========== SMOOTHERS ==========
    def _smooth_u_bar(self, T: int) -> None:
        """
        Smooths the control sequence over time.
        Forward + backward exponential smoothing to reduce oscillation.

        Inputs:
            T:
                Horizon length in steps
        """
        if T <= 2 or not self.u_bar:
            return

        alpha = self._clamp(self.u_smooth_alpha, 0.0, 0.8)
        ub = list(self.u_bar)

        # Forward pass
        for t in range(1, T):
            d_prev, v_prev = ub[t - 1]
            d, v = ub[t]
            ub[t] = ((1.0 - alpha) * d + alpha * d_prev,
                     (1.0 - alpha) * v + alpha * v_prev)

        # Backward pass (reduces lag)
        for t in range(T - 2, -1, -1):
            d_next, v_next = ub[t + 1]
            d, v = ub[t]
            ub[t] = ((1.0 - alpha) * d + alpha * d_next,
                     (1.0 - alpha) * v + alpha * v_next)

        # Clamp result
        self.u_bar = [
            (self._clamp(d, -self.delta_max, self.delta_max),
             self._clamp(v, self.v_min, self.v_max))
            for (d, v) in ub
        ]

    def _smooth_update_seq(self, du_seq, dv_seq):
        """
        Smooths the MPPI control update sequence

        Inputs:
            du_seq:
                Steering sequence
            dv_seq:
                Speed sequence
        Outputs:
            du_seq, dv_seq:
                Smoothed steering and speed update sequences
        """
        alpha = 0.35  # stronger smoothing is ok here
        T = len(du_seq)
        if T <= 2:
            return du_seq, dv_seq

        # forward
        for t in range(1, T):
            du_seq[t] = (1.0 - alpha) * du_seq[t] + alpha * du_seq[t - 1]
            dv_seq[t] = (1.0 - alpha) * dv_seq[t] + alpha * dv_seq[t - 1]
        # backward
        for t in range(T - 2, -1, -1):
            du_seq[t] = (1.0 - alpha) * du_seq[t] + alpha * du_seq[t + 1]
            dv_seq[t] = (1.0 - alpha) * dv_seq[t] + alpha * dv_seq[t + 1]

        return du_seq, dv_seq
    
    # ========== TELEOP LOGIC ==========
    def _teleop_lateral_distance(self, x: float, y: float) -> float:
        """
        Calculates lateral distance to the teleop reference path

        Inputs:
            x:
                Car's x position
            y:
                Car's y position
        Outputs:
            (float value):
                Distance to nearest teleop trajectory point
        """
        if not self.teleop_traj:
            return math.inf

        return min(
            math.hypot(x - xr, y - yr)
            for xr, yr in self.teleop_traj
        )

    # ========== ADDITIONAL ==========
    def _wrap_angle(self, a: float) -> float:
        """Wraps angle to range [-pi, pi]"""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _compute_horizon_steps(self, v):
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

    def _warm_start_controls(self, T):
        """
        Initializes control sequence for warm starting.
        Makes speeds command not stay at 0 when car starts at speed 0.
        
        Inputs:
            T:
                Horizon length in steps
        """
        if not self.u_bar:
            self.u_bar = [(0.0, max(0.5, self.current_speed)) for _ in range(T)]
            return

        prev = list(self.u_bar)
        self.u_bar = []

        for i in range(T):
            if i + 1 < len(prev):
                self.u_bar.append(prev[i + 1])
            else:
                self.u_bar.append(prev[-1])

    def _clamp(self, v, vmin, vmax):
        """Value clamping"""
        return max(vmin, min(vmax, v))

    # ========== CALLBACKS ==========
    def scan_callback(self, msg: LaserScan) -> None:
        """Receives preprocessed LiDAR scan and caches range and geometry data."""
        self.scan_ranges = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_inc = msg.angle_increment

    def odom_callback(self, msg: Odometry) -> None:
        """Receives vehicle odometry and updates current speed and timestamp."""
        self.current_speed = msg.twist.twist.linear.x
        self.last_odom_time = Time.from_msg(msg.header.stamp)

    def teleop_traj_callback(self, msg: Marker) -> None:
        """Receives and stores teleop reference trajectory for cost calculation"""
        if not msg.points:
            self.teleop_traj = None
            return
        self.teleop_traj = [(p.x, p.y) for p in msg.points]
        self.teleop_last_time = self.get_clock().now()  # arrival time

    def teleop_pred_callback(self, msg: DiagnosticArray) -> None:
        """Receives teleoperation prediction diagnostics and updates confidence value."""
        for st in msg.status:
            if st.name != "teleop_pred":
                continue
            for kv in st.values:
                if kv.key == "teleop_confidence":
                    try:
                        self.teleop_confidence = float(kv.value)
                    except ValueError:
                        self.teleop_confidence = 0.0

    # ==============================
    # PUBLISHING
    # ==============================
    def _publish_drive_command(self, delta, v):
        """Publish the selected driving command"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar/base_link'
        msg.drive.steering_angle = delta
        msg.drive.speed = v
        self.plan_pub.publish(msg)

    def _publish_best_marker(self, traj):
        """Publish the best trajectory marker for visualization/debugging."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.02
        marker.color.b = 1.0
        marker.color.a = 1.0

        for x, y in traj:
            marker.points.append(Point(x=x, y=y, z=0.0))

        self.best_traj_pub.publish(marker)

def main():
    rclpy.init()
    node = MppiPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()