#!/usr/bin/env python3

"""
mppi_planner_node.py used to plan, simulate and evaluate trajectories before publishing control actions.

Uses:
- ray-based collision and clearance check from /scan_filtered topic.
- teleoperation information from /teleop_pred topic.
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
from visualization_msgs.msg import Marker, MarkerArray
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

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        # MPPI
        self.K = 24
        self.dt = 0.05
        self.lambd = 0.65

        self.sigma_delta = 0.015
        self.sigma_v = 0.3

        # noise temporal correlation (higher = smoother exploration)
        self.rho_delta = 0.8
        self.rho_v = 0.65

        # control limits & car specs
        self.delta_max = 0.418
        self.v_min = 0.0
        self.v_max = 4.0
        self.wheelbase_L = 0.33
        self.k_v = 2.0

        self.plan_period = 0.05

        # post-update smoothing (reduces trajectory wiggle without "clamping")
        self.u_smooth_alpha = 0.25

        # collision constraint
        self.collision_penalty = 2e6
        self.collision_radius = 0.15

        # cost weights
        self.w_center = 1.0
        self.w_delta = 0.3
        self.w_ddelta = 10.0
        self.w_progress = 15.0
        self.w_speed_risk = 6.0
        self.w_dv = 2.0
        self.w_heading = 4.0
        self.w_turn = 16.0
        self.w_teleop = 4.0
        self.teleop_deadzone = 0.25

        # filters
        self.y_center_filt = 0.0
        self.psi_filt = 0.0
        self.alpha_center = 0.15
        self.alpha_heading = 0.3

        # vehicle state
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

        # teleop state reference
        self.teleop_traj: Optional[List[Tuple[float, float]]] = None
        self.teleop_last_time: Optional[Time] = None

        self.teleop_timeout_s = 0.3
        self.teleop_confidence = 0.0

        self.teleop_mppi_enable = False


        # ========== OPTIONAL ROLLOUT VISUALIZATION ==========
        self.declare_parameter("enable_rollout_viz", False)
        self.declare_parameter("rollout_viz_max_k", 48)          # cap displayed rollouts (perf)
        self.declare_parameter("rollout_viz_t_stride", 1)        # draw every Nth point along a rollout
        self.declare_parameter("rollout_viz_pub_every_n", 1)     # publish every N planner ticks
        self.declare_parameter("rollout_viz_lifetime_s", 0.20)   # marker lifetime to auto-clear

        self.enable_rollout_viz = bool(self.get_parameter("enable_rollout_viz").value)
        self.rollout_viz_max_k = int(self.get_parameter("rollout_viz_max_k").value)
        self.rollout_viz_t_stride = int(self.get_parameter("rollout_viz_t_stride").value)
        self.rollout_viz_pub_every_n = int(self.get_parameter("rollout_viz_pub_every_n").value)
        self.rollout_viz_lifetime_s = float(self.get_parameter("rollout_viz_lifetime_s").value)

        # internal cache for last MPPI samples
        self._last_rollout_trajs = None
        self._last_rollout_costs = None
        self._last_rollout_best_idx = -1
        self._planner_tick = 0


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

        self.create_subscription(
            Bool,
            "/teleop_mppi_enable",
            self._teleop_mppi_enable_cb,
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
        self.rollout_markers_pub = self.create_publisher(
            MarkerArray, 
            "/mppi_rollout_samples", 
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

        # sensor check
        if self.scan_ranges is None or self.last_odom_time is None:
            return

        # turn strength check
        self.turn_strength_now = self._estimate_turn_strength_now()
        plan = self._run_mppi_plan(self.current_speed, self.last_delta_cmd, self.last_speed_cmd)

        # all trajectories invalid: absolute safe 0, 0 command
        if plan is None:
            self._publish_drive_command(0.0, 0.0)
            return


        # ========== OPTIONAL ROLLOUT VISUALIZATION ==========
        # self._publish_rollout_markers()

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


        # ========== K SAMPLES ROLLOUT LOOP ==========
        for k in range(self.K):
            # ========== INIT/REFRESH ==========
            state = (0.0, 0.0, 0.0, current_speed)

            # anchor rate penalties to actual executed last cycle
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


            # ========== T HORIZON STEP LOOP ==========
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

                # ========== ROLLOUT COST STEP ==========
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

            # get best cost
            if valid and cost < best_cost:
                best_cost = cost
                best_idx = k


        # RECOVERY: small forward exploration with slight steering noise
        if best_idx < 0:
            v_seed = max(0.5, 0.6 * self.current_speed)
            self.u_bar = [
                (random.uniform(-0.05, 0.05), v_seed)
                for _ in range(T)
            ]


            # ========== OPTIONAL ROLLOUT VISUALIZATION ==========
            # self._last_rollout_trajs = []
            # self._last_rollout_costs = []
            # self._last_rollout_best_idx = -1

            return self.u_bar, [(0.0, 0.0), (0.2, 0.0)]


        S_min = min(costs)
        inv_lambda = 1.0 / self.lambd

        weights = []
        for c in costs:
            if c < math.inf:
                x = -inv_lambda * (c - S_min)

                # prevent numerical under and overflow
                x = max(-60.0, min(0.0, x))  # exp(-60) ~ 8e-27

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

        # smooth the update itself (prevents wiggle)
        du_seq, dv_seq = self._smooth_update_seq(du_seq, dv_seq)

        for t in range(T):
            d, v = self.u_bar[t]
            self.u_bar[t] = (
                self._clamp(d + du_seq[t], -self.delta_max, self.delta_max),
                self._clamp(v + dv_seq[t], self.v_min, self.v_max),
            )

        self._smooth_u_bar(T)


        # ========== OPTIONAL ROLLOUT VISUALIZATION ==========
        # self._last_rollout_trajs = trajectories
        # self._last_rollout_costs = costs
        # self._last_rollout_best_idx = best_idx

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

        # kinematic bicycle model
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

        if ts_raw < 0.35:
            turn_strength = 0.0
        elif ts_raw > 0.75:
            turn_strength = 1.0
        else:
            turn_strength = (ts_raw - 0.35) / (0.75 - 0.35)


        # ========== FORWARD RISK ==========
        lookahead = self._clamp(1.2 + 0.35 * v, 0.8, 3.5)
        free_frac = self._forward_clearance_fraction(yaw, lookahead)
        risk = (1.0 - free_frac)

        risk_scale = (1.0 - 0.85 * turn_strength)
        speed_risk_cost = (
            risk_scale
            * self.w_speed_risk
            * max(v - 0.6, 0.0)**2
            * max(0.0, risk - 0.25)
        )


        # ========== CORRIDOR CENTER COST ==========
        d_safe = 0.55
        w_side = 18.0

        def barrier(d):
            return max(0.0, d_safe - d) ** 3

        side_barrier = w_side * (barrier(left) + barrier(right))

        center_offset = (right - left) / max(left + right, 1.2)
        w_center_corridor = 6.0
        center_corridor_cost = w_center_corridor * (center_offset ** 2)


        # ========== HEADING TURN COST ==========
        corridor_yaw = math.atan2(
            self._ray_range(yaw + 0.4) - self._ray_range(yaw - 0.4),
            2.0
        )
        yaw_error = self._wrap_angle(yaw - corridor_yaw)

        heading_turn_cost = 5.0 * turn_strength * (yaw_error ** 2)


        # ========== OUTERWALL APPROACH ==========
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


        # ========== CONTINUE COST ==========
        fwd = self._ray_range(yaw)
        fwd_l = self._ray_range(yaw + 0.25)
        fwd_r = self._ray_range(yaw - 0.25)

        corridor_width = min(fwd_l, fwd_r)
        corridor_consistency = corridor_width / max(fwd, 1e-3)

        w_corridor_cont = 12.0
        continuation_cost = w_corridor_cont * max(0.0, 0.6 - corridor_consistency)**2


        # ========== PROGRESS REWARD ==========
        progress = v * math.cos(yaw_error)
        w_progress_eff = self.w_progress * (1.0 - 0.85 * turn_strength)
        progress_reward = -w_progress_eff * max(0.0, progress)


        # ========== SHARP TURN SPEED COST ==========
        v_straight = 3.5
        v_corner = 1.0

        v_turn_max = v_straight - (v_straight - v_corner) * (turn_strength ** 1.5)

        w_turn_speed = 80.0
        turn_speed_cost = w_turn_speed * max(0.0, v - v_turn_max) ** 2


        # ========== VOID HEADING COST ==========
        left_space = left
        right_space = right

        void_dir = math.atan2(left_space - right_space, 1.0)
        yaw_err_void = self._wrap_angle(yaw - void_dir)

        void_asym = abs(left_space - right_space) / max(left_space + right_space, 1.0)

        w_void_heading = 50.0
        speed_gate = max(0.0, 1.2 - v) / 1.2

        void_heading_cost = (
            w_void_heading
            * speed_gate
            * (void_asym ** 1.0)
            * (yaw_err_void ** 2)
        )


        # ========== VOID ATTRACTION COST ==========
        void_ratio = max(left, right) / max(min(left, right), 1e-3)

        w_void = 6.0
        void_cost = w_void * max(0.0, void_ratio - 2.5)**2


        # ===== VOID-BASED STEERING RELAXATION =====
        void_scale = min(1.0, void_asym / 0.6)


        # ========== CONTROL SMOOTHNESS COST ==========
        w_delta_eff = self.w_delta * (1.0 - 0.6 * void_scale)
        steering_cost = w_delta_eff * delta ** 2

        w_ddelta_eff = (
            self.w_ddelta
            * (1.0 - 0.7 * turn_strength)
            * (1.0 - 0.8 * void_scale)
        )
        steering_rate_cost = w_ddelta_eff * (delta - delta_prev) ** 2

        w_vtrack = 0.3
        speed_track_cost = w_vtrack * ((v_cmd - v) ** 2)

        w_dv_eff = self.w_dv * (0.3 + 0.7 * min(1.0, v / 2.0))
        accel_cmd_cost = w_dv_eff * ((v_cmd - v_prev_cmd) ** 2)


        # ========== HEADING AWAY FROM WALL ==========
        heading_wall_cost = 0.0

        if v < 1.0:
            left_fwd = self._ray_range(yaw + 0.6)
            right_fwd = self._ray_range(yaw - 0.6)

            if left_fwd > right_fwd:
                desired_yaw = yaw + 0.6
            else:
                desired_yaw = yaw - 0.6

            yaw_err_free = self._wrap_angle(yaw - desired_yaw)

            if fwd < 0.9:
                gain = 12.0
            elif fwd < 1.2:
                gain = 6.0
            else:
                gain = 0.0

            heading_wall_cost = gain * (yaw_err_free ** 2)


        # ========== STALLING COST ==========
        stall_cost = 0.0
        if v < 0.6 and fwd < 1.0:
            stall_cost = 10.0 * (0.6 - v)


        # ========== ESCAPE TURN DIRECTION ==========
        escape_heading_cost = 0.0

        if fwd < 0.8 and v < 1.2:
            yaw_l = yaw + 0.6
            yaw_r = yaw - 0.6

            fwd_l = self._ray_range(yaw_l)
            fwd_r = self._ray_range(yaw_r)

            desired_yaw = yaw_l if fwd_l > fwd_r else yaw_r
            yaw_err_escape = self._wrap_angle(yaw - desired_yaw)

            w_escape = 80.0
            escape_heading_cost = w_escape * (yaw_err_escape ** 2)


        # ========== TELEOP PATH SIMILARITY (soft) ==========
        teleop_cost = 0.0

        if (
            not self.teleop_mppi_enable
            and self.teleop_traj is not None
            and self.teleop_last_time is not None
        ):
            age = (self.get_clock().now() - self.teleop_last_time).nanoseconds * 1e-9
            if age <= self.teleop_timeout_s:
                d_lat = self._teleop_lateral_distance(x, y)
                w_eff = self.w_teleop * self.teleop_confidence * (1.0 - risk) ** 2
                if d_lat > self.teleop_deadzone:
                    teleop_cost = w_eff * (d_lat - self.teleop_deadzone) ** 2


        # ========== TOTAL COST ==========
        total_cost = (
            speed_risk_cost
            + side_barrier
            + center_corridor_cost
            + heading_turn_cost
            + outer_wall_bias
            + apex_heading_cost
            + steering_cost
            + steering_rate_cost
            + speed_track_cost
            + accel_cmd_cost
            + void_cost
            + continuation_cost
            + progress_reward
            + heading_wall_cost
            + stall_cost
            + turn_speed_cost
            + void_heading_cost
            + escape_heading_cost
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
        Preserves continuity and avoids holding zero speed.
        
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


    def _teleop_mppi_enable_cb(self, msg: Bool):
        self.teleop_mppi_enable = bool(msg.data)

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


    # ========== OPTIONAL ROLLOUT VISUALIZATION ==========
    def _publish_rollout_markers(self) -> None:
        """
        Publish MPPI rollout samples as a MarkerArray for visualization/debugging.

        Uses self._last_rollout_trajs, self._last_rollout_costs, and self._last_rollout_best_idx
        which are populated by _run_mppi_plan().
        """
        if self._last_rollout_trajs is None or self._last_rollout_costs is None:
            return

        trajs = self._last_rollout_trajs
        costs = self._last_rollout_costs
        best_idx = int(self._last_rollout_best_idx)

        if len(trajs) == 0:
            return

        max_k = max(1, int(self.rollout_viz_max_k))
        t_stride = max(1, int(self.rollout_viz_t_stride))

        # subselect rollouts for performance (uniform sampling across K)
        K = len(trajs)
        if K <= max_k:
            k_indices = list(range(K))
        else:
            step = float(K) / float(max_k)
            k_indices = [min(K - 1, int(i * step)) for i in range(max_k)]

        # compute robust cost scaling for alpha mapping
        finite_costs = [c for c in costs if c < math.inf]
        if len(finite_costs) == 0:
            c_min = 0.0
            c_max = 1.0
        else:
            c_min = min(finite_costs)
            c_max = max(finite_costs)
            if abs(c_max - c_min) < 1e-6:
                c_max = c_min + 1e-6

        msg = MarkerArray()

        # clear previous markers (RViz otherwise keeps old ids/namespaces)
        clear = Marker()
        clear.header.frame_id = "ego_racecar/base_link"
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "mppi_rollouts"
        clear.id = 0
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        stamp = self.get_clock().now().to_msg()

        mid = 1
        for k in k_indices:
            traj = trajs[k]
            if not traj:
                continue

            m = Marker()
            m.header.frame_id = "ego_racecar/base_link"
            m.header.stamp = stamp
            m.ns = "mppi_rollouts"
            m.id = mid
            mid += 1

            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.015

            # lifetime so markers decay naturally even if a frame is missed
            m.lifetime.sec = int(self.rollout_viz_lifetime_s)
            m.lifetime.nanosec = int((self.rollout_viz_lifetime_s - int(self.rollout_viz_lifetime_s)) * 1e9)

            # cost > alpha (better cost = brighter/less transparent)
            c = costs[k]
            if c >= math.inf:
                alpha = 0.05
            else:
                u = (c - c_min) / (c_max - c_min)  # 0 best, 1 worst
                u = self._clamp(u, 0.0, 1.0)
                alpha = 0.10 + 0.70 * (1.0 - u)    # 0.80 best, 0.10 worst

            # highlight best trajectory among the rollouts
            if k == best_idx:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 1.0
                m.scale.x = 0.025
            else:
                m.color.r = 0.2
                m.color.g = 0.6
                m.color.b = 1.0
                m.color.a = float(alpha)

            # add points (optionally downsample along time)
            for i, (x, y) in enumerate(traj):
                if (i % t_stride) != 0:
                    continue
                m.points.append(Point(x=float(x), y=float(y), z=0.0))

            # at least 2 points to render line
            if len(m.points) >= 2:
                msg.markers.append(m)

        self.rollout_markers_pub.publish(msg)
        

def main():
    rclpy.init()
    node = MppiPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
