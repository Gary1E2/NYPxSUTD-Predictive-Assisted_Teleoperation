#!/usr/bin/env python3

"""
mppi_planner_node.py used to plan, simulate and evaluate trajectories before publishing control actions.

Uses:
- ray-based collision and clearance check from /scan_filtered topic.
- simulated teleoperation trajectory from /teleop_pred topic.
"""

import math
import time
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
        self.K = 64
        self.dt = 0.05
        self.lambd = 0.65

        self.sigma_delta = 0.015
        self.sigma_v = 0.3

        # Dilation and horizon clamping
        self.dt_base = 0.05          # nominal timestep (low speed)
        self.dt_max = 0.10           # max timestep at high speed
        self.v_dt_scale = 2.5        # speed where dt starts saturating

        self.T_min = 6               # minimum horizon steps
        self.T_max = 12              # maximum horizon steps (HARD CLAMP)


        # Noise temporal correlation (higher = smoother exploration)
        self.rho_delta = 0.9  # 0.6–0.9
        self.rho_v = 0.65     # 0.4–0.8

        # Steering and velocity limits
        self.delta_max = 0.418
        self.v_min = 0.0
        self.v_max = 10.0

        self.wheelbase_L = 0.33
        self.k_v = 2.0

        self.plan_period = 0.05

        # Post-update smoothing (reduces trajectory wiggle without "clamping")
        self.u_smooth_alpha = 0.25  # 0.15–0.35

        # Cost weights
        self.collision_penalty = 2e6
        self.w_delta = 0.3
        self.w_ddelta = 10.0
        self.w_progress = 15.0
        self.w_speed_risk = 6.0

        # speed command smoothness
        self.w_dv = 2.0  # 1–4

        self.w_heading = 4.0
        self.w_turn = 16.0

        # Directional exploration (bias toward free space)
        self.dir_explore_gain = 0.08      # 0.05–0.30
        self.dir_explore_ts_min = 0.25    # activate only when turning
        self.dir_explore_max = 0.7       # clamp (radians)

        # Collision constraint
        self.collision_radius = 0.15

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
        self.teleop_assist_level = 0.0

        self.w_teleop = 20.0        # 15–30
        self.teleop_deadzone = 0.12

        # Teleop-following cost params
        self.w_teleop = 4.0           # 2–6
        self.teleop_deadzone = 0.25   # meters

        self.teleop_mppi_enable = False

        # scan cache for precomputation
        self.scan_len = 0
        self.inv_scan_angle_inc = 0.0
        self.scan_ready = False

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
        t0 = time.perf_counter()

        # Sensor check
        if (not self.scan_ready) or self.last_odom_time is None:
            return

        # Turn strength check (uses new fast _ray_range automatically)
        self.turn_strength_now = self._estimate_turn_strength_now()

        # ===== NEW: local alias (tiny overhead reduction) =====
        run_plan = self._run_mppi_plan
        plan = run_plan(self.current_speed, self.last_delta_cmd, self.last_speed_cmd)

        if plan is None:
            self._publish_drive_command(0.0, 0.0)
            return

        controls, best_traj = plan
        delta_cmd, speed_cmd = controls[0]

        self.last_delta_cmd = float(delta_cmd)
        self.last_speed_cmd = float(speed_cmd)

        dt_s = (time.perf_counter() - t0)
        # self.get_logger().info(f"MPPI compute: {dt_s:.6f} s")

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

        # ===== NEW: local aliases (reduce attribute lookup overhead) =====
        u_bar = self.u_bar
        rollout_step = self._rollout_step
        gauss = random.gauss

        # ===== NEW: inline clamp for hot loop =====
        delta_max = self.delta_max
        v_min = self.v_min
        v_max = self.v_max
        def clamp_fast(v, vmin, vmax):
            return vmin if v < vmin else vmax if v > vmax else v

        # ===== NEW: precompute correlated noise factors once =====
        rho_d = clamp_fast(self.rho_delta, 0.0, 0.95)
        rho_v = clamp_fast(self.rho_v, 0.0, 0.95)
        s_d = math.sqrt(max(1e-6, 1.0 - rho_d * rho_d))
        s_v = math.sqrt(max(1e-6, 1.0 - rho_v * rho_v))

        # ===== NEW: sigma_d depends only on current turn_strength_now (not on k/t) =====
        sigma_d_base = self.sigma_delta * (1.0 + 4.0 * self.turn_strength_now)
        sigma_v = self.sigma_v

        # valid/invalid counts
        K_target = self.K
        K_min_valid = 8        # your chosen minimum
        K_max_attempts = 32  # safety cap

        valid_count = 0
        invalid_count = 0
        attempts = 0

        # ========== DIRECTIONAL EXPLORATION BIAS ==========
        # Estimate preferred steering direction based on current geometry
        # Positive bias → steer left, Negative → steer right

        if self.turn_strength_now > self.dir_explore_ts_min:
            # Reuse fast ray queries
            ray = self._ray_range

            # Look sideways at current pose (yaw = 0 in rollout frame)
            left_space = max(ray(math.pi/2), ray(math.pi/2 + 0.12), ray(math.pi/2 - 0.12))
            right_space = max(ray(-math.pi/2), ray(-math.pi/2 + 0.12), ray(-math.pi/2 - 0.12))

            # Bias toward open side
            dir_bias = self.dir_explore_gain * (left_space - right_space)

            # Clamp so this NEVER becomes a hard turn
            if dir_bias > self.dir_explore_max:
                dir_bias = self.dir_explore_max
            elif dir_bias < -self.dir_explore_max:
                dir_bias = -self.dir_explore_max
        else:
            dir_bias = 0.0

        ts = self.turn_strength_now
        if ts > self.dir_explore_ts_min:
            ts_gate = (ts - self.dir_explore_ts_min) / max(1e-6, 1.0 - self.dir_explore_ts_min)
            if ts_gate > 1.0:
                ts_gate = 1.0
            dir_bias *= ts_gate
        else:
            dir_bias = 0.0

        # ===== DIRECTIONAL BIAS DECAY BY HEADING ALIGNMENT =====
        # If we're already turning in the desired direction, reduce bias

        yaw_err = abs(delta0) / max(self.delta_max, 1e-3)  # normalized [0,1]

        # fade bias as steering angle grows
        heading_gate = max(0.0, 1.0 - yaw_err)
        heading_gate = heading_gate * heading_gate 

        dir_bias *= heading_gate

        min_bias = 0.25 * sigma_d_base
        if abs(dir_bias) < min_bias:
            dir_bias = math.copysign(min_bias, dir_bias)

        self.dir_bias = dir_bias
        # if random.random() < 0.1:
        #     self.get_logger().info(
        #         f"[DIR-EXPLORE] ts={self.turn_strength_now:.2f} "
        #         f"bias={dir_bias:.3f} "
        #         f"sigma_d={sigma_d_base:.3f}"
        #     )

        # NOTE: K == amount of trajectories/rollouts to evaluate
        k = 0
        while k < K_target and attempts < K_max_attempts:
            attempts += 1
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

            # ===== TRAJECTORY-LEVEL METRICS =====
            min_fwd_clear = math.inf
            mean_fwd_clear = 0.0
            fwd_count = 0

            eps_d_prev = 0.0
            eps_v_prev = 0.0

            for t in range(T):
                # ===== NEW: use precomputed sigmas + local gauss =====
                # Directional exploration: biased Gaussian
                # Bias only early in the horizon
                time_gate = math.exp(-1.2 * t / max(1, T))

                w_d = gauss(dir_bias * time_gate, sigma_d_base)
                w_v = gauss(0.0, sigma_v)

                eps_d = rho_d * eps_d_prev + s_d * w_d
                eps_v = rho_v * eps_v_prev + s_v * w_v

                eps_d_prev = eps_d
                eps_v_prev = eps_v
                eps_k.append((eps_d, eps_v))

                d_nom, v_nom = u_bar[t]

                # ===== NEW: inline clamp (no function call) =====
                delta = d_nom + eps_d
                if delta > delta_max:
                    delta = delta_max
                elif delta < -delta_max:
                    delta = -delta_max

                v_cmd = v_nom + eps_v
                if v_cmd > v_max:
                    v_cmd = v_max
                elif v_cmd < v_min:
                    v_cmd = v_min

                state, step_cost, collided, delta_prev, v_prev_cmd = rollout_step(
                    state, delta, v_cmd, delta_prev, v_prev_cmd
                )

                # ----- forward clearance tracking -----
                fwd_clear = self._ray_range(state[2])   # yaw = state[2]
                min_fwd_clear = min(min_fwd_clear, fwd_clear)
                mean_fwd_clear += fwd_clear
                fwd_count += 1

                traj.append((state[0], state[1]))
                cost += step_cost

                if collided:
                    valid = False
                    break
            
            # valid/invalid counts
            if valid:
                valid_count += 1
                k += 1

                # Ensure MPPI invariant
                if len(eps_k) < T:
                    eps_k.extend([(0.0, 0.0)] * (T - len(eps_k)))

                # ===== DEAD-END TRAJECTORY PENALTY =====
                if fwd_count > 0:
                    mean_fwd_clear /= fwd_count

                # Penalize trajectories whose forward clearance collapses
                # Dead ends → very small min_fwd_clear
                if min_fwd_clear < 0.9:
                    w_deadend = 28.0    # START 20–40
                    cost += w_deadend * (0.9 - min_fwd_clear)**2

                # Optional: reward long survival corridors
                cost -= 4.0 * mean_fwd_clear

                trajectories.append(traj)
                costs.append(cost)
                epsilons.append(eps_k)

                if cost < best_cost:
                    best_cost = cost
                    best_idx = len(trajectories) - 1

            else:
                invalid_count += 1
                continue  # DISCARD and resample

        # ========== VALID + INVALID ROLLOUT COUNTS ==========
        # if valid_count < K_min_valid:
        #     self.get_logger().warn(
        #         f"[MPPI LOW-SAMPLES] valid={valid_count} < {K_min_valid}, "
        #         f"proceeding with reduced sample set"
        #     )

        # self.get_logger().info(
        #     f"[MPPI RESAMPLE] valid={valid_count}/{K_target} "
        #     f"attempts={attempts} invalid={invalid_count}"
        # )

        # total = valid_count + invalid_count
        # frac = valid_count / max(total, 1)
        # self.get_logger().info(
        #     f"[MPPI VALIDITY] K={total} | valid={valid_count} | invalid={invalid_count} | valid_frac={frac:.2f}"
        # )

        if best_idx < 0 or best_idx >= len(trajectories):
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
            K_eff = len(epsilons)
            for i in range(K_eff):
                du += weights[i] * epsilons[i][t][0]
                dv += weights[i] * epsilons[i][t][1]

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

        # ===== NEW: local aliases/constants (avoid attribute lookup each time) =====
        dt = self.dt
        k_v = self.k_v
        v_min = self.v_min
        v_max = self.v_max
        L = self.wheelbase_L

        ray = self._ray_range           # fast now
        wrap = self._wrap_angle         # fast now

        def clamp_fast(val, lo, hi):
            return lo if val < lo else hi if val > hi else val

        # ===== NEW: inline clamp dynamics =====
        v = v + k_v * (v_cmd - v) * dt
        v = clamp_fast(v, v_min, v_max)

        # ===== NEW: compute cos/sin once =====
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        x += v * cy * dt
        y += v * sy * dt
        yaw += (v / L) * math.tan(delta) * dt

        if self._collides_disc_ray(x, y, yaw):
            return (x, y, yaw, v), self.collision_penalty, True, delta, v_cmd

        # ========== TURN INTENT (fast, unrolled) ==========
        lp = yaw + math.pi * 0.5
        rp = yaw - math.pi * 0.5

        l0 = ray(lp)
        l1 = ray(lp + 0.12)
        l2 = ray(lp - 0.12)
        left = l0 if l0 > l1 else l1
        left = left if left > l2 else l2

        r0 = ray(rp)
        r1 = ray(rp + 0.12)
        r2 = ray(rp - 0.12)
        right = r0 if r0 > r1 else r1
        right = right if right > r2 else r2

        turn_intent = left - right
        if turn_intent < 0.0:
            turn_intent = -turn_intent

        ts_raw = turn_intent
        if ts_raw > 1.0:
            ts_raw = 1.0

        if ts_raw < 0.35:
            turn_strength = 0.0
        elif ts_raw > 0.75:
            turn_strength = 1.0
        else:
            turn_strength = (ts_raw - 0.35) * (1.0 / 0.4)

        # ===== NEW: local aliases for weights/constants (cuts attribute lookup) =====
        w_speed_risk = self.w_speed_risk
        w_progress   = self.w_progress
        w_delta      = self.w_delta
        w_ddelta     = self.w_ddelta
        w_dv         = self.w_dv

        # ===== NEW: precompute turn-strength gates once =====
        ts = turn_strength
        ts_risk  = 1.0 - 0.85 * ts          # used in speed risk + progress
        ts_gate1 = 1.0 - 0.8 * ts           # first gating
        ts_gate2 = 1.0 - 0.6 * ts
        ts_gate3 = 1.0 - ts                 # second gating
        ts_gate4 = 1.0 - 0.85 * ts

        # Debug logging
        # if random.random() < 0.002:
        #     self.get_logger().info(f"turn_strength={turn_strength:.2f} left={left:.2f} right={right:.2f} yaw={yaw:.2f}")

        # ========== FORWARD RISK ==========
        lookahead = 1.2 + 0.35 * v
        lookahead *= (1.0 - 0.35 * ts)
        if lookahead < 0.8:
            lookahead = 0.8
        elif lookahead > 3.5:
            lookahead = 3.5

        # ===== NEW: unrolled forward clearance (no loop) =====
        free = 0
        free += (ray(yaw - 0.15) > lookahead)
        free += (ray(yaw - 0.07) > lookahead)
        free += (ray(yaw)        > lookahead)
        free += (ray(yaw + 0.07) > lookahead)
        free += (ray(yaw + 0.15) > lookahead)
        free_frac = free * 0.2
        risk = 1.0 - free_frac

        risk_scale = ts_risk  # 1 - 0.85*ts

        dv = v - 0.6
        if dv > 0.0:
            dv2 = dv * dv
        else:
            dv2 = 0.0

        dr = risk - 0.4
        if dr > 0.0:
            speed_risk_cost = risk_scale * w_speed_risk * dv2 * dr
        else:
            speed_risk_cost = 0.0

        straight_conf = free_frac           # 0 → cluttered, 1 → wide open
        speed_gate = max(0.0, (straight_conf - 0.6) / 0.4)  # active only when >60% free

        speed_risk_cost *= (1.0 - speed_gate)

        # ========== CORRIDOR CENTER COST ==========
        # corridor geometry
        d_safe = 0.55  # meters; tune 0.45–0.70 depending on track width
        w_side = 18.0  # start 8–20

        dl = d_safe - left
        if dl < 0.0: dl = 0.0
        dr = d_safe - right
        if dr < 0.0: dr = 0.0
        side_barrier = w_side * (dl * dl * dl + dr * dr * dr)
        side_barrier *= (1.0 - 0.6*ts)

        # If right > left, likely a left turn -> bias slightly RIGHT (positive center_offset).
        turn_sign = 1.0 if (left - right) > 0.0 else -1.0

        k_des = 0.22  # meters-ish proxy scale (start 0.12–0.25)

        desired_offset = (-turn_sign) * (k_des * ts)

        if desired_offset > 0.35:
            desired_offset = 0.35
        elif desired_offset < -0.35:
            desired_offset = -0.35

        # ==========  HEADING TURN COST ==========
        # ===== NEW: linear proxy for corridor yaw (no atan2) =====
        # Equivalent direction cue: positive if more space on +0.4 than -0.4
        d_corr = ray(yaw + 0.4) - ray(yaw - 0.4)
        corridor_yaw = math.atan2(d_corr, 0.8)   # radians
        yaw_error = wrap(yaw - corridor_yaw)
        yaw_error2 = yaw_error * yaw_error

        heading_turn_cost = 5.0 * ts * yaw_error2

        # ========== PROGRESS REWARD ==========
        progress = v * math.cos(yaw_error)

        w_progress_eff = w_progress * ts_risk
        if progress > 0.0:
            progress_reward = -w_progress_eff * progress
        else:
            progress_reward = 0.0

        # ========== SHARP TURN SPEED COST ==========
        # Turn-limited speed target (tight track)
        v_straight = 4      # allowed in straights (tune)
        v_corner   = 2      # allowed in sharp corners (tune)

        # Smoothly interpolate based on turn strength
        v_turn_max = v_straight - (v_straight - v_corner) * (ts * math.sqrt(ts))

        # Penalize only if exceeding turn max
        w_turn_speed = 15.0 * (1.0 - 0.5 * ts)   # strong, tune 40–200
        dv_turn = v - v_turn_max
        if dv_turn > 0.0:
            turn_speed_cost = w_turn_speed * (dv_turn * dv_turn)
        else:
            turn_speed_cost = 0.0

        # ===== LATERAL VOID ATTRACTION =====
        left_space = left
        right_space = right

        # Stronger when slow (need reorientation) and space is asymmetric
        void_asym = abs(left_space - right_space) / max(left_space + right_space, 1.0)

        # ===== VOID-BASED STEERING RELAXATION =====
        # Strong lateral asymmetry → steering should be cheap
        void_scale = min(1.0, void_asym / 0.6)   # 0.6 = "clearly open side"

        # ========== CONTROL SMOOTHNESS COST ==========
        # Steering rate cost (relax even more aggressively)
        w_ddelta_eff = (
            w_ddelta
            * (1.0 - 0.7 * turn_strength)
            * (1.0 - 0.8 * void_scale)
        )
        w_ddelta_eff = w_ddelta_eff * (1.0 - 0.6 * ts)

        dd = (delta - delta_prev)
        steering_rate_cost = w_ddelta_eff * (dd * dd)

        w_dv_eff = w_dv * (0.3 + 0.7 * min(1.0, v / 2.0))
        dv_cmd = (v_cmd - v_prev_cmd)
        accel_cmd_cost = w_dv_eff * (dv_cmd * dv_cmd)

        # ========== STALL PUSH ==========
        stall_push = 0.0
        if v_cmd < 0.4 and ray(yaw) > 1.2:
            # forward is open but you’re commanding crawl/stop
            stall_push = 8.0 * (0.4 - v_cmd)

        # ========== TURN COMMIT COST ==========
        turn_commit_cost = 0.0

        if ts > 0.6:
            # commit direction from corridor geometry
            sign = 1.0 if (left > right) else -1.0

            # target steering (fraction of max)
            desired_delta = 0.55 * self.delta_max * sign

            # weight (start conservative)
            w_commit = 18.0

            # apply cost
            d_err = delta - desired_delta
            turn_commit_cost = w_commit * (d_err * d_err)

        # ========== POST-TURN OVERSTEER DAMPING (NO FWD REQUIRED) ==========
        post_turn_damping = 0.0

        # Exit phase: turn strength is falling but steering persists
        exit_phase = (ts < 0.5 and ts > 0.15)

        if exit_phase:
            same_sign = (delta * delta_prev) > 0.0

            if same_sign:
                # Stronger as turn strength decays
                decay = (0.5 - ts) / 0.35   # ts=0.5→0, ts=0.15→1
                decay = max(0.0, min(1.0, decay))

                w_exit = 8.0    # START 6–12
                post_turn_damping = w_exit * decay * (delta * delta)

        # ========== DEAD-END / CONTINUATION COST ==========
        deadend_cost = 0.0

        # Forward clearance collapse detector
        # fwd_clear is ray(yaw) or equivalent
        fwd_clear = ray(yaw)

        # Expected minimum clearance based on speed
        expected = 0.8 + 0.4 * v    # tune 0.6–1.2 base

        # Penalize if forward space is shrinking too fast
        gap = expected - fwd_clear
        if gap > 0.0:
            w_deadend = 18.0        # START 12–30
            deadend_cost = w_deadend * (gap * gap)

        # ========== TELEOP LOCAL CORRIDOR COST ==========
        teleop_cost = 0.0

        if self._teleop_assist_active():

            assist = self.teleop_assist_level
            assist_eff = assist  # no clearance gating

            # 1) Nearest-point distance pull (LOCAL ONLY)
            d_ref = self._teleop_reference_distance(x, y, max_pts=15)

            if d_ref < 1.0:  # hard locality bound (critical)
                d = max(0.0, d_ref - self.teleop_deadzone)
                teleop_cost += assist_eff * self.w_teleop * (d * d)

            # 2) Local lateral corridor constraint
            lat_err = self._teleop_lateral_error(x, y)

            w_lat = 6.0 * assist_eff
            teleop_cost += w_lat * (lat_err * lat_err)

        # ========== TOTAL COST ==========
        total_cost = (
            speed_risk_cost
            + side_barrier
            + heading_turn_cost
            + steering_rate_cost
            + accel_cmd_cost
            + progress_reward
            + turn_speed_cost
            + stall_push
            + turn_commit_cost
            + post_turn_damping
            + deadend_cost
            + teleop_cost
        )

        return (x, y, yaw, v), total_cost, False, delta, v_cmd
    
    # ==============================
    # UTILITIES & CALLBACKS
    # ==============================

    # ========== RAY LOGIC ==========
    def _ray_range(self, theta: float) -> float:
        """
        Gets LiDAR scan using precomputed inv angle increment and scan length.

        Inputs:
            theta:
                Angle from vehicle frame
        Outputs:
            r:
                Distance to nearest obstacle along the scan line
        """
        if (not self.scan_ready) or (self.scan_ranges is None):
            return math.inf

        # fast index
        idx = int((theta - self.scan_angle_min) * self.inv_scan_angle_inc)

        # early return instead of clamping
        if idx < 0 or idx >= self.scan_len:
            return math.inf

        r = self.scan_ranges[idx]

        #  validity check (assumes <= 0 means invalid)
        return r if r > 0.0 else math.inf

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
        r0 = self._ray_range(theta)
        r1 = self._ray_range(theta + dtheta)
        r2 = self._ray_range(theta - dtheta)
        return max(r0, r1, r2)

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

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        # Pre-rotated offsets (same geometry as yours)
        offsets = (
            (x, y),
            (x - r * sy, y + r * cy),
            (x + r * sy, y - r * cy),
        )

        # local alias
        rayw = self._ray_range_window

        for wx, wy in offsets:
            theta = math.atan2(wy, wx)
            r_obs = rayw(theta)

            # squared compare (no hypot)
            lim = (r_obs - 0.05)
            if wx * wx + wy * wy >= lim * lim:
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
        r = self._ray_range
        free = 0
        free += r(yaw - 0.15) > lookahead_m
        free += r(yaw - 0.07) > lookahead_m
        free += r(yaw) > lookahead_m
        free += r(yaw + 0.07) > lookahead_m
        free += r(yaw + 0.15) > lookahead_m
        return free * 0.2
    
    # ========== TURN STRENGTH LOGIC ==========
    def _estimate_turn_strength_now(self) -> float:
        """
        Estimate turn sharpness from LiDAR side disparity (edges/corners)
        Uses side disparity at +/- 90 degrees, robust to mild turns.

        Outputs:
            turn_strength:
                Normalized value [0.0, 1.0] to represent turn strength
        """
        if self.scan_ranges is None:
            return 0.0

        r = self._ray_range
        lp = math.pi / 2
        rp = -lp

        left = max(
            r(lp), r(lp + 0.12), r(lp - 0.12)
        )
        right = max(
            r(rp), r(rp + 0.12), r(rp - 0.12)
        )

        ts = abs(left - right)
        if ts < 0.35:
            return 0.0
        if ts > 0.75:
            return 1.0
        return (ts - 0.35) * (1.0 / 0.4)
    
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

        best = math.inf
        for xr, yr in self.teleop_traj:
            dx = x - xr
            dy = y - yr
            d2 = dx*dx + dy*dy
            if d2 < best:
                best = d2

        return math.sqrt(best)

    def _teleop_assist_active(self) -> bool:
        if not self.teleop_mppi_enable:
            return False
        if self.teleop_traj is None:
            return False
        if self.teleop_last_time is None:
            return False

        age = (self.get_clock().now() - self.teleop_last_time).nanoseconds * 1e-9
        return age <= self.teleop_timeout_s and self.teleop_assist_level > 0.01

    # ========== ADDITIONAL ==========
    def _wrap_angle(self, a: float) -> float:
        """Wraps angle to range [-pi, pi]"""
        return (a + math.pi) % (2.0 * math.pi) - math.pi


    def _compute_horizon_steps(self, v):
        """
        Computes MPPI horizon steps with speed-based dt dilation
        and HARD min/max step clamping.

        Inputs:
            v:
                Current car speed
        Outputs:
            (int value):
                Number of horizon steps
        """
        # dilation with dt
        v_norm = min(abs(v) / self.v_dt_scale, 1.0)
        dt = self.dt_base + v_norm * (self.dt_max - self.dt_base)

        horizon_time = 1.5 + 0.15 * abs(v)      # same intent as before
        horizon_time = min(horizon_time, 3.0)   # hard cap

        # conversion to steps with clamping
        T = int(math.ceil(horizon_time / dt))
        T = max(self.T_min, min(self.T_max, T))

        # cache dt for rollout
        self.dt = dt

        return T


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

        # precomputation for faster indexing
        if self.scan_ranges is not None and self.scan_angle_inc != 0.0:
            self.scan_len = len(self.scan_ranges)
            self.inv_scan_angle_inc = 1.0 / self.scan_angle_inc
            self.scan_ready = (self.scan_len > 0)
        else:
            self.scan_len = 0
            self.inv_scan_angle_inc = 0.0
            self.scan_ready = False

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
        now = self.get_clock().now().nanoseconds * 1e-9
        if not hasattr(self, "_last_speed_log"):
            self._last_speed_log = 0.0

        if now - self._last_speed_log > 0.2:   # 5 Hz logging
            self.get_logger().info(
                f"[SPEED] v_cmd={v:.2f} m/s"
            )
            self._last_speed_log = now

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
