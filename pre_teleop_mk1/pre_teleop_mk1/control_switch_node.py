#!/usr/bin/env python3

"""
control_switch_node.py used to decide which control method to use while driving the car.

Uses MPPI control when teleop control is determined to be unsafe. Avoids obstacles.
"""

import math
from typing import Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
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

# ========== VARIABLES ==========
MODE_TELEOP = 0
MODE_MPPI = 1

# ==============================
# NODE CLASS
# ==============================
class ControlSwitchNode(Node):
    def __init__(self) -> None:
        super().__init__("control_switch_node")

        # ========== VARIABLES ==========
        # switch-to-MPPI thresholds (more aggressive)
        self.ttc_switch_to_mppi = 0.2
        self.clear_switch_to_mppi = 0.12

        # switch-back-to-teleop thresholds (hysteresis; more lenient)
        self.ttc_switch_to_teleop = 0.8
        self.clear_switch_to_teleop = 0.2

        # "safe" for this long before returning to teleop
        self.safe_confirm_time_s = 0.05

        # danger to persist for N control cycles before switching to MPPI (debounce)
        self.danger_confirm_cycles = 2

        # staleness thresholds
        self.max_cmd_age_s = 0.25
        self.max_pred_age_s = 0.45

        # publish rate
        self.publish_rate_hz = 30.0

        # output clamping/feasibility
        self.delta_abs_max = 0.418
        self.speed_abs_max = 8.0

        # gate collision by time
        self.collision_time_gate_s = 0.12

        # make takeover slightly harder for strong teleop steering
        self.turn_grace_delta_rad = 0.22
        self.turn_grace_scale = 0.65

        # ========== STATE ==========
        self.mode = MODE_TELEOP

        self.last_teleop_cmd: Optional[AckermannDriveStamped] = None
        self.last_teleop_cmd_time: Optional[Time] = None

        self.last_mppi_cmd: Optional[AckermannDriveStamped] = None
        self.last_mppi_cmd_time: Optional[Time] = None

        # teleop prediction
        self.pred_vals: Dict[str, float] = {}
        self.pred_collision_predicted: bool = True  # default conservative
        self.last_pred_time: Optional[Time] = None

        # hysteresis timer
        self.safe_since_time: Optional[Time] = None

        # debouncer for switching into mppi
        self.danger_count: int = 0

        # ========== SUBS & PUBS ==========
        self.create_subscription(
            AckermannDriveStamped, 
            "/drive_teleop", 
            self._teleop_cb, 
            qos_profile_sub
        )

        self.create_subscription(
            AckermannDriveStamped, 
            "/drive_mpc", 
            self._mppi_cb, 
            qos_profile_sub
        )

        self.create_subscription(
            DiagnosticArray, 
            "/teleop_pred", 
            self._pred_cb, 
            qos_profile_sub
        )

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped, 
            "/drive_cmd", 
            qos_profile_pub
        )

        self.state_pub = self.create_publisher(
            DiagnosticArray, 
            "/switch_state", 
            qos_profile_pub
        )

        period = 1.0 / max(1.0, self.publish_rate_hz)
        self.create_timer(period, self._tick)
        self.get_logger().info("control_switch_node up: /drive_cmd arbitration running")

    # ==============================
    # CORE ARBITRATION LOOP
    # ==============================
    def _tick(self) -> None:
        """
        Periodic arbitration loop for shared autonomy control.

        Evaluates teleop safety predictions and command freshness conditions.
        Decide whether teleop or MPPI control should be active.
        Publishes selected drive command and diagnostic state each cycle.
        """
        now = self.get_clock().now()

        teleop_fresh = self._is_fresh(self.last_teleop_cmd_time, now, self.max_cmd_age_s)
        mppi_fresh = self._is_fresh(self.last_mppi_cmd_time, now, self.max_cmd_age_s)
        pred_fresh = self._is_fresh(self.last_pred_time, now, self.max_pred_age_s)

        # If prediction is stale, assume unsafe (conservative behaviour)
        collision_pred = self.pred_collision_predicted if pred_fresh else True
        ttc = float(self.pred_vals.get("ttc_min", 0.0)) if pred_fresh else 0.0
        clear = float(self.pred_vals.get("min_clearance", 0.0)) if pred_fresh else 0.0
        collision_time = float(self.pred_vals.get("collision_time", math.inf)) if pred_fresh else 0.0

        # ========== STEERING GRACE ==========
        # relax takeover thresholds during teleop steering
        delta_now = 0.0
        if self.last_teleop_cmd is not None:
            delta_now = abs(float(self.last_teleop_cmd.drive.steering_angle))

        # mild steering: turn_factor = 0
        # near max steering: turn_factor = 1
        tg = self._clamp(self.turn_grace_delta_rad, 0.0, max(1e-3, self.delta_abs_max))
        den = max(1e-6, (self.delta_abs_max - tg))
        turn_factor = self._clamp((delta_now - tg) / den, 0.0, 1.0)

        # takeover trigger thresholds:
        # smaller thresholds: harder to trip
        ttc_mppi_eff = self.ttc_switch_to_mppi * (1.0 - self.turn_grace_scale * turn_factor)
        clear_mppi_eff = self.clear_switch_to_mppi * (1.0 - self.turn_grace_scale * turn_factor)

        # collision predicted counts only when in close proximity to obstacles
        collision_gate_eff = self.collision_time_gate_s * (1.0 - 0.5 * turn_factor)

        danger_raw = (
            (ttc <= ttc_mppi_eff)
            or (clear <= clear_mppi_eff)
            or (collision_pred and (collision_time <= collision_gate_eff))
        )

        safe = (
            (not collision_pred)
            and (ttc >= self.ttc_switch_to_teleop)
            and (clear >= self.clear_switch_to_teleop)
        )

        # ========== DANGER PERSISTENCE ==========
        if danger_raw:
            self.danger_count = min(self.danger_confirm_cycles, self.danger_count + 1)
        else:
            self.danger_count = 0

        danger = (self.danger_count >= self.danger_confirm_cycles)

        # ========== MODE TRANSITION ==========
        if self.mode == MODE_TELEOP:
            if danger:
                # switch to mppi if available, otherwise continue teleop (AEB safe)
                if mppi_fresh:
                    self.mode = MODE_mppi
                    self.safe_since_time = None
        else: 
            if safe:
                if self.safe_since_time is None:
                    self.safe_since_time = now
                else:
                    dt_ok = (now - self.safe_since_time).nanoseconds * 1e-9
                    if dt_ok >= self.safe_confirm_time_s:
                        # switch back to teleop if available
                        if teleop_fresh:
                            self.mode = MODE_TELEOP
                        self.safe_since_time = None
            else:
                self.safe_since_time = None

        # choose control
        chosen = None
        chosen_src = "none"

        if self.mode == MODE_TELEOP:
            if teleop_fresh and self.last_teleop_cmd is not None:
                chosen = self.last_teleop_cmd
                chosen_src = "teleop"
            elif mppi_fresh and self.last_mppi_cmd is not None:
                chosen = self.last_mppi_cmd
                chosen_src = "mppi_fallback"
        else:
            if mppi_fresh and self.last_mppi_cmd is not None:
                chosen = self.last_mppi_cmd
                chosen_src = "mppi"
            elif teleop_fresh and self.last_teleop_cmd is not None:
                chosen = self.last_teleop_cmd
                chosen_src = "teleop_fallback"

        # Debug Logging
        # if chosen_src == "mppi" or chosen_src == "mppi_fallback":
        #     self.get_logger().info("\033[96mteleop state: \033[0m" + chosen_src)
        # else:
        #     self.get_logger().info("\033[92mteleop state: \033[0m" + chosen_src)

        # if nothing is usable, publish hard stop
        if chosen is None:
            self._publish_cmd(now, steering=0.0, speed=0.0 )
            self._publish_state(
                now, chosen_src,
                teleop_fresh, mppi_fresh, pred_fresh,
                ttc, clear, collision_pred,
                collision_time, delta_now, turn_factor,
                ttc_mppi_eff, clear_mppi_eff, collision_gate_eff
            )
            return

        # pass through selected command with clamps
        delta = float(chosen.drive.steering_angle)
        speed = float(chosen.drive.speed)

        delta = self._clamp(delta, -self.delta_abs_max, self.delta_abs_max)
        speed = self._clamp(speed, 0.0, self.speed_abs_max)

        frame_id = (
            chosen.header.frame_id
            if chosen.header.frame_id
            else "ego_racecar/base_link"
        )

        self._publish_cmd(now, steering=delta, speed=speed, frame_id=frame_id)
        self._publish_state(
            now, chosen_src,
            teleop_fresh, mppi_fresh, pred_fresh,
            ttc, clear, collision_pred,
            collision_time, delta_now, turn_factor,
            ttc_mppi_eff, clear_mppi_eff, collision_gate_eff
        )

    # ==============================
    # UTILITIES & CALLBACKS
    # ==============================

    # ========== UTILITIES ==========
    def _clamp(self, v: float, vmin: float, vmax: float) -> float:
        """Value clamping"""
        return max(vmin, min(vmax, v))

    def _is_fresh(self, last_time: Optional[Time], now: Time, max_age_s: float) -> bool:
        """Check whether a timestamp is within an allowed age window."""
        if last_time is None:
            return False
        return (now - last_time).nanoseconds <= int(max_age_s * 1e9)

    # ========== CALLBACKS ==========
    def _teleop_cb(self, msg: AckermannDriveStamped) -> None:
        """Store latest teleop command and update its receive timestamp."""
        self.last_teleop_cmd = msg
        self.last_teleop_cmd_time = self.get_clock().now()

    def _mppi_cb(self, msg: AckermannDriveStamped) -> None:
        """Store latest MPPI command and update its receive timestamp."""
        self.last_mppi_cmd = msg
        self.last_mppi_cmd_time = self.get_clock().now()

    def _pred_cb(self, msg: DiagnosticArray) -> None:
        """Parse teleop prediction diagnostics and update internal safety state."""
        found = None
        for st in msg.status:
            if st.name == "teleop_pred":
                found = st
                break
        if found is None:
            return

        now = self.get_clock().now()
        self.last_pred_time = now

        # Parse key-values
        vals = {}
        for kv in found.values:
            try:
                # collision_predicted is a bool-like string; keep separately
                if kv.key == "collision_predicted":
                    self.pred_collision_predicted = (kv.value.strip().lower() == "true")
                else:
                    vals[kv.key] = float(kv.value)
            except Exception:
                # ignore bad entries
                pass

        self.pred_vals = vals    

    # ==============================
    # PUBLISHERS
    # ==============================
    def _publish_cmd(
        self,
        now: Time,
        steering: float,
        speed: float,
        frame_id: str = "ego_racecar/base_link",
    ) -> None:
        """Publishes the selected drive command after arbitration and clamping."""
        
        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = frame_id
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)

        self.cmd_pub.publish(msg)
        
    def _publish_state(
        self,
        now: Time,
        chosen_src: str,
        teleop_fresh: bool,
        mppi_fresh: bool,
        pred_fresh: bool,
        ttc: float,
        clear: float,
        collision_pred: bool,
        collision_time: float,
        delta_now: float,
        turn_factor: float,
        ttc_mppi_eff: float,
        clear_mppi_eff: float,
        collision_gate_eff: float,
    ) -> None:
        """Publish diagnostic information describing the current arbitration state."""

        msg = DiagnosticArray()
        msg.header.stamp = now.to_msg()

        st = DiagnosticStatus()
        st.name = "control_switch"
        st.level = DiagnosticStatus.OK
        st.message = "arbitration"

        mode_str = "teleop" if self.mode == MODE_TELEOP else "mppi"
        st.values = [
            KeyValue(key="mode", value=mode_str),
            KeyValue(key="chosen_src", value=chosen_src),
            KeyValue(key="teleop_fresh", value=str(bool(teleop_fresh))),
            KeyValue(key="mppi_fresh", value=str(bool(mppi_fresh))),
            KeyValue(key="pred_fresh", value=str(bool(pred_fresh))),
            KeyValue(key="ttc_min", value=f"{ttc:.6f}"),
            KeyValue(key="min_clearance", value=f"{clear:.6f}"),
            KeyValue(key="collision_predicted", value=str(bool(collision_pred))),
            KeyValue(key="ttc_switch_to_mppi", value=f"{self.ttc_switch_to_mppi:.3f}"),
            KeyValue(key="ttc_switch_to_teleop", value=f"{self.ttc_switch_to_teleop:.3f}"),
            KeyValue(key="clear_switch_to_mppi", value=f"{self.clear_switch_to_mppi:.3f}"),
            KeyValue(key="clear_switch_to_teleop", value=f"{self.clear_switch_to_teleop:.3f}"),
            KeyValue(key="danger_count", value=str(int(self.danger_count))),
            KeyValue(key="danger_confirm_cycles", value=str(int(self.danger_confirm_cycles))),
            KeyValue(key="delta_now_abs", value=f"{delta_now:.4f}"),
            KeyValue(key="turn_factor", value=f"{turn_factor:.4f}"),
            KeyValue(key="ttc_mppi_eff", value=f"{ttc_mppi_eff:.4f}"),
            KeyValue(key="clear_mppi_eff", value=f"{clear_mppi_eff:.4f}"),
            KeyValue(key="collision_time", value=f"{collision_time:.4f}"),
            KeyValue(key="collision_time_gate_eff", value=f"{collision_gate_eff:.4f}")
        ]

        msg.status.append(st)
        self.state_pub.publish(msg)

def main():
    rclpy.init()
    node = ControlSwitchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
