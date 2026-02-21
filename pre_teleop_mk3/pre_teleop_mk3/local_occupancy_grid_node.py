#!/usr/bin/env python3

"""
local_occupancy_grid_node.py used to generate a local grid map consisting of free and occupied space.

Builds a local 2D occupancy grid centered around the ego vehicle
The grid encodes:
- Unknown cells as -1
- Free space as 0
- Occupied space as 100
"""

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

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
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ==============================
# NODE CLASS
# ==============================
class LocalOccupancyGridNode(Node):
    def __init__(self) -> None:
        super().__init__("local_occupancy_grid_node")

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        self.length_m = 12.0
        self.width_m = 8.0
        self.resolution = 0.15
        self.origin_x = -0.5
        self.origin_y = -self.width_m / 2.0
        self.width_cells = math.ceil(self.length_m / self.resolution)
        self.height_cells = math.ceil(self.width_m / self.resolution)
        self.r_max = 8.0


        # ========== SUBS & PUBS ==========
        self.scan_sub = self.create_subscription(
            LaserScan, 
            "/scan_filtered", 
            self.scan_callback, 
            qos_profile_sub
        )

        self.grid_pub = self.create_publisher(
            OccupancyGrid, 
            "/local_occupancy_grid", 
            qos_profile_pub
        )


    # ==============================
    # LiDAR TO GRID CONVERSION
    # ==============================
    def scan_callback(self, scan: LaserScan) -> None:
        """
        Converts a LiDAR scan into a local occupancy grid.

        For each LiDAR ray:
        - Traces free space from vehicle origin to measured range
        - Marks endpoint as occupied if a valid hit occurs
        - Handles rays that terminate outside the grid conservatively

        Inputs:
            scan:
                Filtered LaserScan message in the ego vehicle frame.
        """
        grid = [-1] * (self.width_cells * self.height_cells)

        # precompute effective max range for this local grid
        effective_r_max = min(self.r_max, scan.range_max if scan.range_max > 0.0 else self.r_max)
        range_min = scan.range_min if scan.range_min > 0.0 else 0.0

        angle = scan.angle_min
        for r in scan.ranges:
            # default: assume no hit and trace out to effective_r_max
            has_hit = False
            end_range = effective_r_max

            # only consider finite, positive ranges within valid sensor bounds
            if math.isfinite(r) and r > 0.0:
                # valid hits must be within [range_min, effective_r_max]
                if r >= range_min and r <= effective_r_max:
                    has_hit = True
                    end_range = r
                else:
                    # otherwise no hit: (still trace free space to end_range = effective_r_max)
                    has_hit = False
                    end_range = effective_r_max

            end_x = end_range * math.cos(angle)
            end_y = end_range * math.sin(angle)

            self._trace_ray(grid, end_x, end_y, mark_endpoint=has_hit)
            angle += scan.angle_increment
        
        msg = OccupancyGrid()
        msg.header.frame_id = "ego_racecar/base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = self.width_cells
        msg.info.height = self.height_cells
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        msg.data = grid

        self.grid_pub.publish(msg)


    def _trace_ray(self, grid: List[int], end_x: float, end_y: float, mark_endpoint: bool) -> None:
        """
        Traces a LiDAR ray through the occupancy grid using Bresenham's algorithm.

        Behavior:
        - Marks traversed cells as free (0)
        - Marks endpoint cell as occupied (100) if valid hit occurs
        - If endpoint lies outside the grid, marks last in-bounds cell as occupied 
          when mark_endpoint is True (conservative obstacle handling)

        Inputs:
            grid:
                Flattened occupancy grid buffer to be modified in-place.
            end_x:
                X-coordinate of ray endpoint in meters.
            end_y:
                Y-coordinate of ray endpoint in meters.
            mark_endpoint:
                Whether the ray corresponds to a valid obstacle hit.
        """

        start_ix, start_iy = self._world_to_grid(0.0, 0.0)
        end_ix, end_iy = self._world_to_grid(end_x, end_y)

        dx = abs(end_ix - start_ix)
        dy = abs(end_iy - start_iy)
        sx = 1 if end_ix >= start_ix else -1
        sy = 1 if end_iy >= start_iy else -1

        err = dx - dy
        ix, iy = start_ix, start_iy

        # track last in bounds cell for boundary handling
        last_in_bounds: Tuple[int, int] | None = None
        endpoint_in_bounds = self._in_bounds(end_ix, end_iy)

        while True:
            inb = self._in_bounds(ix, iy)
            if inb:
                last_in_bounds = (ix, iy)
                idx = ix + iy * self.width_cells

                # if true endpoint lies in-bounds is reached, mark occupied if hit
                if mark_endpoint and endpoint_in_bounds and ix == end_ix and iy == end_iy:
                    grid[idx] = 100
                    break

                # otherwise, mark free along the ray (do not overwrite occupied)
                if grid[idx] == -1:
                    grid[idx] = 0
            else:
                # if hit but the endpoint is out of bounds, mark last in bounds cell as occupied (conservative)
                if mark_endpoint and (not endpoint_in_bounds) and last_in_bounds is not None:
                    lix, liy = last_in_bounds
                    lidx = lix + liy * self.width_cells
                    grid[lidx] = 100
                break

            # stop condition if ray traversal reaches endpoint cell
            if ix == end_ix and iy == end_iy:
                # endpoint may be out of bounds or represent free space termination
                # if out of bounds and hit, handle with last_in_bounds
                if mark_endpoint and (not endpoint_in_bounds) and last_in_bounds is not None:
                    lix, liy = last_in_bounds
                    lidx = lix + liy * self.width_cells
                    grid[lidx] = 100
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                ix += sx
            if e2 < dx:
                err += dx
                iy += sy

    # ==============================
    # UTILITIES
    # ==============================
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Converts world-frame coordinates into grid cell indices.

        Inputs:
            x:
                X position in meters relative to grid origin.
            y:
                Y position in meters relative to grid origin.

        Outputs:
            (ix, iy):
                Integer grid indices corresponding to the world position.
        """
        ix = math.floor((x - self.origin_x) / self.resolution)
        iy = math.floor((y - self.origin_y) / self.resolution)
        return int(ix), int(iy)


    def _in_bounds(self, ix: int, iy: int) -> bool:
        """Checks whether a grid index lies within the occupancy grid bounds."""
        return 0 <= ix < self.width_cells and 0 <= iy < self.height_cells


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalOccupancyGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
