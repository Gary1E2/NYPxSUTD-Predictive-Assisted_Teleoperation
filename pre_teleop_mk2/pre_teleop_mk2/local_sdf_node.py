#!/usr/bin/env python3

"""
local_sdf_node.py used to generate a local grid map consisting cells encoded with distance to obstacles.

Actually a unsigned distance field not 'signed'.
"""

import heapq
import math
from typing import List

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# ==============================
# QUALITY OF SERVICE POLICIES
# ==============================
qos_profile_sub = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
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
class LocalSdfNode(Node):
    def __init__(self) -> None:
        super().__init__("local_sdf_node")

        self.get_logger().info(f"Initialized and running.")


        # ========== VARIABLES ==========
        self._processing = False


        # ========== SUBS & PUBS ==========
        self._grid_sub = self.create_subscription(
            OccupancyGrid,
            "/local_occupancy_grid",
            self._occupancy_grid_callback,
            qos_profile_sub,
        )
        self._sdf_pub = self.create_publisher(
            OccupancyGrid,
            "/local_sdf",
            qos_profile_pub,
        )

    # ==============================
    # OCCUPANCY TO SDF GRID CONVERSION
    # ==============================
    def _occupancy_grid_callback(self, msg: OccupancyGrid) -> None:
        """
        Processes an incoming local occupancy grid and generates a local SDF grid.

        Re-entrancy is prevented to avoid overlapping SDF computations.

        - Builds a binary obstacle mask from the occupancy grid
        - Computes distances to the nearest obstacle for each cell
        - Encodes distances into an integer OccupancyGrid representation
        - Publishes the resulting SDF grid

        Inputs:
            msg:
                Local OccupancyGrid message representing free, occupied, and unknown space.
        """
        if self._processing:
            return

        self._processing = True
        try:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution

            obstacle_mask = self._build_obstacle_mask(msg.data)
            distances_cells = self._compute_distances(obstacle_mask, width, height)
            encoded_grid = self._encode_to_grid(distances_cells, resolution)

            sdf_msg = OccupancyGrid()
            sdf_msg.header.stamp = msg.header.stamp
            sdf_msg.header.frame_id = msg.header.frame_id
            sdf_msg.info = msg.info
            sdf_msg.data = encoded_grid

            self._sdf_pub.publish(sdf_msg)
        finally:
            self._processing = False


    def _compute_distances(self, obstacle_mask: List[bool], width: int, height: int) -> List[float]:
        """
        Computes distance to obstacle values for each grid cell using Dijkstra expansion.

        Behavior:
        - Initializes all obstacle cells with zero distance
        - Propagates distances to neighboring cells using an 8 connected grid
        - Approximates Euclidean distance using axial and diagonal step costs

        Inputs:
            obstacle_mask:
                Boolean mask indicating obstacle cells.
            width:
                Grid width in cells.
            height:
                Grid height in cells.

        Outputs:
            distances:
                Distance to the nearest obstacle for each cell, in grid units.
        """
        total_cells = width * height
        distances = [math.inf] * total_cells
        heap: List[tuple[float, int]] = []

        diag_cost = math.sqrt(2.0)
        neighbor_offsets = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diag_cost),
            (-1, 1, diag_cost),
            (1, -1, diag_cost),
            (1, 1, diag_cost),
        ]

        for idx, is_obstacle in enumerate(obstacle_mask):
            if is_obstacle:
                distances[idx] = 0.0
                heapq.heappush(heap, (0.0, idx))

        if not heap:
            return distances

        while heap:
            dist, idx = heapq.heappop(heap)
            if dist > distances[idx]:
                continue

            x = idx % width
            y = idx // width

            for dx, dy, cost in neighbor_offsets:
                nx = x + dx
                ny = y + dy

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                n_idx = nx + ny * width
                new_dist = dist + cost

                if new_dist < distances[n_idx]:
                    distances[n_idx] = new_dist
                    heapq.heappush(heap, (new_dist, n_idx))

        return distances

    # ==============================
    # UTILITIES
    # ==============================
    def _build_obstacle_mask(self, data: List[int]) -> List[bool]:
        """
        Converts occupancy grid values into a binary obstacle mask.

        Inputs:
            data:
                Flattened occupancy grid data array.

        Outputs:
            List[bool]:
                Boolean mask where True indicates obstacle cells.
        """
        return [(value >= 50) or (value < 0) for value in data]


    def _encode_to_grid(self, distances_cells: List[float], resolution: float) -> List[int]:
        """
        Encodes distance values into an OccupancyGrid compatible integer format.

        Encoding rules:
        - Infinite distances are clamped to maximum value (100)
        - Finite distances are converted to meters and scaled
        - Values are clipped to the range [0, 100]

        Inputs:
            distances_cells:
                Distance-to-obstacle values in grid cell units.
            resolution:
                Grid resolution in meters per cell.

        Outputs:
            List[int]:
                Encoded grid data suitable for publishing as OccupancyGrid.
        """

        encoded: List[int] = []

        for dist_cells in distances_cells:
            if math.isinf(dist_cells):
                encoded.append(100)
                continue

            distance_m = dist_cells * resolution
            value = round(distance_m * 10.0)
            value = max(0, min(100, int(value)))
            encoded.append(value)

        return encoded


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalSdfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
