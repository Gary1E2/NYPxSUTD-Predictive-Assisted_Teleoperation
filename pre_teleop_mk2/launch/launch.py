from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="pre_teleop_mk2",
                executable="scan_preprocess_node",
                name="scan_preprocess_node",
                output="screen",
            ),
            # Node(
            #     package="pre_teleop_mk2",
            #     executable="local_occupancy_grid_node",
            #     name="local_occupancy_grid_node",
            #     output="screen",
            # ),
            # Node(
            #     package="pre_teleop_mk2",
            #     executable="local_sdf_node",
            #     name="local_sdf_node",
            #     output="screen",
            # ),
            Node(
                package="pre_teleop_mk2",
                executable="state_estimator_ekf_node",
                parameters=[{'use_sim_time': True}],
                name="state_estimator_ekf_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="teleop_state_node",
                name="teleop_state_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="mppi_planner_node",
                name="mppi_planner_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="command_clamp_node",
                name="command_clamp_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="aeb_node",
                name="aeb_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="control_switch_node",
                name="control_switch_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="command_gate_node",
                name="command_gate_node",
                output="screen",
            ),
            Node(
                package="pre_teleop_mk2",
                executable="teleop_node",
                name="teleop_node",
                output="screen",
            )
        ]
    )
