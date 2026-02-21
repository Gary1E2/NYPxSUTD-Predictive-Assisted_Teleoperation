from setuptools import setup

package_name = "pre_teleop_mk1"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/pre_teleop_launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Gary Chua",
    maintainer_email="1e2.gary.chua@gmail.com",
    description="Predictive teleoperation system for f1tenth_gym_ros simulation version 1",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "scan_preprocess_node = pre_teleop_mk1.scan_preprocess_node:main",
            "state_estimator_ekf_node = pre_teleop_mk1.state_estimator_ekf_node:main",
            "mppi_planner_node = pre_teleop_mk1.mppi_planner_node:main",
            "command_mux_node = pre_teleop_mk1.command_mux_node:main",
            "mpc_tracker_node = pre_teleop_mk1.mpc_tracker_node:main",
            "aeb_node = pre_teleop_mk1.aeb_node:main",
            "teleop_node = pre_teleop_mk1.teleop_node:main",
            "teleop_state_node = pre_teleop_mk1.teleop_state_node:main",
            "control_switch_node = pre_teleop_mk1.control_switch_node:main",
            "command_gate_node = pre_teleop_mk1.command_gate_node:main",
            "rc_test_node = pre_teleop_mk1.rc_test_node:main"
        ],
    },
)
