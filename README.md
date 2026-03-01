# NYPxSUTD-Predictive-Assisted_Teleoperation
## Full Time Semester Project collaboration between NYP and SUTD
Uses:
- Local LiDAR sensor perception of vehicle environment
- Model Predictive Path Integral (MPPI) planner for autonomous motion
- Joystick gamepad control

# Project Overview:
<p align="center">
  <img src="FinalPhysicalTest-PT1.gif" width="30%"/>
  <img src="FinalPhysicalTest-PT3.gif" width="30%"/>
</p>

## EVAM:
EVAM (Electric Vehicle Additive Manufacturing) is a project from SUTD focused on transforming electric vehicle component design and fabrication. It leverages 3D printing additive manufacturing and AI to design and create various car parts, aiming to replace conventional components with more flexible and efficient 3D-printed alternatives. This can help significantly accelerate design cycles and enable complex, lightweight and optimized electric vehicle components.

## AI Assisted Teleoperation:
This project aims to design and implement an AI-assisted teleoperation system for the EVAM vehicle. The system will combine real-time telemetry, remote control, and AI analytics to enhance the vehicle's safety, responsiveness, and situational awareness to obstacles during teleoperation.

The quick start guide can be found below the features.

# Features:
- Teleoperation with joystick gamepad controller
- Obstacle avoidance assistance during teleoperation
- Autonomous navigation without teleoperation

# Dependencies:
## ROS2 Humble:
Use the following install guide or follow the official install instructions: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

1. Install ROS2 Humble by running the following commands in order:
```
# Set locale
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
# Add ROS2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
# Install ROS2 Humble Desktop
sudo apt update
sudo apt install ros-humble-desktop -y 
```
2. Setup the ROS2 environment by running the following commands (generally not recommended for ROS2 developers:
```
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
3. Source the ROS2 environment (recommended action):
```
source /opt/ros/humble/setup.bash
```
4. Install development tools as required:
```
sudo apt install python3-rosdep python3-colcon-common-extensions -y
sudo rosdep init
rosdep update
```
   
## F1Tenth Gym:
Use the following install guide or follow the official Github repo: https://github.com/f1tenth/f1tenth_gym_ros

1. Install F1Tenth Gym:
```
cd ~
git clone https://github.com/f1tenth/f1tenth_gym 
cd f1tenth_gym
```
2. Create python environment as required:
```
python3 -m venv gymenv
```
to activate/enter the environment:
```
source gymenv/bin/activate
```
to leave the environment:
```
deactivate
```

3. Install dependencies for F1Tenth Gym:
```
pip3 install -e .
```

4. Install F1TENTH ROS2 Bridge 
```
cd ~/sim_ws/src 
git clone https://github.com/f1tenth/f1tenth_gym_ros 
cd ~/turtlebot3_ws 
colcon build --packages-select f1tenth_gym_ros 
source install/setup.bash
```

5. Install reactive obstacle avoidance system (F1Tenth standard gap follow)
```
cd ~/turtlebot3_ws/src
git clone https://github.com/CL2-UWaterloo/f1tenth_ws # This includes gap following algorithms 
cd ~/turtlebot3_ws 
colcon build
```

6. Debugging and fixes (python dependencies)
(Try gymenv pip3 install but rely on normal pip3 install instead for system to see packages outside of the env)
```
pip3 install --user f1tenth_gym
pip3 install transforms3d --user
pip3 install --user numba==0.56.4 coverage==6.5.0
sudo apt install ros-humble-xacro
sudo apt install ros-humble-ackermann-msgs
sudo apt install ros-humble-navigation2 -y
sudo apt install ros-humble-nav2-bringup -y
```

# Quick Start Guide:
The control bindings are as follows (uses Logitech F710 Gamepad):
- Left Horizontal Joystick: Speed forward and reverse drive
- Right Horizontal Joystick: Steer left and right
- A button: Emergency stop release
- B button: Emergency stop activate
- Right Button: Autonomous navigation

Using the system will involve running multiple terminals:

## 1st Terminal (Simulator):
Skip if running on a robot car
```
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

## 2nd Terminal (System):
```
ros2 launch pre_teleop_mk3 launch.py
```
Use this template if running the older systems: 
[x] = version number, [launch] = launch file name
```
ros2 launch pre_teleop_mk[x] [launch].py
```

## 3rd Terminal (Controller):
```
ros2 run joy joy_node
```

## To change the configs or world map:
1. Run:
```
cd ~/sim_ws/src/f1tenth_gym_ros/config/sim.yaml
```

2. Find this chunk:
```
    # map parameters
    map_path: '/sim_ws/src/f1tenth_gym_ros/maps/levine'
    map_img_ext: '.png'
```

3. Replace 'map_path:' with the directory to your chosen map:
You may need to use this format if sim_ws is in your home directory:
```
    map_path: '/home/gary/sim_ws/src/f1tenth_gym_ros/maps/icra_2'
```

## To change the world map config:
1. Run:
```
cd ~/sim_ws/src/f1tenth_gym_ros/maps
```

2. Find and modify your the map's .yaml file:
[map] = map name
```
nano [map].yaml
```

# Folder Structure Overview:

```bash
NYPxSUTD-Predictive-Assisted_Teleoperation
├─── demovids
│   ├─── reactive_wall_demo.mp4
│   ├─── avoid_mk1_demo.mp4
│   ├─── ...
├─── pre_teleop_mk3
│   ├─── launch
│   │  └─── launch.py
│   ├─── pre_teleop_mk3
│   │   ├─── __init__.py
│   │   ├─── aeb_node.py
│   │   ├─── command_clamp_node.py
│   │   ├─── command_gate_node.py
│   │   ├─── control_switch_node.py
│   │   ├─── local_occupancy_grid_node.py
│   │   ├─── local_sdf_node.py
│   │   ├─── mpc_tracker_node.py
│   │   ├─── mppi_planner_node.py
│   │   ├─── pid_tracker_node.py
│   │   ├─── scan_preprocess_node.py
│   │   ├─── state_estimator_ekf_node.py
│   │   ├─── teleop_node.py
│   │   └─── teleop_state_node.py
│   ├─── resource
│   │   └─── pre_teleop_mk3
│   ├─── package.xml
│   ├─── setup.cfg
│   └─── setup.py
├─── pre_teleop_mk2
│   ├─── launch
│   │   └─── launch.py
│   ├─── pre_teleop_mk2
│   │   ├─── __init__.py
│   │   └─── ...
│   ├─── resource
│   │   └─── pre_teleop_mk2
│   ├─── package.xml
│   ├─── setup.cfg
│   └─── setup.py
├─── pre_teleop_mk1
│   ├─── launch
│   │   └─── launch.py
│   ├─── pre_teleop_mk1
│   │   ├─── __init__.py
│   │   └─── ...
│   ├─── resource
│   │   └─── pre_teleop_mk1
│   ├─── package.xml
│   ├─── setup.cfg
│   └─── setup.py
├─── LICENSE
└─── README.md                      {project information /THIS FILE/}
```

# System Architecture:
<p align="center">
  <img src="pre_teleop_mk3.drawio.png">
</p>

# How it Works (simplified):
- scan_preprocess_node.py: Downsamples LiDAR scans and limits it's min and max range.
- (REMOVED) local_occupancy_grid_node.py: Converts LiDAR scans into 2D grid map of occupied and free space.
- (REMOVED) local_sdf_node.py: Converts local occupancy grid map into 2D grid map with integer distance from nearest obstacles encoded in each cell.
- state_estimator_ekf_node.py: Estimates vehicle state (velocity and steering) with Extended Kalman Filter and kinematic bicycle model vehicle dynamics using Odometry data.
- mppi_planner_node.py: Samples 'K' amount of control sequences, simulates and accumulates cost for each 'T' control action step and outputs the first control action of the cheapest control sequence. Uses a Model Predictive Path Integral planner.
- command_clamp_node.py: Limits the control action generated from mppi_planner_node.py as a layer of safety.
- (REMOVED) mpc_tracker_node.py: Tracks the high level mppi_planner_node.py control action. Uses a Model Predictive Controller tracker.
- (REMOVED) pid_tracker_node.py: Tracks the high level mppi_planner_node.py control action. Uses a Proportinal Integral Derivative tracker.
- (EXTERNAL) joy_node: Read joystick controller inputs. From the ROS2 Joy node.
- teleop_node.py: Translates the joystick controller inputs into driving commands. Features emergency stop and autonomous navigation.
- teleop_state_node.py: Simulates the future trajectory of the driving command from teleop_node.py. Uses the kinematic bicycle model vehicle dynamics.
- control_switch_node.py: Decides whether teleoperation control or planner control is used. Checks teleoperation state to decide.
- command_gate_node.py: Prevents motion if teleop_node.py or mppi_planner_node.py is down. Engages manual emergency stop activation from teleop_node.py.
- aeb_node.py: Forces the car the brake if the time to collision between the car and an obstacle along 3 separate LiDAR rays is below the threshold. Fallback safety layer.


