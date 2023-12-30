# RealRobot
Real robot interface to connect with ManiSkill2

Runs camera capturing and visualization as separate processes to make it closer to using ROS

:exclamation: This repo is still under heavy development, so API might be changed without notice

## Installation

```bash
python3 -m pip install -U real_robot
```

---

Calibrated camera poses are stored in [real_robot/assets/hec_camera_poses/](real_robot/assets/hec_camera_poses) and
loaded in [real_robot/sensors/camera.py](real_robot/sensors/camera.py).

## Helpful Tools

* Capture Color/Depth/IR images from RS camera, do
  ```bash
  python3 -m real_robot.tools.rs_capture
  ```
* Get detailed device information of connected RealSense device
  (*e.g.*, supported stream configs, sensor intrinsics, extrinsics between sensors), do
  ```bash
  python3 -m real_robot.tools.enumerate_rs_devices > info.log
  ```
* Convert recorded ROS bag file and save rgb & depth images into .npz, do
  ```bash
  python3 -m real_robot.tools.convert_rosbag_to_frames <bag_path>
  ```

## Notes

To fix an unindexed rosbag recorded from `RSDevice`, do
```bash
apt install -y python3-rosbag
rosbag reindex <bag_path>
```

## Known Issue
- [ ] When controlling xArm7 with `motion_mode="cartesian_online"` and `wait=True`,
  sometimes the robot will be stuck indefinitely in `wait_move()` due to `get_state()`
  returns wrong state (i.e., returns `state=1` thinking it's still in motion).
  Simple solution can be just to control briefly via UFACTORY Studio to get it unstuck.

## Changelog

<details>
<summary>0.1.0</summary>
<p>

### New features
* Added `SharedObject` to create/mount objects stored in `SharedMemory`
* Enabled `RSDevice` to run as a separate process (now `Camera` will create
  `RSDevice` as a separate process)
* Enabled `RSDevice` to record camera streams as a rosbag file
* Enabled `XArm7` to run as a separate process (for streaming robot states)
* Enabled `CV2Visualizer` and `O3DGUIVisualizer` to run as separate processes (for visualization)
* Added a default `FileHandler` to all Logger created through `real_robot.utils.logger.get_logger`
* Allow enabling selected camera streams from `RSDevice` and `sensors.camera.Camera`
* Added `sensors.simsense_depth.SimsenseDepth` class to generate depth image from stereo IR images

### API changes
* `real_robot.agents.xarm`
  * Change `XArm7` parameters for clarity (`safety_boundary` => `safety_boundary_mm`, `boundary_clip_eps` => `boundary_clip_mm`)
  * Add `get_gripper_position()` to get gripper opening width in mm or m
  * Add `gripper_speed` parameter to `set_action()` to control gripper speed
* `real_robot.utils.visualization.visualizer`
  * Rename `Visualizer` method `show_observation()` => `show_obs()`
* `real_robot.sensors.camera`
  * `CameraConfig` now accepts a `config` parameter
  * Rename `CameraConfig` parameter `parent_pose_fn` => `parent_pose_so_name`
* `real_robot.utils.realsense`
  * `RSDevice` now accepts `device_sn` instead of an `rs.device`
  * `RSDevice` now accepts `config` as parameter (`width`, `height`, `fps`) instead of `rs.config`

### Other changes
* Switch from `gym` to `gymnasium`
* Rename all `control_mode` by removing `pd_` prefix for clarity. No PD controller is used.
* `real_robot.agents.xarm`
  * `XArm7` will not clear *"Safety Boundary Limit"* error automatically in `set_action()`
  * For `motion_mode == "position"`, switch from using `set_tool_position()` to `set_position()`
  * Enable gripper and set to maximum speed in `reset()`
* Remove all Loggers created as global variables (they will be created
  at import, which might not be saved under `REAL_ROBOT_LOG_DIR`)
* Bugfix in xArm-Python-SDK: enable `wait=True` for modes other than position mode

</p>
</details>

<details>
<summary>0.0.2</summary>
<p>

* Added motion_mode to XArm7 agent
* Added several control_mode: `pd_ee_pos`, `pd_ee_pose_axangle`,
`pd_ee_delta_pose_axangle`, `pd_ee_pose_quat`, `pd_ee_delta_pose_quat`

</p>
</details>
