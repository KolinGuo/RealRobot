# RealRobot
Real robot interface to connect with ManiSkill2

:exclamation: This repo is still under heavy development, so API might be changed without notice

## Installation

```bash
git clone git@github.com:KolinGuo/RealRobot.git
cd RealRobot && pip install -e .
```

---

Calibrated camera poses are stored in [hec_camera_poses/](hec_camera_poses) and
loaded in [real_robot/sensors/camera.py](real_robot/sensors/camera.py).
You will need to specify the environment variable `REAL_ROBOT_ROOT`
if not installing from source.

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

### API changes
* `real_robot.agents.xarm`
  * Change `XArm7` parameters for clarity (`safety_boundary` => `safety_boundary_mm`, `boundary_clip_eps` => `boundary_clip_mm`)
  * Add `get_gripper_position()` to get gripper opening width in mm or m
  * Add `gripper_speed` parameter to `set_action()` to control gripper speed
* `real_robot.sensors.camera`
  * `CameraConfig` now accepts an `fps` parameter
  * Rename `CameraConfig` parameter `parent_pose_fn` => `parent_pose_so_name`
* `real_robot.utils.realsense`
  * `RSDevice` now accepts `device_sn` instead of an `rs.device`
  * `RSDevice` now accepts `color_config` and `depth_config` as parameters
    (`width`, `height`, `fps`) instead of `rs.config`

### Other changes
* `real_robot.agents.xarm`
  * `XArm7` will not clear *"Safety Boundary Limit"* error automatically in `set_action()`
  * For `motion_mode == "position"`, switch from using `set_tool_position()` to `set_position()`
  * Enable gripper and set to maximum speed in `reset()`

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
