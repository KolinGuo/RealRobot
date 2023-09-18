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

### API changes
* `real_robot.sensors.camera`
  * `CameraConfig` now accepts an `fps` parameter
* `real_robot.utils.realsense`
  * `RSDevice` now accepts `device_sn` instead of an `rs.device`
  * `RSDevice` now accepts `color_config` and `depth_config` as parameters
    (`width`, `height`, `fps`) instead of `rs.config`

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
