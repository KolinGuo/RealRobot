# RealRobot
Real robot interface to connect with ManiSkill2

:exclamation: This repo is still under heavy development, so API might be changed without notice

## Changelog

<details>
<summary>0.1.0</summary>
<p>

### New features
* Added `SharedObject` to create/mount objects stored in `SharedMemory`

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
