"""xArm7 Agent Interface
Check user manual at https://www.ufactory.cc/download/
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pyrealsense2 as rs
from gymnasium import spaces
from sapien import Pose
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat
from urchin import URDF
from xarm.wrapper import XArmAPI

from .. import ASSET_DIR
from ..sensors.camera import CameraConfig
from ..utils.common import clip_and_scale_action, vectorize_pose
from ..utils.logger import get_logger
from ..utils.multiprocessing import SharedObject, signal_process_ready

# TODO: remove return code from all functions, add return code checks


class XArm7:
    """
    xArm7 agent class
    Mimics mani_skill2.agents.base_agent.BaseAgent interface
    """

    SUPPORTED_CONTROL_MODES = (
        "ee_pos",
        "ee_delta_pos",
        "ee_pose_axangle",
        "ee_delta_pose_axangle",
        "ee_pose_quat",
        "ee_delta_pose_quat",
    )
    SUPPORTED_MOTION_MODES = (
        "position",
        "servo",
        "joint_teaching",
        "cartesian_teaching (invalid)",
        "joint_vel",
        "cartesian_vel",
        "joint_online",
        "cartesian_online",
    )

    def __init__(
        self,
        ip: str = "192.168.1.229",
        control_mode: str = "ee_delta_pos",
        motion_mode: str = "position",
        *,
        safety_boundary_mm: list[int] = [999, -999, 999, -999, 999, 0],
        boundary_clip_mm: int = 10,
        with_hand_camera: bool = True,
        run_as_process: bool = False,
    ):
        """
        :param ip: xArm7 ip address, see controller box
        :param control_mode: xArm control mode (determines set_action type)
        :param motion_mode: xArm motion mode (determines xArm motion mode)
        :param safety_boundary_mm: [x_max, x_min, y_max, y_min, z_max, z_min] (mm)
        :param boundary_clip_mm: clip action when TCP position to boundary is
                                 within boundary_clip_eps (mm). No clipping if None.
        :param with_hand_camera: whether to include hand camera mount in TCP offset.
        :param run_as_process: whether to run XArm7 as a separate process for
                               streaming robot states and saving trajectory (not done)
            If True, XArm7 needs to be created as a `mp.Process`.
            Several SharedObject are created to control XArm7 and stream data:
            * "join_xarm7_real": If triggered, the XArm7 process is joined.
            * "sync_xarm7_real": If triggered, all processes should fetch data.
                                 Used for synchronizing robot state capture.
            * "start_xarm7_real": If True, starts the XArm7 streams; else, stops it.
            * "xarm7_real_urdf_path": xArm7 URDF path, str
            * "xarm7_real_qpos": xArm7 joint angles, [8,] np.float32 np.ndarray
            * "xarm7_real_qvel": xArm7 joint velocities, [8,] np.float32 np.ndarray
            * "xarm7_real_qf": xArm7 joint torques, [8,] np.float32 np.ndarray
            * "xarm7_real_tcp_pose": tcp pose in world frame (unit: m), sapien.Pose
        """
        self.logger = get_logger("XArm7")
        self.ip = ip
        self.arm = XArmAPI(ip)

        if control_mode not in self.SUPPORTED_CONTROL_MODES:
            raise NotImplementedError(f"Unsupported {control_mode = }")
        if motion_mode not in self.SUPPORTED_MOTION_MODES:
            raise NotImplementedError(f"Unsupported {motion_mode = }")
        self._control_mode = control_mode
        self._motion_mode = motion_mode

        # NOTE: currently still using prismatic fingers in simulation
        # TODO: When gear joint is properly implemented, this is not needed
        self.joint_limits_ms2 = URDF.load(
            f"{ASSET_DIR}/descriptions/xarm7_pris_finger_d435.urdf",
            lazy_load_meshes=True,
        ).joint_limits.astype(np.float32)
        # set joint_limits to correspond to [-10, 850]
        self.joint_limits_ms2[-2:, 0] = self.joint_limits_ms2[-2:, 1] / 850 * -10
        self.gripper_limits = np.asarray([-10, 850], dtype=np.float32)

        self.init_qpos = np.asarray(
            [
                0,
                0,
                0,
                np.pi / 3,
                0,
                np.pi / 3,
                -np.pi / 2,
                0.0453556139430441,
                0.0453556139430441,
            ],
            dtype=np.float32,
        )
        self.pose = Pose()  # base pose in world frame
        self.safety_boundary = np.asarray(safety_boundary_mm)
        self.boundary_clip = boundary_clip_mm
        if self.boundary_clip is not None:
            self.safety_boundary_clip = self.safety_boundary.copy()
            self.safety_boundary_clip[0::2] -= boundary_clip_mm
            self.safety_boundary_clip[1::2] += boundary_clip_mm
        self.with_hand_camera = with_hand_camera

        if run_as_process:
            self.run_as_process()
        else:
            self.reset()

    def __del__(self):
        self.arm.disconnect()

    def get_err_warn_code(self, show=False):
        code, (error_code, warn_code) = self.arm.get_err_warn_code(show=True)
        assert code == 0, "Failed to get_err_warn_code"
        return error_code, warn_code

    def clean_warning_error(self, mode=None):
        error_code, warn_code = self.get_err_warn_code(show=True)
        if warn_code != 0:
            self.arm.clean_warn()
        if error_code != 0:
            self.arm.clean_error()

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(
            self.SUPPORTED_MOTION_MODES.index(self._motion_mode)
            if mode is None
            else mode
        )
        self.arm.set_state(state=0)

    def reset(self, wait=True):
        self.clean_warning_error(mode=0)

        # NOTE: Remove satefy boundary during reset
        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary([999, -999, 999, -999, 999, -999])
        self.set_qpos(self.init_qpos, wait=wait)

        # boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        self.arm.set_reduced_tcp_boundary(self.safety_boundary)
        self.arm.set_tcp_load(0.82, [0.0, 0.0, 48.0])
        if self.with_hand_camera:
            self.arm.set_tcp_offset([0.0, 0.0, 177.0, 0.0, 0.0, 0.0])
        else:
            self.arm.set_tcp_offset([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
        self.arm.set_self_collision_detection(True)
        self.arm.set_collision_tool_model(1)  # xArm Gripper
        self.arm.set_collision_rebound(True)

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(self.SUPPORTED_MOTION_MODES.index(self._motion_mode))
        self.arm.set_state(state=0)

        # Enable gripper and set to maximum speed
        self.arm.set_gripper_enable(enable=True)
        self.arm.set_gripper_speed(speed=5000)

    # ---------------------------------------------------------------------- #
    # Control robot
    # ---------------------------------------------------------------------- #
    def _preprocess_action(
        self, action: np.ndarray, translation_scale: float, axangle_scale: float
    ):
        """Preprocess action:
            * Keep current TCP orientation when action only has position
            * apply translation_scale and axangle_scale for delta control_mode
            * clip tgt_tcp_pose to avoid going out of safety_boundary (optionally)
            * clip and rescale gripper action (action[-1])
        :param action: control action (unit for translation is meters)
                       If in delta control_mode, action needs to apply scale
        :return tgt_tcp_pose: target TCP pose in robot base frame (unit in mm)
        :return gripper_pos: gripper action after rescaling [-10, 850]
        """
        cur_tcp_pose = self.get_tcp_pose(unit_in_mm=True)

        if self._control_mode == "ee_pos":
            tgt_tcp_pose = Pose(p=action[:3] * 1000.0, q=cur_tcp_pose.q)  # m => mm
        elif self._control_mode == "ee_delta_pos":
            delta_tcp_pos = action[:3] * translation_scale  # in milimeters
            tgt_tcp_pose = cur_tcp_pose * Pose(p=delta_tcp_pos)
        # elif self._control_mode == "ee_pose":
        #     tgt_tcp_pose = Pose(action[:3] * 1000.0,
        #                         euler2quat(*action[3:6], axes='sxyz'))
        elif self._control_mode == "ee_pose_axangle":
            tgt_tcp_pose = Pose(
                p=action[:3] * 1000.0,  # m => mm
                q=Rotation.from_rotvec(action[3:6]).as_quat()[[3, 0, 1, 2]],
            )
        elif self._control_mode == "ee_delta_pose_axangle":
            axangle = action[3:6]
            if (theta := math.sqrt(axangle @ axangle)) < 1e-9:
                q = [1, 0, 0, 0]
            else:
                q = axangle2quat(
                    axangle / theta,
                    axangle_scale if theta > 1 else theta * axangle_scale,
                    is_normalized=True,
                )
            delta_tcp_pose = Pose(
                p=action[:3] * translation_scale, q=q
            )  # in milimeters
            tgt_tcp_pose = cur_tcp_pose * delta_tcp_pose
        elif self._control_mode == "ee_pose_quat":
            tgt_tcp_pose = Pose(p=action[:3] * 1000.0, q=action[3:7])  # m => mm
        elif self._control_mode == "ee_delta_pose_quat":
            # TODO: Apply axangle_scale?
            delta_tcp_pose = Pose(
                p=action[:3] * translation_scale, q=action[3:7]  # in milimeters
            )
            tgt_tcp_pose = cur_tcp_pose * delta_tcp_pose
        else:
            raise NotImplementedError(f"{self._control_mode=} not implemented")

        # Clip tgt_tcp_pose.p to safety_boundary_clip
        if self.boundary_clip is not None:
            tgt_tcp_pose.set_p(
                np.clip(
                    tgt_tcp_pose.p,
                    self.safety_boundary_clip[1::2],
                    self.safety_boundary_clip[0::2],
                )
            )

        # [-1, 1] => [-10, 850]
        gripper_pos = clip_and_scale_action(action[-1], self.gripper_limits)

        self.logger.info(f"Setting {tgt_tcp_pose = }, {gripper_pos = }")
        return tgt_tcp_pose, gripper_pos

    def set_action(
        self,
        action: np.ndarray,
        translation_scale=100.0,
        axangle_scale=0.1,
        speed=None,
        mvacc=None,
        gripper_speed=None,
        skip_gripper=False,
        wait=False,
    ):
        """
        :param action: action corresponding to self.control_mode, np.floating np.ndarray
                       action[-1] is gripper action (always has range [-1, 1])
        :param translation_scale: action [-1, 1] maps to [-100mm, 100mm],
                                  Used for delta control_mode only.
        :param axangle_scale: axangle action norm (rotation angle) is clipped by 0.1 rad
                              [-1, 0, 0] => rotate around [1, 0, 0] by -0.1 rad
                              Used for delta control_mode only.
        :param speed: move speed.
                      For TCP motion: range is [0.1, 1000.0] mm/s (default=100)
                      For joint motion: range is [0.05, 180.0] deg/s (default=20)
        :param mvacc: move acceleration.
                      For TCP motion: range [1.0, 50000.0] mm/s^2 (default=2000)
                      For joint motion: range [0.5, 1145.0] deg/s^2 (default=500)
        :param gripper_speed: gripper speed, range [1, 5000] r/min (default=5000)
        :param skip_gripper: whether to skip gripper action
        :param wait: whether to wait for the arm to complete, default is False.
                     Has no effect in "joint_online" and "cartesian_online" motion mode
        """
        # Clean existing warnings / errors
        while self.arm.has_err_warn:
            error_code, warn_code = self.arm.get_err_warn_code()
            # if error_code in [35]:  # 35: Safety Boundary Limit
            #     self.clean_warning_error()
            self.logger.error(f"ErrorCode: {error_code}, need to manually clean it")
            self.arm.get_err_warn_code(show=True)
            _ = input("Press enter after cleaning error")

        # Checks action shape and range
        assert action in self.action_space, f"Wrong {action = }"
        action = np.asarray(action)

        # Preprocess action (apply scaling, clip to safety boundary, rescale gripper)
        tgt_tcp_pose, gripper_pos = self._preprocess_action(
            action, translation_scale, axangle_scale
        )

        # Control gripper position
        ret_gripper = 0
        if not skip_gripper:
            # NOTE: wait=False for gripper position (it should never block)
            ret_gripper = self.arm.set_gripper_position(
                gripper_pos, speed=gripper_speed, wait=False, wait_motion=False
            )

        # Control xArm based on motion mode
        if self._motion_mode == "position":
            # NOTE: when wait=False, the second set_position() call will be blocked
            # until previous motion is completed.
            ret_arm = self.arm.set_position(
                *tgt_tcp_pose.p,
                *quat2euler(tgt_tcp_pose.q, axes="sxyz"),
                speed=speed,
                mvacc=mvacc,
                relative=False,
                is_radian=True,
                wait=wait,
            )
            return ret_arm, ret_gripper
        elif self._motion_mode == "servo":
            raise NotImplementedError("Do not use servo mode! Need fine waypoints")
            ret_arm = self.arm.set_servo_cartesian(
                np.hstack([tgt_tcp_pose.p, quat2euler(tgt_tcp_pose.q, axes="sxyz")]),
                is_radian=True,
                is_tool_coord=False,
            )
            return ret_arm, ret_gripper
        elif self._motion_mode == "joint_teaching":
            raise ValueError("Joint teaching mode enabled (no action needed)")
        elif self._motion_mode == "joint_vel":
            raise NotImplementedError("Unverified")
            # clip delta pos to prevent undesired rotation due to ik solutions
            cur_tcp_pose = self.get_tcp_pose(unit_in_mm=True)
            tgt_tcp_pose.set_p(
                np.clip(tgt_tcp_pose.p, cur_tcp_pose.p - 30, cur_tcp_pose.p + 30)
            )

            _, tgt_qpos = self.arm.get_inverse_kinematics(
                np.hstack([tgt_tcp_pose.p, quat2euler(tgt_tcp_pose.q, axes="sxyz")]),
                input_is_radian=True,
                return_is_radian=True,
            )
            _, (cur_qpos, _, _) = self.arm.get_joint_states(is_radian=True)
            delta_qpos = np.asarray(tgt_qpos) - np.asarray(cur_qpos)

            timestep = 0.5
            qvel = delta_qpos / timestep
            qvel = np.clip(qvel, -0.3, 0.3)  # clip qvel for safety
            ret_arm = self.arm.vc_set_joint_velocity(
                qvel, is_radian=True, is_sync=True, duration=timestep
            )
            # time.sleep(timestep - 0.2)
            return ret_arm, ret_gripper
        elif self._motion_mode == "cartesian_vel":
            raise NotImplementedError(f"{self._motion_mode = } is not yet implemented")

            # [spd_x, spd_y, spd_z, spd_rx, spd_ry, spd_rz]
            speeds_xyz_rpy = np.zeros(6)
            ret_arm = self.arm.vc_set_cartesian_velocity(
                speeds_xyz_rpy, is_radian=True, is_tool_coord=True, duration=timestep
            )
            # time.sleep(timestep - 0.2)
            return ret_arm, ret_gripper
        elif self._motion_mode == "joint_online":
            _, tgt_qpos = self.arm.get_inverse_kinematics(
                np.hstack([tgt_tcp_pose.p, quat2euler(tgt_tcp_pose.q, axes="sxyz")]),
                input_is_radian=True,
                return_is_radian=True,
            )
            ret_arm = self.arm.set_servo_angle(
                angle=tgt_qpos,
                speed=speed,
                mvacc=mvacc,
                relative=False,
                is_radian=True,
                wait=wait,
            )
            return ret_arm, ret_gripper
        elif self._motion_mode == "cartesian_online":
            ret_arm = self.arm.set_position(
                *tgt_tcp_pose.p,
                *quat2euler(tgt_tcp_pose.q, axes="sxyz"),
                speed=speed,
                mvacc=mvacc,
                relative=False,
                is_radian=True,
                wait=wait,
            )
            return ret_arm, ret_gripper
        else:
            raise NotImplementedError()

    def set_qpos(
        self,
        qpos,
        speed=None,
        mvacc=None,
        gripper_speed=None,
        skip_gripper=False,
        wait=False,
    ):
        """Set xarm qpos using maniskill2 qpos
        :param qpos: joint qpos (angles for arm joints, gripper_qpos for gripper)
                     See self.joint_limits_ms2
        :param speed: move speed.
                      For TCP motion: range is [0.1, 1000.0] mm/s (default=100)
                      For joint motion: range is [0.05, 180.0] deg/s (default=20)
        :param mvacc: move acceleration.
                      For TCP motion: range [1.0, 50000.0] mm/s^2 (default=2000)
                      For joint motion: range [0.5, 1145.0] deg/s^2 (default=500)
        :param gripper_speed: gripper speed, range [1, 5000] r/min
        :param skip_gripper: whether to skip gripper action
        :param wait: whether to wait for the arm to complete, default is False.
        """
        assert len(qpos) == 9, f"Wrong qpos shape: {len(qpos)}"
        arm_qpos, gripper_qpos = qpos[:7], qpos[-2:]

        # Control gripper position
        ret_gripper = 0
        if not skip_gripper:
            gripper_qpos = gripper_qpos[0]  # NOTE: mimic action
            gripper_pos = clip_and_scale_action(
                gripper_qpos, self.gripper_limits, self.joint_limits_ms2[-1, :]
            )
            ret_gripper = self.arm.set_gripper_position(
                gripper_pos, speed=gripper_speed, wait=False, wait_motion=False
            )

        ret_arm = self.arm.set_servo_angle(
            angle=arm_qpos,
            speed=speed,
            mvacc=mvacc,
            relative=False,
            is_radian=True,
            wait=wait,
        )
        return ret_arm, ret_gripper

    def set_gripper_position(
        self, gripper_pos, unit_in_mm=False, speed=None, wait=False
    ):
        """Set gripper opening width
        :param gripper_pos: gripper position (default unit is in meters)
        :param unit_in_mm: whether gripper_pos has unit mm or m
        :param speed: gripper speed, range [1, 5000] r/min
        :param wait: whether to wait for the action to complete, default is False.
        """
        ret_gripper = self.arm.set_gripper_position(
            gripper_pos * 10.0 if unit_in_mm else gripper_pos * 10000.0,
            speed=speed,
            wait=wait,
            wait_motion=wait,
        )
        return ret_gripper

    def close_gripper(self, speed=None, wait=False):
        """Close gripper
        :param speed: gripper speed, range [1, 5000] r/min
        :param wait: whether to wait for the action to complete, default is False.
        """
        ret_gripper = self.arm.set_gripper_position(
            self.gripper_limits[0], speed=speed, wait=wait, wait_motion=wait
        )
        return ret_gripper

    def open_gripper(self, speed=None, wait=False):
        """Open gripper
        :param speed: gripper speed, range [1, 5000] r/min
        :param wait: whether to wait for the action to complete, default is False.
        """
        ret_gripper = self.arm.set_gripper_position(
            self.gripper_limits[1], speed=speed, wait=wait, wait_motion=wait
        )
        return ret_gripper

    @staticmethod
    def build_grasp_pose(
        center, approaching=[0.0, 0.0, -1.0], closing=[1.0, 0.0, 0.0]
    ) -> Pose:
        center = np.asarray(center)
        approaching, closing = np.asarray(approaching), np.asarray(closing)
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return Pose(T)

    # ---------------------------------------------------------------------- #
    # Get robot information
    # ---------------------------------------------------------------------- #
    def get_qpos(self) -> np.ndarray:
        """Get xarm qpos in maniskill2 format
        :return qpos: xArm7 joint angles, [9,] np.float32 np.ndarray
        """
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        _, gripper_pos = self.arm.get_gripper_position()  # [-10, 850]

        gripper_qpos = clip_and_scale_action(
            gripper_pos, self.joint_limits_ms2[-1, :], self.gripper_limits
        )
        return np.asarray(qpos + [gripper_qpos, gripper_qpos], dtype=np.float32)

    def get_qvel(self) -> np.ndarray:
        """Get xarm qvel in maniskill2 format
        :return qvel: xArm7 joint velocities, [9,] np.float32 np.ndarray
        """
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        return np.asarray(qvel + [0.0, 0.0], dtype=np.float32)  # No gripper qvel

    def get_qf(self) -> np.ndarray:
        """Get xarm qf (joint torques) in maniskill2 format
        :return qf: xArm7 joint torques, [9,] np.float32 np.ndarray
        """
        _, (qpos, qvel, effort) = self.arm.get_joint_states(is_radian=True)
        return np.asarray(effort + [0.0, 0.0], dtype=np.float32)  # No gripper qf

    def get_tcp_pose(self, unit_in_mm=False) -> Pose:
        """Get TCP pose in world frame
        :return pose: If unit_in_mm, position unit is mm. Else, unit is m.
        """
        # arm.get_position() rounds the values to 6 decimal places
        # call arm_cmd.get_tcp_pose() instead
        ret = self.arm._arm.arm_cmd.get_tcp_pose()
        ret[0] = self.arm._arm._check_code(ret[0])
        for v in ret[1:]:
            if not math.isfinite(v):
                self.logger.critical(f"Return value not finite: {ret[1:]=}")

        xyzrpy = np.asarray(ret[1:], dtype=np.float32)
        pose_base_tcp = Pose(
            p=xyzrpy[:3] if unit_in_mm else xyzrpy[:3] / 1000,
            q=euler2quat(*xyzrpy[3:], axes="sxyz"),
        )
        return self.pose * pose_base_tcp

    def get_gripper_position(self, unit_in_mm=False) -> float:
        """Get gripper opening width
        :return pos: if unit_in_mm, position unit is mm. Else unit is m.
        """
        _, gripper_pos = self.arm.get_gripper_position()
        return gripper_pos / 10.0 if unit_in_mm else gripper_pos / 10000.0

    # ---------------------------------------------------------------------- #
    # Observations
    # ---------------------------------------------------------------------- #
    def get_proprioception(self):
        obs = OrderedDict(qpos=self.get_qpos(), qvel=self.get_qvel())
        # controller_state = self.controller.get_state()
        # if len(controller_state) > 0:
        #     obs.update(controller=controller_state)
        return obs

    def get_state(self) -> dict:
        """Get current state, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        state["robot_root_pose"] = vectorize_pose(self.pose)
        state["robot_qpos"] = self.get_qpos()
        state["robot_qvel"] = self.get_qvel()
        state["robot_qf"] = self.get_qf()

        # controller state
        # state["controller"] = self.controller.get_state()

        return state

    def set_state(self, state: dict, ignore_controller=False):
        # robot state
        pose_array = state["robot_root_pose"]
        self.pose = Pose(p=pose_array[:3], q=pose_array[3:])
        self.set_qpos(state["robot_qpos"])
        # self.robot.set_qvel(state["robot_qvel"])
        # self.robot.set_qacc(state["robot_qacc"])

        # if not ignore_controller and "controller" in state:
        #     self.controller.set_state(state["controller"])

    def run_as_process(self):
        """Run XArm7 as a separate process"""
        self.logger.info(f"Running {self!r} as a separate process")

        # XArm7 control
        so_joined = SharedObject("join_xarm7_real")
        so_sync = SharedObject("sync_xarm7_real")
        so_start = SharedObject("start_xarm7_real", data=False)
        # data
        urdf_path = (
            f"{ASSET_DIR}/descriptions/"
            f"{'xarm7_d435.urdf' if self.with_hand_camera else 'xarm7.urdf'}"
        )
        so_urdf_path = SharedObject("xarm7_real_urdf_path", data=urdf_path)
        so_qpos = SharedObject("xarm7_real_qpos", data=np.zeros(8, dtype=np.float32))
        so_qvel = SharedObject("xarm7_real_qvel", data=np.zeros(8, dtype=np.float32))
        so_qf = SharedObject("xarm7_real_qf", data=np.zeros(8, dtype=np.float32))
        so_tcp_pose = SharedObject("xarm7_real_tcp_pose", data=Pose())

        signal_process_ready()  # current process is ready

        while not so_joined.triggered:
            if so_start.fetch():
                _, (qpos, qvel, qf) = self.arm.get_joint_states(is_radian=True)
                _, gripper_pos = self.arm.get_gripper_position()  # [-10, 850]
                pose_world_tcp = self.get_tcp_pose()

                so_qpos.assign(
                    np.asarray(
                        qpos
                        + [
                            gripper_pos / 1000.0,
                        ],
                        dtype=np.float32,
                    )
                )
                so_qvel.assign(
                    np.asarray(
                        qvel
                        + [
                            0.0,
                        ],
                        dtype=np.float32,
                    )
                )
                so_qf.assign(
                    np.asarray(
                        qf
                        + [
                            0.0,
                        ],
                        dtype=np.float32,
                    )
                )
                so_tcp_pose.assign(pose_world_tcp)

        self.logger.info(f"Process running {self!r} is joined")
        # Unlink created SharedObject
        so_joined.unlink()
        so_sync.unlink()
        so_start.unlink()
        so_urdf_path.unlink()
        so_qpos.unlink()
        so_qvel.unlink()
        so_qf.unlink()
        so_tcp_pose.unlink()

    @property
    def robot(self):
        """An alias for compatibility."""
        return self

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    @property
    def motion_mode(self):
        """Get the currently activated controller uid."""
        return self._motion_mode

    @property
    def action_space(self):
        if self._control_mode == "ee_pos":
            return spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        elif self._control_mode == "ee_delta_pos":
            return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # elif self._control_mode == "ee_pose":
        #     # [x, y, z, r, p, y, gripper], xyz in meters, rpy in radian
        #     return spaces.Box(low=np.array([-np.inf]*3 + [-np.pi]*3 + [-1]),
        #                       high=np.array([np.inf]*3 + [np.pi]*3 + [1]),
        #                       shape=(7,), dtype=np.float32)
        elif self._control_mode == "ee_pose_axangle":
            # [x, y, z, *rotvec, gripper], xyz in meters
            #   rotvec is in axis of rotation and its norm gives rotation angle
            return spaces.Box(
                low=np.array([-np.inf] * 6 + [-1]),
                high=np.array([np.inf] * 6 + [1]),
                shape=(7,),
                dtype=np.float32,
            )
        elif self._control_mode == "ee_delta_pose_axangle":
            # [x, y, z, *rotvec, gripper], xyz in meters
            #   rotvec is in axis of rotation and its norm gives rotation angle
            return spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        elif self._control_mode == "ee_pose_quat":
            # [x, y, z, w, x, y, z, gripper], xyz in meters, wxyz is unit quaternion
            return spaces.Box(
                low=np.array([-np.inf] * 3 + [-1] * 4 + [-1]),
                high=np.array([np.inf] * 3 + [1] * 4 + [1]),
                shape=(8,),
                dtype=np.float32,
            )
        elif self._control_mode == "ee_delta_pose_quat":
            # [x, y, z, w, x, y, z, gripper], xyz in meters, wxyz is unit quaternion
            return spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported {self._control_mode = }")

    @property
    def cameras(self) -> CameraConfig:
        """CameraConfig of cameras attached to agent"""
        pose_tcp_cam = (
            Pose(p=[0, 0, 0.177]).inv()
            * Pose(
                p=[-0.06042734, 0.0175, 0.02915237],
                q=euler2quat(np.pi, -np.pi / 2 - np.pi / 12, np.pi),
            )
            * Pose(p=[0, 0.015, 0])
        )  # camera_color_frame
        return CameraConfig(
            uid="hand_camera",
            device_sn="146322076186",
            pose=pose_tcp_cam,
            config=(848, 480, 30),
            preset="High Accuracy",
            # depth_option_kwargs={rs.option.exposure: 1500},
            parent_pose_so_name="xarm7_real_tcp_pose",
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: ip={self.ip}, "
            f"control_mode={self._control_mode}, motion_mode={self._motion_mode}, "
            f"with_hand_camera={self.with_hand_camera}>"
        )
