from real_robot.utils.registration import register_env

from .base_env import XArmBaseEnv


@register_env("GraspAnything-v0", max_episode_steps=10000,
              obs_mode="rgbd",
              control_mode="ee_pose_quat",
              motion_mode="cartesian_online",
              image_obs_mode="hand")
class GraspAnythingEnv(XArmBaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
