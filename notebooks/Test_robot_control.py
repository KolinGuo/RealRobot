import numpy as np

from real_robot.agents.xarm import XArm7
from real_robot.utils.logger import _log_dir, get_logger

logger = get_logger("robot")
get_logger(log_file=_log_dir / "3rd_party.log")  # root logger log file

robot = XArm7(control_mode="ee_pose_quat", motion_mode="cartesian_online")
# robot = XArm7(control_mode="ee_pose_quat", motion_mode="position")

action_grasp = [0.33488411, -0.24891135, 0.04203321, 0.00724707, 0.04314691, 0.99346465, -0.10542212, 1.]
action_close = [0.33400261, -0.24671784, 0.04204423, -0.00857807, -0.01663382, 0.99924785, -0.03396245, -1.]
action_lift = [0.33488411, -0.24891135, 0.14203322, -0.00724707, -0.0431469, -0.99346465, 0.10542216, -1.]

robot.arm.set_simulation_robot(False)

robot.reset()
robot

import math

way_points = np.asarray([
    [0.03, -0.21, 0.18, 0, 1/math.sqrt(2), 1/math.sqrt(2), 0, -1],
    [0.03, -0.49, 0.18, 0, 1/math.sqrt(2), 1/math.sqrt(2), 0, 1],
    [0.44, -0.21, 0.18, 0, 1/math.sqrt(2), 1/math.sqrt(2), 0, -1],
    [0.44, -0.49, 0.18, 0, 1/math.sqrt(2), 1/math.sqrt(2), 0, 1],
])

for way_point in way_points:
    robot.set_action(way_point, speed=100.0, mvacc=10000.0, wait=False)

robot.set_action(way_points[-1], speed=100.0, mvacc=10000.0, wait=True)

for way_point in way_points:
    robot.set_action(way_point, speed=100.0, mvacc=10000.0, wait=True)
