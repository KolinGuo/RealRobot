import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from sapien.core import Pose

from real_robot.sensors.camera import (
    CALIB_CAMERA_POSES, CameraConfig, Camera
)
from real_robot import ASSET_DIR
from real_robot.utils.visualization import Visualizer
from real_robot.utils.realsense import get_connected_rs_devices
from real_robot.utils.camera import depth2xyz, transform_points


def demo_stream_obs():
    visualizer = Visualizer(run_as_process=True, stream_camera=True, stream_robot=True)
    visualizer.reset()

    visualizer.show_obs({
        # "viso3d_saved_camera|captured_pcd_color": rgb_frame,
        # "viso3d_saved_camera|captured_pcd_pts": pts_camera,
        # "viso3d_saved_camera|captured_pcd_pose": Pose.from_transformation_matrix(T_world_camCV),
        # "viso3d_saved_camera|captured_pcd_color": rgb_frame,
        # "viso3d_saved_camera|captured_pcd_xyzimg": xyz_image,
        # "viso3d_saved_camera|captured_pcd_pose": Pose.from_transformation_matrix(T_world_camCV),
        "vis_saved_camera_color": rgb_frame,
        "vis_saved_camera_depth": depth_frame,
        "vis_saved_camera_intr": K,
        "vis_saved_camera_mask": mask,
        "vis_saved_camera_pose": Pose.from_transformation_matrix(T_world_camCV),
        # Gripper Pose
        "robot_gripper_urdf_path": f"{ASSET_DIR}/descriptions/xarm_floating_pris_finger_d435.urdf",
        "viso3d_CGN_grasps|obj1_gposes": pred_grasps_world,
        "viso3d_CGN_grasps|obj1_gscores": scores,
        "viso3d_CGN_grasps|obj1_gqvals": q_vals,
    })
    visualizer.render()

    while True:
        pass


def demo_sync_obs():
    visualizer = Visualizer(run_as_process=True,
                            stream_camera=False,
                            stream_robot=False)
    visualizer.reset()

    visualizer.show_obs({
        # "viso3d_saved_camera|captured_pcd_color": rgb_frame,
        # "viso3d_saved_camera|captured_pcd_pts": pts_camera,
        # "viso3d_saved_camera|captured_pcd_pose": Pose.from_transformation_matrix(T_world_camCV),
        # "viso3d_saved_camera|captured_pcd_color": rgb_frame,
        # "viso3d_saved_camera|captured_pcd_xyzimg": xyz_image,
        # "viso3d_saved_camera|captured_pcd_pose": Pose.from_transformation_matrix(T_world_camCV),
        "vis_saved_camera_color": rgb_frame,
        "vis_saved_camera_depth": depth_frame,
        "vis_saved_camera_intr": K,
        "vis_saved_camera_mask": mask,
        "vis_saved_camera_pose": Pose.from_transformation_matrix(T_world_camCV),
        # Gripper Pose
        "robot_gripper_urdf_path": f"{ASSET_DIR}/descriptions/xarm_floating_pris_finger_d435.urdf",
        "viso3d_CGN_grasps|obj1_gposes": pred_grasps_world,
        "viso3d_CGN_grasps|obj1_gscores": scores,
        "viso3d_CGN_grasps|obj1_gqvals": q_vals,
    })
    visualizer.render()

    camera.take_picture()
    visualizer.render()

    while True:
        camera.take_picture()
        visualizer.render()


if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    os.environ["REAL_ROBOT_LOG_DIR"] = str(
        cur_dir / f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    data_dir = cur_dir / "data/20230822_164823_cse_pink_mug_front"

    # Camera observation
    rgb_frame = np.load(data_dir / "rgb_frame.npy")
    depth_frame = np.load(data_dir / "depth_frame.npy")
    mask = np.load(data_dir / "mask.npy")
    K = np.load(data_dir / "K.npy")
    T_world_camCV = np.load(data_dir / "T_world_cam.npy")
    xyz_image = depth2xyz(depth_frame, K,
                          1000.0 if depth_frame.dtype == np.uint16 else 1.0)
    pts_camera = xyz_image.reshape(-1, 3)

    # Predicted grasp pose
    pred_grasps_world = np.load(data_dir / "pred_grasps_world.npy")
    scores = np.load(data_dir / "pred_grasps_scores.npy")
    q_vals = np.load(data_dir / "pred_grasps_q_vals.npy")

    device_sn = get_connected_rs_devices()[0]
    camera = Camera(CameraConfig(
        "front_camera", device_sn,
        CALIB_CAMERA_POSES["front_camera"],
        (848, 480, 30), preset="High Accuracy",
    ))

    # demo_stream_obs()
    demo_sync_obs()

    del camera
