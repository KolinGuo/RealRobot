from pathlib import Path
import cv2
import numpy as np
import torch
import sapien.core as sapien
from sapien.core import Pose
from matplotlib import pyplot as plt

from real_robot.utils.visualization import Visualizer
from real_robot.sensors.simsense_depth import SimsenseDepth
from active_zero2.models.cgi_stereo.cgi_stereo import CGI_Stereo

def create_robot_scene(img_width, img_height, intrinsics, render_downsample_factor=1.0):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 100.0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot_actor: sapien.Articulation = loader.load(
        "/home/xuanlin/kolin_maniskill2/rl_benchmark/real_robot/real_robot/assets/descriptions/xarm7_d435.urdf" # xarm_floating_pris_finger_d435.urdf"
    )
    print(robot_actor.get_links())
    print([x.name for x in robot_actor.get_active_joints()])
    rgb_cam_actor = [x for x in robot_actor.get_links() if x.name == 'camera_color_frame'][0]
    
    robot_actor.set_root_pose(sapien.Pose([0, 0, 3], [1, 0, 0, 0]))

    # Set initial joint positions
    qpos = np.zeros(len(robot_actor.get_active_joints()))
    robot_actor.set_qpos(qpos)

    render_width_scaling = img_width // render_downsample_factor / img_width
    render_height_scaling = img_height // render_downsample_factor / img_height
    camera = scene.add_camera(
        name="camera",
        width=int(img_width // render_downsample_factor),
        height=int(img_height // render_downsample_factor),
        fovy=np.deg2rad(78.0), # D435 fovy
        near=0.01,
        far=10.0,
    )
    camera.set_focal_lengths(intrinsics[0,0] * render_width_scaling, intrinsics[1,1] * render_height_scaling)
    camera.set_principal_point(intrinsics[0,2] * render_width_scaling, intrinsics[1,2] * render_height_scaling)
    camera.set_parent(parent=rgb_cam_actor, keep_pose=False)
    
    return scene, robot_actor, camera

def obtain_gripper_gt_depth_img(scene, robot_actor, camera, qpos, orig_width, orig_height):
    import time
    tt = time.time()
    robot_actor.set_qpos(qpos)
    # scene.step() # Note: for sapien2, DO NOT call scene.step(); otherwise the qpos will be wrong.
    scene.update_render()
    camera.take_picture()
    depth_img = camera.get_float_texture('Position') # [H, W, 4]
    depth_img = -depth_img[..., 2] # note: valid depth pixels in depth_img are those whose depth value > 0
    if depth_img.shape[0] != orig_height or depth_img.shape[1] != orig_width:
        depth_img = cv2.resize(depth_img, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    print(time.time() - tt) # rendering first time is slow and takes ~0.2s; second time will be very fast (2.5ms)
    return depth_img

if __name__ == '__main__':
    data_dir = Path("/home/xuanlin/Downloads/capture").resolve()
    image_paths = [p for p in data_dir.glob("*_images.npz") if "unaligned" not in p.name]  # all aligned images

    # ----- ActiveZero++ ----- #
    # ckpt_path = '/home/xuanlin/activezero2_official/model.pth'
    ckpt_path = '/home/xuanlin/activezero2_official/model_oct22_balanced.pth'
    img_resize = (424, 240) # [resize_W, resize_H]
    device = 'cuda:0'
    disp_conf_topk = 2
    disp_conf_thres = 0.8 # 0.95
    MAX_DISP = 256

    model = CGI_Stereo(maxdisp=MAX_DISP)
    model.load_state_dict(torch.load(ckpt_path)['model'])
    model = model.to(device)

    def preprocess_image(image: np.ndarray) -> torch.Tensor:
        img_L = cv2.resize((image / 255.0).astype(np.float32), img_resize, interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(img_L).to(device)[None, None, ...]  # [1, 1, *img_resize]

    # ----- Simsense ----- #
    image_path = image_paths[0]
    images = np.load(image_path)
    params = np.load(image_path.parent / image_path.name.replace("images", "params"))

    ir_shape = images["ir_l"].shape
    rgb_shape = images["rgb"].shape
    assert images["ir_r"].shape == ir_shape, f'Mismatched IR shape: left={ir_shape}, right={images["ir_r"].shape}'
    k_rgb = params["intrinsic_cv"]

    k_irl = k_irr = np.array([[430.13980103,   0.        , 425.1628418 ],
                            [  0.        , 430.13980103, 235.27651978],
                            [  0.        ,   0.        ,   1.        ]])

    # rsdevice.all_extrinsics["Infrared 1=>Infrared 2"]
    T_irr_irl = np.array([[ 1.       ,  0.       ,  0.       , -0.0501572],
                        [ 0.       ,  1.       ,  0.       ,  0.       ],
                        [ 0.       ,  0.       ,  1.       ,  0.       ],
                        [ 0.       ,  0.       ,  0.       ,  1.       ]])
    # rsdevice.all_extrinsics["Infrared 1=>Color"]
    T_rgb_irl = np.array([[ 9.99862015e-01,  1.34780351e-02,  9.70994867e-03, 1.48976548e-02],
                        [-1.35059441e-02,  9.99904811e-01,  2.81448336e-03, 1.15314942e-05],
                        [-9.67109110e-03, -2.94523709e-03,  9.99948919e-01, 1.56505470e-04],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

    depth_engine = SimsenseDepth(
        ir_shape[::-1], k_l=k_irl, k_r=k_irr, l2r=T_irr_irl, k_rgb=k_rgb,
        rgb_size=images["rgb"].shape[1::-1], l2rgb=T_rgb_irl,
        max_disp=1024, median_filter_size=3, depth_dilation=True
    )

    # robot_scene, robot_actor, gripper_camera = create_robot_scene(
    #     img_resize[0], img_resize[1], 
    #     k_irl * np.array([[img_resize[0] / ir_shape[1]], [img_resize[1] / ir_shape[0]], [1]])
    # )
    gt_gripper_render_downsample_factor = 2.0
    robot_scene, robot_actor, gripper_camera = create_robot_scene(rgb_shape[1], rgb_shape[0], k_rgb, gt_gripper_render_downsample_factor)
    qpos = np.zeros(len(robot_actor.get_active_joints()))
    gripper_qpos = 0.85 # robot_actor.get_qlimits()[-1:,-1]
    # given a gripper position, set qpos[-6:] to be the same 
    qpos[-6:] = gripper_qpos
    print(qpos)

    obs_dict = {}

    for image_path in image_paths:
        images = np.load(image_path)
        params = np.load(image_path.parent / image_path.name.replace("images", "params"))

        tag = image_path.stem.replace("_images", "")
        pose_world_camCV = Pose.from_transformation_matrix(params["cam2world_cv"])

        # RS capture
        obs_dict[f"vis_rs|{tag}_hand_camera_color"] = images["rgb"]
        obs_dict[f"vis_rs|{tag}_hand_camera_depth"] = images["depth"]
        obs_dict[f"vis_rs|{tag}_hand_camera_intr"] = params["intrinsic_cv"]
        obs_dict[f"vis_rs|{tag}_hand_camera_infrared_1"] = images["ir_l"]
        obs_dict[f"vis_rs|{tag}_hand_camera_infrared_2"] = images["ir_r"]
        obs_dict[f"vis_rs|{tag}_hand_camera_pose"] = pose_world_camCV

        # ----- ActiveZero++ ----- #
        orig_h, orig_w = images["ir_l"].shape
        with torch.no_grad():
            pred_dict = model({'img_l': preprocess_image(images["ir_l"]),
                            'img_r': preprocess_image(images["ir_r"])})
            torch.cuda.synchronize()
        for k in pred_dict:
            pred_dict[k] = pred_dict[k].detach().cpu().numpy()

        disparity = pred_dict['pred_orig'] # [1, H, W]
        disparity = disparity.squeeze() # [H, W]
        disparity_probs = pred_dict['cost_prob'].squeeze() # [1, disp//4, H, W]
        top_disparity_prob_idx = np.argpartition(-disparity_probs, disp_conf_topk, axis=0)[:disp_conf_topk, :, :]
        disparity_confidence = np.take_along_axis(disparity_probs, top_disparity_prob_idx, axis=0).sum(axis=0) # [H, W]
        disparity_conf_mask = disparity_confidence > disp_conf_thres

        # disparity => depth
        focal_length = k_irl[0, 0] * img_resize[0] / orig_w
        baseline = np.linalg.norm(T_irr_irl[:3, -1])
        depth = focal_length * baseline / (disparity + 1e-5)
        # filter out depth
        depth[~disparity_conf_mask] = 0.0
     
        # Upsample predicted depth image
        depth = cv2.resize(depth, images["ir_l"].shape[::-1], interpolation=cv2.INTER_NEAREST_EXACT)
        from real_robot.utils.camera import register_depth
        depth = register_depth(depth, k_irl, k_rgb, T_rgb_irl, images["rgb"].shape[1::-1], depth_dilation=True)
        
        # Overlay gripper ground truth depth on the depth map obtained by activezero++
        depth_gt_gripper = obtain_gripper_gt_depth_img(robot_scene, robot_actor, gripper_camera, qpos, *images["rgb"].shape[1::-1])
        gripper_overlay_mask = depth_gt_gripper > 0.0
        depth[gripper_overlay_mask] = depth_gt_gripper[gripper_overlay_mask]
        
        green_img = np.zeros_like(images["rgb"])
        green_img[..., 1] = 255
        plt.subplot(1,2,1)
        plt.imshow(depth)
        plt.title(f"{image_path}")
        plt.subplot(1,2,2)
        tmp_img = images["rgb"].copy()
        tmp_img[gripper_overlay_mask] = green_img[gripper_overlay_mask] * 0.2 + tmp_img[gripper_overlay_mask] * 0.8
        plt.imshow(tmp_img)
        plt.show()
        
        # ActiveZero++ results
        obs_dict[f"vis_activezero2|{tag}_hand_camera_color"] = images["rgb"]
        obs_dict[f"vis_activezero2|{tag}_hand_camera_depth"] = depth
        obs_dict[f"vis_activezero2|{tag}_hand_camera_intr"] = params["intrinsic_cv"]
        obs_dict[f"vis_activezero2|{tag}_hand_camera_pose"] = pose_world_camCV

        # No upsample
        # resize_trans = np.array([[img_resize[0] / orig_w, 0, 0],
        #                          [0, img_resize[1] / orig_h, 0],
        #                          [0, 0, 1]], dtype=params["intrinsic_cv"].dtype)
        # # ActiveZero++ results
        # obs_dict[f"vis_activezero2|{tag}_hand_camera_color"] = cv2.resize(images["rgb"], img_resize, interpolation=cv2.INTER_CUBIC)
        # obs_dict[f"vis_activezero2|{tag}_hand_camera_depth"] = depth
        # obs_dict[f"vis_activezero2|{tag}_hand_camera_intr"] = resize_trans @ params["intrinsic_cv"]
        # obs_dict[f"vis_activezero2|{tag}_hand_camera_pose"] = pose_world_camCV

        # ----- Simsense ----- #
        depth_simsense = depth_engine.compute(images["ir_l"], images["ir_r"])
        # SimSense results
        obs_dict[f"vis_simsense|{tag}_hand_camera_color"] = images["rgb"]
        obs_dict[f"vis_simsense|{tag}_hand_camera_depth"] = depth_simsense
        obs_dict[f"vis_simsense|{tag}_hand_camera_intr"] = params["intrinsic_cv"]
        obs_dict[f"vis_simsense|{tag}_hand_camera_pose"] = pose_world_camCV

    obs_dict = dict(sorted(obs_dict.items()))  # sort by key

    visualizer = Visualizer(run_as_process=True)
    visualizer.show_obs(obs_dict)
    while True:
        visualizer.render()

    list(obs_dict.keys())