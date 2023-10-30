from pathlib import Path
import cv2
import numpy as np
import torch
import time
import sapien
from sapien import Pose
from matplotlib import pyplot as plt

from real_robot.utils.visualization import Visualizer
# from real_robot.sensors.simsense_depth import SimsenseDepth
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
    camera = scene.add_mounted_camera(
        "camera",
        rgb_cam_actor.entity,
        pose=Pose(),
        width=int(img_width // render_downsample_factor),
        height=int(img_height // render_downsample_factor),
        fovy=np.deg2rad(78.0), # D435 fovy
        near=0.01,
        far=10.0,
    )
    camera.set_focal_lengths(intrinsics[0,0] * render_width_scaling, intrinsics[1,1] * render_height_scaling)
    camera.set_principal_point(intrinsics[0,2] * render_width_scaling, intrinsics[1,2] * render_height_scaling)
    
    return scene, robot_actor, camera

def obtain_gripper_gt_depth_img(scene, robot_actor, camera, qpos, orig_width, orig_height):
    import time
    tt = time.time()
    robot_actor.set_qpos(qpos)
    # scene.step() # Note: for sapien2, DO NOT call scene.step(); otherwise the qpos will be wrong.
    scene.update_render()
    camera.take_picture()
    depth_img = camera.get_picture('Position') # [H, W, 4]
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
    # ckpt_path = '/home/xuanlin/activezero2_official/model_oct23_balanced.pth' # model_oct24_balanced_reproj2.pth' # model.pth' # model_oct23_balanced.pth' # model.pth'
    # ckpt_path = '/home/xuanlin/activezero2_official/model_oct25_balanced_dtd.pth'
    # disparity_mode = "regular"
    # ckpt_path = '/home/xuanlin/activezero2_official/model_oct27_loglinear_disparity.pth'
    # ckpt_path = '/home/xuanlin/activezero2_official/model_oct27_loglinear_disparity_384_reweight.pth'
    # ckpt_path = '/home/xuanlin/activezero2_official/model_oct28_loglinear_disp384_newdata.pth'
    ckpt_path = '/home/xuanlin/activezero2_official/model_oct28_loglinear_disp384_normal_coeff1_newdata.pth'
    disparity_mode = "log_linear" if 'loglinear' in ckpt_path else 'regular'
    loglinear_disp_c = -0.02 if '384' in ckpt_path else 0.01
    img_resize = (424, 240) # [resize_W, resize_H]
    device = 'cuda:0'
    disp_conf_topk = 2 if 'k4' not in ckpt_path else 4
    disp_conf_thres = 0.0 # 0.8 # 0.95
    MAX_DISP = 384 if '384' in ckpt_path else 256
    ckpt_pred_normal = 'normal' in ckpt_path and not 'normalv2' in ckpt_path
    # ckpt_pred_normal_v2 = 'normalv2' in ckpt_path

    model = CGI_Stereo(maxdisp=MAX_DISP, disparity_mode=disparity_mode, loglinear_disp_c=loglinear_disp_c, 
                       predict_normal=ckpt_pred_normal) # , predict_normal_v2=ckpt_pred_normal_v2)
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

    # depth_engine = SimsenseDepth(
    #     ir_shape[::-1], k_l=k_irl, k_r=k_irr, l2r=T_irr_irl, k_rgb=k_rgb,
    #     rgb_size=images["rgb"].shape[1::-1], l2rgb=T_rgb_irl,
    #     max_disp=1024, median_filter_size=3, depth_dilation=True
    # )

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
        pose_world_camCV = Pose(params["cam2world_cv"])

        # RS capture
        obs_dict[f"vis_rs|{tag}_hand_camera_color"] = images["rgb"]
        obs_dict[f"vis_rs|{tag}_hand_camera_depth"] = images["depth"]
        obs_dict[f"vis_rs|{tag}_hand_camera_intr"] = params["intrinsic_cv"]
        obs_dict[f"vis_rs|{tag}_hand_camera_infrared_1"] = images["ir_l"]
        obs_dict[f"vis_rs|{tag}_hand_camera_infrared_2"] = images["ir_r"]
        obs_dict[f"vis_rs|{tag}_hand_camera_pose"] = pose_world_camCV

        # ----- ActiveZero++ ----- #
        orig_h, orig_w = images["ir_l"].shape
        focal_length = k_irl[0, 0] * img_resize[0] / orig_w
        baseline = np.linalg.norm(T_irr_irl[:3, -1])
        focal_length_arr = torch.tensor([focal_length], device=device).float()
        baseline_arr = torch.tensor([baseline], device=device).float()
        print("focal_length", focal_length, "baseline", baseline, "focal_length * baseline", focal_length * baseline)
        tt = time.time()
        with torch.no_grad():
            pred_dict = model({
                'img_l': preprocess_image(images["ir_l"]),
                'img_r': preprocess_image(images["ir_r"]),
                'focal_length': focal_length_arr,
                'baseline': baseline_arr,
            }, predict_normal=False)
            if disparity_mode == "log_linear":
                pred_dict['pred_orig'] = model.to_raw_disparity(
                    pred_dict['pred_orig'], focal_length_arr, baseline_arr
                )
                pred_dict['pred_div4'] = model.to_raw_disparity(
                    pred_dict['pred_div4'], focal_length_arr, baseline_arr
                )
            # calculate disparity confidence mask; (doing this on gpu is significantly faster than on cpu)
            disparity_confidence = pred_dict['cost_prob'].topk(disp_conf_topk, dim=1).values.sum(dim=1) # [1, H, W]
            pred_dict['disparity_conf_mask'] = disparity_confidence > disp_conf_thres
            torch.cuda.synchronize()
        print("pred time", time.time() - tt)
        for k in pred_dict:
            pred_dict[k] = pred_dict[k].detach().cpu().numpy()
        disparity = pred_dict['pred_orig'] # [1, H, W]
        disparity = disparity.squeeze() # [H, W]
        disparity_conf_mask = pred_dict['disparity_conf_mask'].squeeze() # [H, W]

        # disparity => depth
        depth = focal_length * baseline / (disparity + 1e-6)
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
        
        # green_img = np.zeros_like(images["rgb"])
        # green_img[..., 1] = 255
        # plt.subplot(1,2,1)
        # plt.imshow(depth)
        # plt.title(f"{image_path}")
        # plt.subplot(1,2,2)
        # tmp_img = images["rgb"].copy()
        # tmp_img[gripper_overlay_mask] = green_img[gripper_overlay_mask] * 0.2 + tmp_img[gripper_overlay_mask] * 0.8
        # plt.imshow(tmp_img)
        # plt.show()
        
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

        # # ----- Simsense ----- #
        # depth_simsense = depth_engine.compute(images["ir_l"], images["ir_r"])
        # # SimSense results
        # obs_dict[f"vis_simsense|{tag}_hand_camera_color"] = images["rgb"]
        # obs_dict[f"vis_simsense|{tag}_hand_camera_depth"] = depth_simsense
        # obs_dict[f"vis_simsense|{tag}_hand_camera_intr"] = params["intrinsic_cv"]
        # obs_dict[f"vis_simsense|{tag}_hand_camera_pose"] = pose_world_camCV

    obs_dict = dict(sorted(obs_dict.items()))  # sort by key

    visualizer = Visualizer(run_as_process=True)
    visualizer.show_obs(obs_dict)
    while True:
        visualizer.render()

    list(obs_dict.keys())