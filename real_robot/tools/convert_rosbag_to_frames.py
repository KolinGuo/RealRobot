import argparse
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read recorded bag file and save rgb_image and depth_image into .npz"
        )
    )
    parser.add_argument("bag_path", type=str, help="Path to the rosbag")
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Path to saving directory. If not provided, save to same directory",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable pyrealsense2 logging to console"
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_path).resolve()
    if args.save_dir is None:
        save_npz_path = bag_path.with_suffix(".npz")
    else:
        save_npz_path = Path(args.save_dir) / (bag_path.stem + ".npz")
        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Converting rosbag "{bag_path}"\n')

    if args.debug:
        rs.log_to_console(rs.log_severity.debug)

    # Create pipeline and config from rosbag
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    # Start streaming from file
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Get camera intrinsics
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_profile.intrinsics
    width, height = color_intrinsics.width, color_intrinsics.height
    fx, fy = color_intrinsics.fx, color_intrinsics.fy
    cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
    print(f"Found Color stream with profile {color_profile}")
    print(f"Found Depth stream with profile {depth_profile}")

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align = rs.align(rs.stream.color)
    print("Aligning all frames to color frame\n")

    # Store the frames
    frames_dict = {
        "intrinsics": np.array([fx, fy, cx, cy]),
        "rgb_image": [],
        "depth_image": [],
    }

    with tqdm() as pbar:
        while True:
            # Get time-synchronized frames of each enabled stream in the pipeline
            frames_exist, frames = pipeline.try_wait_for_frames()
            if not frames_exist:
                break

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Verify intrinsics
            aligned_intrinsics = (
                aligned_frames.get_profile().as_video_stream_profile().intrinsics
            )
            np.testing.assert_allclose(
                frames_dict["intrinsics"],
                [
                    aligned_intrinsics.fx,
                    aligned_intrinsics.fy,
                    aligned_intrinsics.ppx,
                    aligned_intrinsics.ppy,
                ],
            )

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # Use copy so frame resources can be released
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()

            frames_dict["rgb_image"].append(color_image)
            frames_dict["depth_image"].append(depth_image)

            pbar.update(1)

    for k, v in frames_dict.items():
        frames_dict[k] = np.stack(v)

    print(
        f"Read {frames_dict['rgb_image'].shape} rgb frames "
        f"and {frames_dict['depth_image'].shape} depth frames\n"
    )

    print("Saving to compressed npz ...")
    np.savez_compressed(save_npz_path, **frames_dict)
    print(f'Saved frames to "{save_npz_path}"')
