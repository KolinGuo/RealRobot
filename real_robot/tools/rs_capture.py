import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from real_robot.sensors.camera import Camera, CameraConfig
from real_robot.utils.logger import get_logger
from real_robot.utils.multiprocessing import (
    SharedObject,
    ctx,
    start_and_wait_for_process,
)
from real_robot.utils.visualization import CV2Visualizer

try:
    from pynput import keyboard
except ImportError as e:
    get_logger("rs_capture.py").warning(f"ImportError: {e}")
    raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture Color/Depth/IR images from RS camera"
    )
    parser.add_argument(
        "--save-dir", type=str, default="capture", help="Path to saving directory."
    )
    parser.add_argument("--prefix", type=str, default="", help="Saving npz file prefix")
    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Saving to "{save_dir}"\n')

    camera = Camera(
        CameraConfig(
            "camera",
            config={
                "Color": (848, 480, 60),
                "Depth": (848, 480, 60),
                "Infrared 1": (848, 480, 60),
                "Infrared 2": (848, 480, 60),
            },
            preset="Default",
            depth_option_kwargs={rs.option.laser_power: 360},
        ),
    )

    # start CV2Visualizer
    cv2vis_proc = ctx.Process(
        target=CV2Visualizer,
        name="CV2Visualizer",
        args=(),
        kwargs=dict(
            run_as_process=True,
            stream_camera=True,
        ),
    )
    start_and_wait_for_process(cv2vis_proc, timeout=30)
    so_cv2vis_joined = SharedObject("join_viscv2")

    print("\n\x1b[36mPress 'c' to capture and 'esc' to end\x1b[0m\n")

    while True:
        with keyboard.Events() as events:
            event = events.get()
            if event is None:
                print("Waited too long, but get nothing")
            elif event.key == keyboard.Key.esc:
                print("Ending capture")
                break
            elif isinstance(
                event, events.Press
            ) and event.key == keyboard.KeyCode.from_char("c"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = str(save_dir / f"{args.prefix}{timestamp}")
                print(f"Captured and saved as {save_path}_[images|params].npz")

                camera.take_picture()
                np.savez_compressed(f"{save_path}_images.npz", **camera.get_images())
                np.savez_compressed(f"{save_path}_params.npz", **camera.get_params())

    so_cv2vis_joined.trigger()
    cv2vis_proc.join()
    del camera
