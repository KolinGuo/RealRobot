import argparse

import numpy as np
import mediapy as media


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create video from extracted .npz file"
    )
    parser.add_argument('src_file', type=str)
    parser.add_argument('tgt_file', type=str)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--frame-skip', type=int, default=1)

    args = parser.parse_args()

    f = np.load(args.src_file)
    if not isinstance(f, np.ndarray):
        f = f['rgb_image']

    media.write_video(args.tgt_file, f[args.start_frame::args.frame_skip], fps=args.fps)
