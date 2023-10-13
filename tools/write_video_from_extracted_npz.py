import argparse
import numpy as np
import mediapy as media

parser = argparse.ArgumentParser()
parser.add_argument('--src-file', type=str, required=True)
parser.add_argument('--tgt-file', type=str, required=True)
parser.add_argument('--start-frame', type=int, default=0)
parser.add_argument('--fps', type=int, default=10)
parser.add_argument('--frame-skip', type=int, default=1)

args = parser.parse_args()

f = np.load(args.src_file)
if not isinstance(f, np.ndarray):
    f = f['rgb_image']

media.write_video(args.tgt_file, f[args.start_frame::args.frame_skip], fps=args.fps)
