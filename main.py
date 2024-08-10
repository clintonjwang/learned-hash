import argparse

import random
import numpy as np
import torch
nn = torch.nn

import train


def seed_everything(args):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def main():
    parser = argparse.ArgumentParser(description='Image Loader')
    parser.add_argument('-n', '--name', type=str, default='default')
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    parser.add_argument('--jpg', action='store_true')
    parser.add_argument('--ngp', action='store_true')
    parser.add_argument('--ours', action='store_true')
    parser.add_argument('--vqrf', action='store_true')
    parser.add_argument('--vqngp', action='store_true')
    parser.add_argument('--vbnerf', action='store_true')
    
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    args = parser.parse_args()
    seed_everything(args)
    train.run_2d(args)
    
if __name__ == '__main__':
    main()