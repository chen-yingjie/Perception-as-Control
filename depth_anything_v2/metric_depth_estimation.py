import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch

import sys
sys.path.append('.')
from .depth_anything_v2.dpt import DepthAnythingV2

def estimate(depth_anything, input_size, imagepath, output_dir, is_vis=False):
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    imagename = imagepath.split('/')[-1].split('.')[0]

    raw_image = cv2.imread(imagepath)
    
    metric_depth = depth_anything.infer_image(raw_image, input_size)
    
    output_path = os.path.join(output_dir, imagename + '_depth_meter.npy')
    np.save(output_path, metric_depth)
    
    relative_depth = (metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min())

    vis_depth = None
    if is_vis:
        vis_depth = relative_depth * 255.0
        vis_depth = vis_depth.astype(np.uint8)
        vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(output_dir, imagename + '_depth.png')
        cv2.imwrite(output_path, vis_depth)

    return metric_depth, relative_depth, vis_depth


def depth_estimation(imagepath, 
              input_size=518, 
              encoder='vitl', 
              load_from='./depth_anything_v2/ckpts/depth_anything_v2_metric_hypersim_vitl.pth',
              max_depth=20,
              output_dir='./tmp',
              is_vis=False):
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    metric_depth, relative_depth, vis_depth = estimate(depth_anything, input_size, imagepath, output_dir, is_vis)

    return metric_depth, relative_depth, vis_depth


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-size', type=int, default=518)
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='../Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)

    # set the gpu
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--output_dir', type=str, default='./tmp', help='output dir')
    parser.add_argument('--imagepath', type=str, default='tmp.jpg', help='imagepath')

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # set the gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    estimate(depth_anything, args.input_size, args.imagepath, args.output_dir)