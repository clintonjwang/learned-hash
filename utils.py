import numpy as np

from PIL import Image
import os
import torch
import math
import numpy as np
import scipy.signal
from typing import List, Optional

import matplotlib.pyplot as plt
from IPython.display import clear_output
import metrics

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def to_jpg(pil_img, quality=50):
    pil_img.save('tmp.jpg', optimize=True, quality=quality)
    size = os.path.getsize('tmp.jpg')
    img = Image.open('tmp.jpg')
    os.remove('tmp.jpg')
    return np.array(img) / 255., size/1024

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))

def load_cfg(runname):
    return np.load(f'results/{runname}.npy', allow_pickle=True).item()

def sample_img(img, coords):
    H,W = img.shape[:2]
    coords = torch.clip(coords, 0, 1)
    y = (coords[..., 0] * (H-1)).long()
    x = (coords[..., 1] * (W-1)).long()
    return img[y,x]

def plot(model, losses, img, iteration, plot_every):
    if plot_every and (iteration+1) % plot_every == 0:
        clear_output(wait=True)
        fig, axes = plt.subplots(1,3, figsize=(10, 2.5), tight_layout=True)
        axes[0].imshow(img)
        axes[0].set_axis_off()
        axes[0].set_title('Ground Truth')
        render = model.render(to_numpy=True)
        ngp_psnr = metrics.psnr(render, img)
        axes[1].imshow(render)
        axes[1].set_title(f'Render ({ngp_psnr:.1f}dB)')
        axes[1].set_axis_off()
        axes[2].plot(losses)
        axes[2].set_ylim(0, 0.01)
        axes[2].set_title(f'Training loss')
        plt.show()


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()