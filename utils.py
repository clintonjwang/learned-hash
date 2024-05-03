import numpy as np

from PIL import Image
import os

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