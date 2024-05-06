import numpy as np

from torchvision.datasets import MNIST

def get_mnist_texture(mnist=None):
    if mnist is None:
        mnist = MNIST('data/mnist')
    h,w = 14,14
    H,W = h*4,w*4
    np.random.seed(64)
    n_textures = 5
    toy_texture = np.zeros((H,W,3))
    textures = np.stack([np.array(mnist[np.random.randint(len(mnist))][0]) for _ in range(n_textures)], axis=0)
    textures = textures[:, ::2, ::2]
    textures[0] = 0
    textures = textures[..., None]
    for x in range(0, H, h):
        for y in range(0, W, w):
            toy_texture[x:x+h, y:y+w] = textures[np.random.randint(n_textures)]
    img = toy_texture / 255.
    return img