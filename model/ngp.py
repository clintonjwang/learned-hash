import numpy as np
import torch
import torch.nn as nn
import pdb

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_layers: int=3,
                 ) -> None:
        super().__init__()
        layers = []
        for _ in range(n_layers-1):
            layers += [nn.Linear(in_channels, in_channels), nn.LeakyReLU(inplace=True)]
        layers.append(nn.Linear(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

class NGP(nn.Module):
    def __init__(self,
                 R: int = 4, # num resolutions
                 N: int = 1024, # num buckets
                 F: int = 2, # num features
                 shape: tuple = None,  # H*W
                 min_resolution: int = 16,
                 max_resolution: int = 150,
                 ) -> None:
        super().__init__()
        self.resolutions = list(np.round(np.logspace(np.log2(min_resolution), np.log2(max_resolution), R, base=2)).astype(int))
        self.N = N
        self.shape = shape
        self.hash_features = nn.Parameter(torch.randn(R, N, F))
        self.mlp = MLP(R*F, 3)
        self.hashmap = None
    
    def bilerp_hash(self, coords: torch.FloatTensor):
        # coords in [0,1] with shape (*, 2)
        feats = []
        for rix, res in enumerate(self.resolutions):
            x = coords * res
            x_ = torch.floor(x)
            w = x - x_
            x,y = x_[..., 0], x_[..., 1]
            feats.append(
                self.hash_lookup_2d(x, y, rix) * (1 - w[..., :1]) * (1 - w[..., 1:]) + \
                self.hash_lookup_2d(x, y + 1, rix) * (1 - w[..., :1]) * w[..., 1:] + \
                self.hash_lookup_2d(x + 1, y, rix) * w[..., :1] * (1 - w[..., 1:]) + \
                self.hash_lookup_2d(x + 1, y + 1, rix) * w[..., :1] * w[..., 1:]
            )
        feats = torch.cat(feats, dim=-1)
        return feats
        
    def hash_lookup_2d(self, x: torch.IntTensor, y: torch.IntTensor, rix: int):
        # x has shape (*, 2)
        idx = torch.bitwise_xor(x, 2654435761*y) % self.N
        if self.hashmap is None:
            return self.hash_features[rix, idx]
        else:
            return self.hash_features[rix, self.hashmap[rix, idx]]
        
    def trilerp_hash(self, x: torch.FloatTensor):
        # x in (*, 3)
        x_ = torch.floor(x)
        w = x - x_
        x,y,z = x_[..., 0], x_[..., 1], x_[..., 2]
        self.hash_lookup_3d(x, y, z) * (1 - w[..., 0]) * (1 - w[..., 1]) * (1 - w[..., 2])
        self.hash_lookup_3d(x, y, z + 1)
        self.hash_lookup_3d(x, y + 1, z)
        self.hash_lookup_3d(x, y + 1, z + 1)
        self.hash_lookup_3d(x + 1, y + 1, z + 1)
        self.hash_lookup_3d(x + 1, y + 1, z + 1)
        self.hash_lookup_3d(x + 1, y + 1, z + 1)
        
    def hash_lookup_3d(self, x: torch.IntTensor, y: torch.IntTensor, z: torch.IntTensor):
        # x in (*, 3)
        a,b = 2654435761, 805459861
        idx = torch.bitwise_xor(torch.bitwise_xor(x, a*y), b*z) % self.N
        return self.hash_features[idx]

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        # x has shape (*, 2)
        feats = self.bilerp_hash(x)
        return self.mlp(feats)
    
    def render(self):
        H,W = self.shape
        x = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), -1).reshape(-1,2)
        rgb = self.forward(x)
        return rgb.reshape(H,W,3)

    def update_hash_feats(self, new_feats, hashmap):
        self.hash_features = nn.Parameter(new_feats)
        self.hashmap = hashmap
        self.compressed_N = new_feats.shape[-2]


def bucket_tensor(self, x, B: int):
    # x has shape (*, N, F)
    # B is buckets per feature dimension, so the total buckets is <=B**F
    # returns:
    #   (*, b, F) tensor of bucket centers
    #   (*, N) tensor of bucket indices for the hash features
    if B <= 256:
        dtype = torch.quint8
    else:
        dtype = torch.qint32
    x = self.hash_features
    xmin, xmax = x.min(), x.max()
    x = (x - xmin)/(xmax - xmin)
    quantized_tensor = torch.quantize_per_tensor(x, scale=1/(B-1), zero_point=0, dtype=dtype)
    unique_values, indices = torch.unique(quantized_tensor.int_repr(), return_inverse=True, dim=-2)
    if unique_values.shape[-2] >= x.shape[-2] * .9:
        raise ValueError(f'failed to compress buckets sufficiently (from {x.shape[-2:]} to {unique_values.shape[-2:]})')
    rescaled_feats = unique_values / (B-1) * (xmax - xmin) + xmin
    return rescaled_feats, indices


def fit_img(img: np.ndarray):
    n_iters = 100
    img = torch.tensor(img)
    ngp = NGP(R=4, N=2**12, F=4, shape=img.shape[:2])
    losses = []
    optimizer = torch.optim.Adam(ngp.parameters(), lr=1e-3)
    for _ in range(n_iters):
        loss = ((ngp.render() - img)**2).mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return ngp.render(), losses