import numpy as np
import torch
import torch.nn as nn
F = nn.functional
import tinycudann as tcnn

from .ngp import INR

class VQRF(INR):
    def __init__(self,
                 N: int = 1024, # num buckets (codebook size)
                 F: int = 4, # num features
                 shape: tuple = None,  # (H,W)
                 output_activation = "sigmoid",
                 ) -> None:
        super().__init__()
        self.N = N
        self.shape = shape
        assert output_activation in ("None", "sigmoid")
        config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": output_activation,
            "n_neurons": 32,
            "n_hidden_layers": 1,
        }
        self.device = 'cuda'
        self.hashmap = nn.Parameter((torch.rand(shape[0]+1, shape[1]+1, N).to(device=self.device) - .5) * 2e-4)
        self.hash_features = nn.Parameter((torch.rand(N, F).to(device=self.device) - .5) * 2e-4) # U(-1e-4, 1e-4)
        self.mlp = tcnn.Network(F, 3, config)
    
    def bilerp_hash(self, coords: torch.FloatTensor):
        # coords in [0,1] with shape (*, 2)
        res = torch.tensor(self.shape, dtype=coords.dtype, device=coords.device)
        x = coords * res
        x_ = torch.floor(x).long()
        w = x - x_
        x,y = x_[..., 0], x_[..., 1]
        return (
            self.hash_lookup_2d(x, y) * (1 - w[..., :1]) * (1 - w[..., 1:]) + \
            self.hash_lookup_2d(x, y + 1) * (1 - w[..., :1]) * w[..., 1:] + \
            self.hash_lookup_2d(x + 1, y) * w[..., :1] * (1 - w[..., 1:]) + \
            self.hash_lookup_2d(x + 1, y + 1) * w[..., :1] * w[..., 1:]
        )
        
    def hash_lookup_2d(self, x: torch.IntTensor, y: torch.IntTensor):
        onehot = F.gumbel_softmax(self.hashmap[x, y], tau=1, hard=True, dim=-1)
        return (onehot[..., None] * self.hash_features).sum(1)
        
    def get_size(self):
        return self.get_mlp_size() + self.get_hash_size() + self.get_hashmap_size()

    def get_hashmap_size(self):
        return self.hashmap.numel() * 2 / 1024
