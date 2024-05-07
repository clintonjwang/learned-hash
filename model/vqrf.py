import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn


class VQRF(nn.Module):
    def __init__(self,
                 N: int = 1024, # num buckets (codebook size)
                 F: int = 8, # num features
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
        self.hashmap = nn.Parameter((torch.rand(*shape, N).to(device=self.device) - .5) * 2e-4)
        self.hash_features = nn.Parameter((torch.rand(N, F).to(device=self.device) - .5) * 2e-4) # U(-1e-4, 1e-4)
        self.mlp = tcnn.Network(F, 3, config)
    
    def bilerp_hash(self, coords: torch.FloatTensor):
        # coords in [0,1] with shape (*, 2)
        feats = []
        res = torch.tensor(self.shape, dtype=coords.dtype, device=coords.device)
        x = coords * res
        x_ = torch.floor(x).long()
        w = x - x_
        x,y = x_[..., 0], x_[..., 1]
        feats.append(
            self.hash_lookup_2d(x, y) * (1 - w[..., :1]) * (1 - w[..., 1:]) + \
            self.hash_lookup_2d(x, y + 1) * (1 - w[..., :1]) * w[..., 1:] + \
            self.hash_lookup_2d(x + 1, y) * w[..., :1] * (1 - w[..., 1:]) + \
            self.hash_lookup_2d(x + 1, y + 1) * w[..., :1] * w[..., 1:]
        )
        return torch.cat(feats, dim=-1)
        
    def hash_lookup_2d(self, x: torch.IntTensor, y: torch.IntTensor):
        idxs = torch.argmax(self.hashmap[x, y], dim=-1)
        return self.hash_features[idxs]
        
    def forward(self, x: torch.FloatTensor, compress=False) -> torch.Tensor:
        # x has shape (*, 2)
        if compress:
            with torch.no_grad():
                self.half()
                x = x.half()
                feats = self.bilerp_hash(x)
                out = self.mlp(feats).float()
                self.float()
                return out
        else:
            feats = self.bilerp_hash(x)
        return self.mlp(feats)
    
    def update_hash_feats(self, new_feats, hashmap):
        self.hash_features = nn.Parameter(new_feats)
        if self.hashmap is None:
            self.hashmap = hashmap
        else:
            for rix in range(self.hashmap.shape[0]):
                for idx in range(self.hashmap.shape[1]):
                    self.hashmap[rix, idx] = hashmap[self.hashmap[rix, idx]]

    def get_size(self):
        return self.get_mlp_size() + self.get_hash_size() + self.get_hashmap_size()

    def get_mlp_size(self):
        return sum(p.numel() for p in self.mlp.parameters()) * 2 / 1024

    def get_hash_size(self):
        return self.hash_features.numel() * 2 / 1024

    def get_hashmap_size(self):
        return self.hashmap.numel() * 2 / 1024
