import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

class NGP(nn.Module):
    def __init__(self,
                 R: int = 4, # num resolutions
                 N: int = 1024, # num buckets
                 F: int = 2, # num features
                 shape: tuple = None,  # H*W
                 min_resolution: int = 16,
                 max_resolution: int = 512,
                 resolution_feature_scaler: float = 1.0,
                 output_activation = "sigmoid",
                 ) -> None:
        # resolution_feature_scaler scales hash feature magnitudes by R^resolution_feature_scaler where R is the resolution
        super().__init__()
        self.resolutions = list(np.round(np.logspace(np.log2(min_resolution), np.log2(max_resolution), R, base=2)).astype(int))
        self.N = N
        self.shape = shape
        assert output_activation in ("None", "sigmoid")
        config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": output_activation,
            "n_neurons": 32,
            "n_hidden_layers": 2,
        }
        self.hash_features = nn.Parameter((torch.rand(R, N, F).cuda() - .5) * 2e-4) # U(-1e-4, 1e-4)
        self.mlp = tcnn.Network(R*F, 3, config)
        self.resolution_feature_scaler = resolution_feature_scaler
    
    def bilerp_hash(self, coords: torch.FloatTensor):
        # coords in [0,1] with shape (*, 2)
        feats = []
        for rix, res in enumerate(self.resolutions):
            x = coords * res
            x_ = torch.floor(x).long()
            w = x - x_
            x,y = x_[..., 0], x_[..., 1]
            feats.append(
                self.hash_lookup_2d(x, y, rix) * (1 - w[..., :1]) * (1 - w[..., 1:]) + \
                self.hash_lookup_2d(x, y + 1, rix) * (1 - w[..., :1]) * w[..., 1:] + \
                self.hash_lookup_2d(x + 1, y, rix) * w[..., :1] * (1 - w[..., 1:]) + \
                self.hash_lookup_2d(x + 1, y + 1, rix) * w[..., :1] * w[..., 1:]
            )
        return torch.cat(feats, dim=-1)
        
    def hash_lookup_2d(self, x: torch.IntTensor, y: torch.IntTensor, rix: int):
        idx = torch.bitwise_xor(x, 2654435761*y) % self.N
        return self.hash_features[rix, idx] * 2**(rix*self.resolution_feature_scaler)
        
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
    
    def grid_coords(self):
        H,W = self.shape
        return torch.stack(torch.meshgrid(torch.linspace(0,1,H), torch.linspace(0,1,W), indexing='ij'), -1).reshape(-1,2).cuda()

    def render(self, compress: bool = False, to_numpy: bool = False):
        rgb = self.forward(self.grid_coords(), compress=compress)
        rgb = rgb.reshape(*self.shape,3)
        if to_numpy:
            return rgb.clamp(0,1).float().cpu().detach().numpy()
        else:
            return rgb

    def render_hash(self, resolution):
        H,W = self.shape
        x = torch.stack(torch.meshgrid(torch.arange(H//resolution), torch.arange(W//resolution), indexing='ij'), -1).reshape(-1,2).cuda()
        rix = self.resolutions.index(resolution)
        x,y = x[..., 0], x[..., 1]
        hash_idxs = torch.bitwise_xor(x, 2654435761*y) % self.N
        return torch.cat(hash_idxs, dim=-1)
        
    def get_size(self):
        return self.get_mlp_size() + self.get_hash_size()

    def get_mlp_size(self):
        return sum(p.numel() for p in self.mlp.parameters()) * 2 / 1024

    def get_hash_size(self):
        return self.hash_features.numel() * 2 / 1024
