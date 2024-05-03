import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn


class CompactNGP(nn.Module):
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
        self.resolution_feature_scaler = resolution_feature_scaler
        self.hashmap = None
        self.hash_features = nn.Parameter((torch.rand(R, N, F).cuda() - .5) * 2e-4) # U(-1e-4, 1e-4)
        self.mlp = tcnn.Network(R*F, 3, config)
    
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
        # if self.hash_features is None:
        #     with torch.no_grad():
        #         features = torch.stack([self.permutations[0][torch.bitwise_xor(x, 2654435761*y) % 374761393] / 374761393,
        #                                 self.permutations[1][torch.bitwise_xor(506832829*x, y) % 25745623] / 25745623,
        #                                 self.permutations[2][torch.bitwise_xor(17674343*x, 135271*y) % 62731] / 62731,
        #                                 ], dim=-1)
        #     return self.pre_mlp(features)

        idx = torch.bitwise_xor(x, 2654435761*y) % self.N
        if self.hashmap is None:
            return self.hash_features[rix, idx] * 2**(rix*self.resolution_feature_scaler)
        else:
            return self.hash_features[self.hashmap[rix, idx]] * 2**(rix*self.resolution_feature_scaler)
        
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
                # feats = feats.float()
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
        if self.hashmap is None:
            return 0
        return self.hashmap.numel() * 2 / 1024

    def check_compression_rate(self, buckets_per_feat: int):
        # figure out how many hash buckets will be produced with a quantization of buckets_per_feat
        if buckets_per_feat <= 256:
            dtype = torch.quint8
        else:
            dtype = torch.qint32
        x = self.hash_features
        xmin, xmax = x.min(), x.max()
        x = (x - xmin)/(xmax - xmin)
        F = x.shape[-1]
        quantized_tensor = torch.quantize_per_tensor(x.reshape(-1,F), scale=1/(buckets_per_feat-1), zero_point=0, dtype=dtype)
        unique_values = torch.unique(quantized_tensor.int_repr(), dim=-2)
        return unique_values.shape[-2] / x.shape[-2], unique_values.shape[-2]
    
    def quantize_table(self, buckets_per_feat: int):
        return quantize_table(self.hash_features, buckets_per_feat)

    def scatter_entries(self):
        if self.hashmap is None:
            return
        expanded_hash_feats = torch.zeros((*self.hashmap.shape, F), dtype=float, device=self.hash_features.device)
        for rix in range(self.hashmap.shape[0]):
            for idx in range(self.hashmap.shape[1]):
                expanded_hash_feats[rix, idx] = self.hash_features[self.hashmap[rix, idx]]
        self.hash_features = nn.Parameter(expanded_hash_feats)
        self.hashmap = None

    def split_table(self, gradients, buckets_per_feat: int):
        # quantize the gradient of the loss wrt the hash features, and for each bucket form a new entry in the hash table
        # each bucket can be split into at most buckets_per_feat**F new buckets
        # returns:
        #   (*, total_buckets, F) tensor of bucket centers, replicated
        #   (*, N) tensor of bucket indices for the hash features
        x = gradients
        xmin, xmax = x.min(), x.max()
        x = (x - xmin)/(xmax - xmin)
        F = x.shape[-1]
        quantized_tensor = torch.quantize_per_tensor(x.reshape(-1,F), scale=1/(buckets_per_feat-1), zero_point=0, dtype=torch.quint8)
        unique_values, indices = torch.unique(quantized_tensor.int_repr(), return_inverse=True, dim=-2)
        quantized_feats = unique_values / (buckets_per_feat-1) * (xmax - xmin) + xmin
        return quantized_feats, indices.reshape(self.hash_features.shape[:-1])
