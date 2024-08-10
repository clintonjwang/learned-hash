import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from .ngp import NGP


class SplitNGP(NGP):
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
        self.hash_features = nn.Parameter((torch.rand(R, N, F).cuda() - .5) * 2e-4) # U(-1e-4, 1e-4)
        self.mlp = tcnn.Network(R*F, 3, config)
        self.resolution_feature_scaler = resolution_feature_scaler
        self.hashmap = None
    
    def hash_lookup_2d(self, x: torch.IntTensor, y: torch.IntTensor, rix: int):
        idx = torch.bitwise_xor(x, 2654435761*y) % self.N
        if self.hashmap is None:
            return self.hash_features[rix, idx] * 2**(rix*self.resolution_feature_scaler)
        else:
            return self.hash_features[self.hashmap[rix, idx]] * 2**(rix*self.resolution_feature_scaler)
        
    def get_size(self):
        return self.get_mlp_size() + self.get_hash_size() + self.get_hashmap_size()

    def get_hashmap_size(self):
        if self.hashmap is None:
            return 0
        return self.hashmap.numel() * 2 / 1024

    def update_hash_feats(self, new_feats, hashmap):
        self.hash_features = nn.Parameter(new_feats)
        if self.hashmap is None:
            self.hashmap = hashmap
        else:
            for rix in range(self.hashmap.shape[0]):
                for idx in range(self.hashmap.shape[1]):
                    self.hashmap[rix, idx] = hashmap[self.hashmap[rix, idx]]

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
