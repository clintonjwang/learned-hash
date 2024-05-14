import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from .ngp import NGP

class VBNeRF(NGP):
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
    