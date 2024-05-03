
# class MLP(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  intermediate_channels: int=64,
#                  n_layers: int=2,
#                  ) -> None:
#         super().__init__()
#         layers = [nn.Linear(in_channels, intermediate_channels), nn.ReLU(inplace=True)]
#         for _ in range(n_layers-2):
#             layers += [nn.Linear(intermediate_channels, intermediate_channels), nn.ReLU(inplace=True)]
#         layers.append(nn.Linear(intermediate_channels, out_channels))
#         self.layers = nn.Sequential(*layers)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layers(x)

