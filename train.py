import numpy as np
import torch

from model import ngp

def init_model(img: np.ndarray, **kwargs):
    kwargs = {**dict(R=5, N=2**10, F=2, lr=1e-2), **kwargs}
    lr = kwargs.pop('lr')
    model = ngp.NGP(shape=img.shape[:2], **kwargs).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9,.99), eps=1e-15)
    return model, optimizer

def fit_img(img: np.ndarray, n_iters: int = 100, **kwargs):
    model, optimizer = init_model(img, **kwargs)
    losses = []
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img).cuda()
    for _ in range(n_iters):
        loss = ((model.render() - img)**2).mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, losses

def fit_img_iterative(img, bucket_schedule, iters_per_bucket, **kwargs):
    model, optimizer = init_model(img, **kwargs)
    losses = []
    for buckets in bucket_schedule:
        for _ in range(iters_per_bucket):
            loss = ((model.render() - img)**2).mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        quantized_feats, indices = ngp.quantize_tensor(model.hash_features, buckets_per_feat=round(buckets**(1/model.F)))
        model.update_hash_feats(quantized_feats, indices)

