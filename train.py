import numpy as np
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os, glob, time

import utils, metrics

from model import ngp, compact_ngp, vbnerf, vqrf



def run_2d(args):
    exp_group_name = args.name
    _jpg = args.jpg
    _ngp = args.ngp
    _cngp = args.ours
    _vqrf = args.vqrf
    _vbnerf = args.vbnerf

    image_paths = sorted(glob.glob('./data/kodak/*.png'))
    results_csv_path = f'results/results.csv'
    if not os.path.exists(results_csv_path):
        results = pd.DataFrame(columns=['path', 'format', 'psnr', 'ssim', 'size'])
    else:
        results = pd.read_csv(results_csv_path)
        
    # fixed hyperparameters
    n_iters = 1000
    n_refinement_iters = 1000
    iteration_budget = n_iters + n_refinement_iters
    R = 8
    F = 6
    batch_size = 240_000
    fixed_hyperparameters = dict(R=R, F=F, iteration_budget=iteration_budget, batch_size=batch_size)
    np.save(f'results/{exp_group_name}', fixed_hyperparameters)

    # for ix, img_path in enumerate(image_paths):
    # print(f'fitting image {ix+1}/{len(image_paths)}...')
    img_path = image_paths[17]
    img = np.array(Image.open(img_path)) / 255.

    # JPG compression
    if _jpg:
        for quality in (50,60,70,80):
            jpg, model_size = utils.to_jpg(Image.open(img_path), quality=quality)
            psnr, ssim = metrics.psnr(jpg, img), metrics.ssim(jpg, img)
            results.loc[len(results.index)] = [img_path, 'JPG', psnr, ssim, model_size]

    if _vqrf:
        for N in (2**8, 2**10):
            losses = []
            model, optimizer = init_model(img, model_type='VQRF', N=N, F=F)
            torch_img = torch.tensor(img).cuda()
            for _ in range(n_iters):
                x = torch.rand((batch_size, 2), device='cuda')
                loss = ((model(x) - utils.sample_img(torch_img, x))**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            render = model.render(compress=True, to_numpy=True)
            model_size = model.get_size()
            psnr, ssim = metrics.psnr(render, img), metrics.ssim(render, img)
            results.loc[len(results.index)] = [img_path, 'VQRF', psnr, ssim, model_size]

    if _vbnerf:
        for N in (2**8, 2**10):
            losses = []
            model, optimizer = init_model(img, model_type='VBNeRF', R=R, N=N, F=F, min_resolution=32, max_resolution=512, resolution_feature_scaler=0.1)
            torch_img = torch.tensor(img).cuda()
            for _ in range(n_iters):
                x = torch.rand((batch_size, 2), device='cuda')
                loss = ((model(x) - utils.sample_img(torch_img, x))**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            render = model.render(compress=True, to_numpy=True)
            model_size = model.get_size()
            psnr, ssim = metrics.psnr(render, img), metrics.ssim(render, img)
            results.loc[len(results.index)] = [img_path, 'VBNeRF', psnr, ssim, model_size]


    # NGP and compressed NGP
    if _ngp or _cngp:
        for N in (2**8, 2**10):
            run_name = f'{exp_group_name}_{N=}'

            # fit image with NGP
            losses = []
            model, optimizer = init_model(img, R=R, N=N, F=F, min_resolution=32, max_resolution=512, resolution_feature_scaler=0.1)
            torch_img = torch.tensor(img).cuda()
            for _ in range(n_iters):
                loss = ((model.render() - torch_img)**2).mean()
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.save({
                'model_state_dict': model.state_dict(),
                'losses': losses,
            }, 'tmp.ckpt')

            if _ngp:
                for _ in range(n_refinement_iters):
                    loss = ((model.render() - torch_img)**2).mean()
                    losses.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                render = model.render(compress=True, to_numpy=True)
                model_size = model.get_size()
                psnr, ssim = metrics.psnr(render, img), metrics.ssim(render, img)
                Image.fromarray((render * 255).astype(np.uint8)).save(f'renders/{run_name}_ngp_{os.path.basename(img_path)}')
                results.loc[len(results.index)] = [img_path, 'iNGP', psnr, ssim, model_size]

            if _cngp:
                # compress and refine hash table
                for buckets_per_feat in (8, 10):
                    model.hashmap = None
                    model.hash_features = nn.Parameter(torch.zeros_like(ckpt['model_state_dict']['hash_features']))
                    ckpt = torch.load('tmp.ckpt')
                    model.load_state_dict(ckpt['model_state_dict'])
                    losses = ckpt['losses']
                    quantized_feats, indices = model.quantize_table(buckets_per_feat)
                    model.update_hash_feats(quantized_feats, indices)

                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(.9,.99), eps=1e-15)
                    for _ in range(n_refinement_iters):
                        loss = ((model.render() - torch_img)**2).mean()
                        losses.append(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    render = model.render(compress=True, to_numpy=True)
                    model_size = model.get_size()
                    psnr, ssim = metrics.psnr(render, img), metrics.ssim(render, img)
                    results.loc[len(results.index)] = [img_path, 'cNGP', psnr, ssim, model_size]
                Image.fromarray((render * 255).astype(np.uint8)).save(f'renders/{run_name}_cngp_{os.path.basename(img_path)}')

    results.to_csv(results_csv_path)



def init_distributed_model(img: np.ndarray, **kwargs):
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.distributed.init_process_group(
        backend='nccl', world_size=torch.cuda.device_count(), init_method='...'
    )
    model = DDP(model, device_ids=[i], output_device=i)


def init_model(img: np.ndarray, **kwargs):
    kwargs = {**dict(N=2**10, F=2, lr=1e-2), **kwargs}
    lr = kwargs.pop('lr')
    model_type = kwargs.pop('model_type', 'ngp').lower()
    if model_type == 'ngp':
        model = ngp.NGP(shape=img.shape[:2], **kwargs).cuda()
    elif model_type == 'cngp':
        model = compact_ngp.CompactNGP(shape=img.shape[:2], **kwargs).cuda()
    elif model_type == 'vqrf':
        model = vqrf.VQRF(shape=img.shape[:2], **kwargs).cuda()
    elif model_type == 'vbnerf':
        model = vbnerf.VBNeRF(shape=img.shape[:2], **kwargs).cuda()
    else:
        raise ValueError(f'{model_type=}')
        
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

