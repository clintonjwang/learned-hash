from PIL import Image
import argparse
import numpy as np
import torch
import pandas as pd
import os, glob

import utils
import train

def main():
    parser = argparse.ArgumentParser(description='Image Loader')
    parser.add_argument('-n', '--name', type=str, default='default')
    args = parser.parse_args()
    exp_group_name = args.name

    image_paths = sorted(glob.glob('./data/kodak/*.png'))
    results_path = f'results/results.csv'
    if not os.path.exists(results_path):
        results = pd.DataFrame(columns=['path', 'format', 'psnr', 'ssim', 'size'])
    else:
        results = pd.read_csv(results_path)
        
    # fixed hyperparameters
    n_iters = 2000
    n_refinement_iters = 1000
    iteration_budget = n_iters + n_refinement_iters
    R = 8
    F = 6
    fixed_hyperparameters = dict(R=R, F=F, iteration_budget=iteration_budget)
    np.save(f'results/{exp_group_name}', fixed_hyperparameters)

    # for ix, img_path in enumerate(image_paths):
    # print(f'fitting image {ix+1}/{len(image_paths)}...')
    img_path = image_paths[17]
    img = np.array(Image.open(img_path)) / 255.

    # JPG compression
    for quality in (50,60,70,80):
        jpg, jpg_size = utils.to_jpg(Image.open(img_path), quality=quality)
        jpg_psnr, jpg_ssim = utils.psnr(jpg, img), utils.ssim(jpg, img)
        results.loc[len(results.index)] = [img_path, 'JPG', jpg_psnr, jpg_ssim, jpg_size]

    # NGP and compressed NGP
    for N in (2**8, 2**10, 2**12):
        run_name = f'{exp_group_name}_{N=}'

        # fit image with NGP
        losses = []
        model, optimizer = train.init_model(img, R=R, N=N, F=F, min_resolution=32, max_resolution=512, resolution_feature_scaler=0.1)
        torch_img = torch.tensor(img).cuda()
        for _ in range(n_iters):
            loss = ((model.render() - torch_img)**2).mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        render = model.render(compress=True, to_numpy=True)
        ngp_size = model.get_size()
        ngp_psnr, ngp_ssim = utils.psnr(render, img), utils.ssim(render, img)
        Image.fromarray((render * 255).astype(np.uint8)).save(f'renders/{run_name}_ngp_{os.path.basename(img_path)}')
        results.loc[len(results.index)] = [img_path, 'iNGP', ngp_psnr, ngp_ssim, ngp_size]

        # compress and refine hash table
        F = model.hash_features.shape[-1]
        buckets_per_feat=8
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
        cngp_size = model.get_size()
        cngp_psnr, cngp_ssim = utils.psnr(render, img), utils.ssim(render, img)
        Image.fromarray((render * 255).astype(np.uint8)).save(f'renders/{run_name}_cngp_{os.path.basename(img_path)}')
        results.loc[len(results.index)] = [img_path, 'cNGP', cngp_psnr, cngp_ssim, cngp_size]

        results.to_csv(f'results/{exp_group_name}.csv')

if __name__ == '__main__':
    main()