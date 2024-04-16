from PIL import Image
import argparse
import numpy as np
import model.ngp as ngp
import torch

def main():
    parser = argparse.ArgumentParser(description='Image Loader')
    # parser.add_argument('image_path', type=str, help='Path to the image file')
    # args = parser.parse_args()
    # image_path = args.image_path
    image_path = 'kodim01t.jpg'
    img = torch.tensor(np.array(Image.open(image_path)) / 255.)
    render = ngp.fit_img(img)
    psnr = np.log10(((render - img)**2).mean())
    output_path = 'render.png'
    Image.fromarray((render * 255).astype(np.uint8)).save(output_path)
    print(f'{psnr=:.2f}')

if __name__ == '__main__':
    main()