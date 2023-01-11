import torch
import argparse
import torch.nn as nn
import math
from ema_pytorch import EMA
from utils.utils import save_all_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from models.customUNet import Unet_custom
import shutil
import os


def get_args():
    parser = argparse.ArgumentParser(description="sample images with checkpoint.")
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "PolyUHKV1"])
    parser.add_argument("-o", "--dataoutdir", type=str, default='data/output')
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--multi_gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, required=True, default=None, dest="ckpt"
    )
    return parser.parse_args()


def build_model(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # load ckpt
    n_dim = 64 # First feature map depth        
    n_channels=3 # num Image channel
    n_dim_mults = (1, 2, 4, 8)
    size = 64 # 
    model = Unet_custom(
            dim = n_dim,
            channels=n_channels,
            dim_mults = n_dim_mults
        )
    diffusion = GaussianDiffusion(
        model,
        image_size = size,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )
    ema = EMA(diffusion)
    checkpoint = torch.load(args.ckpt)
    ema.load_state_dict(checkpoint['ema'])
    diffusion.to(device)
    ema.to(device)
    return (model, diffusion, ema)


def main():
    args = get_args()
    _, __, ema = build_model(args)
    num_ims = 2500
    if not(os.path.exists(f'{args.dataoutdir}/sample_images')):
        os.makedirs(f'{args.dataoutdir}/sample_images')
    ema.ema_model.eval()
    with torch.no_grad():
        for i in range(math.ceil(num_ims/args.batch_size)):
            if args.mgpu == "true":
                sampled_images = ema.ema_model.module.sample(batch_size = args.batch_size)
            else: 
                sampled_images = ema.ema_model.sample(batch_size = args.batch_size)
            sampled_images = (sampled_images*255).byte()
            save_all_image(sampled_images, f'{args.dataoutdir}/sample_images')
            shutil.make_archive(f'{args.dataoutdir}/sample_images', 'zip', args.dataoutdir, "sample_images")
    
    
    
if __name__ == "__main__":
    main() 
    # import shutil
    # shutil.make_archive(f'models', 'zip', '.', 'models')
    # shutil.make_archive(filename, 'zip', compress_dir)
    # # shutil.unpack_archive(filename, extract_dir, archive_format)
    