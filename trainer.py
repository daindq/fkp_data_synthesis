'''
    Implement original training scheme
'''
import argparse
import json
import os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def get_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-d", "--param_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "PolyU HK FKP V1"])
    parser.add_argument("--data_path", type=str, default='data/processed/train')
    parser.add_argument("--model", type=str, required=True, choices=["baseUNet", "ConvNeXtUNet"])
    parser.add_argument(
        "-c", "--checkpoint_milestone", type=str, default=None, dest="milestone"
    )
    return parser.parse_args()


def build(args):
    # load hyperparameters
    with(open(f'{args.param_dir}/param.json')) as f:
        hyperparams = json.load(f)
    batch_size = hyperparams["batch size"]
    n_epoch = hyperparams["num epochs"]
    lr = hyperparams["learning rate"]
    _ = hyperparams["learning rate scheduler"]
    _ = hyperparams["learning rate scheduler minimum"]
    
    # load models
    if args.dataset == "MNIST":
        n_dim = 32         
        n_channels=1
        n_dim_mults = (1, 2, 4, 8)
        size = 32
        model = Unet(
            dim = n_dim,
            channels=n_channels,
            dim_mults = n_dim_mults
        ).cuda()
        diffusion = GaussianDiffusion(
            model,
            image_size = size,
            timesteps = 1000,   # number of steps
            loss_type = 'l2'    # L1 or L2
        )
    elif args.dataset == "PolyU HK FKP V1":
        n_dim = 64         
        n_channels=3
        n_dim_mults = (1, 2, 4, 8)
        size = 128
        model = Unet(
            dim = n_dim,
            channels=n_channels,
            dim_mults = n_dim_mults
        ).cuda()
        diffusion = GaussianDiffusion(
            model,
            image_size = size,
            timesteps = 1000,   # number of steps
            loss_type = 'l2'    # L1 or L2
        )
    # Create trainer object
    trainer = Trainer(
        diffusion,
        args.data_path,
        train_batch_size = batch_size,
        train_lr = lr,
        train_num_steps = n_epoch,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 100,
        num_samples = 5,
        results_folder = args.param_dir,
        amp = True                        # turn on mixed precision
    )
    
    
    # Resume from checkpoint
    if args.milestone != None:
        trainer.load(args.milestone)
    # sample image folder
    if not(os.path.exists(f'{args.param_dir}/sampled_images')):
        os.makedirs(f'{args.param_dir}/sampled_images')
    return (
        trainer
    )


if __name__ == "__main__":
    args = get_args()
    trainer = build(args)
    trainer.train()
    