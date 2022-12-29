import sys
import os
import argparse
import time
import numpy as np
import logging
import json
import torch
import torch.nn as nn
from models.basemodel import Unet, GaussianDiffusion
from models.customUNet import Unet_custom
from models.dataloader import get_data, get_MNIST
from utils.utils import save_images


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def train_epoch(diffusion, device, train_loader, optimizer, epoch, mgpu):
    t = time.time()
    loss_accumulator = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = diffusion(data)
        if mgpu == "true":
            loss = loss.sum()  
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTraining Epoch {}:  [{} batch/{} total batches]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1),
                    len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrained Epoch {}:  [{} batch/{} total batches]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    len(train_loader),
                    len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )
    l = np.mean(loss_accumulator)
    logging.info(f'Train epoch {epoch}, loss: {l}')
    return np.mean(loss_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # load hyperparameters
    with(open(f'{args.param_dir}/param.json')) as f:
        hyperparams = json.load(f)
    batch_size = hyperparams["batch size"]
    n_epoch = hyperparams["num epochs"]
    lr = hyperparams["learning rate"]
    lrs = hyperparams["learning rate scheduler"]
    lrs_min = hyperparams["learning rate scheduler minimum"]
    
    # Load dataloaders...
    if args.dataset == "MNIST":
        dataloader = get_MNIST('data/processed', 32, batch_size)
    elif args.dataset == "PolyU HK FKP V1":
        dataloader = get_data(args.data_path, 128, batch_size)
        
    # get loss function


    # load model
    if args.dataset == "MNIST":
        n_dim = 32         
        n_channels=1
        n_dim_mults = (1, 2, 4, 8)
        size = 32
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
    elif args.dataset == "PolyU HK FKP V1":
        n_dim = 128         
        n_channels=3
        n_dim_mults = (1, 2, 4, 8)
        size = 128
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
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)    
    # Resume from checkpoint
    if args.ckpt != None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        starting_epoch = epoch + 1       
    else: starting_epoch = 1
    # Multi gpu option
    if args.mgpu == "true":
        model = nn.DataParallel(model)
        diffusion = nn.DataParallel(diffusion)
    model.to(device)
    diffusion.to(device)
    optimizer_to(optimizer,device)
    # sample image folder
    if not(os.path.exists(f'{args.param_dir}/sampled_images')):
        os.makedirs(f'{args.param_dir}/sampled_images')
    return (
        device,
        dataloader,
        model,
        diffusion,
        model,
        optimizer,
        n_epoch, 
        lrs, 
        lrs_min,
        starting_epoch
    )


def train(args):
    (
        device,
        dataloader,
        model,
        diffusion,
        model,
        optimizer,
        n_epoch, 
        lrs, 
        lrs_min,
        starting_epoch
    ) = build(args)

    if not os.path.exists(args.param_dir):
        os.makedirs(args.param_dir)

    if lrs == "true":
        if lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    
    for epoch in range(starting_epoch, n_epoch + 1):
        try:
            loss = train_epoch(
                diffusion, device, dataloader, optimizer, epoch, mgpu=args.mgpu
            )
            sampled_images = diffusion.module.sample(batch_size = 4)
            # unnormarlize
            sampled_images = (sampled_images*255).byte()
            save_images(sampled_images, f'{args.param_dir}/sampled_images/{args.dataset}_{epoch}.jpg')
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if lrs == "true":
            scheduler.step(loss)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict()
                if args.mgpu == "false"
                else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            args.param_dir +  '/' + args.dataset  + "_"+ args.model +".pt",
        )


def get_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-d", "--param_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "PolyU HK FKP V1"])
    parser.add_argument("--data_path", type=str, default='data/processed')
    parser.add_argument("--model", type=str, required=True, choices=["baseUNet", "ConvNeXtUNet"])
    parser.add_argument(
        "--multi_gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None, dest="ckpt"
    )
    return parser.parse_args()


def main():
    args = get_args()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=f'{args.param_dir}/train.log', 
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    train(args)


if __name__ == "__main__":
    main()
    