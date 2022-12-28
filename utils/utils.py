from matplotlib import pyplot as plt
import torchvision
import torch
from PIL import Image
import os
import uuid
import numpy as np


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    

def save_all_image(images, dir, **kwargs):
    for num in range(images.shape[0]):
        image = images[num,:,:,:]
        ndarr = image.permute(1, 2, 0).to('cpu').numpy()
        ndarr = np.squeeze(ndarr)
        im = Image.fromarray(ndarr)
        id = uuid.uuid1()
        im.save(f'{dir}/{id.hex}.jpg')        
    
    
def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
    