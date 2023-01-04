import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
import multiprocessing


class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):       
        return image


def get_data(data_path, img_size, batch_size, hvflip:bool = False):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip() if hvflip else NoneTransform, 
        torchvision.transforms.RandomHorizontalFlip() if hvflip else NoneTransform,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((img_size, img_size)),  # args.image_size + 1/4 *args.image_size
        
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.Pool()._processes,)
    return dataloader


def get_MNIST(data_path, img_size, batch_size):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=True, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_size)
        ])),
    batch_size=batch_size, shuffle=True, num_workers=multiprocessing.Pool()._processes,
    )
    return train_loader


if __name__ == "__main__":
    data = get_data('./data/processed/train', 128, 8, 2)
    print(next(iter(data))[0][0])

