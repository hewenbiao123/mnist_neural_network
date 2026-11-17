"""
This script used to:
1. Download MNIST if the dataset doesn't exsit.
2. Return Train DataLoader and Test DataLoader.
"""
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# def get_mnist_dataloaders(data_dir:str="./data", batch_size:int=100) -> tuple[DataLoader, DataLoader]:

#     # Define a pipeline for data processing (transforms.Compose()):
#     #   1. transforms.ToTensor(): change (H,W,C) to (C,H,W) and change pixel value from [0,255] to [0.0,1.0]
#     transform = transforms.Compose([transforms.ToTensor()])

#     # Get MNIST Dataset
#     train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

#     # Get DataLoader (Setting batch_size and shuffle)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


def get_mnist_dataloaders(data_dir=".\data", batch_size=100, use_cache=True):
    """ Return MNIST dataloaders, using cached processed tensors if available. """
    os.makedirs(data_dir, exist_ok=True)          
    cache_file = os.path.join(data_dir, "mnist_processed.pt")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading processed MNIST from cache: {cache_file}")
        data = torch.load(cache_file)
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images  = data['test_images']
        test_labels  = data['test_labels']
    else:
        # 1. Download raw MNIST (only if not cached)
        train_raw = datasets.MNIST(root=data_dir, train=True,  download=True)
        test_raw  = datasets.MNIST(root=data_dir, train=False, download=True)

        # 2. Preprocess once
        train_images = train_raw.data.float() / 255.0    # (60000, 28, 28)
        train_images = train_images.unsqueeze(1)         # (60000, 1, 28, 28)
        test_images  = test_raw.data.float() / 255.0
        test_images  = test_images.unsqueeze(1)          # (10000, 1, 28, 28)

        train_labels = train_raw.targets
        test_labels  = test_raw.targets

        # 3. Save processed tensors to cache
        if use_cache:
            torch.save({
                'train_images': train_images,
                'train_labels': train_labels,
                'test_images':  test_images,
                'test_labels':  test_labels,
            }, cache_file)
            print(f"Cached processed MNIST to: {cache_file}")

    # 4. Build TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset  = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader





if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()
    # 1. iter(): change to iterator
    # 2. next(): get a iterator (get a batch data)
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)