"""
This script used to:
1. Download MNIST if the dataset doesn't exsit.
2. Return Train DataLoader and Test DataLoader.
"""

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


def get_mnist_dataloaders(data_dir="./data", batch_size=100):
    """ Faster version to handle dataset """
    # 1. Download MNIST Dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True) # (data, targets)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True) # (data, targets)

    # 2. Convert to Tensor once: normalize to [0,1] + add channel dimension
    train_images = train_dataset.data.float() / 255.0  # (60000, 28, 28)
    train_images = train_images.unsqueeze(1)           # (60000, 1, 28, 28)
    test_images = test_dataset.data.float() / 255.0    # (10000, 28, 28)
    test_images = test_images.unsqueeze(1)             # (10000, 1, 28, 28)

    # 3. Wrap with TensorDataset (pure tensors, so batching only slices)
    train_dataset = TensorDataset(train_images, train_dataset.targets)
    test_dataset = TensorDataset(test_images, test_dataset.targets)

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader





if __name__ == "__main__":
    train_loader, test_loader = get_mnist_dataloaders()
    # 1. iter(): change to iterator
    # 2. next(): get a iterator (get a batch data)
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)