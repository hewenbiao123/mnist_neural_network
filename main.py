""" Program entry """

from config import MNISTConfig
from data.mnist_loader import get_mnist_dataloaders 
from models.two_layer_net import TwoLayerNet
from train import train
import torch

def main() -> None:
    # 1. Set config, seed and device
    config: MNISTConfig = MNISTConfig() 
    # Seed affects all operations using PyTorch random number generator (e.g. shuffle, nn.Linear, dropout)
    torch.manual_seed(config.random_seed) 
    # In this project, cpu version faster than cuda version
    device: torch.device = torch.device('cpu')
    print(f"Using device: {device}")

    # 2. Get train/test loader
    train_loader, test_loader = get_mnist_dataloaders(data_dir=config.data_dir, batch_size=config.batch_size)

    # 3. Define model
    model: TwoLayerNet = TwoLayerNet(config)
    model.to(device)

    # 4. Train model
    train(model, train_loader, test_loader, config, device)


if __name__ == "__main__":
    main()

