"""" 
This script is used to:
    1. Evalute the model by accuracy
    2. Train the model by BP
"""

from models.two_layer_net import TwoLayerNet
from torch.utils.data import DataLoader
from config import MNISTConfig
import torch

@torch.no_grad() # Not build graph and not save temporary calculation results
def evaluate(model: TwoLayerNet, loader: DataLoader, device: torch.device) -> float:
    """ Return accuracy in total train/test dataset """
    # Change model to Evaluation Mode, which means that it would not dropout neurons randomly
    model.eval()

    correct: int = 0
    total: int = len(loader.dataset) # type: ignore

    # warning：label is not one-hot tensor
    for images, labels in loader:
        # Push tensor into specific device (e.g. CPU, cuda).
        images, labels = images.to(device), labels.to(device) 

        # Get logits without softmax
        logits = model.forward(images) 

        # Get (values,indexs) from torch.max() 
        values, indexs = torch.max(logits, dim=1)

        # Tensor.sum() returns [sum], use item() to get element in tensor which contains only one element.
        correct += (indexs == labels).sum().item()

    return correct / total



def train(model: TwoLayerNet, train_loader: DataLoader, test_loader: DataLoader, config: MNISTConfig, device: torch.device) -> None:
    """ Train model with back propagation (BP)."""
    # Define loss function: CrossEntropyLoss (include softmax).
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer: SGD (w = w - a * w.grad).
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        # Change model to Train Mode, which means that it would dropout neurons randomly with the aim of preventing overfitting
        model.train()
        # total loss
        running_loss: float = 0.0

        for images, labels in train_loader:
            # Push tensor into specific device (e.g. CPU, cuda).
            images, labels = images.to(device), labels.to(device) 

            optimizer.zero_grad()                       # 1. Clean grad
            logits = model.forward(images)              # 2. Forward propagation
            loss = criterion.forward(logits, labels)    # 3. Calculate the loss
            loss.backward()                             # 4. Backward propagation
            optimizer.step()                            # 5. Update grad

            running_loss += loss.item() * images.size(0)

        # Record and output avg_loss train_acc and test_acc.
        avg_loss: float = running_loss / len(train_loader.dataset) # type: ignore
        train_acc: float = evaluate(model, train_loader, device)
        test_acc: float = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1:2d}/{config.epochs} | loss {avg_loss:.4f} | train acc {train_acc:.4f} | test acc {test_acc:.4f}")
    







    

