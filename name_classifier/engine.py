import torch

from tqdm.auto import tqdm
from typing import Tuple

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.optim.Optimizer,
    learning_rate: float,
    device: torch.device
) -> Tuple[float, float]:
    """perform a single training step with a model.

    Args:
        model (torch.nn.Module): the model to train.
        dataloader (torch.utils.data.DataLoader): a dataloader instance to train the model on.
        criterion (torch.optim.Optimizer): a pytorch loss function to minimize.
        learning_rate (float): a rate with which the loss function is minimized.
        device (torch.device): a target device to compute on (eg: "cuda" or "cpu")

    Returns:
        Tuple[float, float]: _description_
    """
    
    model.train()
    
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        hidden = model.init_hidden()
        X, y = X.to(device), y.to(device)
        
        for i in range(X.size()[2]):
            y_pred, hidden = model(X[0, i], hidden)
        
        loss = criterion(y_pred, y[0])
        train_loss += loss.item()
        
        loss.backward()
        
        for p in model.parameters():
            p.data.add_(p.grad.data, alpha = learning_rate)
            
        y_pred_class = torch.argmax(y_pred, dim = 1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc
        