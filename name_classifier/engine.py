import torch

from tqdm.auto import tqdm
from typing import Tuple, Dict, List

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
        Tuple[float, float]: A tuple of training loss and training accuracy.
    """
    
    model.train()
    
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        # initialize the hidden layer
        hidden = model.init_hidden().to(device)
        
        # send the data to the target device
        X, y = X.to(device), y.to(device)
        
        # forward pass 
        for i in range(X.size()[1]):
            y_pred, hidden = model(X[0, i], hidden)
        
        # calculate and accumulate the loss
        loss = criterion(y_pred, y[0])
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
            
        # calculate and accumulate the accuracy
        y_pred_class = torch.argmax(y_pred, dim = 1)
        train_acc += ((y_pred_class == y).sum().item() / len(y_pred_class))
        
    # calculate the average loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """perform a single testing step with a model.

    Args:
        model (torch.nn.Module): the model to be tested.
        dataloader (torch.utils.data.DataLoader): a dataloader instance to test the model on.
        criterion (torch.nn.Module): a pytorch loss function to calculate loss on the test data.
        device (torch.device): a target device to compute on (e.g. "cuda" or "cpu")

    Returns:
        Tuple[float, float]: A tuple of testing loss and testing accuracy.
    """
    model.eval()
    
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for (X, y) in dataloader:
            # send the data to the target device
            X, y = X.to(device), y.to(device)
            
            # forward pass
            for i in range(X.size()[1]):
                test_pred, hidden = model(test_pred, hidden)
            
            # calculate and accumulate the loss
            loss = criterion(test_pred, y[0])
            test_loss += loss.item()
            
            # calculate and accumulate the accuracy
            test_pred_class = torch.argmax(test_pred, dim = 1)
            test_acc += ((test_pred_class == y).sum().item() / len(test_pred_class))
           
    # calculate the average loss and accuracy 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Args:
        model (torch.nn.Module): A model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        criterion (torch.nn.Module): A PyTorch loss function to calculate loss on both datasets.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        epochs (int): The number of epochs to train for.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Dict[str, List]: A dictionary containing training loss and accuracy, and testing loss and accuracy.
    """
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
        
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model,
            train_dataloader,
            criterion,
            optimizer,
            device
        )
        
        test_loss, test_acc = test_step(
            model,
            test_dataloader,
            criterion,
            optimizer,
            device
        )
        
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss} | "
            f"train_acc: {train_acc} | "
            f"test_loss: {test_loss} | "
            f"test_acc: {test_acc}"
        )
        
        # update the results dicionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results

