"""A module containing functions to train a model."""

import torch

from torch.utils.tensorboard.writer import SummaryWriter

from tqdm.auto import tqdm
from typing import Dict, List

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    writer: SummaryWriter,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Args:
    ----
        model (torch.nn.Module): A model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader instance for the model to be tested on.
        criterion (torch.nn.Module): A PyTorch loss function to calculate loss on both datasets.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
        epochs (int): The number of epochs to train for.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    -------
        Dict[str, List]: A dictionary containing training loss and accuracy, and testing loss and accuracy.

    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        # train_loss, train_acc = train_step(
        #     model, train_dataloader, criterion, optimizer, device
        # )

        # test_loss, test_acc = test_step(model, test_dataloader, criterion, device)
        train_loss, train_acc = model.training_step(
            dataloader = train_dataloader,
            criterion = criterion,
            optimizer = optimizer,
            device = device
        )

        val_loss, val_acc = model.validation_step(
            dataloader = test_dataloader,
            criterion = criterion,
            device = device
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {val_loss:.4f} | "
            f"test_acc: {val_acc:.4f}"
        )

        if writer:
            writer.add_scalars(
                main_tag="train/test loss",
                tag_scalar_dict={"train": train_loss, "test": val_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="train/test accuracy",
                tag_scalar_dict={"train": train_acc, "test": val_acc},
                global_step=epoch,
            )
            writer.close()

        # update the results dicionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss)
        results["test_acc"].append(val_acc)

    return results
