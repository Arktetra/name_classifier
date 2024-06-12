"""A module containing a self class for RNN."""

import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple

class RNN(nn.Module):

    """creates a recurrent neural network.

    Args:
    ----
        input_size (int): size of the input layer.
        hidden_size (int): size of the hidden layer.
        output_size (int): size of the output layer.

    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self, input: torch.tensor, hidden: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Applies the RNN to the input and its hidden state.

        Args:
        ----
            input (torch.tensor): text inputs
            hidden (torch.tensor): hidden state

        Returns:
        -------
            torch.tensor: RNN's output, hidden state

        """
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self) -> torch.tensor:
        """Intialize the hidden state of RNN to zero vector.

        Returns
        -------
            torch.tensor: a zero tensor.

        """
        return torch.zeros((1, self.hidden_size))

    def training_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Tuple[float, float]:
        """Perform a single training step.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): A dataloader instance to train the self on.
            criterion (torch.nn.Module): A PyTorch loss function to minimize.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
            device (torch.device): A target device to compute on (e.g. "cuda" or "cpu")

        Returns:
        -------
            Tuple[float, float]: A tuple of training loss and training accuracy.

        """
        self.train()

        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            # initialize the hidden state and the memory
            hidden = self.init_hidden().to(device)

            # send the data to the target device
            X, y = X.to(device), y.to(device)

            # forward pass
            for i in range(X.size()[1]):
                y_pred, hidden = self(X[:, i], hidden)

            # calculate and accumulate the loss
            loss = criterion(y_pred, torch.flatten(y))
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # calculate and accumulate the accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)

            train_acc += (y_pred_class == torch.flatten(y)).sum() / len(y_pred)

        # calculate the average loss and accuracy
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        return train_loss, train_acc

    def validation_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """Perfomr a single validation step.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): a dataloader instance to test the model on.
            criterion (torch.nn.Module): a pytorch loss function to calculate loss on the test data.
            device (torch.device): a target device to compute on (e.g. "cuda" or "cpu")

        Returns:
        -------
            Tuple[float, float]: A tuple of testing loss and testing accuracy.

        """
        self.eval()

        val_loss, val_acc = 0, 0

        with torch.inference_mode():
            for X, y in dataloader:
                # send the data to the target device
                X, y = X.to(device), y.to(device)

                hidden = self.init_hidden().to(device)

                # forward pass
                for i in range(X.size()[1]):
                    val_pred, hidden = self(X[:, i], hidden)

                # calculate and accumulate the loss
                loss = criterion(val_pred, torch.flatten(y))
                val_loss += loss.item()

                # calculate and accumulate the accuracy
                val_pred_class = torch.argmax(val_pred, dim=1)
                val_acc += (val_pred_class == torch.flatten(y)).sum().item() / len(
                    val_pred
                )

        # calculate the average loss and accuracy
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)

        return val_loss, val_acc