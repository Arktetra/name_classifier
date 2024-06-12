"""A module containing a model class for LSTM."""

import torch

from typing import Tuple
import torch.nn.functional as F

class LSTM(torch.nn.Module):

    """Creates a LSTM unit.

    Args:
    ----
        input_size (int): the size of the input layer.
        hidden_size (int): the size of the hidden layer.
        output_size (int): the size of the output layer.

    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        # linear layers for input and hidden state to forget gate
        self.l_fx = torch.nn.Linear(input_size, hidden_size)
        self.l_fa = torch.nn.Linear(hidden_size, hidden_size)

        # linear layers for input and hidden state to input gate
        self.l_ix = torch.nn.Linear(input_size, hidden_size)
        self.l_ia = torch.nn.Linear(hidden_size, hidden_size)

        # linear layers for input and hidden state to candidate memory content (cell gate)
        self.l_gx = torch.nn.Linear(input_size, hidden_size)
        self.l_ga = torch.nn.Linear(hidden_size, hidden_size)

        # linear layers for input and hidden state to output gate
        self.l_ox = torch.nn.Linear(input_size, hidden_size)
        self.l_oa = torch.nn.Linear(hidden_size, hidden_size)

        # an output layer
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        self.softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(
        self, input: torch.tensor, hidden: torch.tensor, memory: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Create a long short term memory unit.

        Args:
        ----
            input (torch.tensor): the input at time t.
            hidden (torch.tensor): the hidden state of the lstm at time t.
            memory (torch.tensor): the memory of the lstm at time t.

        Returns:
        -------
            Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]: a tuple (output, (hidden state, memory))

        """
        # calculate the input gate
        i = F.sigmoid(self.l_ix(input) + self.l_ia(hidden))

        # calculate the forget gate
        f = F.sigmoid(self.l_fx(input) + self.l_fa(hidden))

        # calculate the candidate memory content (cell gate)
        g = F.tanh(self.l_gx(input) + self.l_ga(hidden))

        # calculate the output gate
        o = F.sigmoid(self.l_ox(input) + self.l_oa(hidden))

        # calculate the new memory content
        c = f * memory + i * g  # * is Hadamard product

        # calculate the new hidden state
        a = o * F.tanh(c)

        # calculate the output at time t
        output = self.output_layer(a)
        output = self.softmax(output)

        # return the hidden state and memory
        return output, (a, c)

    def init_hidden(self) -> torch.tensor:
        """Intialize the hidden state of lstm at time 0 to zero vector.

        Returns
        -------
            torch.tensor: a zero tensor.

        """
        return torch.zeros((1, self.hidden_size))

    def init_memory(self) -> torch.tensor:
        """Initialize the memory of lstm at time 0 to zero vector.

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
        device: torch.device,
    ) -> Tuple[float, float]:
        """Perform a single training step.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): A dataloader instance to train the model on.
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
            # Initialize the hidden state and the memory
            hidden = self.init_hidden().to(device)
            memory = self.init_memory().to(device)

            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            for i in range(X.size()[1]):
                y_pred, (hidden, memory) = self(X[:, i], hidden, memory)

            # Calculate and accumulate the loss
            loss = criterion(y_pred, torch.flatten(y))
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Calculate and accumulate the accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == torch.flatten(y)).sum() / len(y_pred)

        # Calculate the average loss and accuracy
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        return train_loss, train_acc

    def validation_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """Performs a single validation step.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): A dataloader instance to test the model on.
            criterion (torch.nn.Module): A PyTorch loss function to calculate loss on the test data.
            device (torch.device): A target device to compute on (e.g. "cuda" or "cpu")

        Returns:
        -------
            Tuple[float, float]: A tuple of validation loss and validation accuracy.

        """
        self.eval()

        val_loss, val_acc = 0, 0

        with torch.inference_mode():
            for X, y in dataloader:
                # send the data to the target device
                X, y = X.to(device), y.to(device)

                hidden = self.init_hidden().to(device)
                memory = self.init_memory().to(device)

                # forward pass
                for i in range(X.size()[1]):
                    val_pred, (hidden, memory) = self(X[:, i], hidden, memory)

                # calculate and accumulate the loss
                loss = criterion(val_pred, torch.flatten(y))
                val_loss += loss.item()

                # calculate and accumulate the accuracy
                val_pred_class = torch.argmax(val_pred, dim=1)
                val_acc += (val_pred_class == torch.flatten(y)).sum().item() / len(val_pred)

        # calculate the average loss and accuracy
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)

        return val_loss, val_acc