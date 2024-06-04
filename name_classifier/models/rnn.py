import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from typing import Tuple

class RNN(nn.Module):
    """creates a recurrent neural network.

    Args:
        input_size (int): size of the input layer.
        hidden_size (int): size of the hidden layer.
        output_size (int): size of the output layer.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
    def forward(self, input: torch.tensor, hidden: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Applies the RNN to the input and its hidden state.

        Args:
            input (torch.tensor): text inputs
            hidden (torch.tensor): hidden state

        Returns:
            torch.tensor: RNN's output, hidden state
        """
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = F.softmax(output)
        return output, hidden
    
    def init_hidden(self) -> torch.tensor:
        return torch.zeros((1, self.hidden_size))