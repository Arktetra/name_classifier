"""A module containing utility functions."""

import torch

from pathlib import Path
from datetime import datetime


def save_model(model: torch.nn.Module, dir: str, model_name: str) -> None:
    """Saves a PyTorch model to the specified directory.

    Args:
    ----
        model (torch.nn.Module): a PyTorch model to save.
        dir (Path): the directory to save the model at.
        model_name (str): the filename (with extension ".pth" or ".pt") for the saved model.

    """
    target_dir = Path(dir)
    # Create the directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if the model_name is valid
    assert model_name.endswith(".pt") or model_name.endswith(
        ".pth"
    ), "model_name should end with either '.pt' or '.pth'."

    # Save the model
    torch.save(model.state_dict(), Path(dir) / model_name)


def load_model(model: torch.nn.Module, path: str) -> None:
    """Loads a PyTorch model from the specified path.

    Args:
    ----
        model (torch.nn.Module): the container model.
        path (Path): a path to a model to load.

    """
    path = Path(path)

    # Check if the path exists
    assert path.exists()

    # Load the model
    model.load_state_dict(torch.load(path))


def create_writer(model_name: str, extra: str) -> torch.utils.tensorboard.SummaryWriter:
    """Creates a Tensorboard summary writer instance saving to a specific directory log_dir.

    log_dir is a combination of runs/timestamp/model_name/extra, where timestamp
    is the current date in YYYY-MM-DD format.

    Args:
    ----
        model_name (str): Name of the model.
        extra (str): Anything extra to add to the directory. Defaults to None.

    Returns:
    -------
        torch.utils.tensorboard.SummaryWriter: An instance of summary writer saving to log_dir.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = Path("runs") / timestamp / model_name / extra
    else:
        log_dir = Path("runs") / timestamp / model_name

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
