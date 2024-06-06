from pathlib import Path

import torch
from torch.utils.data import DataLoader

from name_classifier.data.utils import CustomDataset

def create_dataloader(
    train_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False
) -> DataLoader:
    """Creates a dataloader for the given data.

    Args:
        train_dir (Path): a path to the directory containing training data.
        batch_size (int): number of examples in a batch.
        num_workers (int): number of workers per dataloader.
        persistent_workers (bool): whether the workers are persistent or not.
        pin_memory (bool): whether the memory is pinned or not.

    Returns:
        DataLoader: _description_
    """
    
    dataset = CustomDataset(train_dir)
    
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = persistent_workers,
        pin_memory = True
    )
    
    return dataloader

if __name__ == "__main__":
    dataloader = create_dataloader(
        Path("data/names"),
        1,
        2,
        True,
        True
    )
    
    sample = next(iter(dataloader))
    
    print(sample)