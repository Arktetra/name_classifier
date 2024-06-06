from pathlib import Path
from typing import Tuple

import torch

from utils import CustomDataset

def create_dataloaders(
    root_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Creates a train and test dataloader from the given root directory.

    Args:
        train_dir (Path): a path to the directory containing training data.
        batch_size (int): number of examples in a batch.
        num_workers (int): number of workers per dataloader.
        persistent_workers (bool): whether the workers are persistent or not.
        pin_memory (bool): whether the memory is pinned or not.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    
    # create a custom dataset
    dataset = CustomDataset(root_dir)
    
    # split the dataset into train and test dataset with 80% in train and 
    # 20% in test
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator = torch.Generator().manual_seed(42)
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = persistent_workers,
        pin_memory = pin_memory
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = persistent_workers,
        pin_memory = pin_memory
    )
    
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = create_dataloaders(
        Path("data/names"),
        1
    )
    
    print(len(train_dataloader))
    print(len(test_dataloader))