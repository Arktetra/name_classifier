from pathlib import Path
from typing import Tuple, Optional, Callable

import torch

from name_classifier.data.utils import CustomDataset
from name_classifier.data.utils import custom_collate_function

def create_dataloaders(
    root_dir: Path,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None
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
    
    if collate_fn == None:
        collate_fn = torch.utils.data.default_collate()
    
    # create a custom dataset
    dataset = CustomDataset(root_dir)
    
    # split the dataset into train and test dataset with 80% in train and 
    # 20% in test
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))],
        generator = torch.Generator().manual_seed(42)
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = persistent_workers,
        pin_memory = pin_memory
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn,
        shuffle = True,
        num_workers = num_workers,
        persistent_workers = persistent_workers,
        pin_memory = pin_memory
    )
    
    return train_dataloader, test_dataloader      

if __name__ == "__main__":
    train_dataloader, test_dataloader = create_dataloaders(
        Path("data/names"),
        64,
        collate_fn = custom_collate_function
    )
    
    print(len(train_dataloader))
    print(len(test_dataloader))