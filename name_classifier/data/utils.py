from typing import Dict, List, Tuple

import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
import string
import unicodedata

import name_classifier.metadata.names as metadata

class CustomDataset(Dataset):
    """Creates a custom dataset for name classification.
    
    Args:
        root_dir (Path): a path to the directory containing the data files.
        transform (None): transformation to apply on input.
        target_transform (None): transformation to apply on output.
    """
    
    def __init__(self, root_dir, tranform = None, target_transform = None):
        super().__init__()
        self.root_dir = root_dir
        (self.dataframe, self.categories) = create_dataframe(root_dir)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        (name, language) = (self.dataframe.iloc[idx, 0], self.dataframe.iloc[idx, 1])
        return line_to_tensor(name).squeeze(), torch.tensor([self.categories.index(language)])
    

def unicode_to_data(line: str) -> str:
    """Converts a unicode line to a ascii line.

    Args:
        line (str): a unicode line

    Returns:
        str: an ascii line
    """
    
    return ''.join(
        c for c in unicodedata.normalize("NFD", line)
        if unicodedata.category(c) != "Mn"
        and c in metadata.ALL_LETTERS
    )

def get_names(path: Path) -> Tuple[Dict[str, List], List]:
    """Creates a mapping from language to a list of names and a list of languages.

    Args:
        path (Path): path to the data file.

    Returns:
        Tuple[Dict[str, List], List]: a tuple containing the mapping and categories.
    """
    
    category_line = []
    categories = []
        
    for file_path in path.iterdir():
        categories.append(file_path.stem)
        with open(file_path, encoding = "utf-8") as f:
            # category_line[file_path.stem] = [unicode_to_data(line.strip()) for line in f.readlines()]
            category_line += ((unicode_to_data(line.strip()), file_path.stem) for line in f.readlines())
            
    return category_line, categories
    
def letter_to_index(letter: str) -> int:
    """Returns the index of a letter in ascii_letters.

    Args:
        letter (str): a single character.

    Returns:
        int: the index of the letter in ascii_letters.
    """
    
    return metadata.ALL_LETTERS.find(letter)    

def letter_to_tensor(letter: str) -> torch.tensor:
    """Converts a letter to one hot vector.

    Args:
        letter (str): a single character.

    Returns:
        torch.tensor: a one hot vector.
    """
    
    letter_tensor = torch.zeros((1, metadata.N_LETTERS))
    letter_tensor[0][letter_to_index(letter)] = 1.0
    return letter_tensor

def line_to_tensor(line: str) -> torch.tensor:
    """Converts a line to a collection of one hot vectors.

    Args:
        line (str): a string.

    Returns:
        torch.tensor: a tensor containing one hot vectors representing the string.
    """
    
    line_tensor = torch.zeros((len(line), 1, metadata.N_LETTERS))
    for idx, letter in enumerate(line):
        line_tensor[idx][0][letter_to_index(letter)] = 1.0
    return line_tensor

def tensor_to_category(tensor: torch.tensor) -> str:
    """Get the category of the name from the output of the model.

    Args:
        tensor (torch.tensor): output of the model.

    Returns:
        str: category.
    """
    _, top_idx = tensor.topk(1)
    return metadata.CATEGORIES[top_idx[0].item()]

def create_dataframe(path) -> Tuple[pd.DataFrame, List]:
    """creates a dataframe with names and their corresponding languages.

    Args:
        path (Path): path to the directory containing data files.

    Returns:
        Tuple[pd.DataFrame, List]: A pandas dataframe containing names with their corresponding languages and the list of languages.
    """
    (category_line, categories) = get_names(path)
    category_line_df = pd.DataFrame(category_line)
    category_line_df.columns = ["Name", "Language"]
    global n_categories
    n_categories = len(categories)
    return category_line_df, categories
    
def custom_collate_function(batch: Tuple[torch.tensor, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
    """collates all items in the batch to same size by padding.

    Args:
        batch (Tuple[torch.tensor, torch.tensor]): a batch of (input, target) to collate.

    Returns:
        Tuple[torch.tensor, torch.tensor]: a batch of collated (input, target).
    """
    # Find the maximum length sequence in the batch
    max_len = max(len(item[0]) for item in batch)
    
    batch_x = []
    
    for item in batch:
        pad_len = max_len - len(item[0])
        batch_x.append(torch.concat((item[0], torch.zeros((pad_len, item[0].size()[1])))))
        
    return torch.stack(batch_x, dim = 0), torch.stack([item[1] for item in batch], dim = 0)

if __name__ == "__main__":
    dataset = CustomDataset(
        Path("data/names")
    )
    
    sample = next(iter(dataset))
    
    print(sample)