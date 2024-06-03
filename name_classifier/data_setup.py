from typing import Dict, List, Tuple

import pandas as pd

import torch
from torch.utils.data import Dataset

from pathlib import Path
import string
import unicodedata

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

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
        self.dataframe = create_dataframe(root_dir)
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        (name, language) = (self.dataframe.iloc[idx, 0], self.dataframe.iloc[idx, 1])
        sample = {"name": name, "language": language}
        return sample
    

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
        and c in ALL_LETTERS
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
    
    return ALL_LETTERS.find(letter)    

def letter_to_tensor(letter: str) -> torch.tensor:
    """Converts a letter to one hot vector.

    Args:
        letter (str): a single character.

    Returns:
        torch.tensor: a one hot vector.
    """
    
    letter_tensor = torch.zeros((1, N_LETTERS))
    letter_tensor[0][letter_to_index(letter)] = 1.0
    return letter_tensor

def line_to_tensor(line: str) -> torch.tensor:
    """Converts a line to a collection of one hot vectors.

    Args:
        line (str): a string.

    Returns:
        torch.tensor: a tensor containing one hot vectors representing the string.
    """
    
    line_tensor = torch.zeros((len(line), 1, N_LETTERS))
    for idx, letter in enumerate(line):
        line_tensor[idx][0][letter_to_index(letter)] = 1.0
    return line_tensor

def create_dataframe(path) -> pd.DataFrame:
    """creates a dataframe with names and their corresponding languages.

    Args:
        path (Path): path to the directory containing data files.

    Returns:
        pd.DataFrame: A pandas dataframe containing names with their corresponding languages.
    """
    (category_line, categories) = get_names(path)
    category_line_df = pd.DataFrame(category_line)
    category_line_df.columns = ["Name", "Language"]
    return category_line_df