from typing import Dict, List, Tuple

import torch

from pathlib import Path
import string
import unicodedata

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

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
    
    category_line = {}
    categories = []
        
    for file_path in path.iterdir():
        categories.append(file_path.stem)
        with open(file_path, encoding = "utf-8") as f:
            category_line[file_path.stem] = [unicode_to_data(line.strip()) for line in f.readlines()]
            
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