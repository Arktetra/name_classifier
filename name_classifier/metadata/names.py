"""A module containing metadata."""

import string

from name_classifier.metadata import shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "names"

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

OUTPUT_DIMS = 18

CATEGORIES = [filename.stem for filename in RAW_DATA_DIRNAME.iterdir()]
N_CATEGORIES = len(CATEGORIES)
