from pathlib import Path

import string

import name_classifier.metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "names"

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

OUTPUT_DIMS = 18

CATEGORIES = [filename.stem for filename in RAW_DATA_DIRNAME.iterdir()]