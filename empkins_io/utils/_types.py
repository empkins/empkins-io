"""Some custom helper types to make type hints and type checking easier."""

from pathlib import Path
from typing import Hashable, TypeVar, Union

import numpy as np
import pandas as pd

_Hashable = Union[Hashable, str]

path_t = TypeVar("path_t", str, Path)  # pylint:disable=invalid-name
arr_t = TypeVar("arr_t", pd.DataFrame, pd.Series, np.ndarray)  # pylint:disable=invalid-name
T = TypeVar("T")
