from pathlib import Path
from typing import Optional

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.utils._types import path_t, _check_file_exists


class CenterOfMassData:
    """Class for handling data from center-of-mass text files."""

    def __init__(self, file_path: path_t, frame_time: Optional[float] = 0.017):
        """Create new ``CenterOfMassData`` instance.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to txt file containing center of mass data
        frame_time : float, optional
            time between two consecutive frames in seconds. Default: 0.017

        Raises
        ------
        FileNotFoundError
            if the data path is not valid, or the file does not exist
        :ex:`~biopsykit.utils.exceptions.FileExtensionError`
            if file in ``file_path`` is not a txt file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".txt")
        _check_file_exists(file_path)

        self.sampling_rate: float = 1.0 / frame_time

        # read the file data and filter the data
        data = pd.read_csv(file_path, sep=" ", header=None, names=["x", "y", "z"])
        data.columns.name = "axis"
        data.index = data.index / self.sampling_rate
        data.index.name = "time"
        self.data = data