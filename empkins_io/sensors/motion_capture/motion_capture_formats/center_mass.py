from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM
from empkins_io.utils._types import path_t, _check_file_exists


class CenterOfMassData(_BaseMotionCaptureDataFormat):
    """Class for handling data from center-of-mass text files."""

    axis: Sequence[str]

    def __init__(self, file_path: path_t, system: MOTION_CAPTURE_SYSTEM, frame_time: Optional[float] = 0.017):
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
        _assert_file_extension(file_path, (".txt", ".csv"))
        _check_file_exists(file_path)

        sampling_rate = 1.0 / frame_time
        axis = list("xyz")

        body_parts = ["CenterMass"]
        channels = ["center_mass"]

        # read the file data and filter the data
        if file_path.suffix == ".txt":
            data = pd.read_csv(file_path, sep=" ", header=None, names=["x", "y", "z"])
            data.index = np.around(data.index / sampling_rate, 5)
            data.index.name = "time"
            data.columns = pd.MultiIndex.from_product(
                [body_parts, channels, data.columns], names=["body_part", "channel", "axis"]
            )
        else:
            data = pd.read_csv(file_path, header=list(range(0, 3)), index_col=0)

        super().__init__(data=data, system=system, sampling_rate=sampling_rate, channels=channels, axis=axis)

    def to_csv(self, file_path: path_t):
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".csv")

        self.data.to_csv(file_path, float_format="%.4f")
