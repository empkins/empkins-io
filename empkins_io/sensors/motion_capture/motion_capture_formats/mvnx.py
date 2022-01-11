import gzip
from pathlib import Path

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
import mvnx
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.utils._types import path_t, _check_file_exists


class MvnxData(_BaseMotionCaptureDataFormat):
    num_frames: int = 0
    sampling_rate: float = 0.0
    data: pd.DataFrame = None

    def __init__(self, file_path: path_t):
        file_path = Path(file_path)
        _assert_file_extension(file_path, [".mvnx", ".gz"])
        _check_file_exists(file_path)

        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                _raw_data = mvnx.MVNX(f)
        else:
            with open(file_path, "r") as f:
                _raw_data = mvnx.MVNX(f)

        sampling_rate = _raw_data.frameRate
        data = None

        super().__init__(data=data, sampling_rate=sampling_rate, system="xsens")
