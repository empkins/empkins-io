import pandas as pd
from mvnx import MVNX
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from utils._types import path_t


class MvnxData(_BaseMotionCaptureDataFormat):
    num_frames: int = 0
    sampling_rate: float = 0.0
    data: pd.DataFrame = None

    def __init__(self, file_path: path_t):
        sampling_rate = 0.0
        data = None
        super().__init__(data=data, sampling_rate=sampling_rate, system="xsens")
