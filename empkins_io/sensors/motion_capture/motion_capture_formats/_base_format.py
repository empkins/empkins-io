from typing import Sequence, Optional

import pandas as pd

from empkins_io.sensors.motion_capture.body_parts import BODY_PART
from empkins_io.utils._types import T


class _BaseMotionCaptureDataFormat:

    data: pd.DataFrame
    sampling_rate: float
    body_parts: Sequence[BODY_PART]
    channels: Sequence[str]
    axis: Sequence[str]
    num_frames: int
    rot_drift_data: Optional[pd.DataFrame] = None

    def __init__(
        self,
        data: pd.DataFrame,
        sampling_rate: float,
        body_parts: Sequence[BODY_PART],
        channels: Optional[Sequence[str]] = None,
        axis: Optional[Sequence[str]] = None,
    ):
        self.data = data
        self.sampling_rate = sampling_rate
        self.body_parts = body_parts
        self.channels = channels
        self.axis = axis
        self.num_frames = len(data)

    def cut_data(self, index_start: int, index_end: int) -> T:
        self.data = self.data.iloc[index_start:index_end, :]
        self.num_frames = len(self.data)
        return self
