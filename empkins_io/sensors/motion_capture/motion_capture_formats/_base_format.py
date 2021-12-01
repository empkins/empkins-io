from typing import Sequence

import pandas as pd

from empkins_io.sensors.motion_capture.body_parts import BODY_PART


class _BaseMotionCaptureDataFormat:

    data: pd.DataFrame
    sampling_rate: float
    channels: Sequence[str]
    body_parts: Sequence[BODY_PART]
    num_frames: int

    def __init__(
        self,
        data: pd.DataFrame,
        sampling_rate: float,
        channels: Sequence[str],
        body_parts: Sequence[BODY_PART],
    ):
        self.data = data
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.body_parts = body_parts
        self.num_frames = len(data)
