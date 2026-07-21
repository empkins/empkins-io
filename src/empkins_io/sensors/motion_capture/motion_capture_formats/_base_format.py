from collections.abc import Sequence

import pandas as pd

from empkins_io.sensors.motion_capture.body_parts import get_all_body_parts
from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM
from empkins_io.utils._types import T


class _BaseMotionCaptureDataFormat:
    system: MOTION_CAPTURE_SYSTEM
    data: pd.DataFrame
    sampling_rate_hz: float
    body_parts: Sequence[str]
    channels: Sequence[str]
    axis: Sequence[str]
    num_frames: int
    rot_drift_data: pd.DataFrame | None = None

    def __init__(
        self,
        system: MOTION_CAPTURE_SYSTEM,
        data: pd.DataFrame,
        sampling_rate: float,
        channels: Sequence[str] | None = None,
        axis: Sequence[str] | None = None,
    ):
        self.system = system
        self.data = data
        self.sampling_rate_hz = sampling_rate
        self.body_parts = get_all_body_parts(system)
        self.channels = channels
        self.axis = axis
        self.num_frames = len(data)

    def cut_data(self, index_start: int, index_end: int) -> T:
        self.data = self.data.iloc[index_start:index_end, :]
        self.num_frames = len(self.data)
        return self
