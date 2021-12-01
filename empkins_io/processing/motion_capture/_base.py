import abc
from abc import ABC
from copy import deepcopy
from typing import Sequence, Dict, Any, Optional, Tuple

import pandas as pd
import scipy.signal as ss

from empkins_io.processing.utils.rotations import (
    euler_to_quat_hierarchical,
    rotate_quat_hierarchical,
    quat_to_euler_hierarchical,
)
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat


class _BaseMotionCaptureProcessor(ABC):

    data_dict: Dict[str, _BaseMotionCaptureDataFormat]

    def __init__(self, data: _BaseMotionCaptureDataFormat):
        self.data_dict = {}
        self.add_data("raw", data)
        self.sampling_rate: float = data.sampling_rate

    def add_data(self, key: str, data: _BaseMotionCaptureDataFormat):
        self.data_dict[key] = deepcopy(data)

    @abc.abstractmethod
    def filter_position_drift(self, key: str, Wn: Optional[float] = 0.01) -> _BaseMotionCaptureDataFormat:
        pass

    @abc.abstractmethod
    def filter_rotation_drift(
        self, key: str, filter_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[_BaseMotionCaptureDataFormat, pd.DataFrame]:
        pass

    def _filter_position_drift(self, pos_data: pd.DataFrame, Wn: float):
        # filter data using butterworth filter
        sos = ss.butter(N=3, Wn=Wn, fs=self.sampling_rate, btype="high", output="sos")
        pos_data_filt = ss.sosfiltfilt(sos, x=pos_data, axis=0)

        pos_data_filt = pd.DataFrame(pos_data_filt, columns=pos_data.columns, index=pos_data.index)
        # add the position of time point zero
        pos_data_filt = pos_data_filt.add(pos_data.iloc[0, :])
        return pos_data_filt

    def _filter_rotation_drift(
        self,
        rot_data: pd.DataFrame,
        body_parts: Sequence[str],
        base_filter_params: Dict[str, Any],
        additional_filter_params_list: Sequence[Dict[str, Any]],
        to_euler: Optional[bool] = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get rotation order sequence
        seq = "".join(rot_data.columns.get_level_values("axis").unique())
        rot_data = euler_to_quat_hierarchical(data=rot_data, columns=body_parts, seq=seq)

        drift_data = self._approximate_rotation_drift(rot_data, base_filter_params.get("Wn", 0.01))

        for filter_param_step in additional_filter_params_list:
            drift_data = self._approximate_rotation_drift_update(rot_data, drift_data, filter_param_step)

        rot_data = rotate_quat_hierarchical(rot_data, drift_data, body_parts)
        if to_euler:
            rot_data = quat_to_euler_hierarchical(data=rot_data, columns=body_parts, seq=seq, degrees=True)
        return rot_data, drift_data

    def _approximate_rotation_drift(self, data: pd.DataFrame, Wn: Optional[float] = 0.01) -> pd.DataFrame:
        # loop over body parts
        sos = ss.butter(N=1, Wn=Wn, fs=self.sampling_rate, btype="high", output="sos")
        drift_data = ss.sosfiltfilt(sos=sos, x=data, axis=0)
        drift_data = pd.DataFrame(data=drift_data, columns=data.columns, index=data.index)

        # difference between original quat data and filtered quat data to approximate the drift
        drift_data = data - drift_data
        return drift_data

    def _approximate_rotation_drift_update(
        self,
        data: pd.DataFrame,
        drift_data: pd.DataFrame,
        filter_params: Dict[str, Any] = None,
    ):
        body_parts = filter_params["body_parts"]

        data_before = data.loc[:, body_parts]
        sos = ss.butter(N=filter_params["N"], Wn=filter_params["Wn"], fs=self.sampling_rate, btype="high", output="sos")
        drift_data_update = ss.sosfiltfilt(sos=sos, x=data_before, axis=0)

        drift_data_update = pd.DataFrame(data=drift_data_update, columns=data_before.columns, index=data_before.index)
        drift_data_update = data_before - drift_data_update

        drift_data.loc[:, body_parts] = drift_data_update.loc[:, body_parts]
        return drift_data
