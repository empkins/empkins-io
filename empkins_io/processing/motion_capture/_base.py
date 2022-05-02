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

    def cut_data(self, index_start: int, index_end: int):
        for key in self.data_dict:
            self.data_dict[key].cut_data(index_start, index_end)

    @abc.abstractmethod
    def filter_position_drift(
        self, key: str, filter_params: Optional[Dict[str, Any]] = None
    ) -> _BaseMotionCaptureDataFormat:
        pass

    @abc.abstractmethod
    def filter_rotation_drift(
        self, key: str, filter_params: Optional[Sequence[Dict[str, Any]]] = None
    ) -> _BaseMotionCaptureDataFormat:
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
        filter_params_list: Sequence[Dict[str, Any]],
        to_euler: Optional[bool] = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get rotation order sequence
        seq = "".join(rot_data.columns.get_level_values("axis").unique())
        drift_data = None

        rot_data = euler_to_quat_hierarchical(data=rot_data, seq=seq)
        for filter_params in filter_params_list:
            body_parts = filter_params.get("body_parts", None)
            if body_parts is None:
                body_parts = list(rot_data.columns.get_level_values("body_part").unique())

            drift_data_new = self._approximate_rotation_drift(rot_data, drift_data, body_parts, filter_params)

            if drift_data is None:
                drift_data = drift_data_new
            else:
                drift_data.loc[:, drift_data_new.columns] = drift_data.loc[:, :]

            rot_data_new = rotate_quat_hierarchical(rot_data, drift_data, body_parts)
            rot_data.loc[:, rot_data_new.columns] = rot_data_new.iloc[:, :]

        if to_euler:
            rot_data = quat_to_euler_hierarchical(data=rot_data, seq=seq, degrees=True)

        return rot_data, drift_data

    def _approximate_rotation_drift(
        self,
        data: pd.DataFrame,
        drift_data: pd.DataFrame,
        body_parts: Sequence[str],
        filter_params: Dict[str, Any] = None,
    ) -> pd.DataFrame:

        data_before = data.loc[:, body_parts]
        sos = ss.butter(
            N=filter_params.get("N", 1),
            Wn=filter_params.get("Wn", 0.01),
            fs=self.sampling_rate,
            btype="high",
            output="sos",
        )
        drift_data_update = ss.sosfiltfilt(sos=sos, x=data_before, axis=0)
        drift_data_update = pd.DataFrame(data=drift_data_update, columns=data_before.columns, index=data_before.index)

        # difference between original quat data and filtered quat data to approximate the drift
        drift_data_update = data_before - drift_data_update
        if drift_data is None:
            return drift_data_update
        drift_data.loc[:, body_parts] = drift_data_update.loc[:, body_parts]
        return drift_data
