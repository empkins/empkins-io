from copy import deepcopy
from typing import Dict, Optional, Any, Sequence

import numpy as np
import pandas as pd

from empkins_io.processing.motion_capture._base import _BaseMotionCaptureProcessor
from empkins_io.processing.utils.rotations import quat_to_euler_hierarchical
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.calc import CalcData


class CalcProcessor(_BaseMotionCaptureProcessor):
    def __init__(self, data: CalcData):
        super().__init__(data)

    def filter_position_drift(
        self, key: str, filter_params: Optional[Dict[str, Any]] = None
    ) -> _BaseMotionCaptureDataFormat:
        """Filter positional displacement drift in calc data.

        Parameters
        ----------
        Wn : float, optional
            Wn parameter of filter passed to :func:`scipy.signals.butter`.

        Returns
        -------
        :class:`~empkins_io.sensors.motion_capture.motion_capture_formats.calc.CalcData`
            ``CalcData`` instance with data corrected for positional displacement drift

        """
        calc_data = deepcopy(self.data_dict[key])
        data = calc_data.data

        # extract position data of the calc file object containing the motion data
        pos_data = data.loc[:, pd.IndexSlice[:, "pos", :]].copy()

        # set the model to the origin (only x and y - z goes "up")
        x_slice = pd.IndexSlice[:, "pos", "x"]
        y_slice = pd.IndexSlice[:, "pos", "y"]

        pos_data.loc[:, x_slice] -= pos_data.loc[:, x_slice].iloc[0]
        pos_data.loc[:, y_slice] -= pos_data.loc[:, y_slice].iloc[0]

        pos_data_filt = self._filter_position_drift(pos_data, filter_params.get("Wn", 0.01))

        data.loc[:, pd.IndexSlice[:, "pos", :]] = pos_data_filt.iloc[:, :]

        calc_data.data = data
        return calc_data

    def filter_rotation_drift(
        self, key: str, filter_params: Optional[Sequence[Dict[str, Any]]] = None
    ) -> _BaseMotionCaptureDataFormat:
        calc_data = deepcopy(self.data_dict[key])
        data = calc_data.data

        if filter_params is None:
            filter_params = []
        rot_data = data.filter(like="quat")
        rot_data_euler = quat_to_euler_hierarchical(rot_data.reindex(list("xyzw"), level="axis", axis=1), seq="yxz")

        rot_data = pd.DataFrame(
            np.unwrap(rot_data_euler, axis=0), columns=rot_data_euler.columns, index=rot_data_euler.index
        )

        rot_data, rot_drift_data = self._filter_rotation_drift(rot_data, filter_params, to_euler=False)

        data.loc[:, rot_data.columns] = rot_data.loc[:, :]
        calc_data.data = data
        calc_data.rot_drift_data = rot_drift_data

        return calc_data
