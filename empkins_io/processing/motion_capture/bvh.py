from copy import deepcopy
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd

from empkins_io.processing.motion_capture._base import _BaseMotionCaptureProcessor
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.bvh import BvhData


class BvhProcessor(_BaseMotionCaptureProcessor):
    def __init__(self, data: BvhData):
        super().__init__(data)

    def filter_position_drift(self, key: str, Wn: Optional[float] = 0.01) -> _BaseMotionCaptureDataFormat:
        """Filter positional displacement drift in bvh data.

        Parameters
        ----------
        Wn : float, optional
            Wn parameter of filter passed to :func:`scipy.signals.butter`.

        Returns
        -------
        :class:`~empkins_io.sensors.motion_capture.motion_capture_formats.bvh.BvhData`
            ``BvhData`` instance with data corrected for positional displacement drift

        """
        bvh_data = deepcopy(self.data_dict[key])
        data = bvh_data.data
        # extract position data of the bvh file object containing the motion data of the hip
        pos_data = data.loc[:, pd.IndexSlice["Hips", "pos", :]].copy()

        # set the model to the origin (only x and z - y goes "up")
        pos_data.loc[:, ("Hips", "pos", "x")] = (
            pos_data.loc[:, ("Hips", "pos", "x")] - pos_data[("Hips", "pos", "x")].iloc[0]
        )
        pos_data.loc[:, ("Hips", "pos", "z")] = (
            pos_data.loc[:, ("Hips", "pos", "z")] - pos_data[("Hips", "pos", "z")].iloc[0]
        )

        pos_data_filt = self._filter_position_drift(pos_data, Wn)
        data.loc[:, pd.IndexSlice["Hips", "pos", :]] = pos_data_filt.iloc[:, :]

        bvh_data.data = data
        return bvh_data

    def filter_rotation_drift(
        self, key: str, filter_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[_BaseMotionCaptureDataFormat, pd.DataFrame]:
        bvh_data = deepcopy(self.data_dict[key])
        data = bvh_data.data

        if filter_params is None:
            filter_params = {}
        base_filter_params = filter_params.get("base", {})
        additional_filter_params_list = filter_params.get("additional", [])

        body_parts = base_filter_params.get("body_parts", None)
        if not body_parts:
            body_parts = list(data.columns.get_level_values("body_part").unique())

        rot_data = data[body_parts].filter(like="rot")

        # convert euler angles to rad and unwrap
        rot_data_unwrap = np.unwrap(np.deg2rad(rot_data), axis=0)
        rot_data = pd.DataFrame(rot_data_unwrap, index=rot_data.index, columns=rot_data.columns)

        rot_data, drift_data = self._filter_rotation_drift(
            rot_data, body_parts, base_filter_params, additional_filter_params_list
        )

        data.loc[:, rot_data.columns] = rot_data.loc[:, :]

        bvh_data.data = data
        return bvh_data, drift_data
