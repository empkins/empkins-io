from copy import deepcopy
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import scipy.signal as ss

from empkins_io.processing.utils.rotations import (
    euler_to_quat_hierarchical,
    rotate_quat_hierarchical,
    quat_to_euler_hierarchical,
)


class PerceptionNeuronProcessor:
    data_dict: Dict[str, Dict[str, Any]]

    def __init__(self, data_dict: Dict[str, Any]):
        self.data_dict = {}
        self.add_data("raw", data_dict)

        self.sampling_rate: Dict[str, float] = self._extract_sampling_rate(self.data_dict["raw"])

    def add_data(self, key: str, data_dict: Dict[str, Any]):
        self.data_dict[key] = deepcopy(data_dict)

    def bvh_filter_position_drift(self, key: str, Wn: Optional[float] = 0.01) -> Dict[str, Any]:
        """Filter positional displacement drift in bvh files.

        Parameters
        ----------
        Wn : float, optional
            Wn parameter of filter passed to :func:`scipy.signals.butter`.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with data from bvh file corrected for positional displacement drift

        """
        data_dict_ret = deepcopy(self.data_dict[key])
        data = self.data_dict[key]["bvh"].data
        # extract position data of the bvhFile object containing the motion data
        pos_data = data.loc[:, pd.IndexSlice["Hips", "pos", :]].copy()

        # set the model to the origin (only x and z - y goes "up")
        pos_data.loc[:, ("Hips", "pos", "x")] = (
            pos_data.loc[:, ("Hips", "pos", "x")] - pos_data[("Hips", "pos", "x")].iloc[0]
        )
        pos_data.loc[:, ("Hips", "pos", "z")] = (
            pos_data.loc[:, ("Hips", "pos", "z")] - pos_data[("Hips", "pos", "z")].iloc[0]
        )

        # filter data using butterworth filter
        sos = ss.butter(N=3, Wn=Wn, fs=self.sampling_rate["bvh"], btype="high", output="sos")
        pos_data_filt = ss.sosfiltfilt(sos, x=pos_data, axis=0)

        pos_data_filt = pd.DataFrame(pos_data_filt, columns=pos_data.columns, index=pos_data.index)
        # add the position of time point zero
        pos_data_filt = pos_data_filt.add(pos_data.iloc[0, :])

        data_filt = data.copy()
        data_filt.loc[:, pd.IndexSlice["Hips", "pos", :]] = pos_data_filt.iloc[:, :]

        data_dict_ret["bvh"].data = data_filt
        return data_dict_ret

    def bvh_filter_rotation_drift_bvh(self, key: str, filter_params: Optional[Dict[str, Any]] = None):
        data_dict_ret = deepcopy(self.data_dict[key])
        data = self.data_dict[key]["bvh"].data
        data_filt = data.copy()

        if filter_params is None:
            filter_params = {}
        base_filter_params = filter_params.get("base", {})
        additional_filter_params_list = filter_params.get("additional", [])

        body_parts = base_filter_params.get("body_parts", None)
        if not body_parts:
            body_parts = list(data.columns.get_level_values("body_part").unique())

        rot_data = data[body_parts].filter(like="rot")
        # convert euler angles to rad
        rot_data_unwrap = np.deg2rad(rot_data)
        # unwrap data
        rot_data_unwrap = np.unwrap(rot_data_unwrap, axis=0)

        rot_data = pd.DataFrame(rot_data_unwrap, index=rot_data.index, columns=rot_data.columns)
        # get rotation order sequence
        seq = "".join(rot_data.columns.get_level_values("axis").unique())
        rot_data = euler_to_quat_hierarchical(data=rot_data, columns=body_parts, seq=seq)

        drift_data = self._bvh_approximate_rotation_drift(rot_data, base_filter_params.get("Wn", 0.01))

        for filter_param_step in additional_filter_params_list:
            drift_data = self._bvh_approximate_rotation_drift_update(rot_data, drift_data, filter_param_step)

        rot_data = rotate_quat_hierarchical(rot_data, drift_data, body_parts)
        rot_data = quat_to_euler_hierarchical(data=rot_data, columns=body_parts, seq=seq, degrees=True)

        data_filt.loc[:, rot_data.columns] = rot_data.loc[:, :]

        data_dict_ret["bvh"].data = data_filt
        return data_dict_ret, drift_data

    @classmethod
    def _extract_sampling_rate(cls, data_dict: Dict[str, Any]) -> Dict[str, float]:
        return {key: val.sampling_rate for key, val in data_dict.items()}

    def _bvh_approximate_rotation_drift(self, data: pd.DataFrame, Wn: Optional[float] = 0.01) -> pd.DataFrame:
        # loop over body parts
        sos = ss.butter(N=1, Wn=Wn, fs=self.sampling_rate["bvh"], btype="high", output="sos")
        drift_data = ss.sosfiltfilt(sos=sos, x=data, axis=0)
        drift_data = pd.DataFrame(data=drift_data, columns=data.columns, index=data.index)

        # difference between original quat data and filtered quat data, to approximate the drift
        drift_data = data - drift_data
        return drift_data

    def _bvh_approximate_rotation_drift_update(
        self, data: pd.DataFrame, drift_data: pd.DataFrame, filter_params: Dict[str, Any] = None
    ):
        body_parts = filter_params["body_parts"]

        data_before = data.loc[:, body_parts]
        sos = ss.butter(
            N=filter_params["N"], Wn=filter_params["Wn"], fs=self.sampling_rate["bvh"], btype="high", output="sos"
        )
        drift_data_update = ss.sosfiltfilt(sos=sos, x=data_before, axis=0)

        drift_data_update = pd.DataFrame(data=drift_data_update, columns=data_before.columns, index=data_before.index)
        drift_data_update = data_before - drift_data_update

        drift_data.loc[:, body_parts] = drift_data_update.loc[:, body_parts]
        return drift_data
