from copy import deepcopy
from typing import Dict, Any, Optional, Sequence, Tuple

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
        """Filter positional displacement drift in bvh data.

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
        # extract position data of the bvh file object containing the motion data of the hip
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

    def bvh_filter_rotation_drift(self, key: str, filter_params: Optional[Dict[str, Any]] = None):
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

        # convert euler angles to rad and unwrap
        rot_data_unwrap = np.unwrap(np.deg2rad(rot_data), axis=0)
        rot_data = pd.DataFrame(rot_data_unwrap, index=rot_data.index, columns=rot_data.columns)

        rot_data, drift_data = self._filter_rotation_drift(
            rot_data, body_parts, base_filter_params, additional_filter_params_list
        )

        data_filt.loc[:, rot_data.columns] = rot_data.loc[:, :]

        data_dict_ret["bvh"].data = data_filt
        return data_dict_ret, drift_data

    def calc_filter_position_drift(self, key: str, Wn: Optional[float] = 0.01) -> Dict[str, Any]:
        """Filter positional displacement drift in calc data.

        Parameters
        ----------
        Wn : float, optional
            Wn parameter of filter passed to :func:`scipy.signals.butter`.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with data from calc file corrected for positional displacement drift

        """
        data_dict_ret = deepcopy(self.data_dict[key])
        data = self.data_dict[key]["calc"].data

        # extract position data of the calc file object containing the motion data
        pos_data = data.loc[:, pd.IndexSlice[:, "pos", :]].copy()

        # set the model to the origin (only x and y - z goes "up")
        x_slice = pd.IndexSlice[:, "pos", "x"]
        y_slice = pd.IndexSlice[:, "pos", "y"]

        pos_data.loc[:, x_slice] -= pos_data.loc[:, x_slice].iloc[0]
        pos_data.loc[:, y_slice] -= pos_data.loc[:, y_slice].iloc[0]

        # filter data using butterworth filter
        sos = ss.butter(N=3, Wn=Wn, fs=self.sampling_rate["bvh"], btype="high", output="sos")
        pos_data_filt = ss.sosfiltfilt(sos, x=pos_data, axis=0)

        pos_data_filt = pd.DataFrame(pos_data_filt, columns=pos_data.columns, index=pos_data.index)
        # add the position of time point zero
        pos_data_filt = pos_data_filt.add(pos_data.iloc[0, :])

        data_filt = data.copy()
        data_filt.loc[:, pd.IndexSlice[:, "pos", :]] = pos_data_filt.iloc[:, :]

        data_dict_ret["calc"].data = data_filt
        return data_dict_ret

    def calc_filter_rotation_drift(self, key: str, filter_params: Optional[Dict[str, Any]] = None):
        data_dict_ret = deepcopy(self.data_dict[key])
        data = self.data_dict[key]["calc"].data
        data_filt = data.copy()

        if filter_params is None:
            filter_params = {}
        base_filter_params = filter_params.get("base", {})
        additional_filter_params_list = filter_params.get("additional", [])

        body_parts = base_filter_params.get("body_parts", None)
        if not body_parts:
            body_parts = list(data.columns.get_level_values("body_part").unique())

        rot_data = data[body_parts].filter(like="quat")
        rot_data_euler = quat_to_euler_hierarchical(rot_data.reindex(list("xyzw"), level="axis", axis=1), seq="yxz")
        rot_data = pd.DataFrame(
            np.unwrap(rot_data_euler, axis=0), columns=rot_data_euler.columns, index=rot_data_euler.index
        )

        rot_data, drift_data = self._filter_rotation_drift(
            rot_data, body_parts, base_filter_params, additional_filter_params_list, to_euler=False
        )

        data_filt.loc[:, rot_data.columns] = rot_data.loc[:, :]

        data_dict_ret["calc"].data = data_filt
        return data_dict_ret, drift_data

    def center_mass_filter_position_drift(self, key: str, Wn: Optional[float] = 0.01) -> Dict[str, Any]:
        """Filter positional displacement drift in center-of-mass data.

        Parameters
        ----------
        Wn : float, optional
            Wn parameter of filter passed to :func:`scipy.signals.butter`.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with data from center-of-mass file corrected for positional displacement drift

        """
        data_dict_ret = deepcopy(self.data_dict[key])
        data = self.data_dict[key]["center_mass"].data

        # filter data using butterworth filter
        sos = ss.butter(N=3, Wn=Wn, fs=self.sampling_rate["bvh"], btype="high", output="sos")
        data_filt = ss.sosfiltfilt(sos, x=data, axis=0)

        data_filt = pd.DataFrame(data_filt, columns=data.columns, index=data.index)
        # add the position of time point zero
        data_filt = data_filt.add(data.iloc[0, :])

        data_dict_ret["center_mass"].data = data_filt
        return data_dict_ret

    @classmethod
    def _extract_sampling_rate(cls, data_dict: Dict[str, Any]) -> Dict[str, float]:
        return {key: val.sampling_rate for key, val in data_dict.items() if val is not None}

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

    def _approximate_rotation_drift(
        self, data: pd.DataFrame, Wn: Optional[float] = 0.01, data_type: Optional[str] = "bvh"
    ) -> pd.DataFrame:
        # loop over body parts
        sos = ss.butter(N=1, Wn=Wn, fs=self.sampling_rate[data_type], btype="high", output="sos")
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
        data_type: Optional[str] = "bvh",
    ):
        body_parts = filter_params["body_parts"]

        data_before = data.loc[:, body_parts]
        sos = ss.butter(
            N=filter_params["N"], Wn=filter_params["Wn"], fs=self.sampling_rate[data_type], btype="high", output="sos"
        )
        drift_data_update = ss.sosfiltfilt(sos=sos, x=data_before, axis=0)

        drift_data_update = pd.DataFrame(data=drift_data_update, columns=data_before.columns, index=data_before.index)
        drift_data_update = data_before - drift_data_update

        drift_data.loc[:, body_parts] = drift_data_update.loc[:, body_parts]
        return drift_data
