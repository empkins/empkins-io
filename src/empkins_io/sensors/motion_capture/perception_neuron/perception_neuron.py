import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dir

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.bvh import BvhData
from empkins_io.sensors.motion_capture.motion_capture_formats.calc import CalcData
from empkins_io.sensors.motion_capture.motion_capture_formats.center_mass import CenterOfMassData
from empkins_io.utils._types import path_t, str_t


class PerceptionNeuronDataset:
    """Class representing a measurement dataset from the Perception Neuron motion capture system."""

    _data_dict: Dict[str, _BaseMotionCaptureDataFormat]
    _start_time_unix: pd.Timestamp
    _tz: str
    _sampling_rate_hz: float

    def __init__(
        self,
        data_dict: Dict[str, _BaseMotionCaptureDataFormat],
        start_time_unix: pd.Timestamp,
        tz: str,
        sampling_rate_hz: float,
    ):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_folder` constructor to handle loading recorded Perception Neuron Sessions.

        Parameters
        ----------
        data_dict : dict
            Dictionary with different motion capture data formats as keys and sensor data as values.
        start_time_unix : :class:`pandas.Timestamp`
            Start time of the session in unix time.
        tz : str
            Timezone of the recording.
        sampling_rate_hz : float
            Sampling rate of the motion capture system in Hz.

        """
        self._data_dict = data_dict
        self._start_time_unix = start_time_unix
        self._tz = tz
        self._sampling_rate_hz = sampling_rate_hz

    @property
    def bvh_data(self) -> Optional[BvhData]:
        """Return BVH data."""
        return self._data_dict.get("bvh", None)

    @property
    def calc_data(self) -> Optional[CalcData]:
        """Return Calc data."""
        return self._data_dict.get("calc", None)

    @property
    def center_mass_data(self) -> Optional[CenterOfMassData]:
        """Return Center of Mass data."""
        return self._data_dict.get("center_mass", None)

    @property
    def timezone(self) -> str:
        """Return timezone."""
        return self._tz

    @property
    def sampling_rate_hz(self) -> float:
        """Return sampling rate in Hz."""
        return self._sampling_rate_hz

    @property
    def data_dict(self) -> Dict[str, _BaseMotionCaptureDataFormat]:
        """Return dictionary with all different data formats."""
        return self._data_dict

    @classmethod
    def from_folder(
        cls,
        path: path_t,
        index_start: Optional[int] = 0,
        index_end: Optional[int] = -1,
        start_time: Optional[pd.Timestamp] = None,
        tz: Optional[str] = "Europe/Berlin",
    ) -> "PerceptionNeuronDataset":
        """Load Perception Neuron data from folder.

        Parameters
        ----------
        path : str or Path
            path to folder
        index_start : int, optional
            start index of data to load if data should be cut. Default: 0
        index_end : int, optional
            end index of data to load if data should be cut. Default: -1
        start_time : pandas.Timestamp, optional
            start time of the recording as absolute time stamp. Default: ``None``
        tz : str, optional
            timezone of the recording. Default: "Europe/Berlin"


        Returns
        -------
        :class:`~empkins_io.sensors.motion_capture.perception_neuron.perception_neuron.PerceptionNeuronDataset`
            Perception Neuron Dataset

        """
        return_dict: Dict[str, _BaseMotionCaptureDataFormat] = dict.fromkeys(["bvh", "calc", "center_mass"])

        bvh_data = cls._load_bvh_data(path)
        calc_data = cls._load_calc_data(path)
        center_mass_data = cls._load_center_mass_data(path)

        if bvh_data is not None:
            return_dict["bvh"] = bvh_data
        if calc_data is not None:
            return_dict["calc"] = calc_data
        if center_mass_data is not None:
            return_dict["center_mass"] = center_mass_data

        for key in return_dict:
            if return_dict[key] is not None:
                return_dict[key] = return_dict[key].cut_data(index_start, index_end)

        return cls(
            data_dict=return_dict,
            start_time_unix=start_time,
            tz=tz,
            sampling_rate_hz=return_dict["bvh"].sampling_rate_hz,
        )

    def data_as_df(
        self, data_formats: Optional[str_t] = None, index: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """Export dataset as a single pandas DataFrame.

        Parameters
        ----------
        data_formats : list of str, optional
            List of data formats ("bvh", "calc", "center_mass") to include in the DataFrame or ``None`` to export all
            available data formats. Default: ``None``
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        add_global_pose = False
        if data_formats is None:
            data_formats = list(self._data_dict.keys())
            add_global_pose = True
        if isinstance(data_formats, str):
            data_formats = [data_formats]

        add_global_pose = add_global_pose or ("global_pose" in data_formats)

        data_dict = {key: self._data_dict[key].data for key in data_formats}
        # check for global data file
        if add_global_pose and "bvh" in data_formats:
            data_dict["global_pose"] = self._data_dict["bvh"].data_global

        data = pd.concat(data_dict, names=["data_format"], axis=1).dropna()
        data = self._add_index(data, index)

        return data

    @classmethod
    def _load_bvh_data(cls, path: path_t) -> BvhData:
        bvh_files = _get_files(path, [".bvh", ".bvh.gz"])
        if len(bvh_files) == 1:
            bvh_data = BvhData(bvh_files[0], system="perception_neuron")
            global_pose_files = _get_files(path, ["global_pose.csv", "global_pose.csv.gz"])
            if len(global_pose_files) == 1:
                bvh_data.load_global_pose(global_pose_files[0])
            elif len(global_pose_files) > 1:
                raise ValueError(
                    f"More than one global pose file found in {path}. Please make sure only one global pose "
                    f"file is in the folder!"
                )
            return bvh_data
        if len(bvh_files) > 1:
            raise ValueError(
                f"More than one bvh file found in {path}. Please make sure only one bvh file is in the folder!"
            )

    @classmethod
    def _load_calc_data(cls, path: path_t) -> CalcData:
        calc_files = _get_files(path, [".calc", ".calc.gz"])
        if len(calc_files) == 1:
            return CalcData(calc_files[0], system="perception_neuron")
        if len(calc_files) > 1:
            raise ValueError(
                f"More than one calc file found in {path}. Please make sure only one calc file is in the folder!"
            )

    @classmethod
    def _load_center_mass_data(cls, path: path_t) -> CenterOfMassData:
        center_mass_files = _get_files(path, ["centerOfMass.txt", "centerOfMass.csv"])
        if len(center_mass_files) == 1:
            return CenterOfMassData(center_mass_files[0], system="perception_neuron")
        if len(center_mass_files) > 1:
            raise ValueError(
                f"More than one center of mass file found in {path}. "
                f"Please make sure only one center of mass file is in the folder!"
            )

    def _add_index(self, data: pd.DataFrame, index: str) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")
        if self._start_time_unix is None and index in ["utc", "utc_datetime", "local_datetime"]:
            raise ValueError(
                f"Cannot create index {index} because no start time was supplied when loading the dataset."
            )

        if index is None:
            data = data.reset_index(drop=True)

        # # convert to utc timestamps => for index_type "utc"
        # data.index /= self.sampling_rate_hz
        # data.index += self._start_time_unix
        #
        # if index == "utc_datetime":
        #     data.index = pd.to_datetime(data.index, unit="s")
        #     data.index = data.index.tz_localize("UTC")
        # if index == "local_datetime":
        #     data.index = pd.to_datetime(data.index, unit="s")
        #     data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        index_name = index_names[index]
        data.index.name = index_name

        return data


def _get_files(folder_path: path_t, extensions: Union[Sequence[str], str]):
    if isinstance(extensions, str):
        extensions = [extensions]
    file_list = []
    for ext in extensions:
        files = sorted(folder_path.glob(f"*{ext}"))
        files = [file for file in files if not file.name.startswith("._")]
        file_list.extend(files)
    return file_list


def load_perception_neuron_folder(
    folder_path: path_t, index_start: Optional[int] = 0, index_end: Optional[int] = -1
) -> Dict[str, Any]:
    warnings.warn(
        "The 'load_perception_neuron_folder' function is deprecated and will be removed in a future version. "
        "Use the 'PerceptionNeuronDataset' class instead.",
        DeprecationWarning,
    )

    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)
    return_dict: Dict[str, _BaseMotionCaptureDataFormat] = dict.fromkeys(["bvh", "calc", "center_mass"])

    bvh_files = _get_files(folder_path, [".bvh", ".bvh.gz"])
    if len(bvh_files) == 1:
        bvh_data = BvhData(bvh_files[0], system="perception_neuron")
        global_pose_files = _get_files(folder_path, ["global_pose.csv", "global_pose.csv.gz"])
        if len(global_pose_files) == 1:
            bvh_data.load_global_pose(global_pose_files[0])
        elif len(global_pose_files) > 1:
            raise ValueError(
                f"More than one global pose file found in {folder_path}. Please make sure only one global pose "
                f"file is in the folder!"
            )
        return_dict["bvh"] = bvh_data
    elif len(bvh_files) > 1:
        raise ValueError(
            f"More than one bvh file found in {folder_path}. Please make sure only one bvh file is in the folder!"
        )

    calc_files = _get_files(folder_path, [".calc", ".calc.gz"])
    if len(calc_files) == 1:
        return_dict["calc"] = CalcData(calc_files[0], system="perception_neuron")
    elif len(calc_files) > 1:
        raise ValueError(
            f"More than one calc file found in {folder_path}. Please make sure only one calc file is in the folder!"
        )

    center_mass_files = _get_files(folder_path, ["centerOfMass.txt", "centerOfMass.csv"])
    if len(center_mass_files) == 1:
        return_dict["center_mass"] = CenterOfMassData(center_mass_files[0], system="perception_neuron")
    elif len(center_mass_files) > 1:
        raise ValueError(
            f"More than one center of mass file found in {folder_path}. "
            f"Please make sure only one center of mass file is in the folder!"
        )

    for key in return_dict:
        if return_dict[key] is not None:
            return_dict[key] = return_dict[key].cut_data(index_start, index_end)
    return return_dict
