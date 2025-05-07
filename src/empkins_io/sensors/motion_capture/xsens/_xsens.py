from pathlib import Path

import pandas as pd

from empkins_io.sensors.motion_capture.motion_capture_formats.mvnx import MvnxData
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import InvalidFileFormatError

__all__ = ["XSensDataset"]


class XSensDataset:
    _mvnx_data: MvnxData
    tz: str

    def __init__(self, mvnx_data: MvnxData, tz: str = "Europe/Berlin"):
        self._mvnx_data = mvnx_data
        self.tz = tz

    @classmethod
    def from_mvnx_file(
        cls, file_path: path_t, load_sensor_data: bool = False, tz: str = "Europe/Berlin", *, verbose: bool = True
    ) -> "XSensDataset":
        file_path = Path(file_path)
        if file_path.suffix not in (".mvnx", ".mvnx.gz"):
            raise InvalidFileFormatError(
                f"File {file_path} is not a valid mvnx (or compressed mvnx) file. Please check the file extension."
            )

        mvnx_data = MvnxData(file_path, load_sensor_data=load_sensor_data, verbose=verbose)
        return cls(mvnx_data=mvnx_data, tz=tz)

    def data_as_df(self, index: str | None = None):
        """Return the data as a pandas DataFrame.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        Returns
        -------
        pd.DataFrame
            The data as a pandas DataFrame.
        """
        data = self._mvnx_data.data.copy()

        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.tz})",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")
        index_name = index_names[index]

        if index is None:
            data = data.reset_index(drop=True)
        elif "datetime" in index:
            data.index = pd.to_timedelta(data.index, unit="s")
            data.index += self._mvnx_data.start_time
            data.index = data.index.tz_localize("UTC")
            if index == "local_datetime":
                data.index = data.index.tz_convert(self.tz)

        data.index.name = index_name
        return data


# def _get_files(folder_path: path_t, extensions: Sequence[str] | str):
#     if isinstance(extensions, str):
#         extensions = [extensions]
#     file_list = []
#     for ext in extensions:
#         file_list.extend(sorted(folder_path.glob(f"*{ext}")))
#     return file_list
#
#
# def load_xsens_folder(folder_path: path_t, index_start: int | None = 0, index_end: int | None = -1) -> dict[str, Any]:
#     # ensure pathlib
#     folder_path = Path(folder_path)
#     _assert_is_dir(folder_path)
#     return_dict: dict[str, _BaseMotionCaptureDataFormat] = dict.fromkeys(["mvnx"])
#
#     mvnx_files = _get_files(folder_path, [".mvnx", ".mvnx.gz"])
#     if len(mvnx_files) == 1:
#         mvnx_data = MvnxData(mvnx_files[0])
#         return_dict["mvnx"] = mvnx_data
#
#     for key in return_dict:
#         if return_dict[key] is not None:
#             return_dict[key] = return_dict[key].cut_data(index_start, index_end)
#     return return_dict
