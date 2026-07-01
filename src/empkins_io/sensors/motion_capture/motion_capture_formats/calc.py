import gzip
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.sensors.motion_capture.body_parts import get_all_body_parts
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM
from empkins_io.utils._types import check_file_exists, path_t


class CalcData(_BaseMotionCaptureDataFormat):
    """Class for handling data from calc files."""

    _HEADER_LENGTH = 5
    axis: Sequence[str]

    def __init__(
        self,
        file_path: path_t,
        system: Optional[MOTION_CAPTURE_SYSTEM] = "perception_neuron",
        frame_time: Optional[float] = 0.017,
    ):
        """Create new ``CalcData`` instance.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to calc file
        frame_time : float, optional
            time between two consecutive frames in seconds. Default: 0.017

        Raises
        ------
        FileNotFoundError
            if the data path is not valid, or the file does not exist
        :ex:`~biopsykit.utils.exceptions.FileExtensionError`
            if file in ``file_path`` is not a calc file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, [".calc", ".gz"])
        check_file_exists(file_path)

        channels = ["pos", "vel", "quat", "acc", "ang_vel"]
        body_parts = list(get_all_body_parts(system=system))
        sampling_rate = 1.0 / frame_time
        axis = list("xyz")
        data = self._load_calc_data(file_path, sampling_rate, channels, body_parts, axis)

        super().__init__(data=data, sampling_rate=sampling_rate, channels=channels, system=system, axis=axis)

    def _load_calc_data(
        self,
        file_path: path_t,
        sampling_rate: float,
        channels: Sequence[str],
        body_parts: Sequence[str],
        axis: Sequence[str],
    ):
        """Load and convert bvh data.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to calc file

        Returns
        -------
        :class:`~pandas.DataFrame`
           dataframe from calc file

        """
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                _raw_data_str = f.read().decode("utf8")
        else:
            with open(file_path) as f:
                _raw_data_str = f.read()

        if "contactL\t" in _raw_data_str:
            _raw_data_str = _raw_data_str.replace("contactL\t", "")
        if "contactR\t" in _raw_data_str:
            _raw_data_str = _raw_data_str.replace("contactR\t", "")
        header_str = _raw_data_str.split("\n")[: self._HEADER_LENGTH + 1]
        header_str = "\n".join(header_str)
        self._header_str = header_str

        data = pd.read_csv(file_path, skiprows=self._HEADER_LENGTH, sep="\t")
        # drop unnecessary columns
        data = data.dropna(how="all", axis=1)
        data = data.drop(columns=data.filter(like="contact"))

        multiindex = pd.MultiIndex.from_product([body_parts, channels, axis], names=["body_part", "channel", "axis"])
        multiindex_list = list(multiindex)

        # add the w-component for the quaternions to the multi-index
        for i in body_parts:
            index_x = multiindex_list.index((i, "quat", "x"))
            multiindex_list.insert(index_x, (i, "quat", "w"))

        multiindex_final = pd.MultiIndex.from_tuples(multiindex_list, names=["body_part", "channel", "axis"])
        data.columns = multiindex_final
        data.index = np.around(data.index / sampling_rate, 5)
        data.index.name = "time"
        return data

    def to_gzip_calc(self, file_path: path_t):
        """Export to gzip-compressed calc file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data


        See Also
        --------
        CalcData.to_calc
            Export data as (uncompressed) calc file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".gz")

        with gzip.open(file_path, "w") as fp:
            self._write_calc_fp(fp, encode=True)

    def to_calc(self, file_path: path_t):
        """Export to calc file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data


        See Also
        --------
        BvhData.to_gzip_calc
            Export data as gzip-compressed calc file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".calc")

        with open(file_path, "w") as fp:
            self._write_calc_fp(fp)

    def _write_calc_fp(self, fp, encode: Optional[bool] = False):
        if encode:
            fp.write(self._header_str.encode("utf8"))
            fp.write(
                self.data.to_csv(
                    float_format="%.4f", sep="\t", header=False, index=False, line_terminator=" \n"
                ).encode("utf8")
            )
        else:
            fp.write(self._header_str)
            fp.write(self.data.to_csv(float_format="%.4f", sep="\t", header=False, index=False, line_terminator=" \n"))
