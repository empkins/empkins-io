# -*- coding: utf-8 -*-
"""Module for importing Perception Neuron Motion Capture data saved as .bvh file."""
import re
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.sensors.perception_neuron.body_parts import get_all_body_parts, BODY_PART
from empkins_io.utils._types import _check_file_exists, path_t


class BvhData:
    """Class for handling data from bvh files."""

    def __init__(self, file_path: path_t):
        """Create new ``BvhData`` instance.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to bvh file

        Raises
        ------
        FileNotFoundError
            if the data path is not valid, or the file does not exist
        :ex:`~biopsykit.utils.exceptions.FileExtensionError`
            if file in ``file_path`` is not a bvh file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".bvh")
        _check_file_exists(file_path)

        _raw_data = pd.read_csv(file_path, index_col=False, skip_blank_lines=False)

        index = []
        # calculate the indices for splitting the data
        for i in range(0, len(_raw_data["HIERARCHY"])):  # TODO range needed?
            if "ROOT Hips" in _raw_data["HIERARCHY"][i]:
                index.append(i)
            if "JOINT RightUpLeg" in _raw_data["HIERARCHY"][i]:
                index.append(i)
            if "MOTION" in _raw_data["HIERARCHY"][i]:
                index.append(i)
                break
        self._index_motion = index[2]
        self._channel_indices = [index[0] + 3, index[1] + 3]
        self._hierarchy = _raw_data["HIERARCHY"][0 : index[2] + 1]
        _frame_info_part = _raw_data["HIERARCHY"][index[2] + 1 : index[2] + 3]

        # looking for the original empty line after MOTION and handle it
        if _frame_info_part.isna().any():
            self._frame_info = _raw_data["HIERARCHY"][index[2] + 1 : index[2] + 4]
            self._frame_info = self._frame_info.dropna()
            self._index_motion = self._index_motion + 1
        else:
            raise ValueError("Missing frame info!")

        self.sampling_rate: float = self._frame_info.iloc[1]
        self.sampling_rate = 1.0 / float(re.sub(r"Frame Time: ", "", str(self.sampling_rate)))

        self.body_parts: Sequence[BODY_PART] = get_all_body_parts()
        # set the channels of the bvh files and the multi-index of the dataframe
        self.data: pd.DataFrame = self._load_bvh_data(file_path)
        self.axis: Sequence[str] = list(self.data.columns.get_level_values("axis").unique())

    def _load_bvh_data(self, file_path: path_t):
        """Load and convert bvh data.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to bvh file

        Returns
        -------
        :class:`~pandas.DataFrame`
           dataframe from bvh file

        """
        channels_root, channels = self._get_channels()

        channels_root = [(self.body_parts[0], *channel) for channel in channels_root]
        channels = [(body_part, *channel) for body_part in self.body_parts[1:] for channel in channels]

        multi_index_root = pd.MultiIndex.from_tuples(channels_root, names=["body_part", "channel", "axis"])
        multi_index = pd.MultiIndex.from_tuples(channels, names=["body_part", "channel", "axis"])
        multi_index = multi_index_root.append(multi_index)
        # returns the data in a DataFrame including the multiindex
        data = pd.read_csv(
            file_path,
            sep=" ",
            skiprows=int(self._index_motion + 4),
            names=multi_index,
            index_col=False,
            skip_blank_lines=False,
        )
        data.columns = multi_index
        data.index = data.index / self.sampling_rate
        data.index.name = "time"
        data = data.reindex(columns=list("xyz"), level="axis")
        return data

    def _get_channels(self) -> Tuple[Sequence[Tuple[str, str]], Sequence[Tuple[str, str]]]:
        """Get the correct channels from the bvh data.

        The bvh file from Perception Neuron can either be exported with 6 or 3 channels (with or without position).

        Returns
        -------
        channels_root : list of str
            channels of ROOT node
        channels : list of str
            channels of other nodes

        """
        if not any(channels in self._hierarchy[self._channel_indices[1]] for channels in ["CHANNELS 3", "CHANNELS 6"]):
            raise ValueError("Either 3 or 6 channels must be present in the bvh file!")
        # it's possible to export the bvh data with 3 or 6 channel (with or without position)

        channels_root = self._extract_channels_regex(str(self._hierarchy[self._channel_indices[0]]).strip())
        channels = self._extract_channels_regex(str(self._hierarchy[self._channel_indices[1]]).strip())

        return channels_root, channels

    @staticmethod
    def _extract_channels_regex(channels: str):
        channels = re.sub(r"CHANNELS (\d)", "", channels).strip()
        channels = re.findall(r"(\w)(position|rotation)", channels)
        channels = [(val[1][0:3], val[0].lower()) for val in channels]
        return channels

    def to_bvh(self, file_path: path_t):
        """Export to bvh file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".bvh")

        data_out = self.data.groupby(["body_part", "channel"], sort=False, group_keys=False, axis=1).apply(
            lambda df: self._reindex_axis(df)
        )

        with open(file_path, "w") as fp:
            fp.write(self._hierarchy.to_csv(index=False, line_terminator="\n"))
            # set the empty line after MOTION
            fp.write("\n")
            fp.write(self._frame_info.to_csv(index=False, header=False, line_terminator="\n"))
            fp.write(data_out.round(6).to_csv(sep=" ", header=False, index=False, line_terminator="\n"))

    def _reindex_axis(self, data: pd.DataFrame):
        if "rot" in data.name:
            return data.reindex(columns=list("yxz"), level="axis")
        return data
