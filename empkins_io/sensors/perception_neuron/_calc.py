from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.sensors.perception_neuron.body_parts import get_all_body_parts, BODY_PARTS
from empkins_io.utils._types import path_t, _check_file_exists


class CalcData:
    """Class for handling data from calc files."""

    def __init__(self, file_path: path_t, frame_time: Optional[float] = 0.017):
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
        _assert_file_extension(file_path, ".calc")
        _check_file_exists(file_path)

        self.channels: Sequence[str] = ["pos", "vel", "quat", "accel", "angVel"]
        self.axis: Sequence[str] = ["x", "y", "z"]
        self.body_parts: Sequence[BODY_PARTS] = list(get_all_body_parts())
        self.sampling_rate: float = 1.0 / frame_time

        self.data: pd.DataFrame = self._load_calc_data(file_path)

        self._num_frames = len(self.data)

    def _load_calc_data(self, file_path: path_t):
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
        raw_data = pd.read_csv(file_path, skiprows=5, sep="\t")
        # drop unnecessary columns
        data = raw_data.drop(columns=raw_data.columns[-1])
        data = data.drop(columns=data.filter(like="contact"))

        multiindex = pd.MultiIndex.from_product(
            [self.body_parts, self.channels, self.axis], names=["body_part", "channel", "axis"]
        )
        multiindex_list = list(multiindex)

        # add the w-component for the quaternions to the multi-index
        for i in self.body_parts:
            index_x = multiindex_list.index((i, "quat", "x"))
            multiindex_list.insert(index_x, (i, "quat", "w"))

        multiindex_final = pd.MultiIndex.from_tuples(multiindex_list, names=["body_part", "channel", "axis"])
        data.columns = multiindex_final
        data.index = data.index / self.sampling_rate
        data.index.name = "time"
        data = data.reindex(["x", "y", "z"], level="axis", axis=1)
        return data

    # def get_data_for_body_parts(self, body_part_names, channels: Optional[float] = "default"):
    #     """
    #     Returns the multiindex DataFrame for the given bodyParts or bodyPart
    #
    #     Parameters
    #     ----------
    #     body_part_names : list of strings or string
    #        body parts to load from the calc.File object
    #
    #     channels : list of string or string
    #         channels to load from the calc.File object
    #
    #     Returns
    #     -------
    #     multiindex : :class:`~pd.DataFrame`
    #         Data for the given body parts and channels
    #
    #     Raises
    #     ------
    #     ValueError
    #         if the parameter bodyPartNames ist not a str or list of str
    #     ValueError
    #         if the parameter channels ist not a str or list of str
    #     """
    #
    #     # TODO:rais errors
    #     if isinstance(body_part_names, str):
    #         body_part_names = [body_part_names]
    #     if not isinstance(body_part_names, list):
    #         raise ValueError("the body parts must be given as a list")
    #
    #     is_parameter_valid(self.bodyParts, body_part_names)
    #
    #     if channels == "default":
    #         channels = self.channels
    #     else:
    #         if isinstance(channels, str):
    #             channels = [channels]
    #         if not isinstance(channels, list):
    #             raise ValueError("the channels must be given as a list")
    #
    #     is_parameter_valid(self.channels, channels)
    #
    #     return_body_parts = self.data.loc[:, ((body_part_names), (channels))]
    #
    #     # remove the mutiindex for one bodypart and or channel
    #     if len(body_part_names) == 1:
    #         return_body_parts = return_body_parts[body_part_names[0]]
    #         if len(channels) == 1:
    #             return_body_parts = return_body_parts[channels[0]]
    #
    #     return return_body_parts
