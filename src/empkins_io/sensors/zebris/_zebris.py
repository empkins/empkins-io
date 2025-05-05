import logging
from pathlib import Path
from typing import Dict, Union, Sequence, Optional

import pandas as pd
from typing_extensions import Self

from empkins_io.sensors.zebris._helper import (
    _read_force_curve_csv,
    _read_gait_line_csv,
    _read_parameters_csv,
    _read_patient_info_csv,
    _read_stance_average_csv,
)
from empkins_io.utils._types import path_t, str_t

logging.basicConfig(level=logging.INFO, format="%(message)s")


class ZebrisDataset:
    """_sensor_dict = {
        "force curve": ["values"],  # force-curve.csv
        "force curve L": ["values"],  # force-curve-L.csv
        "force curve R": ["values"],  # force-curve-R.csv
        "force forefoot L %": ["values"],  # parameters.csv
        "force forefoot R %": ["values"],  # parameters.csv
        "force backfoot L %": ["values"],  # parameters.csv
        "force backfoot R %": ["values"],  # parameters.csv
        "total force %": ["values"],  # parameters.csv
        "total force R %": ["values"],  # parameters.csv
        "total force L %": ["values"],  # parameters.csv
        "force curve backfoot L": ["values"],  # force-curve_backfoot-L.csv
        "force curve backfoot R": ["values"],  # force-curve_backfoot-R.csv
        "force curve forefoot L": ["values"],  # force-curve_forefoot-L.csv
        "force curve forefoot R": ["values"],  # force-curve_forefoot-R.csv
        "gait line": ["x", "y"],  # gait-line.csv
        "gait line L": ["x", "y"],  # gait-line-L.csv
        "gait line R": ["x", "y"],  # gait-line-R.csv
        "COP": ["x", "y"],  # parameters.csv in [mm]
        "COP L": ["x", "y"],  # parameters.csv in [mm]
        "COP R": ["x", "y"],  # parameters.csv in [mm]
        "COP forefoot L": ["x", "y"],  # parameters.csv [mm]
        "COP forefoot R": ["x", "y"],  # parameters.csv [mm]
        "COP backfoot L": ["x", "y"],  # parameters.csv [mm]
        "COP backfoot R": ["x", "y"],  # parameters.csv [mm]
        "COP trajectory length": ["values"],  # parameters.csv in [mm]
        "total time": ["values"],  # parameters.csv
        "Area of the 95% confidence ellipse": ["values"],  # parameters.csv [mm2]
        "average velocity": ["values"],  # parameters.csv [mm/s]
    }
    """

    data: pd.DataFrame
    aggregated_data: pd.DataFrame
    raw_data: Optional[pd.DataFrame] = None
    metadata: Optional[pd.DataFrame] = None
    stance_average_data: Optional[pd.DataFrame] = None

    @classmethod
    def from_folder(
        cls,
        folder_path: path_t,
        read_metadata: bool = False,
        read_stance_average: bool = False,
        read_raw_pressure_data: bool = False,
    ) -> Self:
        """
        Creates an instance of the class using data stored in a specified folder containing CSV files.

        The method performs data parsing by reading and optionally processing metadata,
        stance averages, and raw pressure data from the specified folder path. It ensures
        that the provided path is valid and points to a directory before proceeding with
        the data extraction.

        Arguments:
            folder_path (path_t): The folder path from which to read data. It can be provided
                                  as a string or any object supported by Pathlib for file representation.
            read_metadata (bool): Flag indicating whether to include metadata parsing. Defaults to False.
            read_stance_average (bool): Flag indicating whether to include parsing of stance average data.
                                        Defaults to False.
            read_raw_pressure_data (bool): Flag indicating whether to include processing of raw pressure
                                           data. Defaults to False.

        Returns:
            Self: A new instance of the class initialized with the parsed data.

        Raises:
            FileNotFoundError: If the folder path does not exist or is not a directory.
        """
        # ensure pathlib
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Invalid path: '{folder_path}'. Not a directory.")

        file_list = sorted(folder_path.glob("*.csv"))

        data = cls._parse_data(file_list, read_metadata, read_stance_average, read_raw_pressure_data)
        return cls(data)

    def __init__(self, data_dict: dict[str, pd.DataFrame]):
        """
        A class initializer that dynamically sets attributes from a dictionary with DataFrame values.

        Attributes:
        data_dict (dict[str, pd.DataFrame]): A dictionary where keys are attribute names
        and values are pandas DataFrame objects.

        Args:
        data_dict (dict[str, pd.DataFrame]): Dictionary containing attribute names and their
        corresponding pandas DataFrame values.
        """

        for key, val in data_dict.items():
            setattr(self, key, val)

    @classmethod
    def _parse_data(
        cls,
        file_list: Sequence[Path],
        read_metadata: bool = False,
        read_stance_average: bool = False,
        read_raw_pressure_data: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Parses a list of file paths to extract and process specific data types, including time
        series data, aggregated data, metadata, and stance average data. Supports optional
        reading of metadata and stance average data based on the provided arguments. Raises an
        exception if raw pressure data is requested as it is not implemented.

        Args:
            file_list (Sequence[Path]): A sequence of file paths to be processed.
            read_metadata (bool, optional): Whether to read patient metadata files. Defaults to False.
            read_stance_average (bool, optional): Whether to read stance average files. Defaults to False.
            read_raw_pressure_data (bool, optional): Whether to read raw pressure data. Defaults to False.

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing processed data categories as
            DataFrame objects. Keys include "data", "aggregated_data", "metadata", and
            "stance_average_data" depending on the data processed.

        Raises:
            NotImplementedError: If read_raw_pressure_data is set to True as it is not currently implemented.
        """
        return_dict = {}
        time_series_data = []
        aggregated_data = None
        metadata = None
        stance_average_data = None

        for file_path in file_list:
            file_name = file_path.stem.lower()
            if "force-curve" in file_name:
                time_series_data.append(_read_force_curve_csv(file_path))
            if "gait-line" in file_name:
                time_series_data.append(_read_gait_line_csv(file_path))
            if "parameters" in file_name:
                aggregated_data = _read_parameters_csv(file_path)
            if "patient" in file_name and read_metadata:
                metadata = _read_patient_info_csv(file_path)
            if "stance-average" in file_name and read_stance_average:
                stance_average_data = _read_stance_average_csv(file_path)

        if read_raw_pressure_data:
            raise NotImplementedError("Reading raw pressure data is not implemented yet!")

        time_series_data = pd.concat(time_series_data, axis=1)
        return_dict["data"] = time_series_data
        return_dict["aggregated_data"] = aggregated_data
        if metadata is not None:
            return_dict["metadata"] = metadata
        if stance_average_data is not None:
            return_dict["stance_average_data"] = stance_average_data

        return return_dict

    def data_as_df(
        self, *, channel: Optional[str_t] = None, foot: Optional[str_t] = None, foot_region: Optional[str_t] = None
    ) -> pd.DataFrame:
        """
        Transforms the internal data structure to a pandas DataFrame, allowing optional filtering by specific channels,
        feet, or foot regions. Filters are applied if their respective parameters are provided.

        Parameters
        ----------
        channel : Optional[str_t]
            A single channel name or a list of channel names to filter the data. If provided,
            the data will be filtered to include only the specified channels.
        foot : Optional[str_t]
            A single foot name or a list of foot names to filter the data. If provided, the data
            will be filtered to include only the specified feet.
        foot_region : Optional[str_t]
            A single foot region name or a list of foot region names to filter the data. If provided,
            the data will be filtered to include only the specified foot regions.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame object containing the filtered or unfiltered data.
        """
        if isinstance(channel, str):
            channel = [channel]
        if isinstance(foot, str):
            foot = [foot]
        if isinstance(foot_region, str):
            foot_region = [foot_region]

        data = self.data
        if channel is not None:
            data = data.reindex(channel, level="channel", axis=1)
        if foot is not None:
            data = data.reindex(foot, level="foot", axis=1)
        if foot_region is not None:
            data = data.reindex(foot_region, level="foot_region", axis=1)
        return data
