import logging
from collections.abc import Sequence
from pathlib import Path

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
    }.
    """

    data: pd.DataFrame
    aggregated_data: pd.DataFrame
    raw_data: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None
    stance_average_data: pd.DataFrame | None = None

    @classmethod
    def from_folder(
        cls,
        folder_path: path_t,
        read_metadata: bool = False,
        read_stance_average: bool = False,
        read_raw_pressure_data: bool = False,
    ) -> Self:
        """
        Creates a ZebrisDataset instance from a folder containing CSV files.

        Parameters
        ----------
        folder_path : path_t
            Path to the folder containing CSV files.

        Returns
        -------
        ZebrisDataset
            An instance of ZebrisDataset with the data loaded from the specified folder.
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
        Initializes an object that processes CSV files from a given file path or directory.

        The constructor verifies whether the provided path is valid. If a directory is
        provided, it searches for all CSV files within the directory. If it is a single
        file, it checks whether it is a valid CSV file. The object allows optional
        logging to indicate the number of files identified. The data are prepared for
        further processing or analysis.

        Attributes
        ----------
            path (Path): The path to the CSV file or directory containing CSV files.
            explain (bool): Determines whether to log processing details.
            _raw_data (List[Path]): A list of valid CSV file paths found in the path.
            _processed_data (Any): Placeholder for data after processing (implementation dependent).

        Parameters
        ----------
            path (Union[str, Path]): The file path or directory containing CSV files.
            explain (bool): Optional; Defaults to True. If set to True, logs details
                            about the files located and data processing.

        Raises
        ------
            FileNotFoundError: Raised if the provided path is invalid, i.e., not pointing
                               to a valid CSV file or directory.
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
        Processes raw data files to separate them into time-dependent data and single-value data. This method
        parses and categorizes a collection of CSV files based on their metadata attributes, organizing them
        into two separate groups. The method supports optional verbose output for debugging and
        error handling during processing.

        Parameters
        ----------
        file_list : list of Path
            A list of file paths to the CSV files to be processed. The files should contain
            Zebris data with appropriate metadata attributes.
        verbose : bool, optional
            If True, detailed logs are printed during processing. Default is False.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary containing two keys:
            - 'time_dependent_data': DataFrame combining all time-dependent data.
            - 'single_value_data': DataFrame combining all single-value data.
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
        self, *, channel: str_t | None = None, foot: str_t | None = None, foot_region: str_t | None = None
    ) -> pd.DataFrame:
        """Returns the data as a pandas DataFrame."""
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
