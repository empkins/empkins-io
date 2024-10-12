from functools import lru_cache
from typing import Dict, Optional, Sequence, Union, Tuple
from itertools import product
import pandas as pd
from tpcp import Dataset
from pathlib import Path
import numpy as np

from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.sensors.tfm.tfm import TfmLoader
from empkins_io.utils._types import path_t
from empkins_io.datasets.gapvii.dip_study.helper import (
    _load_general_information,
    _load_radar_data,
    _update_dates,
    _load_single_date,
    _load_tfm_data,
)


class DipStudyDataset(Dataset):
    """
    DipStudyDataset is a dataset class for handling and processing data from the DIP study.

    This class provides methods to load, cache, and retrieve various types of data related to the DIP study,
    including subject information, sampling rates, and TFM and Radar data. It also includes functionality to fill missing
    dates and create an index of subjects and phases.
    """

    exlude_failed: bool
    base_path: path_t
    use_cache: bool

    _SAMPLING_RATES: Dict[str, int] = {
        "radar": 8000000 / 4096 / 2,
    }
    SUBJECTS_MISSING: Tuple[str] = ("VP_15", "VP_18")
    RADAR_FAILURE: Tuple[str] = ("VP_03")
    DEF_DATE = "01.01.1970"
    TFM_TIME_OFFSET = pd.DateOffset(hours=3)

    def __init__(
            self,
            base_path: path_t,
            exlude_failed: bool = False,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = False
    ):
        self.base_path = base_path
        self.exclude_failed = exlude_failed
        self.use_cache = use_cache

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _get_ids(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        for subject_id in self.SUBJECTS_MISSING:
            if subject_id in subject_ids:
                subject_ids.remove(subject_id)

        if self.exclude_failed:
            for subject_id in self.RADAR_FAILURE:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)
        return subject_ids

    def create_index(self):
        subject_ids = self._get_ids()

        phases = ["rest_1", "cpt", "rest_2", "straw", "rest_3"]

        index = list(product(subject_ids, phases))
        index = pd.DataFrame(index, columns=["subject", "phase"])
        return index


    def fill_dates(self):
        subject_ids = self._get_ids()
        subject_date_dict = {}

        # Load dates for all subjects from protocol files
        for subject in subject_ids:
            date = _load_single_date(self.base_path, subject)
            subject_date_dict[subject] = date

        # Fill dates to empkins_dip_study.xlsx
        _update_dates(self.base_path, subject_date_dict, "Sheet1")

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._SAMPLING_RATES

    @property
    def condition_order(self):
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="condition_order")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="condition_order")

    @property
    def cpt_duration(self):
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="cpt_duration")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="cpt_duration")

    @property
    def subject(self) -> str:
        if self.is_single(["subject"]):
            return self.index["subject"][0]
        return None

    @property
    def tfm_raw_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        participant_id = self.index["subject"][0]
        data, fs = _load_tfm_data(self.base_path, participant_id, self.date)
        return data

    @property
    def tfm_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_tfm_data(participant_id, phase)
            return data

        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            data, fs = self._get_tfm_data(participant_id, "all")

            return data

        raise ValueError(
            "TFM data can only be accessed for a single participant and a single condition at once!"
        )
    
    @property
    def emrad_raw_data(self) -> Dict[str, np.ndarray]:
        participant_id = self.index["subject"][0]
        data, fs = _load_radar_data(self.base_path, participant_id, self.sampling_rates["radar"])
        return data
    
    @property
    def emrad_data(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_radar_data(participant_id, phase)
            return data

        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            data, fs = self._get_radar_data(participant_id, "all")
            return data

        raise ValueError(
            "Radar data can only be accessed for a single participant and a single condition at once!"
        )

    @property
    def date(self) -> str:
        try:
            if self.is_single(["subject"]):
                return _load_general_information(base_path=self.base_path, column="date")[self.index["subject"][0]]

            return _load_general_information(base_path=self.base_path, column="date")
        except KeyError:
            # Handle the KeyError, e.g., return a default value or log an error
            print(f"Date was not found in empkins_dip_study.xlsx for subject {self.index['subject'][0]}")
            print("Please check its protocol file and run fill_dates() method.")
            return self.DEF_DATE  # or handle the error as needed

    def _get_tfm_data(self, participant_id: str, phase: str) -> tuple[pd.DataFrame, float]:
        # Check if caching is enabled for data retrieval
        if self.use_cache:
            # TODO implement cache logic
            data, fs = None, None # Placeholder for future cache implementation
            pass
        else:
            # Load the TFM data from the dataset for the specified participant and date
            data, fs = _load_tfm_data(self.base_path, participant_id, self.date)

        # Initialize a dictionary to store results
        res_data = {}
        # Check if all phases are requested
        if phase == "all":
            for signal in data:
                phase_dict = {}
                for key in data[signal]:
                    # Convert each start_ phase's data to a DataFrame
                    if key.startswith("start_"):
                        df = pd.DataFrame(data[signal][key])
                        df.index = df.index - self.TFM_TIME_OFFSET
                        phase_dict[key] = df
                 # Store the phase data for each signal
                res_data[signal] = phase_dict
        # Otherwise, only retrieve data for the specified phase
        else:
            for signal in data:
                phase_dict = {}
                for key in data[signal]:
                    # Only retrieve data for the specific start phase
                    if key == f"start_{phase}":
                        df = pd.DataFrame(data[signal][key])
                        df.index = df.index - self.TFM_TIME_OFFSET
                        phase_dict[key] = df
                # Store the specific phase data for each signal
                res_data[signal] = phase_dict

        # Return the structured result data and the sampling rate
        return res_data, fs
    
    def _get_radar_data(self, participant_id: str, phase: str) -> tuple[pd.DataFrame, float]:
        if self.use_cache:
            # TODO implement cache
            data, fs = None, None # Placeholder for future cache implementation
            pass
        else:
            # Load radar data for the given participant_id from the base path
            data, fs = _load_radar_data(self.base_path, participant_id, self.sampling_rates["radar"])

        # Dictionary to store filtered radar data based on the phase
        res_data = {}  
        # If phase is "all", process all available signals
        if phase == "all":
            sig, sig_data = next(iter(self.tfm_data.items()))  # Get first signal and its data
            for key in sig_data:  # Loop over the signal data
                # Get the start and end date from the signal's index
                start_date = sig_data[key].index[0]
                end_date = sig_data[key].index[-1]
                # Filter the radar data using the date range
                df_filtered = data.loc[start_date:end_date]
                # Store the filtered data for the current key in the results dictionary
                res_data[key] = pd.DataFrame(df_filtered)
        # If a specific phase is selected, process only that phase
        else:
            sig, sig_data = next(iter(self.tfm_data.items()))  # Get the signal and its data
            for key in sig_data:
                if key == f"start_{phase}":  # Match the key to the selected phase
                    # Get the start and end date from the signal's index
                    start_date = sig_data[key].index[0]
                    end_date = sig_data[key].index[-1]
                    # Filter the radar data using the date range
                    df_filtered = data.loc[start_date:end_date]
                    # Store the filtered data for the current phase in the results dictionary
                    res_data[key] = pd.DataFrame(df_filtered)

        # Return the filtered data and the sampling frequency
        return res_data, fs
