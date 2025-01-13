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
    _load_b2b_data
)


class DipStudyDataset(Dataset):
    """
    DipStudyDataset is a dataset class for handling and processing data from the DIP study.

    This class provides methods to load, cache, and retrieve various types of data related to the DIP study,
    including subject information, sampling rates, and TFM and Radar data. It also includes functionality to fill missing
    dates and create an index of subjects and phases.
    """

    base_path: path_t
    exclude_failed: bool
    exclude_noisy_tfm: bool
    use_cache: bool

    _PHASES = ["rest_1", "cpt", "rest_2", "straw", "rest_3"]
    _SAMPLING_RATES: Dict[str, int] = {
        "radar": 8000000 / 4096 / 2,
    }
    DEF_DATE = "01.01.1970"
    SUBJECTS_MISSING: Tuple[str] = ("VP_15", "VP_18")
    RADAR_FAILURE: Tuple[str] = ("VP_03")
    TFM_FAILURE: Tuple[Tuple[str, str]] = (
    #   ("VP_05", "cpt"),
    #   ("VP_08", "straw"),
    #   ("VP_12", "straw"),
    )

    def __init__(
            self,
            base_path: path_t,
            exclude_failed: bool = False,
            exclude_noisy_tfm: bool = False,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = False
    ):
        self.base_path = base_path
        self.exclude_failed = exclude_failed
        self.exclude_noisy_tfm = exclude_noisy_tfm
        self.use_cache = use_cache

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _get_ids(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        # Filter out missing subjects
        for subject_id in self.SUBJECTS_MISSING:
            if subject_id in subject_ids:
                subject_ids.remove(subject_id)

        # Filter out failed subjects
        if self.exclude_failed and self.RADAR_FAILURE:
            for subject_id in self.RADAR_FAILURE:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)
        return subject_ids

    def create_index(self):
        subject_ids = self._get_ids()
        index = list(product(subject_ids, self._PHASES))

        # Filter out TFM failures
        if self.exclude_noisy_tfm and self.TFM_FAILURE:
            index = [entry for entry in index if entry not in self.TFM_FAILURE]

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
    def ordered_phases(self):
        # data = self.tfm_data.values()
        # pre_phases = list(next(iter(data)).keys())
        # phases = [phase.replace("start_", "") for phase in pre_phases if phase != 'start_recording']

        if self.condition_order == 'cpt_first':
            return self._PHASES
        elif self.condition_order == 'straw_first':
            res = self._PHASES.copy()
            # exchange second and fourth element
            res[1], res[3] = res[3], res[1]
            return res
        return None

    @property
    def cpt_duration(self):
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="cpt_duration")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="cpt_duration")

    @property
    def straw_duration(self):
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="straw_duration")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="straw_duration")

    @property
    def subject(self) -> str:
        if self.is_single(["subject"]):
            return self.index["subject"][0]
        return None

    @property
    def tfm_raw_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        participant_id = self.index["subject"][0]
        data, fs = _load_tfm_data(self.base_path, participant_id, self.date)
        self._SAMPLING_RATES.update(fs)
        return data

    @property
    def tfm_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_tfm_data(participant_id, phase)
            self._SAMPLING_RATES.update(fs)
            return data

        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            data, fs = self._get_tfm_data(participant_id, "all")
            self._SAMPLING_RATES.update(fs)     
            return data

        raise ValueError(
            "TFM data can only be accessed for a single participant and a single condition at once!"
        )
    
    @property
    def b2b_data(self) -> pd.DataFrame:
        participant_id = self.index["subject"][0]
        data = _load_b2b_data(self.base_path, participant_id, self.date)
        return data

    
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

        res_data = None
        if phase == "all":
            res_data = data
        else:
            res_data = data

        # Return the filtered data and the sampling frequency
        return res_data, fs
