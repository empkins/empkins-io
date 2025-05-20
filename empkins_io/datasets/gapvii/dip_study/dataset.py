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
    _load_b2b_data,
    _load_start_end_times,
    _load_empatica_data,
    _create_avro,
    _save_avro,
    _load_phase_times,
    _save_agg_empatica,
    _create_agg_empatica,
    _check_if_file_exists,
    _save_tfm_csv,
)


class DipStudyDataset(Dataset):
    """
    DipStudyDataset is a dataset class for handling and processing data from the DIP study.

    This class provides methods to load, cache, and retrieve various types of data related to the DIP study,
    including subject information, sampling rates, and TFM and Radar data. It also includes functionality to fill missing
    dates and create an index of subjects and phases.
    """
    # Class attributes describing dataset configurations and metadata
    base_path: path_t  # Base directory path where all dataset files are stored
    exclude_failed: bool # Whether to exclude subjects with failed data (e.g., radar failure)
    exclude_noisy_tfm: bool  # Whether to exclude noisy TFM data for certain subject-phase combinations
    use_cache: bool # Flag to enable caching of loaded data for performance

    # List of Empatica sensor signals of interest
    _EMPATICA = ["eda", "temperature", "respiratory-rate", "pulse-rate", "wearing-detection"]
    # List of AVRO (a type of data format) signals to process
    _AVRO = ["eda", "bvp", "temperature"]
    # The phases used in the study, in order
    _PHASES = ["rest_1", "cpt", "rest_2", "straw", "rest_3"]

    # Sampling rates for different sensors; here radar sampling rate calculated by given formula
    _SAMPLING_RATES: Dict[str, int] = {
        "radar": 8000000 / 4096 / 2,
    }

    # Path to the Empatica data file
    EMPATICA_FILE_PATH = "empatica/cleaned/aggregated_empatica.csv"
    # Path to the AVRO data file
    AVRO_FILE_PATH = "empatica/cleaned/avro_empatica.csv"
    # Path to the TFM data file
    TFM_FILE_PATH = "tfm/cleaned/data_tfm.csv"

    # Default fallback date in case date info is missing
    DEF_DATE = "01.01.1970"
    # Subjects missing from dataset (no data available)
    SUBJECTS_MISSING: list[str] = ["VP_15", "VP_18"]
    # Subjects with radar data failures to optionally exclude
    RADAR_FAILURE: list[str] = ["VP_03"]
    # List of (subject, phase) tuples representing noisy or invalid TFM data to exclude if requested
    TFM_FAILURE: list[Tuple[str, str]] = [
    #   ("VP_05", "cpt"),
    ]

    def __init__(
            self,
            base_path: path_t,
            exclude_failed: bool = True,
            exclude_noisy_tfm: bool = False,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = False
    ):
        # Initialize dataset instance with given parameters
        self.base_path = base_path
        self.exclude_failed = exclude_failed
        self.exclude_noisy_tfm = exclude_noisy_tfm
        self.use_cache = use_cache

        # Call the parent Dataset class initializer with optional grouping and subsetting
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _get_ids(self):
        """
        Get list of subject IDs, excluding missing and failed subjects.
        """
        # Get all subject directories matching the pattern "VP_*"
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
        """
        Create DataFrame index of (subject, phase) pairs, excluding failures.
        """
        # Get all subject IDs and create a Cartesian product with the phases
        subject_ids = self._get_ids()
        index = list(product(subject_ids, self._PHASES))

        # Filter out TFM failures
        if self.exclude_noisy_tfm and self.TFM_FAILURE:
            index = [entry for entry in index if entry not in self.TFM_FAILURE]

        # Create a DataFrame from the index list
        index = pd.DataFrame(index, columns=["subject", "phase"])
        return index


    def fill_dates(self):
        """
        Fill missing dates in the protocol files based on loaded subject dates.
        """
        # Get all subject IDs
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
        """
        Return dictionary of sampling rates for different signals.
        """
        return self._SAMPLING_RATES

    @property
    def condition_order(self):
        """
        Return the experimental condition order per subject or for all.
        """
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="condition_order")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="condition_order")
    
    @property
    def ordered_phases(self):
        """
        Return the phase order adjusted based on the condition order.
        """
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
        """
        Return the duration of the CPT phase per subject or for all.
        """
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="cpt_duration")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="cpt_duration")

    @property
    def straw_duration(self):
        """
        Return the duration of the straw phase per subject or for all.
        """
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="straw_duration")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="straw_duration")
    
    @property
    def subject(self) -> str:
        """
        Return the current subject if a single subject is selected.
        """
        if self.is_single(["subject"]):
            return self.index["subject"][0]
        return None
    
    @property
    def phase_times(self) -> pd.DataFrame:
        """
        Return phase times filtered by subject or all subjects.
        """
        if self.is_single(["subject"]):
            # Load phase times and filter by subject
            df = _load_phase_times(self.base_path)
            subject = self.index["subject"][0]
            # Filter the DataFrame to include only the rows where the "VP" column matches the subject
            return df[df["VP"] == subject]
        return _load_phase_times(self.base_path)
    
    @property
    def date(self) -> str:
        """
        Return date for the current subject or all subjects, with error handling.
        """
        try:
            if self.is_single(["subject"]):
                return _load_general_information(base_path=self.base_path, column="date")[self.index["subject"][0]]

            return _load_general_information(base_path=self.base_path, column="date")
        except KeyError:
            # Handle the KeyError, e.g., return a default value or log an error
            print(f"Date was not found in empkins_dip_study.xlsx for subject {self.index['subject'][0]}")
            print("Please check its protocol file and run fill_dates() method.")
            return self.DEF_DATE  # or handle the error as needed
    
    @property
    def start_end_times(self):
        """
        Return start and end times for phases per subject or all.
        """
        if self.is_single(["subject"]):
            return _load_start_end_times(self.base_path, self.index["subject"][0])

        return {subject: _load_start_end_times(self.base_path, subject) for subject in self.index["subject"]}


    @property
    def empatica_lr(self):
        """
        Return Empatica device side info (left / right) per subject or all.
        """
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="used_empatica_lr")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="used_empatica_lr")
    
    @property
    def empatica_12(self):
        """
        Return Empatica device number (1 / 2) info per subject or all.
        """
        if self.is_single(["subject"]):
            return _load_general_information(base_path=self.base_path, column="used_empatica_12")[self.index["subject"][0]]

        return _load_general_information(base_path=self.base_path, column="used_empatica_12")

    @property
    def empatica_data_raw(self):
        """
        Load raw Empatica data for a single participant and all phases.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(None):
            return #TODO
            
        # Check if data is requested for a single participant
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            data = _load_empatica_data(self.base_path, participant_id, self.date, self.empatica_lr, self.start_end_times, self._EMPATICA)
            return data

        raise ValueError(
            "Empatica data can only be accessed for a single participant and for all phases!"
        )
    
    @property
    def empatica_data(self) -> pd.DataFrame:
        """
        Load or retrieve cached cleaned Empatica aggregated data for one subject.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            
            # Check if the file already exists
            file_exist = _check_if_file_exists(self.base_path, participant_id, self.EMPATICA_FILE_PATH)
            if file_exist is not None:
                # If the file exists, load it and update the sampling rates
                df = file_exist[0]
                fs = file_exist[1]
                self._SAMPLING_RATES.update(fs)
            else:
                # If the file does not exist, create it
                # Load the raw Empatica data
                signal_phase_data, fs = _create_agg_empatica(self.empatica_data_raw, self.phase_times)
                # Save the aggregated data to a CSV file
                df = _save_agg_empatica(self.base_path, participant_id, signal_phase_data, self.EMPATICA_FILE_PATH)
                self._SAMPLING_RATES.update(fs)
            return df

        raise ValueError(
            "Empatica clean data can only be accessed for a single participant and for all phases!"
        )

    @property
    def avro_data(self) -> pd.DataFrame:
        """
        Load or retrieve cached AVRO Empatica cleaned data for one subject.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(None):
            return #TODO
        
        # Check if data is requested for a single participant
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            
            # Check if the file already exists
            file_exist = _check_if_file_exists(self.base_path, participant_id, self.AVRO_FILE_PATH)
            if file_exist is not None:
                # If the file exists, load it and update the sampling rates
                df = file_exist[0]
                fs = file_exist[1]
                self._SAMPLING_RATES.update(fs)
            else:
                # If the file does not exist, create it
                # Load the raw Empatica data
                signal_phase_data, fs = _create_avro(self.base_path, participant_id, self._AVRO, self.phase_times)
                # Save the aggregated data to a CSV file
                df = _save_avro(self.base_path, participant_id, signal_phase_data, self.AVRO_FILE_PATH)
                self._SAMPLING_RATES.update(fs)
            return df
        
        raise ValueError(
            "AVRO Empatica data can only be accessed for a single participant and for all phases!"
        )

    @property
    def tfm_raw_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load raw TFM data dictionary for a single subject.
        """
        participant_id = self.index["subject"][0]
        data, fs = _load_tfm_data(self.base_path, participant_id, self.date)
        self._SAMPLING_RATES.update(fs)
        return data

    @property
    def tfm_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load TFM data for one participant and phase or all phases.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_tfm_data(participant_id, phase)
            self._SAMPLING_RATES.update(fs)
            return data

        # Check if data is requested for a single participant
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
        """
        Load beat-to-beat data for one participant and phase or all phases.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_b2b_data(participant_id, phase)
            self._SAMPLING_RATES.update(fs)
            return data

        # Check if data is requested for a single participant
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            data, fs = self._get_b2b_data(participant_id, "all")
            self._SAMPLING_RATES.update(fs)
            return data

        raise ValueError(
            "B2B data can only be accessed for a single participant and a single condition at once!"
        )

    @property
    def emrad_raw_data(self) -> Dict[str, np.ndarray]:
        """
        Load raw radar data for a single subject.
        """
        participant_id = self.index["subject"][0]
        data, fs = _load_radar_data(self.base_path, participant_id, self.sampling_rates["radar"])
        return data
    
    @property
    def emrad_data(self) -> pd.DataFrame:
        """
        Load radar data for one participant and phase or all phases.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            phase = self.index["phase"][0]
            data, fs = self._get_radar_data(participant_id, phase)
            return data

        # Check if data is requested for a single participant
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]
            data, fs = self._get_radar_data(participant_id, "all")
            return data

        raise ValueError(
            "Radar data can only be accessed for a single participant and a single condition at once!"
        )
    
    def tfm_data_csv(self, tfm_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load or save TFM data CSV for a single participant.
        Its a method not a property to allow saving the data.
        """
        # Check if data is requested for a single participant and a single condition
        if self.is_single(None):
            return #TODO
        
        # Check if data is requested for a single participant
        if self.is_single(["subject"]):
            participant_id = self.index["subject"][0]         

            # Check if the file already exists
            file_exist = _check_if_file_exists(self.base_path, participant_id, self.TFM_FILE_PATH)
            if file_exist is not None and tfm_df is None:
                # If the file exists, load it
                df = file_exist[0]
            else:
                # If the file does not exist, create it
                if tfm_df is not None:
                    df = _save_tfm_csv(self.base_path, participant_id, tfm_df, self.TFM_FILE_PATH)
                else:
                    print("Error: No tfm_df provided for saving to CSV")
            return df
        
        raise ValueError(
            "TFM data can only be accessed for a single participant and for all phases!"
        )

    def _get_tfm_data(self, participant_id: str, phase: str) -> tuple[pd.DataFrame, float]:
        """
        Retrieve and filter TFM data by participant and phase.
        """
        # Check if caching is enabled for data retrieval
        if self.use_cache:
            # TODO implement cache logic
            data, fs = None, None # Placeholder for future cache implementation
            pass
        else:
            # Load the TFM data from the dataset for the specified participant and date
            data, fs = _load_tfm_data(self.base_path, participant_id, self.date)

        # Initialize a dictionary to store results
        tfm_data = {}

        # Iterate through the signals in the data
        for signal in data:
            phase_dict = {}
            # Iterate through the keys in the signal data
            for key in data[signal]:
                # Check if the phase is "all" or matches the current key
                if phase == "all":
                    if key.startswith("start_"):
                        df = pd.DataFrame(data[signal][key])
                        phase_dict[key] = df

                # If the phase matches the key, create a DataFrame
                elif key == f"start_{phase}":
                    df = pd.DataFrame(data[signal][key])
                    phase_dict[key] = df

            # Store the DataFrame in the dictionary for the current signal
            tfm_data[signal] = phase_dict

        return tfm_data, fs
    
    def _get_b2b_data(self, participant_id: str, phase: str) -> tuple[pd.DataFrame, float]:
        """
        Retrieve and filter beat-to-beat data by participant and phase.
        """
        # Check if caching is enabled for data retrieval
        if self.use_cache:
            # TODO implement cache logic
            data, fs = None, None
            pass
        else:
            # Load the TFM data from the dataset for the specified participant and date
            data, fs = _load_b2b_data(self.base_path, participant_id, self.date)

        b2b_data = {}
        # Iterate through the signals in the data
        for signal in data:
            phase_dict = {}
            # Iterate through the keys in the signal data
            for key in data[signal]:
                # Check if the phase is "all" or matches the current key
                if phase == "all":
                    if key.startswith("start_"):
                        df = pd.DataFrame(data[signal][key])
                        phase_dict[key] = df

                # If the phase matches the key, create a DataFrame
                elif key == f"start_{phase}":
                    df = pd.DataFrame(data[signal][key])
                    phase_dict[key] = df

            # Store the DataFrame in the dictionary for the current signal
            b2b_data[signal] = phase_dict

        return b2b_data, fs
    
    def _get_radar_data(self, participant_id: str, phase: str) -> tuple[pd.DataFrame, float]:
        """
        Retrieve and optionally filter radar data by participant and phase.
        """
        if self.use_cache:
            # TODO implement cache
            data, fs = None, None # Placeholder for future cache implementation
            pass
        else:
            # Load radar data for the given participant_id from the base path
            data, fs = _load_radar_data(self.base_path, participant_id, self.sampling_rates["radar"])

        # Initialize a dictionary to store results
        emrad_data = None
        if phase == "all":
            emrad_data = data
        else:
            emrad_data = data

        # Return the filtered data and the sampling frequency
        return emrad_data, fs
