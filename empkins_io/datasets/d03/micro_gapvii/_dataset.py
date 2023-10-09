from functools import cached_property, lru_cache
import glob
from itertools import product
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from biopsykit.io.io import load_long_format_csv
from pandas import DataFrame
from tpcp import Dataset
import warnings

from resampy import resample

from empkins_io.sync import SyncedDataset
from empkins_io.datasets.d03.micro_gapvii.helper import _load_biopac_data, _load_timelog, _load_radar_data
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_radar_data = lru_cache(maxsize=4)(_load_radar_data)


class MicroBaseDataset(Dataset):
    base_path: path_t
    sync_on_load: bool
    use_cache: bool
    phase_fine: bool
    _sampling_rates: Dict[str, float] = {"biopac": 1000}

    PHASE_COARSE = ["Prep", "Pause_1", "Talk", "Pause_2", "Pause_3", "Math", "Pause_4", "Pause_5"]

    PHASE_FINE = ["Prep", "Pause_1", "Talk_1", "Talk_2", "Pause_2", "Pause_3", "Math_1", "Math_2", "Pause_4", "Pause_5"]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Sequence[str] = ["VP_02", "VP_03", "VP_21", "VP_24", "VP_29", "VP_41", "VP_45"]  # Missing data (add participant IDs here)
    
    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            exclude_missing_data: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            phase_fine: Optional[bool] = False,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.phase_fine = phase_fine
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_[0-9]{2}")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        if self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASE_FINE))
            index = pd.DataFrame(index, columns=['subject', 'condition', 'phase'])
            return index
        elif not self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASE_COARSE))
            index = pd.DataFrame(index, columns=['subject', 'condition', 'phase'])
            return index

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def cortisol(self) -> pd.DataFrame:
        cortisol_path = self.base_path.joinpath("data_tabular/saliva/cortisol/cleaned/cortisol_cleaned.csv")
        return load_long_format_csv(cortisol_path)

    @property
    def sex(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/sex.csv"), index_cols=["subject"])

    @property
    def condition_order(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/condition_order.csv"), index_cols=["subject"])

    @property
    def sit_stand(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/sit_stand.csv"), index_cols=["subject"])

    @property
    def condition_day_mapping(self) -> pd.DataFrame:
        return load_long_format_csv(self.base_path.joinpath("data_tabular/extras/processed/day_condition_mapping.csv"), index_cols=["subject", "day"])

    @cached_property
    def biopac(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition", "phase"]):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_biopac_data(participant_id, condition, phase)
            return data

        if self.is_single(["subject", "condition"]):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")
            return data

        if self.is_single(None):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")
            return data

        raise ValueError("Biopac data can only be accessed for all phases or one specific phase!")

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition", "phase"]):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index['phase'][0]
            return self._get_timelog(participant_id, condition, phase)

        if self.is_single(["subject", "condition"]):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant_id, condition, "all")

        raise ValueError("Timelog can only be accessed for a single participant and a single condition!")

    @property
    def emrad(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition", "phase"]):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_radar_data(participant_id, condition, phase)
            return data, fs

        if self.is_single(["subject", "condition"]):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_radar_data(participant_id, condition, "all")
            return data, fs

        if self.is_single(None):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")

            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_radar_data(participant_id, condition, "all")
            return data, fs

        raise ValueError("Emrad data can only be accessed for all phases or one specific phase!")
    
    @property
    def emrad_biopac_synced_and_sr_aligned(self):
        """The synchronized raw data returned as a dictionary containing the rad_i, rad_q and ecg biopac data of all phases. Radar downsampled to
        1000 Hz, now equaling the sample rate of the biopac data. Data cut to begin with Pause 1 and end with Pause 5."""

        # Check if only a single entry is left inside the index
        self.assert_is_single(["subject", "condition"], "emrad_biopac_synced_and_sr_aligned")

        #assert self.is_single(None), "Cannot sync biopac and radar for a specific phase since the sync peak appears before the experiment starts."

        # make a check if processed already for a whole participant and condition and in that case use the existing file
        # go to processed folder and check existence of a file "Biopac_aligned__resampled.pkl" and "rad[1-4]_aligned__resampled.pkl"
        current_vp = list(set(self.create_group_labels(label_cols=["subject", "condition"])))[0][0]
        current_condition = list(set(self.create_group_labels(label_cols=["subject", "condition"])))[0][1]
        existing_processed_files = glob.glob(str(self.base_path) + r"\data_per_subject" + "\\" + current_vp + "\\" + current_condition + r"\emrad\processed\*.pkl")

        if (existing_processed_files):
            res = {}
            for file in existing_processed_files:
                res[Path(file).stem] = pd.read_pickle(file)
            return res

        synced_data = SyncedDataset()

        # Add the available raw radar datasets sensor by sensor
        raw_radar_data, fs = self.emrad
        for radar_sensor in raw_radar_data.columns.get_level_values(0).unique():
            synced_data.add_dataset(radar_sensor, raw_radar_data.xs(radar_sensor, axis=1), "Sync_In", fs)

        # Add the biopac data
        raw_biopac_data = self.biopac
        synced_data.add_dataset("Biopac", raw_biopac_data, "sync", self._sampling_rates['biopac'])

        # cut them to the sync start
        synced_data.cut_to_sync_start()

        synced_data.align_datasets(primary="Biopac", cut_to_shortest=True)

        resampled_data = SyncedDataset()
        
        for aligned_dataset_name, aligned_dataset_vals in synced_data.datasets_aligned.items():
            if "rad" in aligned_dataset_name:
                resampled_data.add_dataset(aligned_dataset_name, aligned_dataset_vals, "Sync_In", fs)
            elif "Biopac" in aligned_dataset_name:
                resampled_data.add_dataset(aligned_dataset_name, aligned_dataset_vals, "sync", self._sampling_rates['biopac'])
        
        resampled_data.resample_datasets(fs_out=1000, method='static')

        # make them represent equal time spans, start at pause 1 when radar is measured first and end at the end of the last pause of the experiment before uncabling
        start_time = self.timelog["Pause_1"]["start"][0]
        end_time = self.timelog["Pause_5"]["end"][0]
        for name in resampled_data.datasets:
            setattr(resampled_data, f"{name}_resampled_", resampled_data.datasets_resampled[f"{name}_resampled_"][start_time:end_time])

        return resampled_data.datasets_resampled

    def _get_biopac_data(self, participant_id: str, condition: str, phase: str) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        else:
            # cut biopac data to specified phase
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs

    def _get_radar_data(self, participant_id: str, condition: str, phase: str) -> tuple[DataFrame, float]:
        if self.use_cache:
            data, fs = _cached_get_radar_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_radar_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        else:
            # cut radar data to specified phase
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, condition, phase, self.phase_fine)

