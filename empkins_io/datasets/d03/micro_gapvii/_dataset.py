from functools import cached_property, lru_cache
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from biopsykit.io.io import load_long_format_csv
from tpcp import Dataset
import warnings

from resampy import resample

from empkins_io.sync import SyncedDataset
from empkins_io.datasets.d03.micro_gapvii.helper import _load_biopac_data, _load_timelog, _load_emrad_data
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_emrad_data = lru_cache(maxsize=4)(_load_emrad_data)


class MicroBaseDataset(Dataset):
    base_path: path_t
    sync_on_load: bool
    use_cache: bool
    phase_fine: bool
    _sampling_rates: Dict[str, float] = {"biopac": 1000}

    PHASE_COARSE = ["Prep", "Pause_1", "Talk", "Pause_2", "Pause_3", "Math", "Pause_4", "Pause_5"]

    PHASE_FINE = ["Prep", "Pause_1", "Talk_1", "Talk_2", "Pause_2", "Pause_3", "Math_1", "Math_2", "Pause_4", "Pause_5"]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Sequence[str] = ["VP_21", "VP_24", "VP_41", "VP_45"]  # Missing data (add participant IDs here)

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

            data, fs = self._get_emrad_data(participant_id, condition, phase)
            return data, fs

        if self.is_single(["subject", "condition"]):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_emrad_data(participant_id, condition, "all")
            return data, fs

        if self.is_single(None):
            if self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_FINE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")
            elif not self.phase_fine and (len(self.index["phase"]) not in [1, len(self.PHASE_COARSE)]):
                warnings.warn("Emrad data can only be accessed for all phases or one specific phase!")

            # get biopac data for specified participant and specified phase and study protocol
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_emrad_data(participant_id, condition, "all")
            return data, fs

        raise ValueError("Emrad data can only be accessed for all phases or one specific phase!")
    
    @property
    def emrad_biopac_synced(self) -> dict:
        """The synchronized raw data returned as a dictionary containing the rad_i, rad_q and ecg biopac data of all phases."""

        # Check if only a single entry is left inside the index
        self.assert_is_single(["subject", "condition"], "emrad_biopac_synced")

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

        # Make them represent equal time spans
        """ min_time = min([v.index[-1] for _,v in synced_data.datasets_cut.items()])
        synced_and_truncated_data = {k:v.truncate(after=min_time) for k, v in synced_data.datasets_cut.items()} """
                
        # Performs a resampling of the synced radar data to 1000 Hz
        """  for key in [key for key in result_dict.keys() if key!="ecg" and result_dict[key] is not None]:        
            result_dict[key] = resampy.resample(result_dict[key], self.sampling_rate_raw_radar_hz, 1000)  """

        return synced_data.datasets_cut
    
    @property
    def emrad_biopac_synced_and_sr_aligned(self):
        """The synchronized raw data returned as a dictionary containing the rad_i, rad_q and ecg biopac data of all phases. Radar downsampled to
        1000 Hz, now equaling the sample rate of the biopac data. Data end cut until end of last phase."""

        # Check if only a single entry is left inside the index
        self.assert_is_single(["subject", "condition"], "emrad_biopac_synced_and_sr_aligned")

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

        resampled_data = SyncedDataset()
        
        for cut_dataset_name, cut_dataset_vals in synced_data.datasets_cut.items():
            if "rad" in cut_dataset_name:
                resampled_data.add_dataset(cut_dataset_name, cut_dataset_vals, "Sync_In", fs)
            elif "Biopac" in cut_dataset_name:
                resampled_data.add_dataset(cut_dataset_name, cut_dataset_vals, "sync", self._sampling_rates['biopac'])
        
        resampled_data.resample_datasets(fs_out=1000, method='static')

        # make them represent equal time spans
        end_time = self.timelog["Pause 5"]["end"][0]
        for resampled_dataset_name, resampled_dataset_vals in resampled_data.datasets_resampled:
            resampled_dataset_vals = resampled_dataset_vals[:end_time]

        return resampled_data.datasets_resampled

    """ 
        # loop over all accessible radar sensors
        for k in [key for key,_ in synced_data.items() if key != None and "rad" in key]:
            synced_data[k].drop(columns=["Sync_In", "Sync_Out"], axis= 1, inplace=True)
            #print(f"Key: {k}")
            #print(synced_data[k].head())
            #print("Old Len")
            #print(len(synced_data[k].xs("I", axis=1).to_numpy()))
            #print("New Len")
            #print(len(resampy.resample(synced_data[k].xs("I", axis=1).to_numpy(), self._sampling_rates['radar'], 1000)))
            resampled_I = resampy.resample(synced_data[k].xs("I", axis=1).to_numpy(), self._sampling_rates['radar'], 1000)
            resampled_Q = resampy.resample(synced_data[k].xs("Q", axis=1).to_numpy(), self._sampling_rates['radar'], 1000)
            new_index = pd.date_range(synced_data[k].index.values[0], periods=len(resampled_I), freq="1L")
            synced_data[k] = pd.DataFrame({'I': resampled_I, 'Q': resampled_Q})
            synced_data[k].set_index(new_index, inplace=True) 
        
        return synced_data 
        """

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

    def _get_emrad_data(self, participant_id: str, condition: str, phase: str) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_emrad_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_emrad_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        else:
            # cut biopac data to specified phase
            timelog = self.timelog
            phase_start = timelog[phase]["start"][0]
            phase_end = timelog[phase]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, condition, phase, self.phase_fine)
    
