import warnings
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import ast
import json
import numpy as np

import pandas as pd
from biopsykit.io.io import load_long_format_csv
from biopsykit.utils.file_handling import get_subject_dirs
from pandas import DataFrame
from tpcp import Dataset

from empkins_io.datasets.d03._utils.check_data_completeness import (
    check_data_completeness,
)
from empkins_io.datasets.d03.micro_gapvii.helper import (
    _build_data_path,
    _load_biopac_data,
    _load_nilspod_session,
    _load_radar_data,
    _load_timelog,
    get_opendbm_derived_features,
    load_opendbm_acoustic_data,
    load_opendbm_acoustic_seg_data,
    load_opendbm_audio_seg_data,
    load_opendbm_facial_data,
    load_opendbm_facial_tremor_data,
    load_opendbm_movement_data,
    load_speaker_diarization,
)
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_radar_data = lru_cache(maxsize=4)(_load_radar_data)
_cached_load_nilspod_data = lru_cache(maxsize=4)(_load_nilspod_session)
_cached_load_opendbm_facial_data = lru_cache(maxsize=4)(load_opendbm_facial_data)
_cached_load_opendbm_acoustic_data = lru_cache(maxsize=4)(load_opendbm_acoustic_data)
_cached_load_opendbm_movement_data = lru_cache(maxsize=4)(load_opendbm_movement_data)
_cached_load_opendbm_acoustic_seg_data = lru_cache(maxsize=4)(load_opendbm_acoustic_seg_data)
_cached_load_opendbm_audio_seg_data = lru_cache(maxsize=4)(load_opendbm_audio_seg_data)
_cached_load_opendbm_facial_tremor_data = lru_cache(maxsize=4)(load_opendbm_facial_tremor_data)
_cached_load_speaker_diarization = lru_cache(maxsize=4)(load_speaker_diarization)


class MicroBaseDataset(Dataset):
    base_path: path_t
    data_tabular_path: path_t
    sync_on_load: bool
    use_cache: bool
    phase_fine: bool
    opendbm_suffix: Optional[str]
    _sampling_rates: Dict[str, float] = {"biopac": 1000, "emrad": 1953.125}
    _sample_times_saliva: Tuple[int] = (-40, -1, 16, 25, 35, 45, 60, 75)
    _sample_times_bloodspot: Tuple[int] = (-40, 60)

    NILSPOD_MAPPING: Dict[str, str] = {
        "chest": "b0c2",  # ecg
        "sternum": "157e",  # sternum
        "sync": "9e02",  # sync with mocap (MASTER)
        "board": "b012",  # sync with video (clapper board)
    }

    PHASE_COARSE = [
        "Prep",
        "Pause_1",
        "Talk",
        "Pause_2",
        "Pause_3",
        "Math",
        "Pause_4",
        "Pause_5",
    ]

    PHASE_FINE = [
        "Prep",
        "Pause_1",
        "Talk_1",
        "Pause_2",
        "Talk_2",
        "Pause_3",
        "Math_1",
        "Pause_4",
        "Math_2",
        "Pause_5",
    ]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Sequence[str] = [
        "VP_045",
    ]  # Missing data (add participant IDs here)

    MISSING_RADAR_SYNC_IN_PEAKS: Sequence[str] = ["VP_002", "VP_003"]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_missing_data: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        phase_fine: Optional[bool] = False,
        opendbm_suffix: Optional[str] = None,
        only_labeled: bool = False,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.data_tabular_path = self.base_path.joinpath("data_tabular")
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.phase_fine = phase_fine
        self.opendbm_suffix = opendbm_suffix
        self.only_labeled = only_labeled
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_[0-9]{3}")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        if self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASE_FINE))
            index = pd.DataFrame(index, columns=["subject", "condition", "phase"])
            return index
        elif not self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASE_COARSE))
            index = pd.DataFrame(index, columns=["subject", "condition", "phase"])
            return index

    @property
    def subset_micro1_0(self):
        # return the subset with subject identifiers > VP_045
        self.subset_index = self.index.query("subject >= 'VP_001' and subject <= 'VP_045'")
        return self

    @property
    def subset_micro1_1(self):
        # return the subset from the second part of the study VP >= 050
        self.subset_index = self.index.query("subject >= 'VP_050'")
        return self

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def subject(self) -> str:
        if self.is_single("subject"):
            return self.index["subject"][0]
        return None

    @property
    def condition(self) -> str:
        if self.is_single("condition"):
            return self.index["condition"][0]
        return None

    @property
    def phase(self) -> str:
        if self.is_single("phase"):
            return self.index["phase"][0]
        return None

    @property
    def id_mapping(self) -> pd.DataFrame:
        return load_long_format_csv(self.data_tabular_path.joinpath("extras/processed/id_mapping.csv"))

    @property
    def subset_demographics(self):
        # TODO
        return NotImplementedError

    @property
    def cortisol(self) -> pd.DataFrame:
        cortisol_path = self.data_tabular_path.joinpath("saliva/cortisol/cleaned/cortisol_cleaned.csv")
        return load_long_format_csv(cortisol_path)

    @property
    def amylase(self) -> pd.DataFrame:
        amylase_path = self.data_tabular_path.joinpath("saliva/amylase/cleaned/amylase.csv")
        return load_long_format_csv(amylase_path)

    @property
    def amylase_features(self) -> pd.DataFrame:
        amylase_features_path = self.data_tabular_path.joinpath("saliva/amylase/processed/amylase_features.csv")
        return load_long_format_csv(amylase_features_path, index_cols=["subject", "condition"])

    @property
    def pep_phase(self) -> pd.DataFrame:
        if self.phase_fine:
            pep_phase_path = self.data_tabular_path.joinpath("pep/pep_phase_fine.csv")
        else:
            pep_phase_path = self.data_tabular_path.joinpath("pep/pep_phase.csv")
        return load_long_format_csv(pep_phase_path, index_cols=["subject", "condition", "phase"])

    @property
    def gender(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("extras/processed/gender.csv"),
            index_cols=["subject"],
        )

    @property
    def condition_order(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("extras/processed/condition_order.csv"),
            index_cols=["subject"],
        )

    @property
    def sit_stand(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("extras/processed/sit_stand.csv"),
            index_cols=["subject"],
        )

    @property
    def condition_day_mapping(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.data_tabular_path.joinpath("extras/processed/day_condition_mapping.csv"),
            index_cols=["subject", "day"],
        )

    @cached_property
    def biopac(self) -> pd.DataFrame:
        if self.is_single(None):

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_biopac_data(participant_id, condition, phase)
            if self.only_labeled:

                data_whole, fs_all = self._get_biopac_data(participant_id, condition, "all")
                start = (data["ecg"].index[0] - data_whole["ecg"].index[0]).total_seconds()
                start_phase = int(start * fs_all)
                transformed_string = participant_id.lower()
                transformed_string = transformed_string.replace("_0", "_", 1)
                path_border = self.base_path.joinpath(
                    "data_per_subject",
                    f"{participant_id}",
                    f"{condition}",
                    "biopac",
                    "manual_labeled",
                    f"{transformed_string}_{condition}_label_borders_saved.csv",
                )
                data_labels = pd.read_csv(path_border)[["pos", "description"]]
                data_labels["description"] = data_labels["description"].apply(lambda s: ast.literal_eval(s))
                rows = data_labels[data_labels["description"].apply(lambda x: phase in x.keys())]

                rows = rows.sort_values(by=["pos"])
                rows = rows.reset_index(drop=True)

                start = rows["pos"][0] - start_phase
                end = rows["pos"][1] - start_phase
                data = data.iloc[start:end]

            return data

        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Biopac data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")

            return data

        raise ValueError("Biopac data can only be accessed for one single participant and condition at once!")

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]
            return self._get_timelog(participant_id, condition, phase)

        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Timelog can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant_id, condition, "all")

        # TODO allow for multiple participants and conditions in the future (return as concatenated dataframe)
        raise ValueError("Timelog can only be accessed for a single participant and a single condition at once!")

    @property
    def emrad(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_radar_data(participant_id, condition, phase)
            return data

        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Radar data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_radar_data(participant_id, condition, "all")
            return data

        raise ValueError("Radar data can only be accessed for a single participant and a single condition at once!")

    @cached_property
    def nilspod(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            self.index["phase"][0]

            # load nilspod data for phase
            return None
        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("NilsPod data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            # load nilspod data for all phases
            data = self._get_nilspod_data(participant_id, condition)
            return data

        raise ValueError("NilsPod data can only be accessed for a single participant in a single condition!")

    @property
    def ecg(self) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            senor_id = self.NILSPOD_MAPPING["chest"]
            data = self.nilspod.xs(senor_id, level=0, axis=1)
            return data[["ecg"]]

    @property
    def pep(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("data_tabular/pep/pep_conditions.csv"),
            index_cols=["subject", "condition"],
        )

    @property
    def face_video_path(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/face/processed/video_face_{participant_id.lower()}_{condition}.mp4")
        return path

    @property
    def body_video_path(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/video_body_{participant_id.lower()}_{condition}.mp4")
        return path

    @property
    def nilspod(self) -> pd.DataFrame:  # TODO: implement
        raise NotImplementedError("NilsPod data is not yet implemented!")

    @property
    def expected_files_list(self) -> pd.DataFrame:
        file_path = self.base_path / "expected_files_per_subject.csv"
        return pd.read_csv(file_path, index_col=2)

    def check_data_completeness(self, save_output: bool = False, output_path: Optional[path_t] = None):
        """Check if all expected files are present for each subject."""
        file_overview = check_data_completeness(
            data_per_subject_folder=self.base_path.joinpath("data_per_subject"),
            expected_files_list=self.expected_files_list,
        )
        if save_output:
            if output_path is None:
                output_path = self.base_path / "data_completeness_check_result.csv"
            file_overview.to_csv(output_path)
        return file_overview

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

    def _get_nilspod_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            data, fs = _cached_load_nilspod_data(self.base_path, subject_id, condition)
        else:
            data, fs = _load_nilspod_session(self.base_path, subject_id, condition)
        return data

    def _get_radar_data(self, participant_id: str, condition: str, phase: str) -> tuple[DataFrame, float]:
        if self.use_cache:
            data, fs = _cached_get_radar_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_radar_data(self.base_path, participant_id, condition)

        if phase == "all":
            timelog = self.timelog
            phase_start = timelog["Pause_1"]["start"][0]
            phase_end = timelog["Pause_5"]["end"][0]
            data = data.loc[phase_start:phase_end]
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

    def _all_phases_selected(self) -> bool:
        # check if all phases are selected
        all_phases_fine = self.phase_fine and (len(self.index["phase"]) == len(self.PHASE_FINE))
        all_phases_coarse = not self.phase_fine and (len(self.index["phase"]) == len(self.PHASE_COARSE))
        return all_phases_fine or all_phases_coarse

    def calculate_pep_manual_labeled(self, ecg_clean, ecg_whole, heartbeats):
        # calculate pep out of the manual labels

        fs = self.sampling_rates["biopac"]

        ecg_start = ecg_whole
        start = (ecg_clean.index[0] - ecg_start.index[0]).total_seconds()
        start = int(start * fs)
        phase = self.index["phase"][0]

        # load the manual labeled data
        data_ICG, data_ECG = self.load_manual_labeled

        if data_ICG is None or data_ECG is None:
            return None, None, None
        row = data_ECG[(data_ECG["Channel"] == phase) & (data_ECG["Label"] == "start")]

        row_end = data_ECG[(data_ECG["Channel"] == phase) & (data_ECG["Label"] == "end")]

        # selct only part of the labels within the specific phase
        data_ECG = data_ECG.iloc[row.index[0] : row_end.index[0] + 2]
        data_ICG = data_ICG.iloc[row.index[0] : row_end.index[0] + 2]
        start_value = row["Samples"].values[0]
        end_value = row_end["Samples"].values[0]

        data_ICG = data_ICG[(data_ICG["Channel"] == "ICG") | (data_ICG["Channel"] == "Artefact")]

        data_ECG = data_ECG[(data_ECG["Channel"] == "ECG") | (data_ECG["Channel"] == "Artefact")]

        heartbeats["start_sample"] = heartbeats["start_sample"] + start
        heartbeats["end_sample"] = heartbeats["end_sample"] + start

        # exclude labeled points that are part of uncomplete heartbeats
        heartbeats = heartbeats[(heartbeats["start_sample"] >= start_value) & (heartbeats["end_sample"] <= end_value)]

        if data_ECG["Samples"].values[0] < heartbeats["start_sample"].values[0]:
            data_ECG = data_ECG[1:]

        if data_ECG["Samples"].values[-1] > heartbeats["end_sample"].values[-1]:
            data_ECG = data_ECG[:-1]

        if data_ICG["Samples"].values[0] < heartbeats["start_sample"].values[0]:
            data_ICG = data_ICG[1:]

        if data_ICG["Samples"].values[-1] > heartbeats["end_sample"].values[-1]:
            data_ICG = data_ICG[:-1]

        # insert nan for all artefacts
        data_ICG.loc[data_ICG["Channel"] == "Artefact", "Samples"] = np.nan
        data_ECG.loc[data_ECG["Channel"] == "Artefact", "Samples"] = np.nan

        b_points = data_ICG["Samples"].values

        q_onset = data_ECG["Samples"].values

        if b_points[0] < q_onset[0]:
            b_points = b_points[1:]
        if b_points[-1] < q_onset[-1]:
            q_onset = q_onset[:-1]

        # insert nan values for heartbeats in which no points were labeled
        count = 0
        average_time = np.mean(heartbeats["rr_interval_samples"])
        outliers = heartbeats[heartbeats["rr_interval_samples"] < (0.7 * average_time)]

        for h in heartbeats.index:
            # ignore if the heartbeat may be not a real heartbeat
            if h in outliers.index:
                count += 1
                continue
            start = heartbeats.loc[h]["start_sample"]
            end = heartbeats.loc[h]["end_sample"]

            if not any(start <= x <= end for x in b_points):
                if not (pd.isna(b_points[count])):
                    b_points = np.insert(b_points, count, np.nan)
            if not any(start <= x <= end for x in q_onset):
                if not (pd.isna(q_onset[count])):
                    q_onset = np.insert(q_onset, count, np.nan)
            count += 1
        # calculate pep from start and end points
        pep_df = pd.DataFrame((b_points - q_onset) / fs * 1000, columns=["pep"])
        pep_df.index = range(len(pep_df))
        pep_df.index.name = "heartbeat_id"

        return b_points, q_onset, pep_df, start_value

    @property
    def load_manual_labeled(self):
        # loadmanually labeled points

        participant = self.index["subject"][0]
        condition = self.index["condition"][0]
        phase = self.index["phase"][0]
        if phase is None:
            raise ValueError("Phase must be specified to load manual labeled data.")
        deploy_type = "local"
        base_path = Path("..")
        data_path = Path(json.load(base_path.joinpath("config.json").open(encoding="utf-8"))[deploy_type]["micro_path"])
        data_path_ICG = data_path.joinpath(
            "data_per_subject",
            f"{participant}",
            f"{condition}",
            "biopac",
            "manual_labeled",
            "icg",
            f"{participant}_{condition}_ICG.csv",
        )
        data_path_ECG = data_path.joinpath(
            "data_per_subject",
            f"{participant}",
            f"{condition}",
            "biopac",
            "manual_labeled",
            "ecg",
            f"{participant}_{condition}_ECG.csv",
        )

        if not data_path_ICG.exists() or not data_path_ECG.exists():
            return None, None
        # read data for ECG and ICG labels
        data_ICG = pd.read_csv(data_path_ICG)

        data_ECG = pd.read_csv(data_path_ECG)

        return data_ICG, data_ECG

    def correct_start_points(
        self, ecg_clean, ecg_start, heartbeats, b_points=[], q_points=[], c_points=[], pep_results=[]
    ):
        # used to correct the start points of the calculated points to match the manually labeled points

        # get the start sample of the phase (needed since all phases are combined after one another and sample count is based on the whole data and not just the phase)
        phase = self.index["phase"][0]
        subject = self.index["subject"][0]
        condition = self.index["condition"][0]
        transformed_string = subject.lower()

        fs = self.sampling_rates["biopac"]
        start = (ecg_clean.index[0] - ecg_start.index[0]).total_seconds()
        start_phase = int(start * fs)
        rows = self.load_annotations()

        # start and end of the random selected part of the phase
        start = rows["pos"][0]
        end = rows["pos"][1]

        heartbeats["start_sample"] = heartbeats["start_sample"] + start_phase
        heartbeats["end_sample"] = heartbeats["end_sample"] + start_phase

        heartbeats = heartbeats.loc[(heartbeats["start_sample"] >= start)]
        heartbeats = heartbeats.loc[(heartbeats["end_sample"] <= end)]
        ids = heartbeats.index

        # correct the sample count of the calculated points to match the manually labeled points
        if len(b_points) != 0:
            b_points = b_points.iloc[ids]
            b_points = b_points + start_phase

        if len(q_points) != 0:
            q_points = q_points.iloc[ids]
            q_points = q_points + start_phase

        if len(c_points) != 0:
            c_points = c_points.iloc[ids]
            c_points = c_points + start_phase
        if len(pep_results) != 0:
            pep_results = pep_results.iloc[ids]

        return b_points, q_points, heartbeats, c_points, pep_results

    def load_annotations(self):
        # load annotations to get the start and end sample of the random part of the phase
        phase = self.index["phase"][0]
        subject = self.index["subject"][0]
        condition = self.index["condition"][0]

        data_path = Path(r"C:\Users\Asus\Uni\Bachelorarbeit\Data\EmpkinS_Micro_Data")
        transformed_string = subject.lower()
        transformed_string = transformed_string.replace("_0", "_", 1)
        path_border = data_path.joinpath(
            "data_per_subject",
            f"{subject}",
            f"{condition}",
            "biopac",
            "manual_labeled",
            f"{transformed_string}_{condition}_label_borders_saved.csv",
        )

        data = pd.read_csv(path_border)[["pos", "description"]]
        data["description"] = data["description"].apply(lambda s: ast.literal_eval(s))
        # select the rows that contain the phase
        rows = data[data["description"].apply(lambda x: phase in x.keys())]

        rows = rows.sort_values(by=["pos"])
        rows = rows.reset_index(drop=True)

        return rows

    # def _get_opendbm_facial_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_facial_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #         )
    #     return load_opendbm_facial_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_acoustic_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_acoustic_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_audio, self.opendbm_suffix
    #         )
    #     return load_opendbm_acoustic_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_audio, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_movement_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_movement_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #         )
    #     return load_opendbm_movement_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_acoustic_seg_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_acoustic_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #     return load_opendbm_acoustic_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_opendbm_audio_seg_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_audio_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #     return load_opendbm_audio_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_opendbm_facial_tremor_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_facial_tremor_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #         )
    #     return load_opendbm_facial_tremor_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_pitch_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     return get_opendbm_pitch_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_opendbm_eyeblink_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     return get_opendbm_eyeblink_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_speaker_diarization(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_speaker_diarization(self.base_path, subject_id, condition)
    #     return load_speaker_diarization(self.base_path, subject_id, condition)

    # @cached_property
    # def opendbm_facial_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_facial_data(subject_id, condition)
    #
    #         # TODO add support for phases
    #         if self.is_single(None):
    #            phase = self.index["phase"].unique()[0]
    #         else:
    #            phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #         data = data.loc[times[0] : times[1]]
    #
    #         if self.normalize_video_time:
    #             data.index -= data.index[0]
    #
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def opendbm_derived_features(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data = get_opendbm_derived_features(self.base_path, subject_id, condition, self.opendbm_suffix)
            return data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    # @cached_property
    # def opendbm_acoustic_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_acoustic_data(subject_id, condition)
    #         data = apply_diarization_aco(data, self.speaker_diarization, self.sampling_rate_audio)
    #
    #         if self.is_single(None):
    #             phase = self.index["phase"].unique()[0]
    #         else:
    #             phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #         data = data.loc[times[0] : times[1]]
    #
    #         if self.normalize_video_time:
    #             data.index -= data.index[0]
    #
    #         data = data.loc[data["diarization"], :]
    #         data = data.drop(["diarization"], axis=1)
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_movement_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_movement_data(subject_id, condition)
    #
    #         if self.is_single(None):
    #             phase = self.index["phase"].unique()[0]
    #         else:
    #             phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #         data = data.loc[times[0] : times[1]]
    #
    #         if self.normalize_video_time:
    #             data.index -= data.index[0]
    #
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_acoustic_seg_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         try:
    #             data = self._get_opendbm_acoustic_seg_data(subject_id, condition)
    #             data = apply_diarization_aco_seg(data, self.speaker_diarization, self.sampling_rate_audio)
    #
    #             if self.is_single(None):
    #                 phase = self.index["phase"].unique()[0]
    #             else:
    #                 phase = "total"
    #
    #             times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #             data = data.loc[(data["start_time"] >= times[0]) & (data["end_time"] <= times[1])]
    #
    #             if self.normalize_video_time:
    #                 data.index -= data.index[0]
    #         except Exception as e:
    #             print(f"loading acoustic_seg opendbm data failed for phase {phase}: {e}")
    #
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_audio_seg_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         try:
    #             data = self._get_opendbm_audio_seg_data(subject_id, condition)
    #
    #             if self.is_single(None):
    #                 phase = self.index["phase"].unique()[0]
    #             else:
    #                 phase = "total"
    #
    #             times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #             data = data.loc[(data["start"] > times[0]) & (data["stop"] < times[1])]
    #             if self.normalize_video_time:
    #                 data.index -= data.index[0]
    #         except Exception as e:
    #             print(f"loading audio_seg opendbm data failed for phase {phase}: {e}")
    #
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_facial_tremor_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_facial_tremor_data(subject_id, condition)
    #
    #         if self.is_single(None):
    #             phase = self.index["phase"].unique()[0]
    #         else:
    #             phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #
    #         try:
    #             data = data.loc[times[0] : times[1]]
    #             if self.normalize_video_time:
    #                 data.index -= data.index[0]
    #         except Exception as e:
    #             print(f"warning: cutting facial tremor data failed (data shape: {data.shape}):", e)
    #
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @property
    # def opendbm_pitch_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         return self._get_opendbm_pitch_data(subject_id, condition)
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @property
    # def opendbm_eyeblink_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         return self._get_opendbm_eyeblink_data(subject_id, condition)
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @property
    # def opendbm_eyeblink_ear_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_eyeblink_data(subject_id, condition)
    #         if len(data.index) != 0:
    #             fps = data.at[0, "fps"]
    #             data.index = data["mov_blinkframes"] / fps
    #             data.index.name = "time [s]"
    #             if self.is_single(None):
    #                 phase = self.index["phase"].unique()[0]
    #             else:
    #                 phase = "total"
    #             times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #             data = data.loc[times[0] : times[1]]
    #
    #         data = data.drop(["vid_dur", "fps", "mov_blinkframes", "mov_blinkdur", "dbm_master_url"], axis=1)
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def speaker_diarization(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_speaker_diarization(subject_id, condition)
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
