import warnings
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.io.io import load_long_format_csv
from biopsykit.utils.file_handling import get_subject_dirs
from pandas import DataFrame
from tpcp import Dataset

from empkins_io.datasets.d03.micro_gapvii.helper import (
    _build_data_path,
    _load_biopac_data,
    _load_nilspod_session,
    _load_radar_data,
    _load_timelog,
    get_opendbm_derived_features,
    get_opendbm_eyeblink_data,
    get_opendbm_pitch_data,
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
_cached_load_opendbm_acoustic_seg_data = lru_cache(maxsize=4)(
    load_opendbm_acoustic_seg_data
)
_cached_load_opendbm_audio_seg_data = lru_cache(maxsize=4)(load_opendbm_audio_seg_data)
_cached_load_opendbm_facial_tremor_data = lru_cache(maxsize=4)(
    load_opendbm_facial_tremor_data
)
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
        "Talk_2",
        "Pause_2",
        "Pause_3",
        "Math_1",
        "Math_2",
        "Pause_4",
        "Pause_5",
    ]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Sequence[str] = [
        # "VP_21",
        # "VP_24",
        # "VP_29",
        # "VP_41",
        "VP_45",
    ]  # Missing data (add participant IDs here)

    MISSING_RADAR_SYNC_IN_PEAKS: Sequence[str] = ["VP_02", "VP_03"]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_missing_data: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        phase_fine: Optional[bool] = False,
        opendbm_suffix: Optional[str] = None,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.data_tabular_path = self.base_path.joinpath("data_tabular")
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.phase_fine = phase_fine
        self.opendbm_suffix = opendbm_suffix
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(
                self.base_path.joinpath("data_per_subject"), "VP_[0-9]{2}"
            )
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
        # return the subset with subject identifiers > VP_45
        self.subset_index = self.index.query(
            "subject >= 'VP_01' and subject <= 'VP_45'"
        )
        return self

    @property
    def subset_micro1_1(self):
        # return the subset from the second part of the study VP >= 50
        self.subset_index = self.index.query("subject >= 'VP_50'")
        return self

    @property
    def sampling_rates(self) -> Dict[str, float]:
        return self._sampling_rates

    @property
    def subject(self) -> str:
        if self.is_single("subject"):
            return self.index["subject"][0]

    @property
    def condition(self) -> str:
        if self.is_single("condition"):
            return self.index["condition"][0]

    @property
    def phase(self) -> str:
        if self.is_single("phase"):
            return self.index["phase"][0]

    @property
    def cortisol(self) -> pd.DataFrame:
        cortisol_path = self.base_path.joinpath(
            "saliva/cortisol/cleaned/cortisol_cleaned.csv"
        )
        return load_long_format_csv(cortisol_path)

    @property
    def amylase(self) -> pd.DataFrame:
        amylase_path = self.data_tabular_path.joinpath(
            "saliva/amylase/cleaned/amylase.csv"
        )
        return load_long_format_csv(amylase_path)

    @property
    def amylase_features(self) -> pd.DataFrame:
        amylase_features_path = self.data_tabular_path.joinpath(
            "saliva/amylase/processed/amylase_features.csv"
        )
        return load_long_format_csv(
            amylase_features_path, index_cols=["subject", "condition"]
        )

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
            self.data_tabular_path.joinpath(
                "extras/processed/day_condition_mapping.csv"
            ),
            index_cols=["subject", "day"],
        )

    @cached_property
    def biopac(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_biopac_data(participant_id, condition, phase)
            return data

        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError(
                    "Biopac data can only be accessed for all phases or one specific phase!"
                )

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_biopac_data(participant_id, condition, "all")
            return data

        raise ValueError(
            "Biopac data can only be accessed for one single participant and condition at once!"
        )

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]
            return self._get_timelog(participant_id, condition, phase)

        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError(
                    "Timelog can only be accessed for all phases or one specific phase!"
                )

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant_id, condition, "all")

        # TODO allow for multiple participants and conditions in the future (return as concatenated dataframe)
        raise ValueError(
            "Timelog can only be accessed for a single participant and a single condition at once!"
        )

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
                raise ValueError(
                    "Radar data can only be accessed for all phases or one specific phase!"
                )

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_radar_data(participant_id, condition, "all")
            return data

        raise ValueError(
            "Radar data can only be accessed for a single participant and a single condition at once!"
        )

    @cached_property
    def nilspod(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            # load nilspod data for phase
            return None
        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError(
                    "NilsPod data can only be accessed for all phases or one specific phase!"
                )

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            # load nilspod data for all phases
            data = self._get_nilspod_data(participant_id, condition)
            return data

        raise ValueError(
            "NilsPod data can only be accessed for a single participant in a single condition!"
        )

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
            raise ValueError(
                "Video can only be accessed for a single participant in a single condition!"
            )
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(
            f"video/face/processed/video_face_{participant_id.lower()}_{condition}.mp4"
        )
        return path

    @property
    def body_video_path(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError(
                "Video can only be accessed for a single participant in a single condition!"
            )
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(
            f"video/body/processed/video_body_{participant_id.lower()}_{condition}.mp4"
        )
        return path

    @property
    def nilspod(self) -> pd.DataFrame:  # TODO: implement
        raise NotImplementedError("NilsPod data is not yet implemented!")

    def _get_biopac_data(
        self, participant_id: str, condition: str, phase: str
    ) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(
                self.base_path, participant_id, condition
            )
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

    def _get_radar_data(
        self, participant_id: str, condition: str, phase: str
    ) -> tuple[DataFrame, float]:
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

    def _get_timelog(
        self, participant_id: str, condition: str, phase: str
    ) -> pd.DataFrame:
        return _load_timelog(
            self.base_path, participant_id, condition, phase, self.phase_fine
        )

    def _all_phases_selected(self) -> bool:
        # check if all phases are selected
        all_phases_fine = self.phase_fine and (
            len(self.index["phase"]) == len(self.PHASE_FINE)
        )
        all_phases_coarse = not self.phase_fine and (
            len(self.index["phase"]) == len(self.PHASE_COARSE)
        )
        return all_phases_fine or all_phases_coarse

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

            data = get_opendbm_derived_features(
                self.base_path, subject_id, condition, self.opendbm_suffix
            )
            return data

        raise ValueError(
            "Data can only be accessed for a single recording of a single participant in the subset"
        )

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
