from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from typing import ClassVar

import pandas as pd
from biopsykit.io import load_long_format_csv
from biopsykit.utils.dataframe_handling import multi_xs
from biopsykit.utils.file_handling import get_subject_dirs, mkdirs
from tpcp import Dataset

from empkins_io.datasets.d03.micro_prestudy.constants import (
    ARTIFACTS,
    CONDITIONS,
    INDEX_LEVELS,
    PHASES,
    STUDY_PARTS,
    SUBPHASES,
)
from empkins_io.datasets.d03.micro_prestudy.helper import (
    get_video_path,
    load_dominant_emotion_data,
    load_ecg_data,
    load_ecg_hr,
    load_emotion_data,
    load_hr_synced,
    load_mis_data,
)
from empkins_io.utils._types import path_t, str_t

_cached_load_emotion_data = lru_cache(maxsize=20)(load_emotion_data)
_cached_load_dominant_emotion_data = lru_cache(maxsize=20)(load_dominant_emotion_data)
_cached_load_ecg_data = lru_cache(maxsize=20)(load_ecg_data)
_cached_load_ecg_hr = lru_cache(maxsize=20)(load_ecg_hr)
_cached_load_mis_data = lru_cache(maxsize=20)(load_mis_data)
_cached_load_hr_synced = lru_cache(maxsize=20)(load_hr_synced)


class MicroPreStudyDataset(Dataset):
    SUBJECTS_WITHOUT_LABELS: tuple[str] = ()

    exclude_without_labels: bool
    _sampling_rate: ClassVar[dict[str, float]] = {"ecg": 256, "emotion": 10}
    _sample_times: ClassVar[Sequence[int]] = [-1, 0, 10, 20, 45]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        exclude_without_labels: bool | None = True,
        use_cache: bool | None = True,
    ):
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)
        # ensure pathlib
        self.base_path = base_path
        self.exclude_without_labels = exclude_without_labels
        self.use_cache = use_cache

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        if self.exclude_without_labels:
            for subject_id in self.SUBJECTS_WITHOUT_LABELS:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index_pre = list(product(["pre"], PHASES["pre"], SUBPHASES["pre"], ARTIFACTS))
        index_mist = list(product(["mist"], PHASES["mist"], SUBPHASES["mist"], ARTIFACTS))
        index_post = list(product(["post"], PHASES["post"], SUBPHASES["post"], ARTIFACTS))
        index = index_pre + index_mist + index_post
        index = [(subject, condition, *i) for subject, condition, i in product(subject_ids, CONDITIONS, index)]
        index = pd.DataFrame(index, columns=INDEX_LEVELS)
        return index

    @property
    def sampling_rate(self) -> dict[str, float]:
        return self._sampling_rate

    @property
    def video_path(self) -> path_t:
        if any([self.is_single("phase"), self.is_single("subphase"), self.is_single(None), self.is_single("artifact")]):
            raise ValueError("Video data not be accessed for individual phases, subphases, or artifacts!")
        if not self.is_single(["subject", "condition", "study_part"]):
            raise ValueError("Video data only be accessed for an individual study part!")
        return get_video_path(
            self.base_path, self.index["subject"][0], self.index["condition"][0], self.index["study_part"][0]
        )

    @property
    def questionnaire(self) -> pd.DataFrame:
        raise NotImplementedError("Questionnaire data not added yet")

    @property
    def sample_times(self) -> Sequence[int]:
        return self._sample_times

    @property
    def cortisol(self) -> pd.DataFrame:
        return self._load_saliva_data("cortisol")

    @property
    def amylase(self) -> pd.DataFrame:
        return self._load_saliva_data("amylase")

    def _load_saliva_data(self, saliva_type: str) -> pd.DataFrame:
        if any(
            [
                self.is_single("study_part"),
                self.is_single("phase"),
                self.is_single("subphase"),
                self.is_single("artifact"),
                self.is_single(None),
            ]
        ):
            raise ValueError(
                f"{saliva_type} data can not be accessed for individual study parts, phases, subphases, or artifacts!"
            )
        data_path = self.base_path.joinpath(f"saliva/processed/empkins_micro_{saliva_type}.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    @property
    def ecg(self) -> dict[str, pd.DataFrame] | pd.DataFrame:
        self._check_access("ECG data")
        artifact = self._check_artifact("ECG data")
        phases, subphases = self._get_phases_subphases()
        ecg_func = _cached_load_ecg_data if self.use_cache else load_ecg_data

        return ecg_func(
            self.base_path,
            self.index["subject"][0],
            self.index["condition"][0],
            self.index["study_part"][0],
            phases,
            subphases,
            artifact,
            self.use_cache,
        )

    @property
    def hr_ecg(self) -> dict[str, pd.DataFrame] | pd.DataFrame:
        self._check_access("ECG heart rate")
        artifact = self._check_artifact("ECG heart rate")
        phases, subphases = self._get_phases_subphases()
        hr_ecg_func = _cached_load_ecg_hr if self.use_cache else load_ecg_hr

        return hr_ecg_func(
            self.base_path,
            self.index["subject"][0],
            self.index["condition"][0],
            self.index["study_part"][0],
            phases,
            subphases,
            artifact,
            self.use_cache,
        )

    @property
    def hr_mis(self) -> dict[str, pd.DataFrame] | pd.DataFrame:
        self._check_access("MIS heart rate")
        artifact = self._check_artifact("MIS heart rate")
        if self.index["study_part"][0] == STUDY_PARTS[-1]:
            raise ValueError("MIS data was not captured for study part post.")

        phases, subphases = self._get_phases_subphases()
        mis_func = _cached_load_mis_data if self.use_cache else load_mis_data

        return mis_func(
            self.base_path,
            self.index["subject"][0],
            self.index["condition"][0],
            self.index["study_part"][0],
            phases,
            subphases,
            artifact,
            self.use_cache,
        )

    @property
    def hr_synced(self):
        self._check_access("Synced heart rate")
        artifact = self._check_artifact("Synced heart rate")
        if self.index["study_part"][0] == STUDY_PARTS[-1]:
            raise ValueError("MIS data was not captured for study part post.")
        phases, subphases = self._get_phases_subphases()
        hr_synced_func = _cached_load_hr_synced if self.use_cache else load_hr_synced
        return hr_synced_func(
            self.base_path,
            self.index["subject"][0],
            self.index["condition"][0],
            self.index["study_part"][0],
            phases,
            subphases,
            artifact,
            self.use_cache,
        )

    @property
    def emotion(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return self._emotion("emotion")

    @property
    def dominant_emotion(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return self._emotion("dominant_emotion")

    def _emotion(self, emotion_type: str):
        if any([self.is_single(None), self.is_single("artifact")]):
            raise ValueError("Emotion data not be accessed for individual artifacts!")
        if not self.is_single(["subject", "condition", "study_part"]):
            raise ValueError("Emotion data only be accessed for an individual study part!")
        if emotion_type == "emotion":
            emotion_func = _cached_load_emotion_data if self.use_cache else load_emotion_data
        elif self.use_cache:
            emotion_func = _cached_load_dominant_emotion_data
        else:
            emotion_func = load_dominant_emotion_data

        phases, subphases = self._get_phases_subphases()

        return emotion_func(
            self.base_path,
            self.index["subject"][0],
            self.index["condition"][0],
            self.index["study_part"][0],
            phases,
            subphases,
            self.use_cache,
        )

    @property
    def export_paths(self) -> dict[str, path_t]:
        if not self.is_single(["subject", "condition"]):
            print("Only supported for a single participant and condition!")
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        video_path_proc = self.base_path.joinpath(f"data_per_subject/{subject_id}/{condition}/video/processed")
        mkdirs(video_path_proc)

        return {"video": video_path_proc}

    def _get_phases_subphases(self) -> tuple[str_t, str_t]:
        study_part = self.index["study_part"][0]
        if all(phase in self.index["phase"].unique() for phase in PHASES[study_part]):
            phases = None
        else:
            phases = tuple(self.index["phase"].unique())
        if all(subphase in self.index["subphase"].unique() for subphase in SUBPHASES[study_part]):
            subphases = None
        else:
            subphases = tuple(self.index["subphase"].unique())

        return phases, subphases

    def _check_access(self, prop: str_t):
        if not self.is_single(["subject", "condition", "study_part"]):
            raise ValueError(f"{prop} can only be accessed for an individual study part!")

    def _check_artifact(self, prop: str_t) -> str_t:
        if any([self.is_single(None), self.is_single("artifact")]):
            artifact = self.index["artifact"][0]
        elif all(artifact in self.index["artifact"].unique() for artifact in ARTIFACTS):
            artifact = None
        else:
            raise ValueError(f"{prop} can only be accessed for either an individual artifact or for all artifacts")
        return artifact
