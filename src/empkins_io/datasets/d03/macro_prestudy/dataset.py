import json
from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path

import pandas as pd
from biopsykit.io import load_long_format_csv, load_questionnaire_data
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

from empkins_io.datasets.d03._utils.dataset_utils import get_cleaned_openpose_data
from empkins_io.datasets.d03.macro_prestudy.helper import (
    apply_diarization_aco,
    apply_diarization_aco_seg,
    compress_opendbm_data,
    extract_opendbm_data,
    get_audio_path,
    get_opendbm_derived_features,
    get_opendbm_eyeblink_data,
    get_opendbm_pitch_data,
    get_times_for_mocap,
    get_times_for_video,
    get_video_path,
    load_mocap_data,
    load_opendbm_acoustic_data,
    load_opendbm_acoustic_seg_data,
    load_opendbm_audio_seg_data,
    load_opendbm_facial_data,
    load_opendbm_facial_tremor_data,
    load_opendbm_movement_data,
    load_speaker_diarization,
    write_file_to_opendbm_tar,
)
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import SyncDataNotFoundError

_cached_load_mocap_data = lru_cache(maxsize=4)(load_mocap_data)
_cached_load_opendbm_facial_data = lru_cache(maxsize=4)(load_opendbm_facial_data)
_cached_load_opendbm_acoustic_data = lru_cache(maxsize=4)(load_opendbm_acoustic_data)
_cached_load_opendbm_movement_data = lru_cache(maxsize=4)(load_opendbm_movement_data)
_cached_load_opendbm_acoustic_seg_data = lru_cache(maxsize=4)(load_opendbm_acoustic_seg_data)
_cached_load_opendbm_audio_seg_data = lru_cache(maxsize=4)(load_opendbm_audio_seg_data)
_cached_load_opendbm_facial_tremor_data = lru_cache(maxsize=4)(load_opendbm_facial_tremor_data)
_cached_load_speaker_diarization = lru_cache(maxsize=4)(load_speaker_diarization)


class MacroPreStudyDataset(Dataset):
    SUBJECTS_WITHOUT_MOCAP: tuple[str] = ("VP_01",)
    SUBJECTS_DIARIZATION_FAILED: tuple[str] = ("VP_19",)
    SUBJECTS_WITHOUT_OPENPOSE: tuple[str] = ("VP_01",)

    base_path: path_t
    exclude_without_mocap: bool
    exclude_diarization_failed: bool
    normalize_mocap_time: bool
    normalize_video_time: bool
    normalize_openpose_time: bool
    use_cache: bool
    opendbm_suffix: str
    _sampling_rate: float = 1.0 / 0.017
    _sampling_rate_video: float = 25  # frames per second
    _sampling_rate_audio: float = 1000  # down sampled fs of extracted features in opendbm
    _sample_times: tuple[int] = (-20, -1, 0, 10, 20, 45)

    def __init__(  # noqa: PLR0913
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        exclude_without_mocap: bool = True,
        exclude_diarization_failed: bool = False,
        exclude_without_openpose: bool = True,
        normalize_mocap_time: bool = True,
        normalize_video_time: bool = True,
        normalize_openpose_time: bool = True,
        use_cache: bool = True,
        opendbm_suffix: str | None = None,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_without_mocap = exclude_without_mocap
        self.exclude_without_openpose = exclude_without_openpose
        self.exclude_diarization_failed = exclude_diarization_failed
        self.use_cache = use_cache
        self.normalize_mocap_time = normalize_mocap_time
        self.normalize_openpose_time = normalize_openpose_time
        self.normalize_video_time = normalize_video_time
        self.opendbm_suffix = opendbm_suffix
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        conditions = ["ftsst", "tsst"]
        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)
        if self.exclude_without_openpose:
            for subject_id in self.SUBJECTS_WITHOUT_OPENPOSE:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)
        if self.exclude_diarization_failed:
            for subject_id in self.SUBJECTS_DIARIZATION_FAILED:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index = list(product(conditions, ["talk"]))
        index.append(("tsst", "math"))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]
        index = pd.DataFrame(index, columns=["subject", "condition", "phase"])
        return index

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def sampling_rate_openpose(self):
        """Sampling rate of videos and openpose data."""
        return self._sampling_rate_video

    @property
    def sampling_rate_video(self) -> float:
        return self._sampling_rate_video

    @property  # down sampled fs of extracted features in opendbm
    def sampling_rate_audio(self) -> float:
        return self._sampling_rate_audio

    @property
    def sample_times(self) -> Sequence[int]:
        return self._sample_times

    @property
    def questionnaire(self):
        if self.is_single(["phase"]):
            raise ValueError("questionnaire data can not be accessed for individual phases!")
        if self.is_single(["condition"]):
            raise ValueError("questionnaire data can not be accessed for a single condition!")
        return self._load_questionnaire_data()

    @property
    def condition_first(self) -> pd.DataFrame:
        data = self.questionnaire[["TSST_first"]].replace({True: "TSST first", False: "fTSST first"})
        data.columns = ["condition_first"]
        return data

    @property
    def cortisol_non_responder(self) -> pd.DataFrame:
        non_responder = self.cortisol_features.xs("tsst", level="condition")
        non_responder = non_responder.xs("max_inc", level="saliva_feature") <= 1.5
        non_responder.columns = ["non_responder"]
        return non_responder

    @property
    def cortisol(self) -> pd.DataFrame:
        return self._load_saliva_data("cortisol")

    @property
    def cortisol_features(self) -> pd.DataFrame:
        return self._load_saliva_features("cortisol")

    @property
    def amylase(self) -> pd.DataFrame:
        return self._load_saliva_data("amylase")

    @property
    def amylase_features(self) -> pd.DataFrame:
        return self._load_saliva_features("amylase")

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        phase = self._determine_phase()
        data = self._get_mocap_data(subject_id, condition)
        times = get_times_for_mocap(self.base_path, self.sampling_rate, subject_id, condition, phase)
        data = data.loc[times[0] : times[1]]
        if self.normalize_mocap_time:
            data.index -= data.index[0]
        return data

    @cached_property
    def opendbm_facial_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_opendbm_facial_data(subject_id, condition)

            phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"

            times = get_times_for_video(self.base_path, subject_id, condition, phase)
            data = data.loc[times[0] : times[1]]

            if self.normalize_video_time:
                data.index -= data.index[0]

            return data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def opendbm_derived_features(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            feature_list = []
            for phase in self.index["phase"].unique():
                data = get_opendbm_derived_features(self.base_path, subject_id, condition, phase, self.opendbm_suffix)
                feature_list.append(data)
            feature_data = pd.concat(feature_list, axis=0)

            return feature_data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @cached_property
    def opendbm_acoustic_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_opendbm_acoustic_data(subject_id, condition)
            data = apply_diarization_aco(data, self.speaker_diarization, self.sampling_rate_audio)

            phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"

            times = get_times_for_video(self.base_path, subject_id, condition, phase)
            data = data.loc[times[0] : times[1]]

            if self.normalize_video_time:
                data.index -= data.index[0]

            data = data.loc[data["diarization"], :]
            data = data.drop(["diarization"], axis=1)
            return data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @cached_property
    def opendbm_movement_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_opendbm_movement_data(subject_id, condition)

            phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"

            times = get_times_for_video(self.base_path, subject_id, condition, phase)
            data = data.loc[times[0] : times[1]]

            if self.normalize_video_time:
                data.index -= data.index[0]

            return data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @cached_property
    def opendbm_acoustic_seg_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            try:
                data = self._get_opendbm_acoustic_seg_data(subject_id, condition)
                data = apply_diarization_aco_seg(data, self.speaker_diarization, self.sampling_rate_audio)

                phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"

                times = get_times_for_video(self.base_path, subject_id, condition, phase)
                data = data.loc[(data["start_time"] >= times[0]) & (data["end_time"] <= times[1])]

                if self.normalize_video_time:
                    data.index -= data.index[0]
            except ValueError as e:
                print(f"loading acoustic_seg opendbm data failed for phase {phase}: {e}")

            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @cached_property
    def opendbm_audio_seg_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            try:
                data = self._get_opendbm_audio_seg_data(subject_id, condition)

                phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"

                times = get_times_for_video(self.base_path, subject_id, condition, phase)
                data = data.loc[(data["start"] > times[0]) & (data["stop"] < times[1])]
                if self.normalize_video_time:
                    data.index -= data.index[0]
            except ValueError as e:
                print(f"loading audio_seg opendbm data failed for phase {phase}: {e}")
            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @cached_property
    def opendbm_facial_tremor_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_opendbm_facial_tremor_data(subject_id, condition)

            phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"

            times = get_times_for_video(self.base_path, subject_id, condition, phase)

            try:
                data = data.loc[times[0] : times[1]]
                if self.normalize_video_time:
                    data.index -= data.index[0]
            except ValueError as e:
                print(f"warning: cutting facial tremor data failed (data shape: {data.shape}):", e)

            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def opendbm_pitch_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_opendbm_pitch_data(subject_id, condition)

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def opendbm_eyeblink_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_opendbm_eyeblink_data(subject_id, condition)
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def opendbm_eyeblink_ear_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_opendbm_eyeblink_data(subject_id, condition)
            if len(data.index) != 0:
                fps = data.loc[0, "fps"]
                data.index = data["mov_blinkframes"] / fps
                data.index.name = "time [s]"
                phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"
                times = get_times_for_video(self.base_path, subject_id, condition, phase)
                data = data.loc[times[0] : times[1]]

            data = data.drop(["vid_dur", "fps", "mov_blinkframes", "mov_blinkdur", "dbm_master_url"], axis=1)
            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @cached_property
    def speaker_diarization(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_speaker_diarization(subject_id, condition)
            return data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    def _determine_phase(self) -> str:
        """Checks if single phase or both phases from a single condition are selected and sets phase accordingly."""
        if self.is_single(None):
            phase = self.index["phase"][0]
        elif self.is_single(["subject", "condition"]):
            phase = "total"
        else:
            raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
        return phase

    @property
    def panas_diff(self) -> pd.DataFrame:
        panas_data = wide_to_long(self.questionnaire, "PANAS", levels=["subscale", "condition", "time"]).dropna()
        panas_data = panas_data.drop("Total", level="subscale")
        panas_data = panas_data.reindex(["ftsst", "tsst"], level="condition").reindex(["pre", "post"], level="time")
        panas_data = panas_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return self._apply_indices(panas_data).reorder_levels(
            ["subject", "condition", "condition_first", "non_responder", "subscale"]
        )

    @property
    def stadi_state_diff(self) -> pd.DataFrame:
        stadi_data = wide_to_long(self.questionnaire, "STADI_State", levels=["subscale", "condition", "time"]).dropna()
        stadi_data = stadi_data.reindex(["pre", "post"], level="time")
        stadi_data = stadi_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return self._apply_indices(stadi_data).reorder_levels(
            ["subject", "condition", "condition_first", "non_responder", "subscale"]
        )

    @property
    def pasa(self) -> pd.DataFrame:
        pasa_data = wide_to_long(self.questionnaire, "PASA", levels=["subscale"]).dropna()
        return self._apply_indices(pasa_data).reorder_levels(
            ["subject", "condition_first", "non_responder", "subscale"]
        )

    @property
    def openpose_data(self) -> pd.DataFrame:
        """Returns the dataframe of the cleaned openpose data per phase (or for "total" phase)."""
        phase = self._determine_phase()
        file_path = self._openpose_data_path
        timings = self.sync_data["video"][phase]
        data = get_cleaned_openpose_data(file_path, timings["begin_sec"], timings["end_sec"])
        if self.normalize_openpose_time:
            data.index -= data.index[0]
        return data

    @property
    def sync_data(self):
        self.assert_is_single(["subject", "condition"], "sync_data")
        subject_id = self.group.subject
        condition = self.group.condition
        file_path = self.base_path.joinpath(
            f"data_per_subject/{subject_id}/{condition}/{subject_id}_times_{condition}.json"
        )
        if not file_path.exists():
            raise SyncDataNotFoundError(f"Sync data not found for subject {subject_id} and condition {condition}.")
        with file_path.open() as f:
            sync_data = json.load(f)
        return sync_data

    @property
    def _video_folder_path(self):
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Path can only be accessed for a single condition of a single participant!")
        data_path = self.base_path.joinpath("data_per_subject").joinpath(
            f"{self.group.subject}/{self.group.condition}/video"
        )
        return data_path

    @property
    def _openpose_data_path(self) -> Path:
        file_path = self._openpose_folder_path / "openpose.csv"
        return file_path

    @property
    def _openpose_folder_path(self) -> Path:
        self.assert_is_single(["subject", "condition"], "_openpose_folder_path")
        file_path = self._video_folder_path / "cleaned"
        return file_path

    @property
    def video_path(self) -> Path:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_video_path(subject_id, condition)

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def audio_path(self) -> Path:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_audio_path(subject_id, condition)

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    def extract_opendbm_data(self, suffix: str | None = None):
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        return extract_opendbm_data(self.base_path, subject_id, condition, suffix)

    def compress_opendbm_data(self, suffix: str | None = None):
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        return compress_opendbm_data(self.base_path, subject_id, condition, suffix)

    def write_file_to_opendbm(
        self,
        data: pd.DataFrame,
        data_type: str | None = None,
        raw: bool | None = False,
        derived: bool | None = False,
        group: str | None = None,
        subgroup: str | None = None,
    ):
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        phase = self.index["phase"].unique()[0] if self.is_single(None) else "total"
        write_file_to_opendbm_tar(
            base_path=self.base_path,
            subject_id=subject_id,
            condition=condition,
            data=data,
            data_type=data_type,
            raw=raw,
            derived=derived,
            phase=phase,
            group=group,
            subgroup=subgroup,
        )

    def _apply_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.join(self.condition_first).join(self.cortisol_non_responder)
        data = data.set_index(["condition_first", "non_responder"], append=True)
        return data

    def _get_mocap_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition)
        return load_mocap_data(self.base_path, subject_id, condition)

    def _get_opendbm_facial_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_opendbm_facial_data(
                self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
            )
        return load_opendbm_facial_data(
            self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
        )

    def _get_opendbm_acoustic_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_opendbm_acoustic_data(
                self.base_path, subject_id, condition, self.sampling_rate_audio, self.opendbm_suffix
            )
        return load_opendbm_acoustic_data(
            self.base_path, subject_id, condition, self.sampling_rate_audio, self.opendbm_suffix
        )

    def _get_opendbm_movement_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_opendbm_movement_data(
                self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
            )
        return load_opendbm_movement_data(
            self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
        )

    def _get_opendbm_acoustic_seg_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_opendbm_acoustic_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
        return load_opendbm_acoustic_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)

    def _get_opendbm_audio_seg_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_opendbm_audio_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
        return load_opendbm_audio_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)

    def _get_opendbm_facial_tremor_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_opendbm_facial_tremor_data(
                self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
            )
        return load_opendbm_facial_tremor_data(
            self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
        )

    def _get_opendbm_pitch_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        return get_opendbm_pitch_data(self.base_path, subject_id, condition, self.opendbm_suffix)

    def _get_opendbm_eyeblink_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        return get_opendbm_eyeblink_data(self.base_path, subject_id, condition, self.opendbm_suffix)

    def _get_speaker_diarization(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_speaker_diarization(self.base_path, subject_id, condition)
        return load_speaker_diarization(self.base_path, subject_id, condition)

    def _get_video_path(self, subject_id: str, condition: str) -> Path:
        return get_video_path(self.base_path, subject_id, condition)

    def _get_audio_path(self, subject_id: str, condition: str) -> Path:
        return get_audio_path(self.base_path, subject_id, condition)

    def _load_questionnaire_data(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("questionnaire_total/processed/empkins_macro_questionnaire_data.csv")
        data = load_questionnaire_data(data_path)
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    def _load_saliva_data(self, saliva_type: str) -> pd.DataFrame:
        if self.is_single("phase"):
            raise ValueError(f"{saliva_type} data can not be accessed for individual phases!")
        data_path = self.base_path.joinpath(f"saliva_total/processed/empkins_macro_{saliva_type}.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    def _load_saliva_features(self, saliva_type: str) -> pd.DataFrame:
        if self.is_single("phase"):
            raise ValueError(f"{saliva_type} features can not be accessed for individual phases!")
        data_path = self.base_path.joinpath(f"saliva_total/processed/empkins_macro_{saliva_type}_features.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    def add_cortisol_index(self, cort_data: pd.DataFrame) -> pd.DataFrame:
        index_levels = list(cort_data.index.names)
        new_index_levels = ["condition_first", "non_responder"]
        cort_data = cort_data.join(self.condition_first).join(self.cortisol_non_responder)
        cort_data = cort_data.set_index(new_index_levels, append=True)
        cort_data = cort_data.reorder_levels(index_levels[:-1] + new_index_levels + [index_levels[-1]])

        return cort_data
