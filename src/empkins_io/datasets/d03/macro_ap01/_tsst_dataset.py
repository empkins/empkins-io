import contextlib
import json
from collections.abc import Sequence
from functools import cached_property, lru_cache
from pathlib import Path

import pandas as pd

from empkins_io.datasets.d03._utils.dataset_utils import get_cleaned_openpose_data
from empkins_io.datasets.d03.macro_ap01._base_dataset import MacroBaseDataset
from empkins_io.datasets.d03.macro_ap01.helper import _get_times_for_mocap, _load_tsst_mocap_data
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import SyncDataNotFoundError, TimestampDataNotFoundError

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_tsst_mocap_data)


class MacroStudyTsstDataset(MacroBaseDataset):
    """Class to conveniently access the data of the macro study dataset for subject and condition.
    If access is required per-phase, use :class:`MacroStudyTsstDatasetPerPhase` instead.
    """

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        exclude_complete_subjects_if_error: bool = True,
        exclude_without_mocap: bool = True,
        exclude_without_openpose: bool = False,
        exclude_with_arm_errors: bool = False,
        exclude_without_prep: bool = False,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose

        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            exclude_complete_subjects_if_error=exclude_complete_subjects_if_error,
            exclude_without_mocap=exclude_without_mocap,
            exclude_without_openpose=exclude_without_openpose,
            exclude_with_arm_errors=exclude_with_arm_errors,
            exclude_without_prep=exclude_without_prep,
        )

    @property
    def sampling_rate_openpose(self):
        """Sampling rate of cleaned openpose data."""
        return 30

    @property
    def ecg(self) -> pd.DataFrame:
        return self._load_ecg_data()

    @property
    def ecg_baseline(self) -> pd.DataFrame:
        return self._load_ecg_data(True)

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        data, start = self._get_mocap_data(subject_id, condition, verbose=self.verbose)

        times = _get_times_for_mocap(self.timelog_test, start, phase="total")
        times = times.loc["total"]
        data_total = data.loc[times["start"] : times["end"]]

        return data_total

    def _get_mocap_data(self, subject_id: str, condition: str, *, verbose: bool = True) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition, verbose=verbose)
        return _load_tsst_mocap_data(self.base_path, subject_id, condition, verbose=verbose)

    @property
    def sync_path(self) -> Path:
        if not (self.is_single(None) or self.is_single(["subject", "condition"])):
            raise ValueError("Path can only be accessed for a single condition of a single participant!")
        data_path = self.base_path.joinpath("data_per_subject").joinpath(
            f"{self.group.subject}/{self.group.condition}/sync.json"
        )
        return data_path

    @property
    def sync_data(self) -> dict:
        """Dict that contains timestamps of important events for synchronisation, e.g. claps in the video.

        ! Watch out: The video timestamps are NOT realtime timestamps! They do correspond to the `cfr_video_timestamps`
        column (and the `cfr_time_seconds` column) in the timestamps.csv file/video_timestamps property.
        """
        if not (self.is_single(None) or self.is_single(["subject", "condition"])):
            raise ValueError("Path can only be accessed for a single condition of a single participant!")

        class JSONDecoder(json.JSONDecoder):
            """Class to properly convert ISO timestamps in json file to pd.Timestamps."""

            def __init__(self, *args, **kwargs):
                json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)  # noqa: B026

            def object_hook(self, obj):
                if type(obj) is dict:
                    for key, data in obj.items():
                        with contextlib.suppress(Exception):
                            obj[key] = pd.to_timedelta(data)
                    return obj
                try:
                    return pd.to_timedelta(obj)
                except ValueError:
                    return obj

        if not self.sync_path.exists():
            subject_id = self.group.subject
            condition = self.group.condition
            raise SyncDataNotFoundError(f"Sync data not found for subject {subject_id} and condition {condition}.")

        with self.sync_path.open() as f:
            sync_data = json.load(f, cls=JSONDecoder)
        return sync_data

    @property
    def body_video_path(self):
        if not (self.is_single(None) or self.is_single(["subject", "condition"])):
            raise ValueError("Path can only be accessed for a single condition of a single participant!")
        data_path = self.base_path.joinpath("data_per_subject").joinpath(
            f"{self.group.subject}/{self.group.condition}/video/body"
        )
        return data_path

    @property
    def _openpose_cleaned_path(self) -> Path:
        if not (self.is_single(None) or self.is_single(["subject", "condition"])):
            raise ValueError("OpenPose data can only be accessed for a single condition of a single participant!")

        file_path = self.body_video_path.joinpath("cleaned/openpose.csv")
        return file_path

    @property
    def openpose_data(self) -> pd.DataFrame:
        """Returns the cleaned openpose data for a single subject and position.
        The data is cut to the claps specified in sync_data. Additionally, the index was set to a constant frame rate
        and the data was interpolated linearly and lowpass filtered.
        """
        if not (self.is_single(None) or self.is_single(["subject", "condition"])):
            raise ValueError("OpenPose data can only be accessed for a single condition of a single participant!")
        file_path = self._openpose_cleaned_path
        return get_cleaned_openpose_data(file_path)

    @property
    def video_timestamps(self) -> pd.DataFrame:
        if not (self.is_single(None) or self.is_single(["subject", "condition"])):
            raise ValueError(
                "Video timestamps data can only be accessed for a single condition of a single participant!"
            )

        file_path = self.body_video_path.joinpath("processed/timestamps.csv")
        if not file_path.exists():
            subject_id = self.group.subject
            condition = self.group.condition
            raise TimestampDataNotFoundError(
                f"Timestamp data not found for subject {subject_id} and condition {condition}."
            )
        df = pd.read_csv(file_path, header=0, index_col=0)
        return df
