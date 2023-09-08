import json
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.datasets.d03._utils.dataset_utils import get_cleaned_openpose_data
from empkins_io.datasets.d03.macro_ap01._base_dataset import MacroBaseDataset
from empkins_io.datasets.d03.macro_ap01.helper import _get_times_for_mocap, _load_tsst_mocap_data
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import SyncDataNotFoundException, TimestampDataNotFoundException

_cached_load_mocap_data = lru_cache(maxsize=4)(_load_tsst_mocap_data)


class MacroStudyTsstDataset(MacroBaseDataset):
    """Class to conveniently access the data of the macro study dataset for subject and condition.
    If access is required per-phase, use :class:`MacroStudyTsstDatasetPerPhase` instead."""

    SUBJECTS_WITHOUT_MOCAP = (
        "VP_03",
        "VP_31",
    )

    SUBSETS_WITHOUT_OPENPOSE_DATA = (
        ("VP_01", "ftsst"),
        ("VP_01", "tsst"),
        ("VP_03", "tsst"),
        ("VP_05", "tsst"),
        ("VP_07", "ftsst"),
        ("VP_08", "ftsst"),
        ("VP_17", "ftsst"),
        ("VP_18", "ftsst"),
        ("VP_19", "ftsst"),
        ("VP_21", "ftsst"),
        ("VP_25", "tsst"),
        ("VP_33", "ftsst"),
        ("VP_34", "tsst"),
        ("VP_35", "ftsst"),
        ("VP_35", "tsst"),
        ("VP_36", "ftsst"),
        ("VP_36", "tsst"),
        ("VP_37", "ftsst"),
        ("VP_37", "tsst"),
        ("VP_38", "ftsst"),
        ("VP_38", "tsst"),
        ("VP_41", "ftsst"),
        ("VP_41", "tsst"),
    )

    conditions = ("ftsst", "tsst")

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_without_mocap: bool = True,
        exclude_without_openpose: bool = True,
        exclude_missing_data: bool = False,
        use_cache: bool = True,
    ):
        self.exclude_without_mocap = exclude_without_mocap
        self.exclude_without_openpose = exclude_without_openpose
        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            exclude_missing_data=exclude_missing_data,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        if self.exclude_missing_data:
            for missing_type, sids in self.MISSING_DATA.items():
                for sid in sids:
                    if sid in subject_ids:
                        subject_ids.remove(sid)

        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index = list(product(subject_ids, self.conditions))
        if self.exclude_without_openpose:
            for subject_id, condition in self.SUBSETS_WITHOUT_OPENPOSE_DATA:
                index.remove((subject_id, condition))

        index = pd.DataFrame(
            index,
            columns=["subject", "condition"],
        )

        return index

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
        if self.is_single(None):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = "total"
            data_total = self._get_mocap_data_per_phase(subject_id, condition, phase)

            return data_total
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    def _get_mocap_data_per_phase(self, subject_id: str, condition: str, phase: str = "total"):
        data, start = self._get_mocap_data(subject_id, condition)
        timelog = self.timelog_test
        times = _get_times_for_mocap(timelog, start, phase)
        data_total = {}

        if phase == "total":
            for idx, phase in times.iterrows():
                data_total[idx] = data.loc[phase["start"] : phase["end"]]
            data_total = pd.concat(data_total).droplevel(0)
        else:
            data_total = data.loc[times["start"] : times["end"]]

        return data_total

    def _get_mocap_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition)
        return _load_tsst_mocap_data(self.base_path, subject_id, condition)

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
                json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

            def object_hook(self, obj):
                if type(obj) is dict:
                    for key, data in obj.items():
                        try:
                            obj[key] = pd.to_timedelta(data)
                        except:
                            pass
                    return obj
                else:
                    try:
                        return pd.to_timedelta(obj)
                    except:
                        return obj

        if not self.sync_path.exists():
            subject_id = self.group.subject
            condition = self.group.condition
            raise SyncDataNotFoundException(f"Sync data not found for subject {subject_id} and condition {condition}.")

        with open(self.sync_path) as f:
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
        and the data was interpolated linearly and lowpass filtered."""
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
            raise TimestampDataNotFoundException(
                f"Timestamp data not found for subject {subject_id} and condition {condition}."
            )
        df = pd.read_csv(file_path, header=0, index_col=0)
        return df
