from collections.abc import Sequence
from itertools import product
from typing import ClassVar

import biopsykit
import pandas as pd
from tpcp import Dataset

from empkins_io.utils._types import path_t


class VadasDataset(Dataset):
    ECG_SAMPLING_RATE = 256

    DAYS: ClassVar[Sequence[str]] = ["U1", "U2"]
    TEST: ClassVar[Sequence[str]] = ["baseline", "cpt"]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
    ):
        # ensure pathlib
        self.base_path = base_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            "_".join(file.name.split("_")[:2]) for file in self.base_path.joinpath("Sensordaten Luca").glob("*")
        ]

        subject_ids = sorted(set(subject_ids))

        index_cols = ["subject", "day", "test"]
        index = product(subject_ids, self.DAYS, self.TEST)

        index = pd.DataFrame(index, columns=index_cols)

        return index

    @property
    def subject(self) -> str:
        if not self.is_single("subject"):
            raise ValueError("Dataset is not single-subject dataset!")

        return self.index["subject"].iloc[0]

    @property
    def day(self) -> str:
        if not self.is_single("day"):
            raise ValueError("Dataset is not single-day dataset!")
        return self.index["day"].iloc[0]

    @property
    def test(self) -> str:
        if not self.is_single("test"):
            raise ValueError("Dataset is not single-test dataset!")
        return self.index["test"].iloc[0]

    @property
    def ecg_data(self) -> pd.DataFrame:
        if not self.is_single("subject") or not self.is_single("day"):
            raise ValueError("Dataset is not single-subject/day dataset!")
        return self._load_nilspod_session(self.subject, self.day)[0]

    @property
    def start_time_nilspod(self) -> pd.Timestamp:
        if not self.is_single("subject") or not self.is_single("day"):
            raise ValueError("Dataset is not single-subject/day dataset!")
        timelog = pd.read_csv(self.base_path.joinpath("timelog_cleaned.csv"))
        timelog = timelog.set_index(["subject", "day"])
        start_nilspod = timelog.loc[(self.subject, self.day), "start_nilspod"]
        date = timelog.loc[(self.subject, self.day), "date"].split(",")[0]
        return pd.to_datetime(date + "2024 " + str(start_nilspod), format="%d.%m.%Y %H:%M:%S")

    @property
    def start_time_cpt(self) -> pd.Timestamp:
        if not self.is_single("subject") or not self.is_single("day"):
            raise ValueError("Dataset is not single-subject/day dataset!")
        timelog = pd.read_csv(self.base_path.joinpath("timelog_cleaned.csv"))
        timelog = timelog.set_index(["subject", "day"])
        start_nilspod = timelog.loc[(self.subject, self.day), "start_cpt"]
        date = timelog.loc[(self.subject, self.day), "date"].split(",")[0]
        return pd.to_datetime(date + "2024 " + str(start_nilspod), format="%d.%m.%Y %H:%M:%S")

    def _load_nilspod_session(self, subject_id: str, day: str) -> tuple[pd.DataFrame, float]:
        nilspod_files = sorted(self.base_path.joinpath("Sensordaten Luca").glob(f"{subject_id}_{day}_*.bin"))

        if len(nilspod_files) == 0:
            raise ValueError("No NilsPod file found!")

        if len(nilspod_files) > 1:
            raise ValueError("More than one NilsPod file found!")

        return biopsykit.io.nilspod.load_dataset_nilspod(nilspod_files[0])
