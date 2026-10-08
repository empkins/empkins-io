from functools import lru_cache
from typing import Dict, Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd
from tpcp import Dataset
from pathlib import Path
from itertools import product
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.utils._types import path_t

from .helper import (
    load_timelog,
    load_raw_emrad_data,
    load_raw_biopac_data,
    load_raw_psg_data,
    get_respiration_data_synced,
)

from .plotting import (
    activities_overview_plotting,
    respiration_overview_synced_plotting,
    sync_raw_plot,
    sync_synced_plot
)

class RespirationActivityStudyDataset(Dataset):

    base_path: path_t

    SAMPLING_RATES = {
        "psg": 256,
        "emrad": (8000000 / 4096) / 2,
        "resampled": 1000,
        "biopac": 2000,
    }

    BIOPAC_CHANNEL_MAPPING = {
        "ecg": "ecg",
        "breathing_belt": "breathing_belt",
        "sync": "sync",
    }

    PSG_DATASTREAMS = ["Flow", "RIP Abdom", "SpO2", "Pulse", "Pleth", "AUX"]

    SYNC_SEQUENCE_LENGTH_s = 8

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
    ):

        self.base_path = base_path

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in
            get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
            ]

        data_type = ["breathing", "activity"]

        index = list(product(subject_ids, data_type))

        index = pd.DataFrame(index, columns=["subject", "data_type"])

        return index

    @property
    def subject(self) -> str:
        if not self.is_single(["subject"]):
            raise ValueError("Subject can only be accessed for one single participant at once")
        return self.index["subject"][0]

    @property
    def data_type(self) -> str:
        if not self.is_single(["data_type"]):
            raise ValueError("Data type can only be accessed for one single participant at once")
        return self.index["data_type"][0]

    @property
    def timelog(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")

        if not self.is_single(["data_type"]):
            return load_timelog(self.base_path, self.subject, "all")
        else:
            return load_timelog(self.base_path, self.subject, self.data_type)

    @property
    def raw_emrad_data(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "Raw EMRAD data can only be accessed for one single participant-data type combination at once"
            )

        return load_raw_emrad_data(self.base_path, self.subject, self.data_type, self.SAMPLING_RATES["emrad"])

    @property
    def raw_biopac_data(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError(
                "Raw BIOPAC data can only be accessed for one single participant at once"
            )

        # the .acq file of one recording does not contain a start time. In that case, the start time of the
        # radar recording is used instead (only loaded if it is actually needed).
        start_time = (lambda: self.raw_emrad_data.index[0]) if self.is_single(None) else None

        return load_raw_biopac_data(
            self.base_path, self.subject, self.BIOPAC_CHANNEL_MAPPING, start_time=start_time
        )

    @property
    def raw_psg_data(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError(
                "Raw PSG data can only be accessed for one single participant at once"
            )

        return load_raw_psg_data(self.base_path, self.subject, self.PSG_DATASTREAMS)

    def respiration_data_synced(self, perform_sync: bool = False) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError(
                "Respiration data synced can only be accessed for one single participant at once"
            )

        if not self.data_type == "breathing":
            raise ValueError(
                "Respiration data synced can only be accessed for participants with breathing data type"
            )

        return get_respiration_data_synced(
            self.base_path,
            self.subject,
            sync_sequence_length_s=self.SYNC_SEQUENCE_LENGTH_s,
            fs=self.SAMPLING_RATES,
            biopac_channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            psg_datastreams=self.PSG_DATASTREAMS,
            perform_sync=perform_sync,
        )

    @property
    def activities_overview_plot(self):
        if not self.is_single(None):
            raise ValueError(
                "Activities overview plot can only be accessed for one single participant at once"
            )

        return activities_overview_plotting(self.raw_emrad_data, self.timelog, self.subject)

    @property
    def sync_raw_plot(self):
        if not self.is_single(["subject"]):
            raise ValueError(
                "Sync raw plot can only be accessed for one single participant at once"
            )

        return sync_raw_plot(self.raw_emrad_data, self.raw_biopac_data, self.raw_psg_data, self.subject)

    @property
    def sync_synced_plot(self):
        if not self.is_single(["subject"]):
            raise ValueError(
                "Sync synced plot can only be accessed for one single participant at once"
            )

        return sync_synced_plot(self.respiration_data_synced(), self.subject)

    @property
    def respiration_overview_synced_plot(self):
        if not self.is_single(None):
            raise ValueError(
                "Respiration overview synced plot can only be accessed for one single "
                "participant-data type combination at once"
            )

        return respiration_overview_synced_plotting(
            self.respiration_data_synced(), self.timelog, self.subject
        )






