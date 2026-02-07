from pathlib import Path
from typing import Dict, Optional, Sequence
import pandas as pd
import neurokit2 as nk

from empkins_io.datasets.radarstenosis.helper import (
    _get_locations_from_index,
    _load_atimelogger_file,
    _calc_biopac_timelog_shift,
    _load_radar_raw,
    _load_biopac_raw,
    _ensure_synced,
    _load_radar_synced,
    _load_biopac_synced,
)

from itertools import product
from biopsykit.utils.file_handling import get_subject_dirs

from empkins_io.utils._types import path_t

from tpcp import Dataset


class RadarCardiaStenosisTest(Dataset):
    base_path: path_t
    _SAMPLING_RATES: Dict[str, float] = {
        "radar_original": 8000000 / 4096 / 2,
        "biopac_original": 2000,
        "resampled": 1000,
    }
    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
        "sync": "sync",
    }
    bp_tl_shift: pd.Timedelta | None

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
    ):
        self.base_path = base_path
        self.bp_tl_shift = None
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        indices = pd.DataFrame()
        for subject_id in subject_ids:
            timelog_file_path = self.base_path.joinpath(
                f"data_per_subject/{subject_id}/timelog/processed/{subject_id}_timelog.csv"
            )
            timelogs = pd.read_csv(timelog_file_path, encoding="utf-8")
            measurements = timelogs["Aktivitätstyp"]
            index = pd.DataFrame({"subject": subject_id, "measurement": measurements})
            indices = pd.concat([indices, index], axis=0)
        indices = indices[indices["measurement"] != "sync"]

        return indices

    @property
    def sampling_rates(self) -> dict[str, float]:
        return self._SAMPLING_RATES

    @property
    def sampling_rate(self) -> float:
        return self._SAMPLING_RATES["resampled"]

    @property
    def subject(self) -> str:
        if not self.is_single(["subject"]):
            raise ValueError("Subject can only be accessed for one single participant at once")
        return self.index["subject"][0]

    @property
    def measurement(self) -> str:
        if not self.is_single(["measurement"]):
            raise ValueError("Measurement can only be accessed for a single measurement at once")
        return self.index["measurement"][0]

    @property
    def timelog(self) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Timelog can only be accessed for one single participant at once")
        locations = _get_locations_from_index(index=self.index)
        subject = self.index["subject"][0]
        timelog_file_path = self.base_path.joinpath(
            f"data_per_subject/{subject}/timelog/processed/{subject}_timelog.csv"
        )
        timelog = _load_atimelogger_file(file_path=timelog_file_path, timezone="Europe/Berlin")
        return timelog[locations]

    @property
    def biopac_timelog_shift(self) -> pd.Timedelta:
        if not self.bp_tl_shift:
            self.bp_tl_shift = _calc_biopac_timelog_shift(base_path=self.base_path, subject=self.subject)
        return self.bp_tl_shift

    @property
    def emrad_raw(self) -> pd.DataFrame:
        # radar data unsynchronized
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        radar = _load_radar_raw(
            base_path=self.base_path, subject=self.subject, fs=self._SAMPLING_RATES["radar_original"]
        )
        return radar

    @property
    def biopac_raw(self) -> pd.DataFrame:
        # biopac data unsynchronized
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        biopac = _load_biopac_raw(
            base_path=self.base_path, subject=self.subject, channel_mapping=self.BIOPAC_CHANNEL_MAPPING
        )
        tl = self.timelog
        return biopac

    @property
    def emrad_synced(self) -> pd.DataFrame:
        # radar data synchronized
        if not self.is_single(["subject"]):
            raise ValueError("Radar data can only be accessed for one single participant at once")

        emrad = _load_radar_synced(
            base_path=self.base_path,
            subject=self.subject,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            fs=self._SAMPLING_RATES,
            index=self.index,
        )
        return emrad

    @property
    def biopac_synced(self) -> pd.DataFrame:
        # biopac data synchronized
        if not self.is_single(["subject"]):
            raise ValueError("BIOPAC data can only be accessed for one single participant at once")

        biopac = _load_biopac_synced(
            base_path=self.base_path,
            subject=self.subject,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            fs=self._SAMPLING_RATES,
            index=self.index,
        )
        return biopac

    @property
    def biopac_data(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("BIOPAC data can only be accessed for one single location at once")
        location = _get_locations_from_index(index=self.index)[0]
        data_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/data_per_location/{location}/{self.subject}_biopac_data.h5"
        )
        _ensure_synced(
            base_path=self.base_path,
            subject=self.subject,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            fs=self._SAMPLING_RATES,
            index=self.index,
            resample=False,
            location=location,
        )
        data = pd.read_hdf(data_path, key=f"biopac_data")

        if self.is_single(None):
            location = _get_locations_from_index(index=self.index)[0]
            tl = self.timelog
            start = tl[location]["start"].iloc[0] + self.biopac_timelog_shift
            end = tl[location]["end"].iloc[0] + self.biopac_timelog_shift
            return data.loc[start:end]
        return data

    @property
    def emrad_data(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("BIOPAC data can only be accessed for one single location at once")
        location = _get_locations_from_index(index=self.index)[0]
        data_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/data_per_location/{location}/{self.subject}_emrad_data.h5"
        )
        _ensure_synced(
            base_path=self.base_path,
            subject=self.subject,
            channel_mapping=self.BIOPAC_CHANNEL_MAPPING,
            fs=self._SAMPLING_RATES,
            index=self.index,
            resample=False,
            location=location,
        )
        data = pd.read_hdf(data_path, key=f"emrad_data")

        if self.is_single(None):
            location = _get_locations_from_index(index=self.index)[0]
            tl = self.timelog
            start = tl[location]["start"].iloc[0] + self.biopac_timelog_shift
            end = tl[location]["end"].iloc[0] + self.biopac_timelog_shift
            return data.loc[start:end]
        return data

    def save_data_to_location(self, data: pd.DataFrame, file_name: str) -> None:
        locations = self.index.drop(columns="subject").columns.tolist()
        if not self.is_single(locations):
            raise ValueError("Data can only be saved for a single location-breathing combination")

        location = _get_locations_from_index(index=self.index)[0]
        data_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/data_per_location/{location}/{self.subject}_{file_name}.h5"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_hdf(data_path, mode="w", key="data", index=True)

    def load_data_from_location(self, file_name: str, data_format: str = "h5") -> pd.DataFrame:
        locations = self.index.drop(columns="subject").columns.tolist()
        if not self.is_single(locations):
            raise ValueError("Data can only be loaded for a single location-breathing combination")

        location = _get_locations_from_index(index=self.index)[0]
        data_path = self.base_path.joinpath(
            f"data_per_subject/{self.subject}/data_per_location/{location}/{self.subject}_{file_name}.{data_format}"
        )
        if data_format == "h5":
            data = pd.read_hdf(data_path, key="data")
        elif data_format == "csv":
            data = pd.read_csv(data_path)
        else:
            raise ValueError("Unable to read data format")
        return data

    def get_subsets_by_measurement_position(self, measurement_position: str) -> pd.DataFrame:
        if not self.is_single(["subject"]):
            raise ValueError("Data can only be accessed for one single participant at once")

        measurement_names = []
        for subset in self:
            if subset.measurement.startswith(measurement_position):
                measurement_names.append(subset.measurement)
        data = self.get_subset(measurement=measurement_names)
        return data

    def get_r_peaks(self, data: pd.DataFrame) -> pd.DataFrame:
        signals, r_peaks = nk.ecg_process(ecg_signal=data.ecg, sampling_rate=self.sampling_rate)
        r_peaks = r_peaks["ECG_R_Peaks"]
        return r_peaks
