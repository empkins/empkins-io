import warnings
from pathlib import Path
from pprint import pprint
from typing import Optional

import pandas as pd
from nilspodlib.exceptions import CorruptedPackageWarning, SynchronisationWarning

from empkins_io.datasets.d07.pilot_study._custom_synced_session import CustomSyncedSession
from empkins_io.sensors.motion_capture.xsens import XSensDataset
from empkins_io.utils._types import path_t, str_t
from empkins_io.utils.exceptions import NilsPodDataNotFoundError
from nilspodlib import Dataset
from packaging.version import Version


def _load_xsens_data(
    file_path: Path,
) -> pd.DataFrame:
    """Load Xsens data for a specific participant."""
    dataset = XSensDataset.from_mvnx_file(file_path, tz="Europe/Berlin")
    data = dataset.data_as_df(index="local_datetime")

    return data


def _load_nilspod_session(data_path: path_t) -> pd.DataFrame:
    nilspod_files = sorted(data_path.glob("NilsPodX-*.bin"))
    if len(nilspod_files) == 0:
        raise NilsPodDataNotFoundError("No NilsPod files found in directory!")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SynchronisationWarning)
        warnings.simplefilter("ignore", CorruptedPackageWarning)

        session = CustomSyncedSession.from_folder_path(data_path)
        # fix for "classical nilspod bug" where last sample counter is corrupted
        session = session.cut(stop=-10)
        session = session.align_to_syncregion()

    # _handle_counter_inconsistencies_session(session, handle_counter_inconsistency="ignore")

    # convert dataset to dataframe and localize timestamp
    df = session.data_as_df(index="local_datetime", concat_df=True)
    df.index.name = "time"
    return df


def load_ecg_raw_data(
    data_path: path_t,
    tz: str,
    datastreams: Optional[str_t] = None,
) -> pd.DataFrame:
    """Load raw ECG data from a participant folder."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist!")
    file_list = sorted(data_path.glob("*.bin"))
    file_list = [f for f in file_list if f.is_file() and not f.name.startswith("._")]
    for fp in file_list:
        print(fp)
        try:
            d = Dataset.from_bin_file(fp, tz=tz)
        except Exception:
            d = Dataset.from_bin_file(fp, tz=tz, legacy_support="resolve", force_version=Version("0.17.0"))
        print(d.active_sensors)

    pprint(file_list)
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .bin files found in {data_path}.")
    if len(file_list) > 1:
        raise ValueError(f"More than one .bin file found in {data_path}.")

    file_path = file_list[0]
    dataset = Dataset.from_bin_file(file_path, tz=tz)
    return dataset.data_as_df(index="local_datetime", datastreams=datastreams)


# def load_hr_data(data_path: path_t, tz: str) -> pd.DataFrame:
#     data = pd.read_csv(data_path, index_col=0)
#     data = data.assign(r_peak_time=pd.to_datetime(data["r_peak_time"], format="ISO8601").dt.tz_convert(tz))
#     return data
