from datetime import time
import datetime
from typing import Optional

import pandas as pd

from empkins_io.utils._types import path_t


def _load_ecg_data(
    base_path: path_t, subject: str, to_local_time: Optional[bool] = True
) -> pd.DataFrame:
    df = pd.read_csv(
        base_path.joinpath("data_per_subject", subject, f"{subject}.txt"),
        sep="\t",
        skiprows=11,
        header=0,
        skipfooter=2,
        engine="python",
        encoding="unicode_escape",
    )

    df.set_index("Zeit", inplace=True)
    df.index.names = ["sample"]

    df = df.loc[:, ["Sensor-B:EKG", "Ereignisse"]]
    df.columns = ["ecg", "event"]
    if to_local_time:
        return _convert_to_timestamps(df, start_time(base_path, subject))
    else:
        return df


def _load_raw_log(base_path: path_t, subject: str) -> pd.DataFrame:
    df = pd.read_csv(
        base_path.joinpath("data_per_subject", subject, f"LogID{subject}.csv"),
        skiprows=4,
        sep="\t",
    )

    df.set_index("time", inplace=True)

    df.index = df.index.str.rsplit(":", n=1).str.get(0)

    df.index = pd.to_timedelta(df.index) + pd.to_datetime(
        start_time(base_path, subject).date()
    )

    # convert utc to local time
    df.index = df.index.tz_localize("UTC").tz_convert("Europe/Berlin")

    # erase timezone awareness
    df.index = df.index.tz_localize(None)

    return pd.DataFrame(df["phase"])


def _convert_to_timestamps(df: pd.DataFrame, start_time: pd.Timestamp) -> pd.DataFrame:
    # convert samples to seconds
    df.index = df.index / 256

    # add start time to seconds to get timestamps
    df.index = pd.to_timedelta(df.index, unit="s") + start_time
    df.index.names = ["time"]

    return df


def start_time(base_path: path_t, subject: str) -> pd.Timestamp:
    df = pd.read_csv(
        base_path.joinpath("data_per_subject", subject, f"{subject}.txt"),
        sep="\t",
        engine="python",
        nrows=4,
    )

    datetime_string = df.loc["Datum:"][0] + " " + df.loc["Zeit:"][0]

    return pd.to_datetime(datetime_string, format="%Y-%m-%d %H:%M:%S")
