import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from biopsykit.io.nilspod import _handle_counter_inconsistencies_session

from empkins_io.datasets.d03._utils.dataset_utils import get_uncleaned_openpose_data
from empkins_io.datasets.d03.macro_ap01._custom_synced_session import (
    CustomSyncedSession,
)
from empkins_io.sensors.motion_capture.motion_capture_formats import mvnx

from empkins_io.utils._types import path_t, str_t
from empkins_io.utils.exceptions import NilsPodDataNotFoundError


def _load_mocap_data(
    base_path: path_t, participant: str, condition: str, *, verbose: bool = True
) -> (pd.DataFrame, datetime):
    # phase = self.index["phase"][0] if self.is_single(None) else list(self.index["phase"])
    data_path = base_path.joinpath("xsens/processed/test")
    mocap_file = data_path.joinpath(f"{participant}_{condition}_TEST.mvnx")
    if not mocap_file.exists():
        mocap_file_1 = data_path.joinpath(f"{participant}_{condition}_TEST1.mvnx")
        if mocap_file_1.exists():
            mocap_file_2 = data_path.joinpath(f"{participant}_{condition}_TEST2.mvnx")
            mvnx_data_1 = mvnx.MvnxData(mocap_file_1, verbose=True)
            mvnx_data_2 = mvnx.MvnxData(mocap_file_2, verbose=True)
            data1 = mvnx_data_1.data
            data2 = mvnx_data_2.data
            start1 = mvnx_data_1.start_time
            start1 = start1.tz_localize("UTC")
            start1 = start1.tz_convert("Europe/Berlin")
            start2 = mvnx_data_2.start_time
            start2 = start2.tz_localize("UTC")
            start2 = start2.tz_convert("Europe/Berlin")

            dt = (start2 - start1).total_seconds()

            data2 = data2.copy()
            data2.index = data2.index + dt
            data = pd.concat([data1, data2])
            start = start1

        else:
            raise FileNotFoundError(f"File '{mocap_file}' not found!")

    else:
        mvnx_data = mvnx.MvnxData(mocap_file, verbose=True)
        data = mvnx_data.data
        start = mvnx_data.start_time
        start = start.tz_localize("UTC")
        start = start.tz_convert("Europe/Berlin")

        # raise ValueError("Mocap recording shorter than phase")
    return data,start


_NILSPOD_FILE_PATTERN = re.compile(r"NilsPodX-(?P<sensor>[0-9A-Fa-f]{4})_(?P<session>\d{8}_\d{6})")


def _load_nilspod_session(
    base_path: path_t, participant: str, condition: str, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> pd.DataFrame:
    """Load the NilsPod session of one participant and condition that covers the test.

    Some folders contain several sessions (e.g. aborted restarts); the session whose chest sensor (the one recording
    ECG) overlaps ``test_start`` - ``test_end`` the most is used. The sensors are aligned to the sync region; if the
    sync region does not cover the test (e.g. sensors only synchronized at the very end), only the chest sensor is
    returned, on its own clock.

    Returns a DataFrame with (sensor id, channel) columns and the local time (Europe/Berlin) as index.
    """
    folder = Path(base_path).joinpath(f"nilspod/raw/{participant}/{condition}")
    sessions: Dict[str, list] = {}
    for file in sorted(folder.glob("NilsPodX-*.bin")):
        match = _NILSPOD_FILE_PATTERN.search(file.name)
        if match:
            sessions.setdefault(match["session"], []).append(file)
    if not sessions:
        raise NilsPodDataNotFoundError(f"No NilsPod files found for {condition} condition of {participant}!")

    best = None
    for files in sessions.values():
        # CustomSyncedSession loads the sync pod (9E02) with its custom firmware version
        datasets = list(CustomSyncedSession.from_file_paths(files, tz="Europe/Berlin").datasets)
        chest = [d for d in datasets if "ecg" in d.info.enabled_sensors]
        if not chest:
            continue
        start = pd.Timestamp(chest[0].info.utc_datetime_start).tz_convert("Europe/Berlin")
        stop = pd.Timestamp(chest[0].info.utc_datetime_stop).tz_convert("Europe/Berlin")
        overlap = (min(stop, test_end) - max(start, test_start)).total_seconds()
        if best is None or overlap > best[0]:
            best = (overlap, datasets, chest[0].info.sensor_id)
    if best is None:
        raise NilsPodDataNotFoundError(f"No NilsPod with ECG found for {condition} condition of {participant}!")
    _, datasets, chest_id = best

    try:
        session = CustomSyncedSession(datasets).cut(stop=-10).align_to_syncregion()
        _handle_counter_inconsistencies_session(session, handle_counter_inconsistency="ignore")
        data = session.data_as_df(index="local_datetime", concat_df=True)
        if data.index[0] > test_start or data.index[-1] < test_end:
            raise ValueError("Sync region does not cover the test.")
    except Exception:  # noqa: BLE001 - fall back to the (unsynchronized) chest sensor on its own
        chest = next(d for d in datasets if d.info.sensor_id == chest_id)
        data = chest.cut(stop=-10).data_as_df(index="local_datetime")
        data = pd.concat({chest_id: data}, axis=1)

    data.index.name = "time"
    return data