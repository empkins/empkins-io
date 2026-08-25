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
    data_path = base_path.joinpath(f"xsens/processed")
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