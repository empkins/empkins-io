from datetime import datetime

import pandas as pd
from empkins_io.sensors.motion_capture.xsens._xsens import XSensDataset
from empkins_io.utils._types import path_t


def _load_mocap_data(base_path: path_t, p_id: str) -> XSensDataset:
    file_path = base_path.joinpath(f"data_per_participant/{p_id}/mocap/processed/{p_id}-002.mvnx")
    dataset = XSensDataset.from_mvnx_file(file_path, tz="Europe/Berlin")
    return dataset.data_as_df(index="local_datetime")

