from pathlib import Path

import pandas as pd

from empkins_io.sensors.motion_capture.xsens import XSensDataset


def _load_mocap_data(
    file_path: Path,
) -> pd.DataFrame:
    """Load Xsens data for a specific participant."""

    dataset = XSensDataset.from_mvnx_file(file_path, tz="Europe/Berlin")
    data = dataset.data_as_df(index="local_datetime")

    return data
