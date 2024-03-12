import pandas as pd
from typing import Dict

from empkins_io.utils._types import path_t
from empkins_io.datasets.d05.guardian._tfm_tilt_table_loader import TFMTiltTableLoader


def _load_tfm_data(tfm_path: path_t) -> Dict[str, pd.DataFrame]:
    tfm_data = TFMTiltTableLoader.from_mat_file(tfm_path)
    return tfm_data.data_as_dict(index="local_datetime")
