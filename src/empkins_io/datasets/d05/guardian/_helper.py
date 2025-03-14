from typing import Dict

import pandas as pd

from empkins_io.datasets.d05.guardian._tfm_tilt_table_loader import TFMTiltTableLoader
from empkins_io.utils._types import path_t


def _load_tfm_data(base_path: path_t, tfm_path: path_t) -> Dict[str, pd.DataFrame]:
    tfm_data = TFMTiltTableLoader.from_mat_file(base_path, tfm_path)
    return tfm_data.data_as_dict(index="local_datetime")
