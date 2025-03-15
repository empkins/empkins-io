from pathlib import Path

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from src.empkins_io.utils._types import path_t


def load_data(file_path: path_t) -> pd.DataFrame:
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, ".npz")
    file = np.load(file_path.name, allow_pickle=True)  # load npz file
    data = file["data"]
    t = data[:, 0, :].ravel()  # time base
    ya = data[:, 1, :].ravel()  # channel 1 (diffuse reflection)
    yb = data[:, 2, :].ravel()  # channel 2 (specular reflection)

    return pd.DataFrame({"ch_a": ya, "ch_b": yb}, index=pd.Index(t, name="time"))
