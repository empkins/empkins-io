import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Union
from biopsykit.utils.time import tz
from pandas import DataFrame
from empkins_io.utils._types import path_t
from empkins_io.sensors.emrad import EmradDataset

import pandas as pd
import numpy as np


def _build_data_path(base_path: path_t, participant_id: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}")
    assert data_path.exists()
    return data_path


def _build_tabular_data_path(base_path: path_t) -> Path:
    data_path = base_path.joinpath(f"data_tabular")
    assert data_path.exists()
    return data_path


def _build_general_tabular_path(base_path: path_t) -> Path:
    data_path = _build_tabular_data_path(base_path)
    data_path = data_path.joinpath("processed/empkins_dip_study.xlsx")
    assert data_path.exists()
    return data_path


def _load_general_information(base_path: path_t, column: str) -> DataFrame:
    file_path = _build_general_tabular_path(base_path)
    df = pd.read_excel(file_path, index_col=0)
    return df[column]


