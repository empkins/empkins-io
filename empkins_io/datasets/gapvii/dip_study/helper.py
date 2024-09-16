import datetime
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
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
    data_path = base_path.joinpath("data_tabular")
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

def _build_datetime_path(base_path: path_t, participant_id: str) -> Path:
    date_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "protocol/cleaned"
    )
    date_file_path = date_dir_path.joinpath(f"{participant_id}_protocol.xlsx")
    assert date_file_path.exists()
    return date_file_path

def _load_single_date(base_path: path_t, subject_id: str) -> pd.Timestamp:
    data_path =_build_datetime_path(base_path=base_path, participant_id=subject_id)
    df = pd.read_excel(data_path, sheet_name="Allgemein", header=None)
    # Convert the cell to DataFrame indices (C3)
    cell_value = df.iloc[2, 2]
    return pd.to_datetime(cell_value, dayfirst=True)

def _update_dates(base_path: path_t, subject_date_dict: dict, sheet_name: str = "Sheet1"):
    # Load the workbook and select the specified sheet
    file_path = _build_general_tabular_path(base_path)
    workbook = load_workbook(file_path)
    sheet = workbook[sheet_name]
    sheet["H1"] = "date"

    # Loop through the names and update dates
    for subject, date in subject_date_dict.items():
        # Ensure the date is in the desired format
        if isinstance(date, pd.Timestamp):
            formatted_date = date.strftime("%d.%m.%Y")
        else:
            raise TypeError("Date should be a pd.Timestamp object")

        # Find the row with the matching name
        for row in sheet.iter_rows(min_row=2, max_row=None, min_col=1, max_col=1):
            cell = row[0]
            if cell.value == subject:
                # Write the date to the corresponding cell in column H
                sheet[f"H{cell.row}"] = formatted_date
                break

    # Save the changes to the workbook
    workbook.save(file_path)
