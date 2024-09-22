import datetime
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pathlib import Path
from typing import Dict, Literal, Optional, Union
from biopsykit.utils.time import tz
from pandas import DataFrame
from empkins_io.utils._types import path_t
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sensors.tfm import TfmLoader

import pandas as pd
import numpy as np

PHASE_MAPPING = {
        "Beginn der Aufzeichnung": "start_recording",
        "Ende der Aufzeichnung": "end_recording",
        "Beginn Ruhe 1": "start_rest_1",
        "Ende Ruhe 1": "end_rest_1",
        "Beginn Ruhe 2": "start_rest_2",
        "Ende Ruhe 2": "end_rest_2",
        "Beginn Ruhe 3": "start_rest_3",
        "Ende Ruhe 3": "end_rest_3",
        "Beginn CPT": "start_cpt",
        "Ende CPT": "end_cpt",
        "Beginn Atmung": "start_straw",
        "Ende Atmung": "end_straw",
}

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

def _load_tfm_data(base_path: path_t, participant_id: str, date: str) -> tuple[pd.DataFrame, float]:
    # Build the path to the TFM data directory for the given participant
    tfm_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "tfm/cleaned"
    )

    # Convert the input date string to a datetime object using the specified format
    pd_date = pd.to_datetime(date, format='%d.%m.%Y')
    form_date = pd_date.strftime('%Y-%m-%d')

    # Construct the full path to the TFM data file for the participant
    tfm_file_path = tfm_dir_path.joinpath(f"{participant_id}_tfm_data.mat")
    # Load the TFM data from the specified .mat file
    loader = TfmLoader.from_mat_file(
        path=tfm_file_path,
        recording_date=form_date,  # Use the reformatted date
        phase_mapping=PHASE_MAPPING  # Map phases according to the dataset's mapping
    )

    # Extract the raw phase data as a dictionary of DataFrames, indexed by local datetime
    data = loader.raw_phase_data_as_df_dict(index="local_datetime")
    # Retrieve the sampling rates from the loader
    fs = loader.sampling_rates_hz
    return data, fs

