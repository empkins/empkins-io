import datetime
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from pathlib import Path
from typing import Dict, Literal, Optional, Union, List
from biopsykit.utils.time import tz
from pandas import DataFrame
from empkins_io.utils._types import path_t
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sensors.tfm import TfmLoader
from empkins_io.sensors.empatica import EmpaticaDataset

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

# Define folder names for left and right devices
FOLDER_LEFT = "LEFT1-3YK33141NH"
FOLDER_RIGHT = "RIGHT-3YK34142G6"

# Define prefix for Empatica data files
PREFIX_LEFT = "1-1-LEFT1"
PREFIX_RIGHT = "1-1-RIGHT"

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

def _build_phase_times_path(base_path: path_t) -> Path:
    data_path = _build_tabular_data_path(base_path)
    file_path = data_path / "processed/phase_times_synchronized.csv"
    assert file_path.exists(), f"{file_path} does not exist"
    return file_path

def _load_phase_times(base_path: path_t) -> DataFrame:
    file_path = _build_phase_times_path(base_path)
    df = pd.read_csv(file_path)

    # Convert `timestamp_unix` from s to ms
    if "start_time" in df.columns and 'end_time' in df.columns:
        df["start_time"] = (df["start_time"] * 1000).astype("int64")
        df["end_time"] = (df["end_time"] * 1000).astype("int64")

    return df

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

def _load_start_end_times(base_path: path_t, participant_id: str) -> (datetime.datetime, datetime.datetime):
    data_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "protocol/cleaned"
    )
    data_path = data_path.joinpath(f"{participant_id}_protocol.xlsx")
    df = pd.read_excel(data_path, sheet_name="Messung", header=None)

    # Convert the cell to DataFrame indices (C39, C175)
    start_time = df.iloc[38, 2]
    end_time = df.iloc[174, 2]
    return (start_time, end_time)

def _create_loader(base_path: path_t, participant_id: str, date: str) -> TfmLoader:
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

    return loader

def _load_tfm_data(base_path: path_t, participant_id: str, date: str) -> tuple[pd.DataFrame, float]:
    # Create a TfmLoader object for the specified participant and date
    loader = _create_loader(base_path, participant_id, date)
    # Extract the raw phase data as a dictionary of DataFrames, indexed by local datetime
    data = loader.raw_phase_data_as_df_dict(index="local_datetime")
    # Retrieve the sampling rates from the loader
    fs = loader.sampling_rates_hz
    return data, fs


def _load_b2b_data(base_path: path_t, participant_id: str, date: str) -> tuple[pd.DataFrame, float]:
    # Create a TfmLoader object for the specified participant and date
    loader = _create_loader(base_path, participant_id, date)
    # Extract the B2B phase data as a dictionary of DataFrames, indexed by local datetime
    data = loader.b2b_phase_data_as_df_dict(index="local_datetime")
    # Retrieve the sampling rates from the loader
    fs = loader.sampling_rates_hz
    return data, fs

def _load_radar_data(base_path: path_t, participant_id: str, sampling_rate_hz: float) -> tuple[pd.DataFrame, float]:
    # Build the directory path for radar data based on the base path and participant ID
    radar_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath(
        "emrad/raw"
    )
    
    # Build the file path to the radar data (HDF5 format) using the participant ID
    radar_file_path = radar_dir_path.joinpath(f"{participant_id}_emrad_data.h5")
    # Load radar data from the HDF5 file using the EmradDataset class
    dataset_radar = EmradDataset.from_hd5_file(radar_file_path, sampling_rate_hz=sampling_rate_hz)
    
    # Convert the radar data to a pandas DataFrame, using "local_datetime" as the index and adding a sync output column
    data = dataset_radar.data_as_df(index="local_datetime", add_sync_out=True)
    # Retrieve the sampling rate (Hz) from the dataset
    fs = dataset_radar.sampling_rate_hz
    # Return the DataFrame (data) and the sampling rate (fs)
    return data, fs

def _check_if_file_exists(base_path: path_t, subject_id: str, path_to_file) -> Optional[pd.DataFrame]:
    # Skip if file already exists
    data_path = _build_data_path(base_path, participant_id=subject_id)
    output_path = data_path.joinpath(path_to_file)
    if output_path.exists():
        # print(f"File already exists: {output_path}")
        # load the existing file
        existing_df = pd.read_csv(output_path)
        return existing_df
    else:
        return None

def _load_empatica_data(base_path: path_t, participant_id: str, date: str, empatica_lr: str, start_end_times: tuple[datetime.datetime, datetime.datetime], signal_type: list[str]):
    data = {}
    
    # Convert date from "dd.mm.yyyy" to "yyyy-mm-dd" for folder matching
    date_parts = date.split(".")
    folder_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"

    # Determine LEFT or RIGHT device usage
    device_side = FOLDER_LEFT if empatica_lr == "L" else FOLDER_RIGHT

    # Determine the filename based on the device side, date, and signal type
    file_prefix = PREFIX_LEFT if empatica_lr == "L" else PREFIX_RIGHT

    for signal in signal_type:
        # Determine the filename based on the device side, date, and signal type
        filename = f"{file_prefix}_{folder_date}_{signal}.csv"

        # Build the path to the Empatica data directory for the given participant
        final_path = base_path.joinpath(f"Empatica/{folder_date}/{device_side}/digital_biomarkers/aggregated_per_minute/{filename}")

        # Check if file exists
        if not final_path.exists():
            print(f"File not found: {final_path}")
            continue

        # Load the Empatica data from the specified CSV file
        df = pd.read_csv(final_path, index_col=1, parse_dates=True)

        # Ensure index is in datetime format
        df.index = pd.to_datetime(df.index, utc=True)  # Ensure timestamps are timezone-aware in UTC

        # Convert UTC index to CEST (Europe/Berlin, GMT+2 in summer)
        df.index = df.index.tz_convert("Europe/Berlin")

        # Filter the data to the specified start and end times
        df = df.loc[start_end_times[0]:start_end_times[1]]

        # Append the filtered data to the dict
        data[signal] = df
        
    return data

def _create_agg_empatica(empatica_data: dict[str, pd.DataFrame], phase_times: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    empatica_data_by_phase = {}

    for signal, df in empatica_data.items():
        # Ensure correct type
        df["timestamp_unix"] = pd.to_numeric(df["timestamp_unix"])
        df.columns.values[2] = "value"

        phase_dict = {}
        for _, row in phase_times.iterrows():
            phase = row["phase"]
            start = row["start_time"]
            end = row["end_time"]

            sliced = df[
                (df["timestamp_unix"] >= start) &
                (df["timestamp_unix"] <= end)
            ]
            phase_dict[phase] = sliced

        empatica_data_by_phase[signal] = phase_dict
    return empatica_data_by_phase

def _save_agg_empatica(base_path: path_t, subject_id: str, signal_phase_data: dict[str, dict[str, pd.DataFrame]], path_to_file: str) -> pd.DataFrame:
    # Otherwise, create the directory
    rows = []
    for signal, phases in signal_phase_data.items():
        for phase, df in phases.items():
            df_copy = df.copy()
            df_copy["timestamp"] = df_copy.index
            df_copy["signal"] = signal
            df_copy["phase"] = phase
            rows.append(df_copy)

    # Concatenate all rows into one big DataFrame
    full_df = pd.concat(rows, ignore_index=True)
    # order by timestamp
    full_df = full_df.sort_values(by=["signal","timestamp_unix"])

    # Reorder columns for clarity
    cols = ["signal", "phase", "timestamp"] + [col for col in full_df.columns if col not in ["signal", "phase", "timestamp"]]
    full_df = full_df[cols]

    # Save to CSV
    data_path = _build_data_path(base_path, participant_id=subject_id)
    output_path = data_path.joinpath(path_to_file)
    full_df.to_csv(output_path, index=False)
    print(f"Saved all data to: {output_path}")

    return full_df

def _create_avro(base_path: path_t, participant_id: str, signal_type: list[str], phase_times: pd.DataFrame) -> tuple[pd.DataFrame, float]: 
    # Build the path to the Empatica data directory for the given participant
    empatica_dir_path = _build_data_path(base_path, participant_id=participant_id).joinpath("empatica/raw")

    # Load the Empatica data from the specified CSV file
    loader = EmpaticaDataset(path=empatica_dir_path, index_type="local_datetime", tz="Europe/Berlin")
    sampling_rates = loader._sampling_rates_hz.copy()
    
    avro_data_by_phase = {}
    # Convert the timestamp_unix column to datetime
    # phase_times_datetime["start_time"] = pd.to_datetime(phase_times_datetime["start_time"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
    # phase_times_datetime["end_time"] = pd.to_datetime(phase_times_datetime["end_time"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")

    # Filter the data to the specified start and end times
    for signal in signal_type:
        # load the data
        df = loader.data_as_df(sensor=signal)
        # Ensure index is in datetime format
        df.columns.values[0] = "value"
        df.index.name = "timestamp"
        df["timestamp_unix"] = df.index.astype("int64") // 10**6 

        #calculate sampling rate
        delta = (df.index[1] - df.index[0]).total_seconds()
        # Sampling rate in Hz
        sampling_rate = 1 / delta
        sampling_rates[signal] = sampling_rate

        # Convert the timestamp_unix column to datetime
        phase_dict = {}
        for _, row in phase_times.iterrows():
            phase = row["phase"]
            start = row["start_time"]
            end = row["end_time"]

            sliced = df[(df["timestamp_unix"] >= start) & (df["timestamp_unix"] <= end)]
            phase_dict[phase] = sliced

        avro_data_by_phase[signal] = phase_dict

    return avro_data_by_phase, sampling_rates

def _save_avro(base_path: path_t, subject_id: str, signal_phase_data: dict[str, dict[str, pd.DataFrame]], path_to_file: str) -> pd.DataFrame:
    # Otherwise, create the directory
    rows = []
    for signal, phases in signal_phase_data.items():
        for phase, df in phases.items():
            df_copy = df.copy()
            df_copy["timestamp"] = df_copy.index
            df_copy["signal"] = signal
            df_copy["phase"] = phase
            rows.append(df_copy)

    # Concatenate all rows into one big DataFrame
    full_df = pd.concat(rows, ignore_index=True)
    # order by timestamp
    full_df = full_df.sort_values(by=["signal","timestamp_unix"])

    # Reorder columns for clarity
    cols = ["signal", "phase", "timestamp"] + [col for col in full_df.columns if col not in ["signal", "phase", "timestamp"]]
    full_df = full_df[cols]

    # Save to CSV
    data_path = _build_data_path(base_path, participant_id=subject_id)
    output_path = data_path.joinpath(path_to_file)
    full_df.to_csv(output_path, index=False)
    print(f"Saved all data to: {output_path}")

    return full_df
