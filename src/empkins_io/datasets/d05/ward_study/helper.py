from pathlib import Path
import pandas as pd
from datetime import timedelta

from ._columns import COMPLETED_COLS

def get_raw_data_file(base_path: Path, load_only_completed: bool):
    # TODO: low_memory=False to suppress the warnings of different dtypes in columns
    data = pd.read_csv(base_path, low_memory=False)

    # Remove Part 53 since he does not have any values
    data = data[data["record_id"] != 53]

    # Correct the timestamps in the ipos_i_time column -> right now not necessarily a date in the usual format
    ipos_self_time_col = "ipos_i_time"
    data[ipos_self_time_col] = pd.to_datetime(
        data[ipos_self_time_col],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    if load_only_completed:
        return get_completed_rows_only(raw_data=data)    
    else:
        return data

def get_completed_rows_only(raw_data: pd.DataFrame):
    """
    Get only the completed entries where the uebersicht instrument is marked as complete &
    filters the rows which are marked as unverified
    """
    # Get only ids where uebersicht instrument is complete
    id_col = "record_id"
    gt_overview_column = "uebersicht_complete"

    completed_ids = raw_data.loc[
        raw_data[gt_overview_column] == 2.0,
        id_col
    ].unique()

    filtered_data = raw_data[
        raw_data[id_col].isin(completed_ids)
    ]

    # Filter unverified rows
    cols_to_check = [
        col for col in COMPLETED_COLS
        if col != "record_id"
    ]

    mask = ~(filtered_data[cols_to_check] == 1.0).any(axis=1)

    return filtered_data.loc[mask]

def calculate_participant_ids(base_path: Path, load_only_completed: bool):
    """
    Calculates the amount of participants in the study.
    """
    data = get_raw_data_file(base_path, load_only_completed)
    _participant_ids = data["record_id"].unique()

    subjects = []
    for i in _participant_ids:
        subjects.append(f"{i}")
    return subjects

def get_property(base_path: Path, load_only_completed: bool, property_name: str, cols):

    property_filters = {'uebergabe': 'uebergabe', 
                        'radar': 'messung_radar',
                        'karnofsky_index': 'ground_truth_karnofsky_index',
                        'epa': 'ground_truth_epa',
                        'palliativphase': 'ground_truth_palliativphase',
                        'demand_medication': 'ground_truth_bedarfsgabe',
                        'ipos_external': 'ground_truth_fremderfassung_ipos_0208',
                        'intervention': 'ground_truth_komplexe_interventionen',
                        'medication': 'regelmedikation',
                        'progress': 'verlaufsbericht', 
                        'vital_param': 'ground_truth_vitalparamater',
                        'empatica': 'messung_empatica', 
                        'tfc': 'messung_tfc',
                        'ipos_self': 'ground_truth_selbsterfassung_ipos'
    }

    data = get_raw_data_file(base_path=base_path, load_only_completed=load_only_completed)

    instrument = property_filters.get(property_name)

    if instrument:
        data = data.loc[data.redcap_repeat_instrument == instrument, cols]
    else:
        data = data.loc[:, cols]

    #remove NaN rows
    data = data.dropna(
        subset=[col for col in data.columns if col != "record_id"],
        how="all"
    )

    return data

def find_after(data: pd.DataFrame, reference_property_col: str, event_property_col: str, time_window: int) -> pd.DataFrame:

    print("Reference: ", reference_property_col)
    print("Event: ", event_property_col)

    #convert objects to time entries
    data[reference_property_col] = pd.to_datetime(data[reference_property_col])
    data[event_property_col] = pd.to_datetime(data[event_property_col])

    # checks for the last time end_time_property occurs + first time event_property occurs
    participants = (
        data.groupby("record_id")
        .agg({
            reference_property_col : "max",
            event_property_col : "min"
        }).reset_index()
    )

    time_diff = participants[event_property_col] - participants[reference_property_col]

    return participants[
        time_diff.between(
            pd.Timedelta(0),
            pd.Timedelta(minutes=time_window)
        )
    ]
