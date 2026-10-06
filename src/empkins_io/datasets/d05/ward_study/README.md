# WardStudyDataset Overview

`WardStudyDataset` provides access to the different study instruments and helper functions for filtering, querying, and summarizing the dataset.

## Usage

- Import WardStudyDataset:

```python
from empkins_io.datasets.d05.ward_study import WardStudyDataset
```

- Create WardStudyDataset:

```python
"""
Parameters:
    base_path: path to the csv file.
    load_only_completed: Boolean to specify whether all data (False) or only the completed entries (True) should be loaded.
"""

tpcp_dataset = WardStudyDataset(base_path=tpcp_data_path, load_only_completed=False)
```

- Optional: Create subset:

```python
"""
If a subset is created all properties except 

- raw_data
- total_participants
- instruments
- radar_total_worn_time
- radar_mean_worn_time
- empatica_total_worn_time
- empatica_mean_worn_time
- demand_medication_total
- Multi-Instrument Properties
- total_empatica_measurements
- total_radar_measurements
- total_tfc_measurements
- age_statistics
- sex_statistics
- study_end_overview
- death_sanity_check

return only the subject/s specific data.

Parameters:
    subject: The subject identifier for which the subject should be created.
"""

subset_data = tpcp_dataset.get_subset(record_id=["1"])
```

## Dataset Access

These properties return the data for the corresponding instrument.

| Property | Description |
|---|---|
| `raw_data` | Returns raw data. |
| **General Properties** |  |
| `total_participants` | Returns the number of participants. |
| `instruments` | Returns all RedCap columns. |
| `record_ids` | Returns the record_ids of the dataset. |
| **RedCap Instruments** |  |
| `uebergabe` | Returns handover data. |
| `uebersicht` | Returns overview data. |
| `radar` | Returns radar measurements. |
| `empatica` | Returns Empatica measurements. |
| `tfc` | Returns TFC measurements. |
| `karnofsky_index` | Returns Karnofsky index data. |
| `epa` | Returns EPA data. |
| `palliativphase` | Returns palliative phase data. |
| `vital_parameters` | Returns vital parameter measurements. |
| `demand_medication` | Returns demand medication entries. |
| `ipos_self` | Returns self-reported IPOS data. |
| `ipos_external` | Returns externally assessed IPOS data. |
| `intervention` | Returns complex intervention data. |
| `death` | Returns death information. |
| `saliva_base` | Returns saliva base data. |
| `saliva_symptom` | Returns saliva symptom data. |
| `medication` | Returns regular medication data. |
| `progress` | Returns progress report data. |
| **Radar Specific** |  |
| `radar_total_worn_time_per_part` | Returns the total radar wear time per participants. |
| `radar_total_worn_time` | Returns the total radar wear time. |
| `radar_mean_worn_time` | Returns the mean radar wear time across all participants. |
| **Empatica Specific** |  |
| `empatica_total_worn_time_per_part` | Returns the total empatica wear time per participants. |
| `empatica_total_worn_time` | Returns the total empatica wear time. |
| `empatica_mean_worn_time` | Returns the mean empatica wear time across all participants. |
| **Demand Medication Specific** |  |
| `demand_medication_total` | Returns the total amount of demand medications documented. |
| `demand_medication_per_participant` | Returns the total amount of demand medications documented per participant. |
| **Additional Properties** |  |
| `diagnosis` | Returns diagnosis information. |
| `study_end_cause_information` | Returns study end cause information. |
| `consent` | Returns consent information. |
| **Multi-Instrument Properties** |  |
| `radar_empatica_ids` | Returns all record ids which have radar and empatica data. |
| `radar_tfc_ids` | Returns all record ids which have radar and tfc data. |
| `empatica_tfc_ids` | Returns all record ids which have empatica and tfc data. |
| `radar_empatica_intervals` | Returns all intervals during which radar and empatica data were recorded simultaneously. |
| **Statistical Analysis** |  |
| `total_empatica_measurements` | Returns the amount of participant where empatica data was recorded. |
| `total_radar_measurements` | Returns the amount of participant where radar data was recorded. |
| `total_tfc_measurements` | Returns the amount of participant where tfc data was recorded. |
| `age` | Returns the age per participant. |
| `age_statistics` | Returns the minimum, maximum and mean age over all participants. |
| `sex` | Returns the gender per participant. |
| `sex_statistics` | Returns overall gender distribution. |
| `study_end_overview` | Returns an overview of the study end causes and the corresponding record ids, i.e. which ids discharged, died, revocation, other. |
| **Debugging** |  |
| `unverified` | Returns all record ids which have unverified entries, together with the affected column. |
| `incomplete` | Returns all record ids which have incomplete entries, together with the affected column. |
| `death_sanity_check` | Returns the record ids and corresponding death information where study_ennd_cause is death but no death_time is documented. |


---

## Functions

### `get_subset(subject: list[str] | str)`

Creates a subset of the raw data which only contains data for the subjects specified.

```python
"""
Parameters:
    subject: The subject or subjects the subset should be created for.
"""
subset = tpcp_dataset.get_subset(subject=["EMP_10XX"])
```

### `cut(start_time=None, end_time=None)`

Restricts all time-dependent properties to a given time window.

For measurements with a start and end time, entries are extracted if they overlap with the requested time window.

For measurements with a single timestamp, entries are extracted if the timestamp lies inside the requested time window.

If both parameters are set to None, there is no restriction for the time-dependent properties. If one of the parameters is None, the data will only be restricted by the parameter which is not none.

```python
# Restricts all time-dependent properties to the given time window.
dataset.cut(
    start_time="2026-01-01",
    end_time="2026-02-01",
)
# Restricts all time-dependent properties so that entries which end before start_time are filtered. Entries which start before start_time but end after start_time are included.
dataset.cut(
    start_time="2026-01-01",
    end_time=None,
)
```

### `find_all_events_after_reference(reference: str, event: str, time_window: int, reference_time: str, event_time: str) -> pd.DataFrame`

Returns all occurrences of an event that happened within a certain time window after the reference.

```python
"""
Parameters:
    reference: Reference event from which the occurence of the event is checked.
    event: Event for which the occurence is checked within the time_window. 
    time_window: Length of the time window in minutes.
    reference_time: time-property marking the reference time, e.g. "start", "end", "time".
    event_time: time-property marking the event time, e.g. "start", "end", "time".
"""
tpcp_dataset.find_all_events_after_reference(reference="radar", event="death", time_window=1440, reference_time="end", event_time="time")
```

### `get_instrument_time_window(instrument: str) -> pd.DataFrame`

Returns the start & end time for the specified instrument. Returns only entries where both a valid start & end time exist. 

```python
"""
Parameters:
    instrument: Instrument for which the start & end time is requested, e.g. "radar", "empatica", "tfc".
"""
tpcp_dataset.get_instrument_time_window("radar")
```

### `get_record_ids_with_data(*instruments: str) -> pd.DataFrame`

Returns the ids, where data for the specified instruments was recorded, e.g. ids for which both radar and empatica data were recorded.

```python
"""
Parameters:
    instrument: Instruments which should be checked, e.g. "radar", "empatica", "tfc".
"""
tpcp_dataset.get_record_ids_with_data("empatica", "radar")
```

### `get_overlapping_intervals(*instruments: str) -> pd.DataFrame`

Returns the overlapping intervals, where data for all the specified instruments was recorded, e.g. overlapping interval in which both radar and empatica data were recorded.

```python
"""
Parameters:
    instrument: Instruments for which overlapping data availability should be checked, e.g. "radar", "empatica", "tfc".
"""
tpcp_dataset.get_overlapping_intervals("empatica", "radar")
```

### `get_record_ids_with_diagnosis(diagnosis: str, specific_diagnosis: str)`

Returns the record_id of all participants who had a specific diagnosis.

```python
"""
Parameters:
    diagnosis: The type of diagnosis, either "add_diagnosis" for secondary diagnosis or "heart" for heart diseases.
    specific_diagnosis: The specific diagnosis within the diagnosis category. Possible options, status 15.09.26 are:
        - add_diagnosis: "visual", "arrhythmias", "insufficiency", "pulmonary", "kidney", "psychiatric"
        - heart: "aFib", "av", "vitien", "pacemaker"
"""
heart_diagnosis_ids = tpcp_dataset.get_record_ids_with_diagnosis("heart", "av")
```

### `get_record_ids_with_device(instrument: str, device_name: str)`

Returns the record_id of all participants who had a specific device.

```python
"""
Parameters:
    instrument: The type of device, either "radar" or "empatica".
    device_name: The name of the device. Possible options, status 09.09.26 are:
        - radar: "Master_id_4", "Master_id_5", "Master_id_6", "Master_id_7", "Master_id_8", "Master_id_10", "Master_id_11"
        - empatica: "C9", "TM"
"""
tpcp_dataset.get_record_ids_with_device("radar", "Master_id_4")
```

### `get_device_for_id(instrument: str, query_id: int)`

Returns the device for a specific participant id.

```python
"""
Parameters:
    instrument: The type of device, either "radar" or "empatica".
    query_id: The record id of the participant.
"""
tpcp_dataset.get_device_for_id("radar", 62)
```