from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tpcp import Dataset
import warnings

from empkins_io.utils._types import path_t

from .helper import (
    get_raw_data_file,
    calculate_participant_ids,
    get_property,
    find_after
)

from ._columns import (
    # Completed cols
    COMPLETED_COLS,
    # All cols
    UEBERGABE_COLS,
    UEBERSICHT_COLS,
    MESSUNG_RADAR_COLS,
    EMPATICA_COLS,
    TFC_COLS,
    KARNOFSKY_COLS,
    EPA_COLS,
    PALLIATIVPHASE_COLS,
    VITAL_PARAM_COLS,
    DEMAND_MEDICATION_COLS,
    IPOS_SELF_COLS,
    IPOS_E_COLS,
    INTERVENTION_COLS,
    TIME_OF_DEATH_COLS,
    SALIVA_BASE_COLS,
    SALIVA_SYMPTOM_COLS,
    MEDICATION_COLS,
    PROGRESS_COLS,
    # Specific properties
    DIAGNOSIS_COLS,
    STUDY_INFO_COLS,
    CONSENT_COLS,
)


class WardStudyDataset(Dataset):
    base_path: path_t

    TIME_PROPERTIES = {
        "radar": {
            "start": "radar_start_time",
            "end": "radar_end_time"
        },
        "empatica": {
            "start": "empatica_start_time",
            "end": "empatica_end_time"
        },      
        "tfc": {
            "start": "tfc_start_time",
            "end": "tfc_end_time"
        },
        # medication only has date without exact time, time in form of numbers, e.g. 1-0-1-0
        "medication": {
            "start": "regular_med_time_start",
            "end": "regular_med_time_end"
        },   
        "intervention": {
            "start": "complex_int_start_time",
            "end": "complex_int_end_time"
        },
        "consent": {
            #this only has a date without exact time!!!
            "time": "consent_date"
        },
        "study": {
            "time": "study_end_time"
        },   
        "karnofsky": {
            "time": "karnofsky_time"
        },
        "epa": {
            "time": "epa_time"
        },
        "palliativ": {
            "time": "palliative_phase_time"
        },
        "vital_parameters": {
            "time": "vitalparam_time"
        },
        "demand_medication": {
            "time": "ondemand_time"
        },  
        "ipos_self": {
            "time": "ipos_i_time" 
        },
        "ipos_external": {
            "time": "ipos_e_time" 
        },  
        "death": {
            "time": "death_time" 
        },     
        "progress": {
            "time": "progress_report_time" 
        }
    }

    INSTRUMENT_DEVICES = {
        "radar": "radar_device",
        "empatica": "empatica_device"
    }

    DEVICE_NR_RADAR = {
        "Master_id_4": 1.0,
        "Master_id_5": 3.0,
        "Master_id_6": 4.0,
        "Master_id_7": 6.0,
        "Master_id_8": 2.0,
        "Master_id_10": 7.0,
        "Master_id_11": 5.0,
    }

    DEVICE_NR_EMPATICA = {
        "C9": 1.0,
        "TM": 2.0,
    }

    DIAGNOSIS_CATEGORY = {
        "add_diagnosis": "cat_add_diagnosis___",
        "heart": "cat_heart_diagnosis___"
    }

    ADD_DIAGNOSIS_CATEGORY = {
        "visual": {
            "Seheinschränkungen": 1.0,
        },
        "arrhythmia": {
            "Herzrhythmusstörungen": 2.0,
        },
        "insufficiency": {
            "Herzinsuffizienz": 3.0,
        },
        "Pulmonary": {
            "Pulmonale Erkrankungen": 4.0,
        },
        "kidney": {
            "Niereninsuffizienz": 5.0,
        },  
        "psychiatric": {
            "Psychiatrische Vorerkrankungen: Depression oder Angststörungen": 6.0,
        },        
    }

    HEART_DIAGNOSIS_CATEGORY = {
        "aFib": {
            "Vorhofflimmern": 1.0,
        },
        "av": {
            "AV-Block": 2.0,
        },
        "vitien": {
            "Vitien": 3.0,
        },
        "pacemaker": {
            "Herzschrittmacher": 4.0,
        }
    }

    # Completed cols
    COMPLETED_COLS = COMPLETED_COLS
    # All cols
    UEBERGABE_COLS = UEBERGABE_COLS
    UEBERSICHT_COLS = UEBERSICHT_COLS
    MESSUNG_RADAR_COLS = MESSUNG_RADAR_COLS
    EMPATICA_COLS = EMPATICA_COLS
    TFC_COLS = TFC_COLS
    KARNOFSKY_COLS = KARNOFSKY_COLS
    EPA_COLS = EPA_COLS
    PALLIATIVPHASE_COLS = PALLIATIVPHASE_COLS
    VITAL_PARAM_COLS =VITAL_PARAM_COLS
    DEMAND_MEDICATION_COLS = DEMAND_MEDICATION_COLS
    IPOS_SELF_COLS = IPOS_SELF_COLS
    IPOS_E_COLS = IPOS_E_COLS
    INTERVENTION_COLS = INTERVENTION_COLS
    TIME_OF_DEATH_COLS = TIME_OF_DEATH_COLS
    SALIVA_BASE_COLS = SALIVA_BASE_COLS
    SALIVA_SYMPTOM_COLS = SALIVA_SYMPTOM_COLS
    MEDICATION_COLS = MEDICATION_COLS
    PROGRESS_COLS = PROGRESS_COLS
    # Specific properties
    DIAGNOSIS_COLS = DIAGNOSIS_COLS
    STUDY_INFO_COLS = STUDY_INFO_COLS
    CONSENT_COLS = CONSENT_COLS
   

    def __init__(
        self,
        base_path: path_t,
        load_only_completed: bool,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ):

        self.base_path = base_path
        self.load_only_completed = load_only_completed
        self.start_time = start_time
        self.end_time = end_time

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:

        subjects = calculate_participant_ids(self.base_path, self.load_only_completed)
        index = pd.DataFrame(subjects, columns=["record_id"])

        return index

    def filter_data_to_subset(self, data: pd.DataFrame) -> pd.DataFrame:
        # No subset -> return everything
        if self.index_is_unchanged:
            return data

        new_data = data.copy()
        new_subset = self.index.copy()

        index_cols = self.index.columns.tolist()

        # Match the raw-data dtype to the index dtype
        for col in index_cols:
            if col == "record_id":
                new_data[col] = new_data[col].astype(str)
                new_subset[col] = new_subset[col].astype(str)

        return new_data.merge(
            new_subset[index_cols].drop_duplicates(),
            on=index_cols,
            how="inner",
        )


    def cut(self, start_time: str, end_time: str):
        """
        Restricts all time-dependent properties to a given time window.
        """

        dataset = self.clone().set_params(
            start_time = pd.to_datetime(start_time),
            end_time = pd.to_datetime(end_time),
        )

        return dataset

    def _cut_data(self, data: pd.DataFrame, instrument: str) -> pd.DataFrame:

        if self.start_time is None and self.end_time is None:
            return data 

        data = data.copy()

        time_properties = self.TIME_PROPERTIES[instrument]

        if "start" in time_properties and "end" in time_properties:
            start_time_property = time_properties["start"]
            end_time_property = time_properties["end"]

            #convert objects to time entries
            data[start_time_property] = pd.to_datetime(data[start_time_property])
            data[end_time_property] = pd.to_datetime(data[end_time_property])

            mask = pd.Series(True, index=data.index)

            if self.start_time is not None:
                mask &= (
                    (data[end_time_property].isna())
                    | (data[end_time_property] >= self.start_time)
                )

            if self.end_time is not None:
                mask &= data[start_time_property] <= self.end_time

        # Instrument has one timestamp
        elif "time" in time_properties:
            time_col = time_properties["time"]

            data[time_col] = pd.to_datetime(
                data[time_col],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )

            mask = pd.Series(True, index=data.index)

            if self.start_time is not None:
                mask &= data[time_col] >= self.start_time

            if self.end_time is not None:
                mask &= data[time_col] <= self.end_time

        else:
            raise ValueError(f"No time for instrument: '{instrument}'")

        return data.loc[mask]

  
    def find_all_events_after_reference(
            self,
            reference: str, 
            event: str, 
            time_window: int,
            reference_time: str,
            event_time: str
        ) -> pd.DataFrame:
        """
        Returns all occurrences of an event that happened within a certain time_window after the reference.

        Parameters:
            reference: Reference event from which the occurence of the event is checked.
            event: Event for which the occurence is checked within the time_window. 
            time_window: Length of the time window in minutes.
            reference_time: time-property marking the reference time, e.g. "start", "end", "time".
            event_time: time-property marking the event time, e.g. "start", "end", "time".
        """

        reference_property = self.TIME_PROPERTIES[reference]
        event_property = self.TIME_PROPERTIES[event]

        if reference_time not in reference_property:
            raise ValueError(
                f"{reference!r} has no '{reference_time}'-timestamp."
            )

        if event_time not in event_property:
            raise ValueError(
                f"{event!r} has no '{event_time}'-timestamp."
            )

        return find_after(self.raw_data, reference_property[reference_time], event_property[event_time], time_window)


    def get_instrument_time_window(self, instrument: str) -> pd.DataFrame:
        """
        Returns the start & end time for the specified instrument. Returns only entries where both a valid start & end time exist. 
        TODO: Only if a valid start & end date exists? 

        Parameters:
            instrument: Instrument for which the start & end time is requested, e.g. "radar", "empatica", "tfc".
        """

        data = get_raw_data_file(
            base_path=self.base_path,
            load_only_completed=self.load_only_completed,
        )

        if instrument not in self.TIME_PROPERTIES:
            raise ValueError(f"{instrument} is not a time property.")

        property = self.TIME_PROPERTIES[instrument]

        if "start" not in property or "end" not in property:
            raise ValueError(
                f'{instrument} does not contain start or end. Choose valid instrument, e.g. "radar", "empatica", "tfc".'
            )

        start_col = property["start"]
        end_col = property["end"]

        filtered_data = data.loc[data[start_col].notna() & data[end_col].notna(),
                        ["record_id", start_col, end_col]]

        return self.filter_data_to_subset(data=filtered_data)


    def get_record_ids_with_data(
        self,
        *instruments: str,
    ) -> pd.DataFrame:
        """
        Returns the ids, where data for the specified *instruments was recorded, e.g. ids for which both radar and empatica data were recorded.
        TODO: Only if a valid start & end date exists? 

        Parameters:
            instrument: Instruments which should be checked, e.g. "radar", "empatica", "tfc".
        """
        data = get_raw_data_file(
            base_path=self.base_path,
            load_only_completed=self.load_only_completed,
        )

        valid_ids = None

        for instrument in instruments:
            if instrument not in self.TIME_PROPERTIES:
                raise ValueError(f"{instrument} is not a time property.")

            start_col = self.TIME_PROPERTIES[instrument]["start"]
            end_col = self.TIME_PROPERTIES[instrument]["end"]

            ids = data.loc[
                data[start_col].notna() & data[end_col].notna(),
                "record_id"
            ].unique()

            if valid_ids is None:
                valid_ids = ids
            else:
                valid_ids = np.intersect1d(valid_ids, ids)

        return pd.DataFrame({
            "record_id": valid_ids
        })

    def get_overlapping_intervals(
        self,
        *instruments: str,
    ) -> pd.DataFrame:
        """
        Returns the overlapping intervals, where data for all the specified *instruments was recorded, e.g. overlapping interval in which both radar and empatica data were recorded.
        TODO: Only if a valid start & end date exists? 

        Parameters:
            instrument: Instruments for which overlapping data availability should be checked, e.g. "radar", "empatica", "tfc".
        """

        if len(instruments) < 2 | len(instruments) > 3:
            raise ValueError(f"Too few/many instruments: Provide either 2 or 3 instruments.")

        data = get_raw_data_file(base_path=self.base_path, load_only_completed=self.load_only_completed)

        def get_intervals(instrument: str) -> pd.DataFrame:
            if instrument not in self.TIME_PROPERTIES:
                raise ValueError(f"{instrument!r} is not a time property.")

            start_col = self.TIME_PROPERTIES[instrument]["start"]
            end_col = self.TIME_PROPERTIES[instrument]["end"]

            intervals = data.loc[
                data[start_col].notna() & data[end_col].notna(),
                ["record_id", start_col, end_col],
            ].copy()

            intervals[start_col] = pd.to_datetime(intervals[start_col])
            intervals[end_col] = pd.to_datetime(intervals[end_col])

            return intervals.rename(
                columns={
                    start_col: "start",
                    end_col: "end",
                }
            )

        first_interval = get_intervals(instruments[0])

        for instrument in instruments[1:]:
            next_interval = get_intervals(instrument)

            merged = first_interval.merge(
                next_interval,
                on="record_id",
                suffixes=("_current", "_next"),
            )

            merged["start"] = merged[
                ["start_current", "start_next"]
            ].max(axis=1)

            merged["end"] = merged[
                ["end_current", "end_next"]
            ].min(axis=1)

            first_interval = merged.loc[
                merged["start"] <= merged["end"],
                ["record_id", "start", "end"],
            ].drop_duplicates()

        return (
            first_interval
            .sort_values(["record_id", "start"])
            .reset_index(drop=True)
        )

        
    def get_record_ids_with_diagnosis(self, diagnosis: str, specific_diagnosis: str):
        """
        Returns the record_id of all participants who had a specific diagnosis.

        Parameters:
            diagnosis: The type of diagnosis, either "add_diagnosis" for secondary diagnosis or "heart" for heart diseases.
            specific_diagnosis: The specific diagnosis within the diagnosis category. Possible options, status 15.09.26 are:
                - add_diagnosis: "visual", "arrhythmias", "insufficiency", "pulmonary", "kidney", "psychiatric"
                - heart: "aFib", "av", "vitien", "pacemaker"
        """
        add_diag_options = ["visual", "arrhythmia", "insufficiency", "pulmonary", "kidney", "psychiatric"]
        heart_diag_options = ["aFib", "av", "vitien", "pacemaker"]

        data = get_property(self.base_path, self.load_only_completed, "uebersicht", self.UEBERSICHT_COLS)
        diag_col = self.DIAGNOSIS_CATEGORY[diagnosis]

        if diagnosis == "add_diagnosis":
            if specific_diagnosis not in add_diag_options:
                raise ValueError(f'For diagnosis {diagnosis}: Only "visual", "arrhythmias", "insufficiency", "pulmonary", "kidney", "psychiatric" allowed.')
            diagnosis_name, diagnosis_nr = next(iter(self.ADD_DIAGNOSIS_CATEGORY[specific_diagnosis].items()))
        elif diagnosis == "heart":
            if specific_diagnosis not in heart_diag_options:
                raise ValueError(f'For diagnosis {diagnosis}: Only "aFib", "av", "vitien", "pacemaker" allowed.')
            diagnosis_name, diagnosis_nr = next(iter(self.HEART_DIAGNOSIS_CATEGORY[specific_diagnosis].items()))
        else: 
            raise ValueError(f"No diagnosis for {diagnosis}")

        diag_col = diag_col + str(int(diagnosis_nr))

        # Return Array with [diagnosis_name, [record_ids]]
        record_ids = data.loc[
            data[diag_col] == 1.0,
            "record_id"
        ].unique()

        return [diagnosis_name, record_ids.tolist()]


    def get_record_ids_with_device(self, instrument: str, device_name: str):
        """
        Returns the record_id of all participants who had a specific device.

        Parameters:
            instrument: The type of device, either "radar" or "empatica".
            device_name: The name of the device. Possible options, status 09.09.26 are:
                - radar: "Master_id_4", "Master_id_5", "Master_id_6", "Master_id_7", "Master_id_8", "Master_id_10", "Master_id_11"
                - empatica: "C9", "TM"
        """
        radar_options = ["Master_id_4", "Master_id_5", "Master_id_6", "Master_id_7", "Master_id_8", "Master_id_10", "Master_id_11"]
        empatica_options = ["C9", "TM"]

        if instrument == "radar":
            data = get_property(self.base_path, self.load_only_completed, "radar", self.MESSUNG_RADAR_COLS)
            dev_col = self.INSTRUMENT_DEVICES[instrument]
            if device_name not in radar_options:
                raise ValueError(f"For instrument {instrument}: Only radar with device name Master_id_ 4-8 or 10-11 allowed.")
            dev_nr = self.DEVICE_NR_RADAR[device_name]
        elif instrument == "empatica":
            data = get_property(self.base_path, self.load_only_completed, "empatica", self.EMPATICA_COLS)
            dev_col = self.INSTRUMENT_DEVICES[instrument]
            if device_name not in empatica_options:
                raise ValueError(f"For instrument {instrument}: Only empatica with device name \"C9\" and \"TM\" allowed.")
            dev_nr = self.DEVICE_NR_EMPATICA[device_name]
        else: 
            raise ValueError(f"No device for {instrument}")

        # Return Array with [radar_device, [record_ids]]
        record_ids = data.loc[
            data[dev_col] == dev_nr,
            "record_id"
        ].unique()

        return [device_name, record_ids.tolist()]

    def get_device_for_id(self, instrument: str, query_id: int):
        """
        Returns the device for a specific participant id.

        Parameters:
            instrument: The type of device, either "radar" or "empatica".
            query_id: The record id of the participant.
        """

        if instrument == "radar":
            data = get_property(self.base_path, self.load_only_completed, "radar", self.MESSUNG_RADAR_COLS)
            device_mapping = self.DEVICE_NR_RADAR
        elif instrument == "empatica":
            data = get_property(self.base_path, self.load_only_completed, "empatica", self.EMPATICA_COLS)
            device_mapping = self.DEVICE_NR_RADAR
        else: 
            raise ValueError(f"No device for {instrument}")

        id_specific_data = data[data["record_id"] == query_id].copy()
        dev_col = self.INSTRUMENT_DEVICES[instrument]

        reverse_mapping = {
            value: key
            for key, value in device_mapping.items()
        }

        id_specific_data["device_name"] = id_specific_data[dev_col].map(reverse_mapping)

        return id_specific_data[["record_id", dev_col, "device_name"]]

    # ======================================================================
    #                           PROPERTIES
    # ======================================================================

    @property
    def raw_data(self) -> pd.DataFrame:
        return get_raw_data_file(self.base_path, self.load_only_completed)

    # --------------------------------------
    #   General Information regarding Data
    # --------------------------------------

    @property
    def total_participants(self):
        subjects = calculate_participant_ids(self.base_path, self.load_only_completed)
        return len(subjects)

    @property
    def instruments(self) -> list[str]:
        return list(self.raw_data.columns)

    @property
    def record_ids(self):
        record_ids = self.index["record_id"].unique()
        return record_ids

    # --------------------------------------
    #       RedCap Instruments
    # --------------------------------------

    @property
    def uebergabe(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebergabe", self.UEBERGABE_COLS)
        return self.filter_data_to_subset(data=data)

    @property
    def uebersicht(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", self.UEBERSICHT_COLS)
        return self.filter_data_to_subset(data=data)

    @property
    def radar(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "radar", self.MESSUNG_RADAR_COLS)

        data = self._cut_data(
            data,
            instrument="radar"
        )

        return self.filter_data_to_subset(data=data)

    @property
    def empatica(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "empatica", self.EMPATICA_COLS)

        data = self._cut_data(
                data,
                instrument="empatica"
        )
        
        return self.filter_data_to_subset(data=data)

    @property
    def tfc(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "tfc", self.TFC_COLS)

        data = self._cut_data(
                data,
                instrument="tfc"
        )
        
        return self.filter_data_to_subset(data=data)

    @property
    def karnofsky_index(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "karnofsky_index", self.KARNOFSKY_COLS)

        data = self._cut_data(
                    data,
                    instrument="karnofsky"
                )
        
        return self.filter_data_to_subset(data=data)

    @property
    def epa(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "epa", self.EPA_COLS)

        data = self._cut_data(
                    data,
                    instrument="epa"
                )
        
        return self.filter_data_to_subset(data=data)

    @property
    def palliativphase(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "palliativphase", self.PALLIATIVPHASE_COLS)

        data = self._cut_data(
                    data,
                    instrument="palliativ"
                )

        return self.filter_data_to_subset(data=data)

    @property
    def vital_parameters(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "vital_param", self.VITAL_PARAM_COLS)

        data = self._cut_data(
                    data,
                    instrument="vital_parameters"
                )
        
        return self.filter_data_to_subset(data=data)
  
    @property
    def demand_medication(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "demand_medication", self.DEMAND_MEDICATION_COLS)

        data = self._cut_data(
                    data,
                    instrument="demand_medication"
                )
        
        return self.filter_data_to_subset(data=data)
        
    @property
    def ipos_self(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "ipos_self", self.IPOS_SELF_COLS)

        data = self._cut_data(
                    data,
                    instrument="ipos_self"
                )

        return self.filter_data_to_subset(data=data)
        
    @property
    def ipos_external(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "ipos_external", self.IPOS_E_COLS)

        data = self._cut_data(
                    data,
                    instrument="ipos_external"
                )
        
        return self.filter_data_to_subset(data=data)

    @property
    def intervention(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "intervention", self.INTERVENTION_COLS)

        data = self._cut_data(
                    data,
                    instrument="intervention"
                )

        return self.filter_data_to_subset(data=data)

    @property
    def death(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "death", self.TIME_OF_DEATH_COLS)

        data = self._cut_data(
                    data,
                    instrument="death"
                )

        return self.filter_data_to_subset(data=data)  

    @property
    def saliva_base(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "saliva", self.SALIVA_BASE_COLS)
        return self.filter_data_to_subset(data=data)

    @property
    def saliva_symptom(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "saliva", self.SALIVA_SYMPTOM_COLS)
        return self.filter_data_to_subset(data=data)

    @property
    def medication(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "medication", self.MEDICATION_COLS)

        data = self._cut_data(
                    data,
                    instrument="medication"
                )
        
        return self.filter_data_to_subset(data=data)

    @property
    def progress(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "progress", self.PROGRESS_COLS)

        data = self._cut_data(
                    data,
                    instrument="progress"
                )
        
        return self.filter_data_to_subset(data=data)

    # --------------------------------------
    #   Radar Specific Properties
    # --------------------------------------

    @property
    def radar_total_worn_time_per_part(self) -> pd.DataFrame:
        data = self.radar.copy()
        data["radar_start_time"] = pd.to_datetime(data["radar_start_time"])
        data["radar_end_time"] = pd.to_datetime(data["radar_end_time"])

        time_diff = data["radar_end_time"] - data["radar_start_time"] 

        data["time_diff"] = (
            time_diff
        )

        return (
            data.groupby("record_id")["time_diff"]
            .sum()
            .reset_index()
        )

    @property
    def radar_total_worn_time(self) -> float:
        data = get_property(self.base_path, self.load_only_completed, "radar", self.MESSUNG_RADAR_COLS)

        data["radar_start_time"] = pd.to_datetime(data["radar_start_time"])
        data["radar_end_time"] = pd.to_datetime(data["radar_end_time"])

        time_diff = data["radar_end_time"] - data["radar_start_time"] 

        data["time_diff"] = (
            time_diff
        )

        data = data.groupby("record_id")["time_diff"].sum().reset_index()

        # Gives a warning and skips entries which have a negative time difference 
        negative_time_values = data[data["time_diff"] < pd.Timedelta(0)]
        for record_id in negative_time_values["record_id"]:
            print(
                f"\033[93mWARNING: The time duration that participant \033[0m "
                f"\033[91m{record_id}\033[0m "
                f"\033[93m has worn radar is negative and will be ignored!\033[0m"
            )

        # Currently filters the entries which are potentially wrong
        data = data[data["time_diff"] > pd.Timedelta(0)]
        mean = data["time_diff"].sum()
        return mean

    @property
    def radar_mean_worn_time(self) -> float:
        data = get_property(self.base_path, self.load_only_completed, "radar", self.MESSUNG_RADAR_COLS)
        
        data["radar_start_time"] = pd.to_datetime(data["radar_start_time"])
        data["radar_end_time"] = pd.to_datetime(data["radar_end_time"])

        time_diff = data["radar_end_time"] - data["radar_start_time"] 

        data["time_diff"] = (
            time_diff
        )

        data = data.groupby("record_id")["time_diff"].sum().reset_index()

        # Gives a warning and skips entries which have a negative time difference 
        negative_time_values = data[data["time_diff"] < pd.Timedelta(0)]
        for record_id in negative_time_values["record_id"]:
            print(
                f"\033[93mWARNING: The time duration that participant \033[0m "
                f"\033[91m{record_id}\033[0m "
                f"\033[93m has worn empatica is negative and will be ignored!\033[0m"
            )

        # Currently filters the entries which are potentially wrong
        data = data[data["time_diff"] > pd.Timedelta(0)]
        mean = data["time_diff"].mean()
        return mean


    # --------------------------------------
    #   Empatica Specific Properties
    # --------------------------------------

    @property
    def empatica_total_worn_time_per_part(self):
        data = self.empatica.copy()
        data["empatica_start_time"] = pd.to_datetime(data["empatica_start_time"])
        data["empatica_end_time"] = pd.to_datetime(data["empatica_end_time"])

        time_diff = data["empatica_end_time"] - data["empatica_start_time"] 

        data["time_diff"] = (
            time_diff
        )

        return (
            data.groupby("record_id")["time_diff"]
            .sum()
            .reset_index()
        )

    @property
    def empatica_total_worn_time(self) -> float:
        data = get_property(self.base_path, self.load_only_completed, "empatica", self.EMPATICA_COLS)
        data["empatica_start_time"] = pd.to_datetime(data["empatica_start_time"])
        data["empatica_end_time"] = pd.to_datetime(data["empatica_end_time"])

        time_diff = data["empatica_end_time"] - data["empatica_start_time"] 

        data["time_diff"] = (
            time_diff
        )

        data = data.groupby("record_id")["time_diff"].sum().reset_index()

        # Gives a warning and skips entries which have a negative time difference 
        negative_time_values = data[data["time_diff"] < pd.Timedelta(0)]
        for record_id in negative_time_values["record_id"]:
            print(
                f"\033[93mWARNING: The time duration that participant \033[0m "
                f"\033[91m{record_id}\033[0m "
                f"\033[93m has worn empatica is negative and will be ignored!\033[0m"
            )

        # Currently filters the entries which are potentially wrong
        data = data[data["time_diff"] > pd.Timedelta(0)]
        mean = data["time_diff"].sum()
        return mean

    @property
    def empatica_mean_worn_time(self) -> float:
        data = get_property(self.base_path, self.load_only_completed, "empatica", self.EMPATICA_COLS)
        data["empatica_start_time"] = pd.to_datetime(data["empatica_start_time"])
        data["empatica_end_time"] = pd.to_datetime(data["empatica_end_time"])

        time_diff = data["empatica_end_time"] - data["empatica_start_time"] 

        data["time_diff"] = (
            time_diff
        )

        data = data.groupby("record_id")["time_diff"].sum().reset_index()

        # Gives a warning and skips entries which have a negative time difference 
        negative_time_values = data[data["time_diff"] < pd.Timedelta(0)]
        for record_id in negative_time_values["record_id"]:
            print(
                f"\033[93mWARNING: The time duration that participant \033[0m "
                f"\033[91m{record_id}\033[0m "
                f"\033[93m has worn empatica is negative and will be ignored!\033[0m"
            )

        # Currently filters the entries which are potentially wrong
        data = data[data["time_diff"] > pd.Timedelta(0)]
        mean = data["time_diff"].mean()
        return mean

    # --------------------------------------
    #   TFC Specific Properties
    # --------------------------------------

    # --------------------------------------
    #   Demand Medication Specific Properties
    # --------------------------------------

    @property
    def demand_medication_total(self) -> int:
        data = get_property(self.base_path, self.load_only_completed, "demand_medication", self.DEMAND_MEDICATION_COLS)
        return data.shape[0]

    @property 
    def demand_medication_per_participant(self) -> pd.DataFrame:
        return self.demand_medication.groupby("record_id").size().reset_index(name="amount")


    # --------------------------------------
    #       Additional Properties
    # --------------------------------------

    @property
    def diagnosis(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", self.DIAGNOSIS_COLS)
        return self.filter_data_to_subset(data=data)

    @property
    def study_end_cause_information(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", self.STUDY_INFO_COLS)

        data = self._cut_data(
            data,
            instrument="study"
        )

        return self.filter_data_to_subset(data=data)

    @property
    def consent(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", self.CONSENT_COLS)

        data = self._cut_data(
            data,
            instrument="consent"
        )

        return self.filter_data_to_subset(data=data)

    # --------------------------------------
    #   Multi-Instrument Properties
    # --------------------------------------

    @property
    def radar_empatica_ids(self):
        return self.get_record_ids_with_data(
            "radar",
            "empatica"
        )

    @property
    def radar_tfc_ids(self):
        return self.get_record_ids_with_data(
            "radar",
            "tfc"
        )

    @property
    def empatica_tfc_ids(self):
        return self.get_record_ids_with_data(
            "empatica",
            "tfc"
        )

    @property
    def radar_empatica_intervals(self):
        return self.get_overlapping_intervals(
            "radar",
            "empatica"
        )

    # --------------------------------------
    #       Statistical Analysis
    # --------------------------------------

    @property
    def total_empatica_measurements(self):
        # Also include entries which are not completed yet
        data = get_property(self.base_path, self.load_only_completed, "empatica", self.EMPATICA_COLS)
        
        data = self._cut_data(
                data,
                instrument="empatica"
        )

        count = len(data[data["messung_empatica_complete"] != 1.0]["record_id"].unique())
        return count

    @property
    def total_radar_measurements(self):
        # Also include entries which are not completed yet
        data = get_property(self.base_path, self.load_only_completed, "radar", self.MESSUNG_RADAR_COLS)
        
        data = self._cut_data(
            data,
            instrument="radar"
        )
        count = len(data[data["messung_radar_complete"] != 1.0]["record_id"].unique())
        return count

    @property
    def total_tfc_measurements(self):
        # Also include entries which are not completed yet
        data = get_property(self.base_path, self.load_only_completed, "tfc", self.TFC_COLS)
        
        data = self._cut_data(
                data,
                instrument="tfc"
        )
        count = len(data[data["messung_tfc_complete"] != 1.0]["record_id"].unique())
        return count

    @property
    def age(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", ["record_id", "age"])
        return self.filter_data_to_subset(data=data)

    @property
    def age_statistics(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", ["age"])
        age_range = pd.DataFrame({
            "age_min": [data.age.min()],
            "age_max": [data.age.max()],
            "age_mean": [data.age.mean()],
        })
        return age_range

    @property
    def sex(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", ["record_id", "sex"])
        data["gender"] = data["sex"].map({
            1.0: "female",
            2.0: "male",
        })
        return self.filter_data_to_subset(data=data)

    @property
    def sex_statistics(self) -> pd.DataFrame:
        #1 = female, 2 = male
        data = get_property(self.base_path, self.load_only_completed, "uebersicht", ["sex"])
        # data.sex.dtype: float64
        age_range = pd.DataFrame({
            "female": [(data.sex == 1.0).sum()],
            "male": [(data.sex == 2.0).sum()],
        })

        return age_range

    @property
    def study_end_overview(self) -> pd.DataFrame:
        data = get_raw_data_file(self.base_path, load_only_completed=self.load_only_completed)

        causes = {
            1.0: "discharged",
            2.0: "died",
            3.0: "revocation",
            4.0: "other"
        }

        rows = []

        for cause, name in causes.items():
            ids = (
                data.loc[data["study_end_cause"] == cause, "record_id"]
                .drop_duplicates()
                .astype("Int64")
                .tolist()
            )

            rows.append({
                "study_end_cause": cause,
                "cause": name,
                "count": len(ids),
                "record_ids": ids,
            })

        return pd.DataFrame(rows)

    # --------------------------------------
    #             Debugging
    # --------------------------------------

    @property
    def unverified(self):
        data = get_property(self.base_path, self.load_only_completed, "", COMPLETED_COLS)

        cols_to_check = [
            col for col in COMPLETED_COLS
            if col != "record_id"
        ]

        data["affected_cols"] = data[cols_to_check].apply(lambda row: row[row == 1.0].index.tolist(), axis=1)

        affected_cols_data = data.loc[
            data["affected_cols"].str.len() > 0,
            ["record_id", "affected_cols"]
        ]

        return self.filter_data_to_subset(affected_cols_data)

    @property
    def incomplete(self) -> pd.DataFrame:
        data = get_property(self.base_path, self.load_only_completed, "", COMPLETED_COLS)

        # Get only ids where uebersicht instrument is complete
        id_col = "record_id"
        gt_death_column = "ground_truth_todeszeitpunkt_complete"

        cols_to_check = [
            col for col in COMPLETED_COLS
            if col not in (id_col, gt_death_column)
        ]

        data["affected_cols"] = data[cols_to_check].apply(lambda row: row[row == 0.0].index.tolist(), axis=1)
        
        affected_cols_data = data.loc[
            data["affected_cols"].str.len() > 0,
            ["record_id", "affected_cols"]
        ]
        
        return self.filter_data_to_subset(affected_cols_data)

    @property
    def death_sanity_check(self):
        data = get_raw_data_file(self.base_path, self.load_only_completed)
        invalid_mask = (data["study_end_cause"] == 2.0) & (data["death_time"].isna())
        return data[invalid_mask][["record_id", "study_end_cause", "death_time", "ground_truth_todeszeitpunkt_complete"]]