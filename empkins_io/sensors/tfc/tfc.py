from typing import Dict, Optional, List, Union, Literal

import numpy as np
import pandas as pd

from typing_extensions import Self

import warnings

from pathlib import Path

import os
import glob
import re


class TfcLoader:
    _SAMPLING_RATES_HZ = {
        "sync": 1000,
        "ecg": 500,
        "cnap": 10,

    }

    _timezone: str
    _start_time_unix: float
    _phase_information: pd.DataFrame
    _data = Dict
    _raw_signal_information: pd.DataFrame

    _RAW_SIGNAL_GROUPS = {
        "nbp_cuff_p": {
            "columns": ["NBP_Cuff_P"],
            "fs": 5
        },
        "nbp_data": {
            "columns": ["NBP_Sys", "NBP_Dia", "NBP_MAP", "Time_to_NBP"],
            "fs": 1
        },
        "cnap_data": {
            "columns": ["CNAP_Sys", "CNAP_Dia", "CNAP_MAP", "CNAP_PR", "CNAP_no_Avg_Sys", "CNAP_no_Avg_Dia",
                        "CNAP_no_Avg_MAP", "CNAP_no_Avg_PR", "CNAP_CO", "CNAP_SV", "CNAP_SVR", "CNAP_CI", "CNAP_SI",
                        "CNAP_SVRI", "CNAP_IBI", "CNAP_time_to_CF", "CNAP_uncal_Sys", "CNAP_uncal_Dia",
                        "CNAP_uncal_MAP"],
            "fs": 10
        },
        "cnap_add_data": {
            "columns": ["CNAP_BP_Info", "CNAP_BP", "CNAP_uncal_BP", "CNAP_HD_Info"],
            "fs": 100
        },
        "ecg_data": {
            "columns": ["ECG_I", "ECG_II", "ECG_III", "ECG_aVR", "ECG_aVL", "ECG_aVF", "ECG_V1", "ECG_V2", "ECG_V3",
                        "ECG_V4", "ECG_V5", "ECG_V6", "ECG_R_Wave", "ECG_Info"],
            "fs": 500
        },
        "ecg_hr": {
            "columns": ["ECG_HR"],
            "fs": 10
        }
    }

    def __init__(
            self,
            tz: str,
            start_time_unix: float,
            phase_information: pd.DataFrame,
            data: Dict,
            raw_signal_information

    ):
        self._timezone = tz
        self._start_time_unix = start_time_unix
        self._phase_information = phase_information
        self._data = data
        self._raw_signal_information = raw_signal_information

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._timezone

    @property
    def recording_start_time_unix(self) -> float:
        """Start time of the recording."""
        return self._start_time_unix

    @property
    def sampling_rates_hz(self) -> Dict[str, float]:
        """Sampling rates of the different sensors in Hz."""
        return self._SAMPLING_RATES_HZ

    @property
    def raw_signal_information(self) -> pd.DataFrame:
        return self._raw_signal_information

    @classmethod
    def from_text_files(
            cls,
            path: Union[str, Path],
            tz: Optional[str] = "Europe/Berlin",
    ) -> Self:

        data_edf = None
        data_cs = None
        signal_info = None

        if isinstance(path, str):
            path = Path(path)

        # check which type of files exist
        edf_browser, cardio_science = cls._assert_existing_files(path=path)
        if not (edf_browser or cardio_science):
            raise FileExistsError("No TFC recording found.")

        # load data exported from edf browser
        if edf_browser:
            data_edf = {}
            start_time_unix_edf = cls._get_header_information(path=path, tz=tz)
            phases_edf, osc_bp = cls._get_annotation_information(path=path)
            signal_info, data_raw = cls._get_raw_data(path=path)
            data_edf["osc_bp"] = osc_bp
            data_edf.update(data_raw)

        # load data exported from cardio science software
        if cardio_science:
            start_time_unix_cs, phases_cs = cls._get_cardio_science_general_info(path=path, tz=tz)
            cls._clean_cardio_science_files(path=path)
            data_cs = cls._get_cardio_science_data(path=path)

        if edf_browser and cardio_science:
            if start_time_unix_cs != start_time_unix_edf:
                warnings.warn("Start times extracted from edf browser and cardio science files are not the same,"
                              "taking start time from edf browser")
            start_time_unix = start_time_unix_edf

            if not((phases_edf == phases_cs).all().all()):
                warnings.warn("Phases extracted from edf browser and cardio science files are not the same,"
                              "taking phases from edf browser")
            phases = phases_edf
            data_edf.update(data_cs)
            data = data_edf

        elif edf_browser:
            start_time_unix = start_time_unix_edf
            phases = phases_edf
            data = data_edf

        elif cardio_science:
            start_time_unix = start_time_unix_cs
            phases = phases_cs
            data = data_cs

        return cls(tz, start_time_unix, phases, data, signal_info)

    def data_as_df(
            self,
            data_type: "str",
            index: Optional[str] = None,
    ):

        if data_type not in self._data.keys():
            raise ValueError(
                f"Supplied value for data_type ({data_type}) is not allowed. Allowed values: {self._data.keys()}"
            )

        data = self._data[data_type].copy()
        data = self._add_index(data=data, index=index)

        return data

    def _add_index(self, data: pd.DataFrame, index: str) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")

        index_name = index_names[index]
        data.index.name = index_name

        if index is None:
            data = data.reset_index(drop=True)
            data.index.name = index_name
            return data
        if index == "time":
            return data

        # convert to utc timestamps => for index_type "utc"
        data.index += self._start_time_unix

        if index == "utc_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC").tz_convert(self._timezone)

        return data

    @classmethod
    def _assert_existing_files(cls, path: Path):

        data_path = path.joinpath("raw")

        edf_browser = False
        cardio_science = False

        # check for completeness of edf browser exported raw_data input files
        files = ["annotations", "header", "raw_signals", "signal_mapping"]
        path_export_files = data_path.joinpath("edf_browser_export")
        available_files = [f for f in os.listdir(path_export_files) if f.endswith("txt")]
        missing_files = [keyword for keyword in files if not any(keyword in file for file in available_files)]

        if len(missing_files) == 0:
            edf_browser = True
        else:
            warnings.warn(
                f"Required files exported from edf browser with suffix(es) {missing_files} "
                f"are not available in {path_export_files}."
            )

        # check for completeness of cardio science exported input files
        files = ["BeatToBeat", "BPV", "BPVsBP", "BRS_BRS0", "BRS_BRS1", "BRS_BRS2", "CardiacParams", "HRV", "NBP"]
        path_export_files = data_path.joinpath("cardio_science_export")
        available_files = [f for f in os.listdir(path_export_files) if f.endswith("csv")]
        missing_files = [keyword for keyword in files if not any(keyword in file for file in available_files)]

        if len(missing_files) == 0:
            cardio_science = True
        else:
            warnings.warn(
                f"Required files exported from cardio science software with suffix(es) {missing_files} "
                f"are not available in {path_export_files}."
            )

        return edf_browser, cardio_science

    @classmethod
    def _get_header_information(cls, path: Path, tz: str):

        data_path = path.joinpath("raw")

        header_path = glob.glob(os.path.join(data_path.joinpath("edf_browser_export"), "*_header.txt"))[0]
        header_data = pd.read_table(header_path, delimiter=',')

        timestamp = pd.to_datetime(f"{header_data.Startdate[0]} {header_data.Startime[0]}", format="%d.%m.%y %H.%M.%S")
        timestamp = timestamp.tz_localize(tz=tz)

        return timestamp.timestamp()

    @classmethod
    def _get_annotation_information(cls, path: Path):

        data_path = path.joinpath("raw")

        annotation_path = glob.glob(os.path.join(data_path.joinpath("edf_browser_export"), "*_annotations.txt"))[0]
        annotation_data = pd.read_table(annotation_path, delimiter=',')

        # get phase information
        phase_data = annotation_data.loc[annotation_data["Annotation"].str.contains("UM:")]
        phase_data = phase_data.drop(columns="Duration").reset_index(drop=True)
        phase_data["Annotation"] = phase_data["Annotation"].str.replace("UM:", "", regex=True)
        phase_data.columns = ["relative_time", "phase"]

        # get oscillatory bp measurements
        bp_data = annotation_data.loc[annotation_data["Annotation"].str.contains("SE1:NBP finished")]
        bp_data = bp_data.drop(columns="Duration").reset_index(drop=True)
        bp_data['bp_systolic'] = bp_data['Annotation'].str.extract(r'(\d+)/\d+')
        bp_data['bp_diastolic'] = bp_data['Annotation'].str.extract(r'\d+/(\d+)')
        bp_data['bp_systolic'] = pd.to_numeric(bp_data["bp_systolic"])
        bp_data["bp_diastolic"] = pd.to_numeric(bp_data["bp_diastolic"])
        bp_data.columns = ["relative_time", "annotation_original", "bp_systolic", "bp_diastolic"]
        bp_data.index = bp_data["relative_time"]

        return phase_data, bp_data

    @classmethod
    def _get_raw_data(cls, path: Path):

        data_path = glob.glob(os.path.join(path.joinpath("raw/edf_browser_export"), "*_raw_signals.txt"))[0]
        raw_data = pd.read_table(data_path, delimiter=',', index_col=0)

        mapping_path = glob.glob(os.path.join(path.joinpath("raw/edf_browser_export"), "*_signal_mapping.txt"))[0]
        signal_mapping = pd.read_table(mapping_path, delimiter=',', index_col=0)

        _signal_mapping = signal_mapping.copy()
        _signal_mapping.index = _signal_mapping.index.astype(str)
        _signal_mapping["Label"] = [label.replace(" ", "") for label in _signal_mapping["Label"].to_list()]
        dict_labels = _signal_mapping["Label"].to_dict()
        raw_data = raw_data.rename(columns=dict_labels)

        for fs, analog_group in _signal_mapping.loc[_signal_mapping["Label"].str.startswith("AI_")].groupby("Smp/Rec"):
            cls._RAW_SIGNAL_GROUPS[f"analog_{fs}"] = {
                "columns": analog_group["Label"].to_list(),
                "fs": fs
            }

        data = {}

        for key_group, value_group in cls._RAW_SIGNAL_GROUPS.items():
            fs = value_group["fs"]
            index_group = np.arange(start=0, stop=int(np.ceil(raw_data.index[-1])), step=(1/fs)).round(4)
            data[key_group] = raw_data[value_group["columns"]].loc[index_group]

        return signal_mapping, data

    @classmethod
    def _clean_cardio_science_files(cls, path: Path):
        file_path = path.joinpath("raw/cardio_science_export")
        file_path_cleaned = file_path.parent.parent.joinpath("cleaned/cardio_science_export")

        if file_path_cleaned.exists():
            return
        else:
            file_path_cleaned.mkdir()

        available_files = [f for f in os.listdir(file_path) if f.endswith("csv")]

        for file in available_files:
            path_temp = file_path.joinpath(file)
            data = cls._clean_cs_file(file_path=path_temp)

            file_split = file.split('.')
            file_path_cleaned_new = file_path_cleaned.joinpath(f"{file_split[0]}_{file_split[1]}.{file_split[2]}")

            with open(file_path_cleaned_new, "w") as csv_file:
                for line in data:
                    csv_file.write(f"{line}\n")

    @classmethod
    def _clean_cs_file(cls, file_path: Path):

        with open(file_path, "r", encoding="ISO-8859-1") as file:
            rows = file.readlines()

        # remove all general information
        rows = rows[11:]

        # filter all lines including intervention phases which are set by the user
        exclude = [
            not ("Start Recording" in x or "Stop Recording" in x or "UM:" in x or "\n" == x)
            for x in rows
        ]
        rows_filtered = [x for x, keep in zip(rows, exclude) if keep]

        # generate column names including variable name and unit
        var_names = rows_filtered[0].split(" ")
        var_names = [x for x in var_names if x]  # remove all empty strings
        var_names[-1] = var_names[-1].replace("\n", "")  # remove linebreak at the end

        var_units = rows_filtered[1].split(" ")
        var_units = [x for x in var_units if x]  # remove all empty strings

        if len(var_names) != len(var_units):  # make sure that all units are separated (due to CardiacParams file)
            var_units = "".join(var_units).split("[")
            var_units = ["["+x for x in var_units if x]

        var_units = [x.replace(chr(178), "^2") for x in var_units if x]  # replace superscript two
        var_units[-1] = var_units[-1].replace("\n", "")  # remove linebreak at the end

        var = [f"{x} {y}" for x, y in zip(var_names, var_units)]

        # prepare data for saving as csv file ([:-1] is important that the last comma is removed and no empty column is created)
        data = [x.replace(" ", "").replace("\n", "").replace(";", ",").replace("NAN", "")[:-1] for x in rows_filtered[2:]]
        var = ",".join(var)
        final_data = [var] + data

        return final_data

    @classmethod
    def _get_cardio_science_general_info(cls, path: Path, tz: str):

        file_path = path.joinpath("raw/cardio_science_export")
        available_files = [f for f in os.listdir(file_path) if f.endswith("csv")]
        file_path = file_path.joinpath(available_files[0])

        with open(file_path, "r", encoding="ISO-8859-1") as file:
            rows = file.readlines()

        # get unix start time of the recording
        date = re.search(r"\d{4}-\d{2}-\d{2}", rows[0]).group(0)
        start_time = re.search(r"\d{2}:\d{2}:\d{2}", rows[1]).group(0)
        start_time_unix = pd.Timestamp(f"{date} {start_time}", tz=tz).timestamp()

        # get all interventions that are set by a user
        um_rows = [row.strip() for row in rows if "UM:" in row]
        interventions = {
            row.split()[0]: row.split("UM:")[1].strip()
            for row in um_rows
        }
        interventions = pd.DataFrame(list(interventions.items()), columns=["relative_time", "phase"])
        interventions["relative_time"] = interventions["relative_time"].astype(float)

        return start_time_unix, interventions

    @classmethod
    def _get_cardio_science_data(cls, path: Path):

        file_path = path.joinpath("cleaned/cardio_science_export")
        files = [f for f in os.listdir(file_path) if f.endswith("csv")]

        data = {}

        for file in files:
            file = Path(file)
            data_type = file.stem.split("_")[-1]
            data[data_type] = pd.read_csv(file_path.joinpath(file), index_col=0)

        return data



