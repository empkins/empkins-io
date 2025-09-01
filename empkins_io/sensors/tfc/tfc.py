from typing import Dict, Optional, List, Union

import numpy as np
import pandas as pd

from typing_extensions import Self

import warnings

from pathlib import Path

import os
import re


class TfcLoader:
    """Class representing a measurement with the Task Force Cardio (TFC).

    Attributes
    ----------
    _timezone :  str
        Timezone of the recording (if available)
    _start_time_unix : float
        Unix start time of the recording
    _phase_information : pd.DataFrame
        Information about each of the phases (interventions) including duration, relative, and absolute time
    _data : Dict
        Dictionary containing all the data
    _raw_signal_information : pd.DataFrame
        Contains information about the recording of the single signals, i.e., units, min and max values,
        sampling frequency, ...

    """

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
        """Get new data loader instance.

            note::
                Usually you shouldn't use this init directly.
                Use the provided `from_text_files` constructor to handle loading recorded TFC data.

            Parameters
            ----------
            tz : str
                Timezone of the recording (if available)
            start_time_unix : float
                Unix start time of the recording
            phase_information : pd.DataFrame
                Information of the phases (interventions) including duration, relative and absolute time
            data : Dict
                Dictionary containing all the data
            raw_signal_information : pd.DataFrame
                Contains information about the recording of the single signals, i.e., units, min and max values,
                sampling frequency, ...

        """
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
        """Unix start time of the recording."""
        return self._start_time_unix

    @property
    def sampling_rates_hz(self) -> Dict[str, float]:
        """Sampling rates of the different raw signal groups in Hz."""
        return {key: value["fs"] for key, value in self._RAW_SIGNAL_GROUPS.items()}

    @property
    def raw_signal_information(self) -> pd.DataFrame:
        """Information on the recording properties of the different signals."""
        return self._raw_signal_information

    @property
    def phase_information(self) -> pd.DataFrame:
        """Information on the start times and durations of the different recording phases."""
        return self._phase_information

    @property
    def data_available(self) -> List:
        """Overview of all available data groups."""
        return list(self._data.keys())

    @classmethod
    def from_text_files(
            cls,
            path: Union[str, Path],
            tz: Optional[str] = "Europe/Berlin",
    ) -> Self:

        """Create a new TFC Loader instance from text files.

        note::
            TfcLoader expects the data to be structured as follows:
            <data_recording>/
            └── raw
                ├── cardio_science_export/
                │   ├── <prefix>.BeatToBeat.csv
                │   ├── <prefix>.BPV.csv
                │   ├── <prefix>.BPVsBP.csv
                │   ├── <prefix>.BRS_BRS0.csv
                │   ├── <prefix>.BRS_BRS1.csv
                │   ├── <prefix>.BRS_BRS2.csv
                │   ├── <prefix>.CardiacParams.csv
                │   ├── <prefix>.HRV.csv
                │   └── <prefix>.NBP.csv
                └── edf_browser_export/
                    ├── <prefix>_annotations.csv
                    ├──<prefix>_data.csv
                    ├──<prefix>_header.csv
                    └── <prefix>_signals.csv

        Parameters
        ----------
        path : str
            Path to the folder
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        """

        data_edf = None
        data_cs = None
        signal_info = None

        if isinstance(path, str):
            path = Path(path)

        # check which type of exported files exist (edf_browser and/or cardio_science)
        edf_data_exist, cs_data_exist = cls._assert_existing_files(path=path)
        if not (edf_data_exist or cs_data_exist):
            raise FileExistsError("No TFC recording found.")

        # load data exported from edf browser
        if edf_data_exist:
            data_edf = {}
            start_time_unix_edf, rel_stop_time_edf = cls._get_header_information(path=path, tz=tz)
            phases_edf, osc_bp = cls._get_annotation_information(path=path)
            signal_info, data_raw = cls._get_raw_data(path=path)
            data_edf["osc_bp"] = osc_bp
            data_edf.update(data_raw)

        # load data exported from cardio science software
        if cs_data_exist:
            start_time_unix_cs, phases_cs, rel_stop_time_cs = cls._get_cardio_science_general_info(path=path, tz=tz)
            cls._clean_cardio_science_files(path=path)
            data_cs = cls._get_cardio_science_data(path=path)

        # if both edf browser and cardio science files exist
        if edf_data_exist and cs_data_exist:
            if start_time_unix_cs != start_time_unix_edf:
                warnings.warn("Start times extracted from edf browser and cardio science files are not the same,"
                              "taking start time from edf browser")
            start_time_unix = start_time_unix_edf

            if not((phases_edf == phases_cs).all().all()):
                warnings.warn("Phases extracted from edf browser and cardio science files are not the same,"
                              "taking phases from edf browser")
            phases = phases_edf

            if rel_stop_time_cs != rel_stop_time_edf:
                warnings.warn("Relative stop times extracted from edf browser and cardio science files are not the same,"
                              "taking stop time from edf browser")
            rel_stop_time = rel_stop_time_edf

            data_edf.update(data_cs)
            data = data_edf

        # if only edf browser files exist
        elif edf_data_exist:
            start_time_unix = start_time_unix_edf
            phases = phases_edf
            rel_stop_time = rel_stop_time_edf
            data = data_edf

        # if only cardio science files exist
        elif cs_data_exist:
            start_time_unix = start_time_unix_cs
            phases = phases_cs
            rel_stop_time = rel_stop_time_cs
            data = data_cs

        # this should never happen
        else:
            raise FileExistsError("No TFC recording found.")

        # calculate duration and absolute timings of the phases
        phases = cls._calculate_phase_information(
            phases=phases,
            start_time=start_time_unix_edf,
            tz=tz,
            rel_stop_time=rel_stop_time
        )

        return cls(tz, start_time_unix, phases, data, signal_info)

    def data_as_df(
            self,
            data_type: "str",
            index: Optional[str] = None,
    ):
        """Create pandas dataframe from recorded data with a specified index.

        Parameters
        ----------
        data_type : str
            Indicates which types of data should be returned. Must be one of 'osc_bp', 'nbp_cuff_p', 'nbp_data',
            'cnap_data', 'cnap_add_data', 'ecg_data', 'ecg_hr', 'analog_1000', 'NBP', 'BeatToBeat', 'BPVsBP',
            'CardiacParams', 'HRV', 'BPV', 'BRS1', 'BRS0', 'BRS2'
        index : str, optional
            Indicates which type of index the dataframe has. Must be one of None, 'time', 'utc', 'utc_datetime',
            'local_datetime'

        """

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

        edf_browser = False
        cardio_science = False

        # check for completeness of edf browser exported raw_data input files
        files = ["annotations", "header", "data", "signals"]  # those 4 files usually exists in a complete tfc dataset
        path_export_files = path.joinpath("raw", "edf_browser_export")
        if path_export_files.exists():
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
        path_export_files = path.joinpath("raw", "cardio_science_export")
        if path_export_files.exists():
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

        header_path = next((path / "raw" / "edf_browser_export").glob("*_header.txt"))
        header_data = pd.read_table(header_path, delimiter=',')

        # calculate unix start time
        timestamp = pd.to_datetime(f"{header_data.Startdate[0]} {header_data.Startime[0]}", format="%d.%m.%y %H.%M.%S")
        timestamp = timestamp.tz_localize(tz=tz)

        # get duration of recording = relative stop time in seconds
        rel_stop_time = header_data.NumRec[0]  # Error in the tfc header file, duration column only returns an 1

        return timestamp.timestamp(), rel_stop_time

    @classmethod
    def _get_annotation_information(cls, path: Path):

        annotation_path = next((path / "raw" / "edf_browser_export").glob("*_annotations.txt"))
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
    def _calculate_phase_information(cls, phases: pd.DataFrame, start_time: float, tz: str, rel_stop_time: float):
        _phases = phases.copy()
        _phases = _phases.rename(columns={"relative_time": "relative_start_time"})

        # add unix and absolute start time columns
        _phases["unix_start_time"] = _phases["relative_start_time"] + start_time
        _phases["absolute_start_time"] = pd.to_datetime(_phases["unix_start_time"], unit="s", utc=True).dt.tz_convert(
           tz
        )

        # calculate duration of each phase
        durations = _phases["relative_start_time"].to_numpy()
        durations = np.append(durations, rel_stop_time)
        _phases["duration"] = np.diff(durations)

        # sort columns
        _phases = _phases[["phase", "duration", "relative_start_time", "unix_start_time", "absolute_start_time"]]
        _phases = _phases.set_index("phase")
        return _phases

    @classmethod
    def _get_raw_data(cls, path: Path):

        data_path = next((path / "raw" / "edf_browser_export").glob("*_data.txt"))
        raw_data = pd.read_table(data_path, delimiter=',', index_col=0)

        # includes mapping from signal identifier to signal name and unit
        mapping_path = next((path / "raw" / "edf_browser_export").glob("*_signals.txt"))
        signal_mapping = pd.read_table(mapping_path, delimiter=',', index_col=0)

        # rename columns accordingly
        _signal_mapping = signal_mapping.copy()
        _signal_mapping.index = _signal_mapping.index.astype(str)
        _signal_mapping["Label"] = [label.replace(" ", "") for label in _signal_mapping["Label"].to_list()]
        dict_labels = _signal_mapping["Label"].to_dict()
        raw_data = raw_data.rename(columns=dict_labels)

        # add analog signal groups to RAW_SIGNAL_GROUPS
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

        # create new directory for cleaned files if necessary
        if file_path_cleaned.exists():
            return
        else:
            file_path_cleaned.mkdir()

        available_files = [f for f in os.listdir(file_path) if f.endswith("csv")]

        # clean each file separately and save as file
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

        file_path = path.joinpath("raw", "cardio_science_export")
        available_files = [f for f in os.listdir(file_path) if f.endswith("csv")]
        file_path = file_path.joinpath(available_files[0])  # all files include the same header information

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

        # get length of recording
        stop_rec_row = [row.strip() for row in rows if "Stop Recording" in row]
        relative_stop_time = float(stop_rec_row[0].split()[0])

        return start_time_unix, interventions, relative_stop_time

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



