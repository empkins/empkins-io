from typing import Dict, Optional, List
import numpy as np

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from typing_extensions import Self

from scipy.io import loadmat

import warnings


class TfmLoader:
    """Class representing a measurement with the Task Force Monitor (TFM).

    Attributes
    ----------
    _timezone :  str
        Timezone of the recording (if available)
    _recording_date : str
        Date of the recording (if available)
    _phase_information: pd.DataFrame
        Information about each of the phases (interventions) including duration, relative and absolute time
    _oscillatory_blood_pressure : pd.DataFrame
        Oscillatory blood pressure measurements
    _tfm_b2b_phase_data
        Dictionary containing Beat-to-Beat data recorded with the TFM separated into phases
    _tfm_raw_phase_data
        Dictionary containing raw data recorded with the TFM separated into phases

    """

    _SAMPLING_RATES_HZ = {
        "bp": 100,
        "ecg_1": 1000,
        "ecg_2": 1000,
        "icg_der": 500,
        "ext_1": 1000,
        "ext_2": 1000,
        "icg_raw": 50,
    }

    _RAW_SIGNAL_CHANNEL_MAPPING = {
        "bp": "BPCONT_NOTCH_UC",
        "ecg_1": "rawECG1",
        "ecg_2": "rawECG2",
        "icg_der": "rawICG",
        "ext_1": "EXT1",
        "ext_2": "EXT2",
        "icg_raw": "Z0"
    }

    _HEMODYNAMIC_PARAMETERS_MAPPING = {
        "relative_time": "Zeit",
        "beat": "Beat",
        "CI": "CI",
        "HR": "HR",
        "HZV": "HZV",
        "RRI": "RRI",
        "SI": "SI",
        "SV": "SV",
        "TPR": "TPR",
        "TPRI": "TPRI",
        "dBP": "dBP",
        "mBP": "dBP",
        "sBP": "sBP"
    }

    _BPV_PARAMETERS_MAPPING = {
        "relative_time": "Zeit",
        "beat": "Beat",
        "HF_dBP": "HF_dBP",
        "HFnu_dBP": "HFnu_dBP",
        "LF_HF": "LF_HF",
        "LF_HF_dBP": "LF_HF_dBP",
        "LF_dBP": "LF_dBP",
        "LFnu_dBP": "LFnu_dBP",
        "PSD_dBP": "PSD_dBP",
        "VLF_dBP": "VLF_dBP"
    }

    _CARDIAC_PARAMETERS_MAPPING = {
        "relative_time": "Zeit",
        "beat": "Beat",
        "ACI": "ACI",
        "CI": "CI",
        "EDI": "EDI",
        "HR": "HR",
        "IC": "IC",
        "LVET": "LVET",
        "LVWI": "LVWI",
        "SI": "SI",
        "TFC": "TFC",
        "TPRI": "TPRI",
        "dBP": "dBP",
        "mBP": "mBP",
        "sBP": "sBP"
    }

    _HRV_PARAMETERS_MAPPING = {
        "relative_time": "Zeit",
        "beat": "Beat",
        "HF_RRI": "HF_RRI",
        "HFnu_RRI": "HFnu_RRI", "LF_HF": "LF_HF",
        "LF_HF_RRI": "LF_HF_RRI",
        "LF_RRI": "LF_RRI",
        "LFnu_RRI": "LFnu_RRI",
        "PSD_RRI": "PSD_RRI",
        "VLF_RRI": "VLF_RRI",
    }

    _timezone: str
    _recording_date: str
    _tfm_b2b_phase_data: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    _tfm_raw_phase_data: Dict[str, Dict[str, np.ndarray]]
    _phase_information: pd.DataFrame
    _oscillatory_blood_pressure: pd.DataFrame

    def __init__(
            self,
            tz: str,
            recording_date: str,
            phase_information: pd.DataFrame,
            osc_blood_pressure: pd.DataFrame,
            tfm_b2b_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
            tfm_raw_data: Dict[str, Dict[str, np.ndarray]]
    ):
        """Get new data loader instance.

        note::
            Usually you shouldn't use this init directly.
            Use the provided `from_mat_file` constructor to handle loading recorded TFM data.

        Parameters
        ----------
        tz :
            Timezone of the recording (if available)
        recording_date :
            Date of the recording (if available)
        phase_information :
            Information about each of the phases (interventions) including duration, relative and absolute time
        osc_blood_pressure :
            Oscillatory blood pressure measurements
        tfm_b2b_data :
            Beat-to-Beat data recorded with the TFM separated into phases
        tfm_raw_data :
            Raw data recorded with the TFM separated into phases

        """
        self._timezone = tz
        self._recording_date = recording_date
        self._phase_information = phase_information
        self._oscillatory_blood_pressure = osc_blood_pressure
        self._tfm_b2b_phase_data = tfm_b2b_data
        self._tfm_raw_phase_data = tfm_raw_data

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._timezone

    @property
    def recording_date(self) -> str:
        """Date of the recording."""
        return self._recording_date

    @property
    def sampling_rates_hz(self) -> Dict[str, float]:
        """Sampling rates of the different sensors in Hz."""
        return self._SAMPLING_RATES_HZ

    @property
    def phase_information(self) -> pd.DataFrame:
        """Information about each of the phases (interventions) including duration, relative and absolute time."""
        return self._phase_information

    @property
    def oscillatory_blood_pressure(self) -> pd.DataFrame:
        """Oscillatory blood pressure measurements."""
        return self._oscillatory_blood_pressure

    @property
    def b2b_data_dict(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """Dictionary containing Beat-to-Beat data recorded with the TFM separated into phases."""
        return self._tfm_b2b_phase_data

    @property
    def raw_data_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Dictionary containing raw data recorded with the TFM separated into phases."""
        return self._tfm_raw_phase_data

    @property
    def phase_timestamps(self) -> pd.DataFrame:
        """Phase (Intervention) absolute timestamps ."""
        phase_timestamps = pd.DataFrame()
        phase_timestamps.index = self.phase_information.index
        phase_timestamps.index.name = "phase"
        date_times = self.phase_information["absolute_time"].tolist()

        times_utc = []
        times_local = []
        for time in date_times:
            time_unix = self._convert_time_to_unix(time)
            time_unix = pd.to_datetime(time_unix, unit="s")
            times_utc.append(time_unix.tz_localize("UTC"))
            times_local.append(time_unix.tz_localize("UTC").tz_convert(self.timezone))

        phase_timestamps["date (UTC)"] = times_utc
        phase_timestamps[f"date ({self.timezone})"] = times_local
        return phase_timestamps

    @classmethod
    def from_mat_file(
            cls,
            path: str,
            tz: Optional[str] = "Europe/Berlin",
            recording_date: Optional[str] = "1970-01-01",
            phase_mapping: Optional = None
    ) -> Self:
        """Create a new TFM Loader instance from a valid MATLAB file.

        Parameters
        ----------
        path : str
            Path to the file
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.
        recording_date : str, optional
            Date str of the recording (YYYY-MM-DD)
        phase_mapping : dict, optional
            Dictionary containing a mapping of the original phase (intervention) names (dict keys) and the given phase
            (intervention) names (dict values).

        """
        _assert_file_extension(path, ".mat")

        data = loadmat(path, struct_as_record=False, squeeze_me=True)

        df_iv = cls._load_intervention_variables(data, phase_mapping)
        intervention_names = list(df_iv.index)
        df_osc = cls._load_osc_blood_pressure(data)
        dict_beat2beat = cls._load_beat_to_beat_parameters(data, intervention_names)
        dict_raw = cls._load_raw_data(data, intervention_names)

        return cls(tz, recording_date, df_iv, df_osc, dict_beat2beat, dict_raw)

    @classmethod
    def _load_intervention_variables(cls, data, phase_mapping) -> pd.DataFrame:
        """Load descriptive phase (intervention) parameters.

        Parameters
        ----------
        data : dict
            Dictionary containing the data from MATLAB file.
        phase_mapping : dict or None
            Dictionary containing the mapping of the original phase (intervention) names (dict keys) and the given
            phase (intervention) names (dict values).

        Raises
        ------
        ValueError
            If the loaded phase (intervention) names do not match the phase mapping

        """
        df_iv = pd.DataFrame()

        if phase_mapping is not None:
            if set(list(data["IV"].Name)) != set(list(phase_mapping.keys())):
                raise ValueError("Phase mapping values are not compatible with TFM intervention names")
            df_iv.index = [phase_mapping.get(name) for name in list(data["IV"].Name)]
        else:
            df_iv.index = list(data["IV"].Name)

        df_iv.index.name = "phase"
        df_iv["duration"] = data["IV"].Duration
        df_iv["absolute_time"] = data["IV"].AbsTime
        df_iv["relative_time"] = data["IV"].Reltime

        if len(df_iv.index) != len(set(df_iv.index)):

            df_iv.index = df_iv.index.where(
                ~df_iv.index.duplicated(), df_iv.index + "_" + df_iv.groupby(level=0).cumcount().astype(str)
            )

            warnings.warn("The TFM Dataset contains duplicate intervention names. "
                          "Intervention names where renamed to avoid overwriting of data."
                          )

        return df_iv

    @classmethod
    def _load_osc_blood_pressure(cls, data) -> pd.DataFrame:
        """Load oscillatory blood pressure measurements.

        Parameters
        ----------
        data : dict
            Dictionary containing the data from MATLAB file.

        """
        df_osc = pd.DataFrame()
        data_tmp = data["OscBP"]
        df_osc.index = cls._unwrap_osc_data(data_tmp.Zeit)
        df_osc.index.name = "relative_time"
        df_osc["duration"] = cls._unwrap_osc_data(data_tmp.Messdauer)
        df_osc["diastolic_pressure"] = cls._unwrap_osc_data(data_tmp.DBP)
        df_osc["systolic_pressure"] = cls._unwrap_osc_data(data_tmp.SBP)
        df_osc["heart_rate"] = cls._unwrap_osc_data(data_tmp.HR)
        return df_osc

    @classmethod
    def _unwrap_osc_data(cls, var) -> List:
        """Unwrap oscillatory blood pressure measurements parameter.

        Parameters
        ----------
        var : array

        """
        info_tmp = [t.tolist() if isinstance(t, np.ndarray) else t for t in var]
        info_tmp = [
            item for sublist in info_tmp for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        return info_tmp

    @classmethod
    def _load_raw_data(cls, data, intervention_names) -> Dict[str, Dict[str, np.ndarray]]:
        """Load raw signals from MATLAB dict.

        Parameters
        ----------
        data : dict
            Dictionary containing the data from MATLAB file.
        intervention_names : List
            Phase (intervention) names.

        """

        dict_raw = cls._load_data(
            data=data["RAW_SIGNALS"],
            intervention_names=intervention_names,
            mapping=cls._RAW_SIGNAL_CHANNEL_MAPPING
        )

        return dict_raw

    @classmethod
    def _load_beat_to_beat_parameters(cls, data, intervention_names) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """Load beat-to-beat parameters from MATLAB dict.

        Parameters
        ----------
        data : dict
            Dictionary containing the data from MATLAB file.
        intervention_names : List
            Phase (intervention) names.

        """

        dict_beat_parameters = {
            "hemodynamic_parameters": cls._load_data(
                data=data["BeatToBeat"],
                intervention_names=intervention_names,
                mapping=cls._HEMODYNAMIC_PARAMETERS_MAPPING
            ), "bpv_parameters": cls._load_data(
                data=data["BPV"],
                intervention_names=intervention_names,
                mapping=cls._BPV_PARAMETERS_MAPPING
            ), "hrv_parameters": cls._load_data(
                data=data["HRV"],
                intervention_names=intervention_names,
                mapping=cls._HRV_PARAMETERS_MAPPING
            ), "cardiac_parameters": cls._load_data(
                data=data["CardiacParams"],
                intervention_names=intervention_names,
                mapping=cls._CARDIAC_PARAMETERS_MAPPING
            )
        }

        return dict_beat_parameters

    @classmethod
    def _load_data(cls, data, intervention_names, mapping):
        """Load data and assign correct parameters and phase names.

        Parameters
        ----------
        data : dict
            Dictionary containing the data from MATLAB file.
        intervention_names : List
            Phase (intervention) names.
        mapping : dict
            Dictionary containing the mapping of the original and the given parameter names

        """

        data_dict_tmp = {key: getattr(data, value) for key, value in mapping.items()}
        data_dict = {}

        for key, value in data_dict_tmp.items():
            data_dict[key] = {}
            for i in range(len(intervention_names) - 1):
                index = intervention_names[i]
                data_dict[key][index] = value[i]

        return data_dict

    def raw_phase_data_as_df_dict(self, index: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate dictionary containing the raw data as dataframes separated in phases.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        """

        dict_df = {}
        for sig in self.raw_data_dict.keys():
            dict_df[sig] = {}
            for phase in self.raw_data_dict[sig].keys():
                data = pd.DataFrame(self.raw_data_dict[sig][phase])
                data.columns = [sig]
                data = self._add_index_raw(
                    data=data,
                    index=index,
                    start_time=self.phase_information.at[phase, "absolute_time"]
                )
                dict_df[sig][phase] = data

        return dict_df

    def b2b_phase_data_as_df_dict(self, index: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate dictionary containing the beat-to-beat data as dataframes separated in phases.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        """

        data_dict_inv = {}
        for params_group in self.b2b_data_dict.keys():
            data_dict = self.b2b_data_dict[params_group]

            data_dict_inv[params_group] = {}

            for parameter in data_dict.keys():
                for phase in data_dict[parameter].keys():

                    if phase not in data_dict_inv[params_group].keys():
                        data_dict_inv[params_group][phase] = {}

                    data_dict_inv[params_group][phase][parameter] = data_dict[parameter][phase]

            for phase in data_dict_inv[params_group].keys():
                df = pd.DataFrame.from_dict(data_dict_inv[params_group][phase])
                df = self._add_index_beat(
                    df, index=index,
                    start_time=self.phase_information.at[phase, "absolute_time"],
                    relative_start_time=self.phase_information.at[phase, "relative_time"],
                )
                data_dict_inv[params_group][phase] = df

        return data_dict_inv

    def raw_data_as_df_dict(self, index: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Generate dictionary containing the raw data as dataframes.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        """
        data_dict = self.raw_phase_data_as_df_dict(index=index)
        return self._concat_data_over_time(data_dict)

    def b2b_data_as_df_dict(self, index: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Generate dictionary containing the beat-to-beat data as dataframes.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        """
        data_dict = self.b2b_phase_data_as_df_dict(index=index)
        return self._concat_data_over_time(data_dict)

    def _concat_data_over_time(self, data):
        data_dict_concat = {}
        for group in data.keys():
            df = pd.DataFrame()
            for value in data[group].values():
                df = pd.concat([df, value], axis=0)

            if df.index.duplicated().sum() > 0:
                warnings.warn("Duplicated index values found. Dropping duplicates.")
                df = df.loc[~df.index.duplicated()]

            data_dict_concat[group] = df

        return data_dict_concat

    def _convert_time_to_unix(self, time):
        start_date_time = self._recording_date + " " + time
        timestamp = pd.to_datetime(start_date_time, format='%Y-%m-%d %H:%M:%S.%f')
        timestamp = timestamp.tz_localize(self.timezone)
        return timestamp.timestamp()

    def _add_index_raw(self, data: pd.DataFrame, index: str, start_time: str) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }

        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")

        data.index.name = index_names[index]

        if index is None:
            return data
        if index == "time":
            data.index -= data.index[0]
            data.index /= self.sampling_rates_hz[data.columns[0]]
            return data

        # convert to utc timestamps
        data.index /= self.sampling_rates_hz[data.columns[0]]

        return self._convert_index_to_datetime(data, start_time, index)

    def _add_index_beat(
            self,
            data: pd.DataFrame,
            index: str,
            start_time: str,
            relative_start_time: float
    ) -> pd.DataFrame:

        index_names = {
            None: "n_beats",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }

        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")

        if index is None:
            data = data.set_index("beat")
        elif index == "time":
            data = data.set_index("relative_time")
        else:
            data = data.set_index("relative_time", drop=False)
            data.index -= relative_start_time
            data = self._convert_index_to_datetime(data, start_time, index)

        data.index.name = index_names[index]
        return data

    def _convert_index_to_datetime(self, data, start_time, index):
        data.index += self._convert_time_to_unix(start_time)

        if index == "utc_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="s")
            data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        return data
