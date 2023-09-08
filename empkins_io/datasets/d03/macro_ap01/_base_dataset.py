import warnings
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
from biopsykit.io import (
    load_atimelogger_file,
    load_codebook,
    load_long_format_csv,
    load_pandas_dict_excel,
    load_questionnaire_data,
)
from biopsykit.utils.file_handling import get_subject_dirs
from tpcp import Dataset

from empkins_io.datasets.d03.macro_ap01.helper import _build_data_path, _load_nilspod_session
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import HeartRateDataNotFoundException, TimelogNotFoundException

_cached_load_nilspod_data = lru_cache(maxsize=4)(_load_nilspod_session)


class MacroBaseDataset(Dataset):
    base_path: path_t
    use_cache: bool
    _sample_times_saliva: Tuple[int] = (-40, -1, 16, 25, 35, 45, 60, 75)
    _sample_times_bloodspot: Tuple[int] = (-40, 60)

    NILSPOD_MAPPING: Dict[str, str] = {
        "chest": "56bb",  # ecg
        "sync": "9e02",  # sync with mocap
        "board": "e76b",  # sync with video (clapper board)
    }

    MISSING_DATA: Dict[str, Sequence[str]] = {"no_math": ["VP_03"]}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_missing_data: bool = False,
        use_cache: bool = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        if self.exclude_missing_data:
            for missing_type, sids in self.MISSING_DATA.items():
                for sid in sids:
                    if sid in subject_ids:
                        subject_ids.remove(sid)

        conditions = ["ftsst", "tsst"]

        index = list(product(subject_ids, conditions))
        index = pd.DataFrame(index, columns=["subject", "condition"])
        return index

    @property
    def subject(self) -> str:
        if not self.is_single("subject"):
            raise ValueError("Subject data can only be accessed for a single participant!")
        return self.index["subject"][0]

    @property
    def condition(self) -> str:
        if not self.is_single("condition"):
            raise ValueError("Condition data can only be accessed for a single condition!")
        return self.index["condition"][0]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the MoCap system"""
        return 60

    @property
    def sample_times_saliva(self) -> Sequence[int]:
        return self._sample_times_saliva

    @property
    def sample_times_bloodspot(self) -> Sequence[int]:
        return self._sample_times_bloodspot

    @cached_property
    def nilspod(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("NilsPod data can only be accessed for a single participant in a single condition!")
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        data = self._get_nilspod_data(subject_id, condition)
        return data

    @property
    def ecg(self) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            senor_id = self.NILSPOD_MAPPING["chest"]
            data = self.nilspod.xs(senor_id, level=0, axis=1)
            return data[["ecg"]]

    @property
    def heart_rate(self) -> Dict[str, pd.DataFrame]:
        if not self.is_single(None):
            raise ValueError("Heart rate data can only be accessed for a single participant in a single condition!")

        subject_id = self.group.subject
        condition = self.group.condition
        ecg_path = self.ecg_output_path

        file_path = ecg_path.joinpath(f"hr_result_{subject_id}_{condition}_total.xlsx")
        if not file_path.exists():
            raise HeartRateDataNotFoundException(f"No heart rate data for {subject_id} {condition}.")
        data = load_pandas_dict_excel(file_path)
        return data

    @property
    def hrv(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("Heart rate data can only be accessed for a single participant in a single condition!")

        subject_id = self.group.subject
        condition = self.group.condition
        ecg_path = self.ecg_output_path
        file_path = ecg_path.joinpath(f"hrv_result_{subject_id}_{condition}.csv")
        if not file_path.exists():
            raise HeartRateDataNotFoundException(f"No HRV data for {subject_id} {condition}.")
        return pd.read_csv(file_path, index_col="phase")

    @property
    def timelog_ecg_baseline(self):
        data = self.ecg
        data = data.drop(index=data.first("1min").index)
        timelog = pd.DataFrame(data.first("5min").index[[0, -1]], index=["start", "end"]).T
        timelog.columns.name = "start_end"
        timelog = pd.concat({"ECG_Baseline": timelog}, names=["phase"], axis=1)
        return timelog

    @property
    def timelog_test(self) -> pd.DataFrame:
        return self._load_time_log("test")

    @property
    def timelog_gait(self) -> pd.DataFrame:
        return self._load_time_log("gait")

    def _load_time_log(self, timelog_type: str):
        subject_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        data_path = _build_data_path(self.base_path.joinpath("data_per_subject"), subject_id, condition)
        file_path = data_path.joinpath(f"timelog/cleaned/{subject_id}_{condition}_timelog_{timelog_type}.csv")
        if not file_path.exists():
            raise TimelogNotFoundException(
                f"No time log data was found for {timelog_type} in the {condition} condition of {subject_id}!"
            )
        timelog = load_atimelogger_file(file_path, timezone="Europe/Berlin")
        # convert all column names of the multi-level column index to lower case
        timelog.columns = timelog.columns.set_levels([level.str.lower() for level in timelog.columns.levels])

        return timelog

    @property
    def timelog_total(self) -> pd.DataFrame:
        timelog = pd.concat([self.timelog_ecg_baseline, self.timelog_gait, self.timelog_test], axis=1)
        return timelog.sort_values(by="time", axis=1)

    @property
    def questionnaire(self) -> pd.DataFrame:
        if self.is_single(["condition"]):
            raise ValueError(f"Questionnaire data can not be accessed for a single condition!")
        data = load_questionnaire_data(self.base_path.joinpath("questionnaires/merged/questionnaire_data.xlsx"))
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    @property
    def gender(self) -> pd.Series:
        return self.questionnaire["Gender"]

    @property
    def questionnaire_scores(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("questionnaires/processed/questionnaire_data_processed.csv")
        if not data_path.exists():
            raise ValueError(
                "Processed questionnaire data not available! "
                "Please run the 'data_processing/Questionnaire_Processing.ipynb' notebook first!"
            )
        data = load_long_format_csv(data_path)
        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(conditions, level="condition")

    @property
    def pasa(self) -> pd.DataFrame:
        return self._extract_questionnaire_score("PASA")

    @property
    def stadi(self) -> pd.DataFrame:
        return self._extract_questionnaire_score("STADI")

    @property
    def panas(self) -> pd.DataFrame:
        return self._extract_questionnaire_score("PANAS")

    @property
    def codebook(self) -> pd.DataFrame:
        return load_codebook(self.base_path.joinpath("questionnaires/codebook.csv"))

    @property
    def condition_order(self) -> pd.DataFrame:
        data = pd.read_csv(self.base_path.joinpath("extras/condition_order.csv"))
        data = data.set_index("subject")[["condition_order"]]
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    @property
    def day_condition_map(self) -> pd.DataFrame:
        data = pd.read_csv(self.base_path.joinpath("extras/condition_order.csv"))
        data = data.set_index("subject")[["T1", "T2"]].stack()
        data.index = data.index.set_names("day", level=-1)
        data = pd.DataFrame(data, columns=["condition"])
        return data

    @property
    def cort_non_responder(self) -> pd.Series:
        non_responder = self.cortisol_features.xs("tsst", level="condition")
        non_responder = non_responder.xs("max_inc", level="saliva_feature") <= 1.5
        non_responder.columns = ["non_responder"]
        subject_ids = self.index["subject"].unique()
        return non_responder.loc[subject_ids]

    @property
    def cortisol(self) -> pd.DataFrame:
        return self._load_saliva_data("cortisol")

    @property
    def cortisol_features(self) -> pd.DataFrame:
        return self._load_saliva_features("cortisol")

    @property
    def amylase(self) -> pd.DataFrame:
        return self._load_saliva_data("amylase")

    @property
    def amylase_features(self) -> pd.DataFrame:
        return self._load_saliva_features("amylase")

    @property
    def progesterone(self) -> pd.DataFrame:
        return self._load_estradiol_progesterone()[["progesterone"]]

    @property
    def estradiol(self) -> pd.DataFrame:
        return self._load_estradiol_progesterone()[["estradiol"]]

    @property
    def ecg_output_path(self) -> Path:
        if not self.is_single(None):
            raise ValueError("Path can only be accessed for a single condition of a single participant!")
        data_path = self.base_path.joinpath("data_per_subject").joinpath(
            f"{self.group.subject}/{self.group.condition}/nilspod/processed/ecg"
        )
        return data_path

    def _get_nilspod_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            data, fs = _cached_load_nilspod_data(self.base_path, subject_id, condition)
        else:
            data, fs = _load_nilspod_session(self.base_path, subject_id, condition)
        return data

    def _load_estradiol_progesterone(self):
        data_path = self.base_path.joinpath("saliva/processed/progesterone_estradiol_samples.csv")
        if not data_path.exists():
            raise ValueError(
                "Processed saliva data not available! "
                "Please run the 'data_processing/Saliva_Processing.ipynb' notebook first!"
            )
        data = pd.read_csv(data_path)
        data = data.set_index("subject")
        subject_ids = self.index["subject"].unique()
        return data.reindex(subject_ids).dropna()

    def _load_questionnaire_data(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("questionnaire_total/processed/empkins_macro_questionnaire_data.csv")
        data = load_questionnaire_data(data_path)
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    def _extract_questionnaire_score(self, score_type: str):
        data = self.questionnaire_scores
        return data["data"].unstack("type")[[score_type]].dropna()

    def _load_saliva_data(self, saliva_type: str) -> pd.DataFrame:
        data_path = self.base_path.joinpath(f"saliva/processed/{saliva_type}_samples.csv")
        if not data_path.exists():
            raise ValueError(
                "Processed saliva data not available! "
                "Please run the 'data_processing/Saliva_Processing.ipynb' notebook first!"
            )
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(conditions, level="condition")

    def _load_saliva_features(self, saliva_type: str) -> pd.DataFrame:
        data_path = self.base_path.joinpath(f"saliva/processed/{saliva_type}_features.csv")
        data = load_long_format_csv(data_path)
        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(conditions, level="condition")
