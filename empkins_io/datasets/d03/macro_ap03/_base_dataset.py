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

from empkins_io.datasets.d03.macro_ap01.helper import (
    _build_data_path,
    _load_nilspod_session,
)
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import (
    HeartRateDataNotFoundException,
    TimelogNotFoundException,
)

_cached_load_nilspod_data = lru_cache(maxsize=4)(_load_nilspod_session)


class MacroBaseDataset(Dataset):
    base_path: path_t
    index_from_folder: bool
    data_tabular_path: path_t
    use_cache: bool
    _sample_times_saliva: Tuple[int] = (-40, -1, 15, 25, 35, 45, 60, 75)
    _sample_times_bloodspot: Tuple[int] = (-40, 60)

    SAMPLING_RATE_XSENS_HZ: float = 60.0
    SAMPLING_RATE_NILSPOD_HZ: float = 256.0
    CONDITIONS = ["ftsst", "tsst"]

    # TODO
    NILSPOD_MAPPING: Dict[str, str] = {
        "chest": "56bb",  # ecg
        "sync": "9e02",  # sync with mocap
        "board": "e76b",  # sync with video (clapper board)
    }

    SUBSETS_WITHOUT_MOCAP = ()

    # TODO check for incomplete videos
    SUBSETS_WITHOUT_BODY_VIDEO = (
        ("VP_033", "tsst"),
        ("VP_039", "tsst"),
        ("VP_039", "ftsst"),
        ("VP_042", "tsst"),
        ("VP_057", "tsst"),
        ("VP_058", "ftsst"),
        ("VP_066", "ftsst"),
        ("VP_080", "tsst"),
        ("VP_086", "tsst"),
        ("VP_086", "ftsst"),
        ("VP_087", "tsst"),
        ("VP_087", "ftsst"),
        ("VP_091", "tsst"),
        ("VP_091", "ftsst"),
        ("VP_094", "tsst"),
        ("VP_094", "ftsst"),
        ("VP_097", "tsst"),
        ("VP_101", "tsst"),
        ("VP_102", "ftsst"),  # incomplete video
        ("VP_112", "ftsst"),
        ("VP_114", "tsst"),  # multiple partial videos
        ("VP_115", "ftsst"),  # multiple partial videos
        ("VP_120", "tsst"),  # incomplete video
        ("VP_121", "ftsst"),
        ("VP_123", "tsst"),
        # ("VP_126", "ftsst"), to be checked
    )

    # extend SUBSETS_WITHOUT_BODY_VIDEO if audio is also missing
    SUBSETS_WITHOUT_AUDIO = SUBSETS_WITHOUT_BODY_VIDEO + (
        ("VP_003", "tsst"),
        ("VP_013", "ftsst"),
        ("VP_028", "tsst"),
        ("VP_040", "tsst"),
        ("VP_098", "tsst"),  # unvollständig
        ("VP_108", "ftsst"),  # silent
        ("VP_119", "tsst"),  # silent
    )

    SUBSETS_WITH_ARM_ERRORS = ()

    SUBSETS_WITHOUT_PREP = ()

    SUBSETS_WITHOUT_GAIT_TESTS = ()

    def __init__(
        self,
        base_path: path_t,
        index_from_folder: bool = True,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_complete_subjects_if_error: bool = True,
        exclude_without_mocap: bool = True,
        exclude_with_arm_errors: bool = False,
        exclude_without_prep: bool = False,
        exclude_without_gait_tests: bool = False,
        exclude_without_body_video: bool = False,
        exclude_without_audio: bool = False,
        use_cache: bool = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.index_from_folder = index_from_folder
        self.data_tabular_path = base_path.joinpath("data_tabular")
        self.exclude_complete_subjects_if_error = exclude_complete_subjects_if_error
        self.exclude_without_mocap = exclude_without_mocap
        self.exclude_with_arm_errors = exclude_with_arm_errors
        self.exclude_without_prep = exclude_without_prep
        self.exclude_without_gait_tests = exclude_without_gait_tests
        self.exclude_without_body_video = exclude_without_body_video
        self.exclude_without_audio = exclude_without_audio
        self.data_to_exclude = self._find_data_to_exclude(
            exclude_complete_subjects_if_error
        )
        self.use_cache = use_cache

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        if self.index_from_folder:
            subject_ids = [
                subject_dir.name
                for subject_dir in get_subject_dirs(
                    self.base_path.joinpath("data_per_subject"), "VP_*"
                )
            ]
        else:
            # list from VP_001 to VP_142
            subject_ids = [f"VP_{i:03d}" for i in range(1, 142)]

        index_cols = ["subject", "condition"]
        index = list(product(subject_ids, self.CONDITIONS))

        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()

        return index

    def _find_data_to_exclude(self, exclude_complete_subjects_if_error: bool):
        data_to_exclude = []
        if self.exclude_without_mocap:
            data_to_exclude += self.SUBSETS_WITHOUT_MOCAP
        if self.exclude_with_arm_errors:
            data_to_exclude += self.SUBSETS_WITH_ARM_ERRORS
        if self.exclude_without_prep:
            data_to_exclude += self.SUBSETS_WITHOUT_PREP
        if self.exclude_without_gait_tests:
            data_to_exclude += self.SUBSETS_WITHOUT_GAIT_TESTS
        if self.exclude_without_body_video:
            data_to_exclude += self.SUBSETS_WITHOUT_BODY_VIDEO
        if self.exclude_without_audio:
            data_to_exclude += self.SUBSETS_WITHOUT_AUDIO

        if exclude_complete_subjects_if_error:
            data_to_exclude = [x[0] for x in data_to_exclude]

        return data_to_exclude

    @property
    def subject(self) -> str:
        if not self.is_single("subject"):
            raise ValueError(
                "Subject data can only be accessed for a single participant!"
            )
        return self.index["subject"][0]

    @property
    def condition(self) -> str:
        if not self.is_single("condition"):
            raise ValueError(
                "Condition data can only be accessed for a single condition!"
            )
        return self.index["condition"][0]

    @property
    def language(self) -> str:
        if not self.is_single("subject"):
            raise ValueError(
                "Language data can only be accessed for a single participant!"
            )
        language = self.meta_data.loc[self.subject, "language"]

        if language == "deutsch":
            return "de"
        elif language == "englisch":
            return "en"
        else:
            raise ValueError(
                f"Unknown language '{language}' for subject {self.subject}!"
            )

    @property
    def sample_times_saliva(self) -> Sequence[int]:
        return self._sample_times_saliva

    @property
    def sample_times_bloodspot(self) -> Sequence[int]:
        return self._sample_times_bloodspot

    @property
    def meta_data(self) -> pd.DataFrame:
        path = self.base_path.joinpath("data_tabular/meta_data.xlsx")

        data = pd.read_excel(path, index_col="subject")

        return data

    @property
    def body_video_path(self) -> Path:
        if not self.is_single(None):
            raise ValueError(
                "Path can only be accessed for a single condition of a single participant!"
            )
        data_path = self.base_path.joinpath("body_videos").joinpath(
            f"video_body_{self.subject}_{self.condition}.mp4"
        )
        return data_path

    @property
    def audio_path(self) -> Path:
        if not self.is_single(None):
            raise ValueError(
                "Path can only be accessed for a single condition of a single participant!"
            )
        data_path = self.base_path.joinpath("body_videos/audio").joinpath(
            f"audio_{self.subject}_{self.condition}.wav"
        )
        return data_path

    @property
    def transcript_path(self):
        if not self.is_single(None):
            raise ValueError(
                "Path can only be accessed for a single condition of a single participant!"
            )
        data_path = self.base_path.joinpath("body_videos/transcript").joinpath(
            f"transcript_{self.subject}_{self.condition}.csv"
        )
        return data_path

    @property
    def diarization_path(self, method: str = "pyannote") -> Path:
        if not self.is_single(None):
            raise ValueError(
                "Path can only be accessed for a single condition of a single participant!"
            )
        dia_path = self.base_path.joinpath("body_videos/diarization").joinpath(
            f"diarization_{method}_{self.subject}_{self.condition}.csv"
        )
        return dia_path

    @property
    def speech_path(self):
        if not self.is_single(None):
            raise ValueError(
                "Path can only be accessed for a single condition of a single participant!"
            )
        data_path = self.base_path.joinpath("body_videos/transcript/speech").joinpath(
            f"speech_{self.subject}_{self.condition}.csv"
        )
        return data_path

    @cached_property
    def nilspod(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "NilsPod data can only be accessed for a single participant in a single condition!"
            )
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
            raise ValueError(
                "Heart rate data can only be accessed for a single participant in a single condition!"
            )

        subject_id = self.group.subject
        condition = self.group.condition
        ecg_path = self.ecg_output_path

        file_path = ecg_path.joinpath(f"hr_result_{subject_id}_{condition}_total.xlsx")
        if not file_path.exists():
            raise HeartRateDataNotFoundException(
                f"No heart rate data for {subject_id} {condition}."
            )
        data = load_pandas_dict_excel(file_path)
        return data

    @property
    def hrv(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "Heart rate data can only be accessed for a single participant in a single condition!"
            )

        subject_id = self.subject
        condition = self.condition
        ecg_path = self.ecg_output_path
        file_path = ecg_path.joinpath(f"hrv_result_{subject_id}_{condition}.csv")
        if not file_path.exists():
            raise HeartRateDataNotFoundException(
                f"No HRV data for {subject_id} {condition}."
            )
        return pd.read_csv(file_path, index_col="phase")

    @property
    def timelog_ecg_baseline(self):
        data = self.ecg
        data = data.drop(index=data.first("1min").index)
        timelog = pd.DataFrame(
            data.first("5min").index[[0, -1]], index=["start", "end"]
        ).T
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
        subject_id = self.subject
        condition = self.condition
        data_path = _build_data_path(
            self.base_path.joinpath("data_per_subject"), subject_id, condition
        )
        file_path = data_path.joinpath(
            f"timelog/cleaned/{subject_id}_{condition}_timelog_{timelog_type}.csv"
        )
        if not file_path.exists():
            raise TimelogNotFoundException(
                f"No time log data was found for {timelog_type} in the {condition} condition of {subject_id}!"
            )
        timelog = load_atimelogger_file(file_path, timezone="Europe/Berlin")
        # convert all column names of the multi-level column index to lower case
        timelog.columns = timelog.columns.set_levels(
            [level.str.lower() for level in timelog.columns.levels]
        )

        return timelog

    @property
    def timelog_total(self) -> pd.DataFrame:
        timelog = pd.concat(
            [self.timelog_ecg_baseline, self.timelog_gait, self.timelog_test], axis=1
        )
        return timelog.sort_values(by="time", axis=1)

    @property
    def questionnaire(self) -> pd.DataFrame:
        if self.is_single(["condition"]):
            raise ValueError(
                "Questionnaire data can not be accessed for a single condition!"
            )
        data = load_questionnaire_data(
            self.data_tabular_path.joinpath(
                "_archive/old/merged_total/questionnaire_data.xlsx"
            )
        )
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    @property
    def gender(self) -> pd.Series:
        return self.questionnaire["Gender"]

    @property
    def handedness(self) -> pd.Series:
        return self.questionnaire["handedness"]

    @property
    def questionnaire_scores(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath(
            "questionnaires/processed/questionnaire_data_processed.csv"
        )
        if not data_path.exists():
            raise ValueError(
                "Processed questionnaire data not available! "
                "Please run the 'questionnaires/Questionnaire_Processing.ipynb' notebook first!"
            )
        data = load_long_format_csv(data_path)
        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(
            conditions, level="condition"
        )

    @property
    def questionnaire_scores_relative(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath(
            "questionnaires/processed/questionnaire_data_processed_relative.csv"
        )
        if not data_path.exists():
            raise ValueError(
                "Processed relative questionnaire data not available! "
                "Please run the 'questionnaires/Questionnaire_Processing.ipynb' notebook first!"
            )
        data = load_long_format_csv(data_path)
        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(
            conditions, level="condition"
        )

    @property
    def pasa(self) -> pd.DataFrame:
        return self._extract_questionnaire_score("PASA")

    @property
    def stadi_state(self) -> pd.DataFrame:
        return self._extract_questionnaire_score("STADI_State")

    @property
    def panas(self) -> pd.DataFrame:
        return self._extract_questionnaire_score("PANAS")

    @property
    def panas_diff(self) -> pd.DataFrame:
        panas_data = self.panas
        panas_data = panas_data.drop("Total", level="subscale")
        panas_data = panas_data.reindex(["ftsst", "tsst"], level="condition").reindex(
            ["pre", "post"], level="time"
        )
        panas_data = panas_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return panas_data.reorder_levels(["subject", "condition", "subscale"])

    @property
    def stadi_state_diff(self) -> pd.DataFrame:
        stadi_data = self.stadi_state
        stadi_data = stadi_data.reindex(["pre", "post"], level="time")
        stadi_data = stadi_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return stadi_data.reorder_levels(["subject", "condition", "subscale"])

    @property
    def codebook(self) -> pd.DataFrame:
        return load_codebook(self.base_path.joinpath("questionnaires/codebook.csv"))

    @property
    def condition_order(self) -> pd.DataFrame:
        data = pd.read_csv(
            self.data_tabular_path.joinpath("_extras/condition_order.csv")
        )
        data = data.set_index("subject")[["condition_order"]]
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    @property
    def day_condition_map(self) -> pd.DataFrame:
        data = pd.read_csv(
            self.data_tabular_path.joinpath("_extras/condition_order.csv")
        )
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
        return self._load_saliva_features("cortisol_features")

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
    def blood_spots(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("bloodspots/processed/crp_samples.csv")
        if not data_path.exists():
            raise ValueError(
                "Processed bloodspot data not available! "
                "Please run the 'biomarker/Bloodspot_Processing.ipynb' notebook first!"
            )
        data = load_long_format_csv(data_path)
        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(
            conditions, level="condition"
        )

    @property
    def ecg_output_path(self) -> Path:
        if not self.is_single(None):
            raise ValueError(
                "Path can only be accessed for a single condition of a single participant!"
            )
        data_path = self.base_path.joinpath("data_per_subject").joinpath(
            f"{self.group.subject}/{self.group.condition}/nilspod/processed/ecg"
        )
        return data_path

    def add_cortisol_index(self, cort_data: pd.DataFrame) -> pd.DataFrame:
        index_levels = list(cort_data.index.names)
        new_index_levels = ["condition_order", "non_responder"]
        cort_data = cort_data.join(self.condition_order).join(self.cort_non_responder)
        cort_data = cort_data.set_index(new_index_levels, append=True)
        cort_data = cort_data.reorder_levels(
            index_levels[:-1] + new_index_levels + [index_levels[-1]]
        )

        return cort_data

    def _get_nilspod_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            data, fs = _cached_load_nilspod_data(self.base_path, subject_id, condition)
        else:
            data, fs = _load_nilspod_session(self.base_path, subject_id, condition)
        return data

    def _load_estradiol_progesterone(self):
        data_path = self.base_path.joinpath(
            "saliva/processed/progesterone_estradiol_samples.csv"
        )
        if not data_path.exists():
            raise ValueError(
                "Processed saliva data not available! "
                "Please run the 'biomarker/Saliva_Processing.ipynb' notebook first!"
            )
        data = pd.read_csv(data_path)
        data = data.set_index("subject")
        subject_ids = self.index["subject"].unique()
        return data.reindex(subject_ids).dropna()

    def _load_questionnaire_data(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath(
            "questionnaire_total/processed/empkins_macro_questionnaire_data.csv"
        )
        data = load_questionnaire_data(data_path)
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    def _extract_questionnaire_score(self, score_type: str):
        data = self.questionnaire_scores
        return data["data"].unstack("type")[[score_type]].dropna()

    def _load_saliva_data(self, saliva_type: str) -> pd.DataFrame:
        data_path = self.data_tabular_path.joinpath(f"saliva/final/{saliva_type}.csv")
        if not data_path.exists():
            raise ValueError(
                "Processed saliva data not available! "
                "Please run the 'biomarker/Saliva_Processing.ipynb' notebook first!"
            )
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return data.reindex(subject_ids, level="subject").reindex(
            conditions, level="condition"
        )

    def _load_saliva_features(self, saliva_type: str) -> pd.DataFrame:
        data_path = self.data_tabular_path.joinpath(f"saliva/final/{saliva_type}.csv")

        data = pd.read_csv(data_path)

        data_long = pd.melt(
            data, id_vars=["subject"], var_name="saliva_feature", value_name="data"
        )

        data_long[["prefix1", "prefix2", "feature", "condition"]] = data_long[
            "saliva_feature"
        ].str.split("-", expand=True)

        data_long.drop(["prefix1", "prefix2", "saliva_feature"], axis=1, inplace=True)
        data_long.set_index(["subject", "condition", "feature"], inplace=True)
        data_long.columns.name = f"{saliva_type}"

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return (
            data_long.reindex(subject_ids, level="subject")
            .reindex(conditions, level="condition")
            .sort_index()
        )
