from functools import cached_property, lru_cache
from itertools import product
from typing import Optional, Sequence, Tuple

import pandas as pd
from biopsykit.io import load_long_format_csv, load_questionnaire_data
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.utils._types import path_t
from tpcp import Dataset

from empkins_io.datasets.d03.macro_prestudy.helper import get_times_for_mocap, load_mocap_data

_cached_load_mocap_data = lru_cache(maxsize=4)(load_mocap_data)


class MacroPreStudyDataset(Dataset):

    SUBJECTS_WITHOUT_MOCAP: Tuple[str] = ("VP_01",)

    base_path: path_t
    exclude_without_mocap: bool
    normalize_mocap_time: bool
    use_cache: bool
    _sampling_rate: float = 1.0 / 0.017
    _sample_times: Tuple[int] = (-20, -1, 0, 10, 20, 45)

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_without_mocap: Optional[bool] = True,
        normalize_mocap_time: Optional[bool] = True,
        use_cache: Optional[bool] = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_without_mocap = exclude_without_mocap
        self.use_cache = use_cache
        self.normalize_mocap_time = normalize_mocap_time
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]
        conditions = ["ftsst", "tsst"]
        if self.exclude_without_mocap:
            for subject_id in self.SUBJECTS_WITHOUT_MOCAP:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        index = list(product(conditions, ["talk"]))
        index.append(("tsst", "math"))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]
        index = pd.DataFrame(index, columns=["subject", "condition", "phase"])
        return index

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def sample_times(self) -> Sequence[int]:
        return self._sample_times

    @property
    def questionnaire(self):
        if self.is_single(["phase"]):
            raise ValueError(f"questionnaire data can not be accessed for individual phases!")
        if self.is_single(["condition"]):
            raise ValueError(f"questionnaire data can not be accessed for a single condition!")
        return self._load_questionnaire_data()

    @property
    def condition_first(self) -> pd.DataFrame:
        data = self.questionnaire[["TSST_first"]].replace({True: "TSST first", False: "fTSST first"})
        data.columns = ["condition_first"]
        return data

    @property
    def cortisol_non_responder(self) -> pd.DataFrame:
        non_responder = self.cortisol_features.xs("tsst", level="condition")
        non_responder = non_responder.xs("max_inc", level="saliva_feature") <= 1.5
        non_responder.columns = ["non_responder"]
        return non_responder

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

    @cached_property
    def mocap_data(self) -> pd.DataFrame:
        if self.is_single(None) or self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            data = self._get_mocap_data(subject_id, condition)
            if self.is_single(None):
                phase = self.index["phase"].unique()[0]
            else:
                phase = "total"
            times = get_times_for_mocap(self.base_path, self.sampling_rate, subject_id, condition, phase)
            data = data.loc[times[0] : times[1]]
            if self.normalize_mocap_time:
                data.index -= data.index[0]
            return data
        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def panas_diff(self) -> pd.DataFrame:
        panas_data = wide_to_long(self.questionnaire, "PANAS", levels=["subscale", "condition", "time"]).dropna()
        panas_data = panas_data.drop("Total", level="subscale")
        panas_data = panas_data.reindex(["ftsst", "tsst"], level="condition").reindex(["pre", "post"], level="time")
        panas_data = panas_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return self._apply_indices(panas_data).reorder_levels(
            ["subject", "condition", "condition_first", "non_responder", "subscale"]
        )

    @property
    def stadi_state_diff(self) -> pd.DataFrame:
        stadi_data = wide_to_long(self.questionnaire, "STADI_State", levels=["subscale", "condition", "time"]).dropna()
        stadi_data = stadi_data.reindex(["pre", "post"], level="time")
        stadi_data = stadi_data.unstack("time").diff(axis=1).stack().droplevel(-1)
        return self._apply_indices(stadi_data).reorder_levels(
            ["subject", "condition", "condition_first", "non_responder", "subscale"]
        )

    @property
    def pasa(self) -> pd.DataFrame:
        pasa_data = wide_to_long(self.questionnaire, "PASA", levels=["subscale"]).dropna()
        return self._apply_indices(pasa_data).reorder_levels(
            ["subject", "condition_first", "non_responder", "subscale"]
        )

    def _apply_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.join(self.condition_first).join(self.cortisol_non_responder)
        data = data.set_index(["condition_first", "non_responder"], append=True)
        return data

    def _get_mocap_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.base_path, subject_id, condition)
        return load_mocap_data(self.base_path, subject_id, condition)

    def _load_questionnaire_data(self) -> pd.DataFrame:
        data_path = self.base_path.joinpath("questionnaire_total/processed/empkins_macro_questionnaire_data.csv")
        data = load_questionnaire_data(data_path)
        subject_ids = self.index["subject"].unique()
        return data.loc[subject_ids]

    def _load_saliva_data(self, saliva_type: str) -> pd.DataFrame:
        if self.is_single("phase"):
            raise ValueError(f"{saliva_type} data can not be accessed for individual phases!")
        data_path = self.base_path.joinpath(f"saliva_total/processed/empkins_macro_{saliva_type}.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    def _load_saliva_features(self, saliva_type: str) -> pd.DataFrame:
        if self.is_single("phase"):
            raise ValueError(f"{saliva_type} features can not be accessed for individual phases!")
        data_path = self.base_path.joinpath(f"saliva_total/processed/empkins_macro_{saliva_type}_features.csv")
        data = load_long_format_csv(data_path)

        subject_ids = self.index["subject"].unique()
        conditions = self.index["condition"].unique()
        return multi_xs(multi_xs(data, subject_ids, level="subject"), conditions, level="condition")

    def add_cortisol_index(self, cort_data: pd.DataFrame) -> pd.DataFrame:
        index_levels = list(cort_data.index.names)
        new_index_levels = ["condition_first", "non_responder"]
        cort_data = cort_data.join(self.condition_first).join(self.cortisol_non_responder)
        cort_data = cort_data.set_index(new_index_levels, append=True)
        cort_data = cort_data.reorder_levels(index_levels[:-1] + new_index_levels + [index_levels[-1]])

        return cort_data