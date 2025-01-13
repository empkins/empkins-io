import ast
import json
from pathlib import Path
from itertools import product
from typing import Optional, Sequence, Union, Dict

from functools import cached_property, lru_cache

import pandas as pd
import numpy as np
from tpcp import Dataset

from empkins_io.utils._types import path_t
from empkins_io.datasets.d05.guardian._tfm_tilt_table_loader import TFMTiltTableLoader
from empkins_io.datasets.d05.guardian._helper import _load_tfm_data
from empkins_micro.feature_extraction.pep.algorithms.ecg.extraction_heartbeats import HeartBeatExtraction

_cached_get_tfm_data = lru_cache(maxsize=4)(_load_tfm_data)


class GuardianTiltTableDataset(Dataset):
    """Dataset for the Task Force Monitor (TFM) data from the Guardian dataset."""

    base_path: path_t
    use_cache: bool

    SAMPLING_RATES = {"ecg_1": 500, "ecg_2": 500, "icg_der": 500}
    PHASES = ["Pause", "Valsalva", "HoldingBreath", "TiltUp", "TiltDown"]

    SUBSET_NO_RECORDED_DATA = (
        ("GDN0006", "HoldingBreath"),
        ("GDN0009", "HoldingBreath"),
        ("GDN0010", "Valsalva"),
        ("GDN0017", "Pause"),
        ("GDN0018", "TiltDown"),
        ("GDN0020", "TiltUp"),
        ("GDN0022", "TiltUp"),
        ("GDN0024", "TiltDown"),
        ("GDN0025", "Valsalva"),
        ("GDN0028", "TiltUp"),
        ("GDN0030", "Pause"),
        ("GDN0030", "TiltUp"),
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        exclude_no_recorded_data: bool = True,
        use_cache: bool = True,
        only_labeled: bool = False,
    ):
        self.base_path = base_path
        self.exclude_no_recorded_data = exclude_no_recorded_data
        self.data_to_exclude = self._find_data_to_exclude()
        self.use_cache = use_cache
        self.only_labeled = only_labeled
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        overview_df = pd.read_csv(self.base_path.joinpath("dataset_overview.csv"), sep=";")
        pids = list(overview_df["participant"])
        index = list(product(pids, self.PHASES))
        index = pd.DataFrame(index, columns=["participant", "phase"])
        for item in self.data_to_exclude:
            index = index.drop(index[(index["participant"] == item[0]) & (index["phase"] == item[1])].index)
        index = index.reset_index(drop=True)

        return index

    def _find_data_to_exclude(self) -> pd.DataFrame:
        data_to_exclude = []
        if self.exclude_no_recorded_data:
            data_to_exclude = self.SUBSET_NO_RECORDED_DATA

        return data_to_exclude

    @property
    def tfm_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        lower_letters = self.index["participant"][0].lower()
        participant = self.index["participant"][0]
        phases = self.index["phase"]

        if not (self.is_single(None) or len(phases) == len(self.PHASES)):
            raise ValueError(
                "TFM data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )
        if participant == "GDN0025":
            tfm_path = Path(f"Data_raw/{participant}/TFM-Daten/{lower_letters}_no02.mat")
        else:
            tfm_path = Path(f"Data_raw/{participant}/TFM-Daten/{lower_letters}_no01.mat")
        if self.use_cache:
            tfm_data = _cached_get_tfm_data(self.base_path, tfm_path)
        else:
            tfm_data = _load_tfm_data(self.base_path, tfm_path)
        if self.only_labeled:

            label_path = self.base_path.joinpath(
                f"Data_raw/{participant}/TFM-Daten/manual_labeling/labeling_borders_{participant}.csv"
            )
            data = pd.read_csv(label_path)[["description", "time"]]
            data["description"] = data["description"].apply(lambda s: ast.literal_eval(s))

            if self.is_single(None):
                rows = data[data["description"].apply(lambda x: phases[0] in x.keys())]
                start = rows["time"].iloc[0]
                end = rows["time"].iloc[1]
                tfm_data = tfm_data[phases[0]]
                tfm_data = tfm_data.loc[(tfm_data.index >= start) & (tfm_data.index <= end)]
                return tfm_data

        else:
            if self.is_single(None):
                tfm_data = tfm_data[phases[0]]
                return tfm_data
        return tfm_data

    def calculate_pep_manual_labeled(self, ecg_clean):
        # calculate the pep from manually labeled points
        fs = self.SAMPLING_RATES["ecg_2"]
        phase = self.index["phase"][0]
        heartbeat_algo = HeartBeatExtraction()
        heartbeat_algo.extract(ecg_clean=ecg_clean, sampling_rate_hz=fs)
        heartbeats = heartbeat_algo.heartbeat_list_

        # load manually labeled points
        data_ICG, data_ECG = self.load_manual_labeled

        if data_ICG is None or data_ECG is None:
            return None, None, None
        row = data_ECG[(data_ECG["Channel"] == phase) & (data_ECG["Label"] == "start")]

        # get start of first heartbeat
        start_value = row["Samples"].values[0]

        data_ICG.loc[data_ICG["Channel"] == "Artefact", "Samples"] = np.nan
        data_ECG.loc[data_ECG["Channel"] == "Artefact", "Samples"] = np.nan
        data_ICG = data_ICG[(data_ICG["Channel"] == "ICG") | (data_ICG["Channel"] == "Artefact")]

        data_ECG = data_ECG[(data_ECG["Channel"] == "ECG") | (data_ECG["Channel"] == "Artefact")]
        b_points = data_ICG["Samples"].values

        q_onset = data_ECG["Samples"].values

        if b_points[0] < q_onset[0]:

            b_points = b_points[1:]
        if b_points[-1] < q_onset[-1]:

            q_onset = q_onset[:-1]

        # calculate pep from manually labeled points
        pep_df = pd.DataFrame((b_points - q_onset) / fs * 1000, columns=["pep"])
        pep_df.index = range(len(pep_df))
        pep_df.index.name = "heartbeat_id"

        # correcct sample number to match the start of the phase (needed since the heartbeats are only calculated inside the random selected part of the phase and not in realtion to the whole phase)
        heartbeats["start_sample"] = heartbeats["start_sample"] + start_value
        heartbeats["end_sample"] = heartbeats["end_sample"] + start_value

        # exclude labeled points that are part of uncomplete heartbeats

        if q_onset[0] < heartbeats["start_sample"].values[0]:

            q_onset = q_onset[1:]
            b_points = b_points[1:]
            pep_df = pep_df[1:]

        if q_onset[-1] > heartbeats["end_sample"].values[-1]:

            q_onset = q_onset[:-1]
            b_points = b_points[:-1]
            pep_df = pep_df[:-1]

        return b_points, q_onset, pep_df, start_value

    @property
    def load_manual_labeled(self):
        participant = self.index["participant"][0]
        phase = self.index["phase"][0]
        deploy_type = "local"
        base_path = Path("..")
        data_path = Path(json.load(base_path.joinpath("config.json").open(encoding="utf-8"))[deploy_type]["base_path"])
        data_path_ICG = data_path.joinpath(
            "Data_raw", f"{participant}", "TFM-Daten", "manual_labeling", f"{participant}_{phase}_ICG.csv"
        )
        data_path_ECG = data_path.joinpath(
            "Data_raw", f"{participant}", "TFM-Daten", "manual_labeling", f"{participant}_{phase}_ECG.csv"
        )

        if not data_path_ICG.exists() or not data_path_ECG.exists():
            return None, None

        # read files of manually labeled points
        data_ICG = pd.read_csv(data_path_ICG)

        data_ECG = pd.read_csv(data_path_ECG)

        return data_ICG, data_ECG

    def correct_start_point(self, heartbeats, b_points=[], q_points=[], c_points=[], pep_results=[]):
        # correct samples such manually labeled and calculated ones match

        rows = self.load_annotations()

        start = rows["pos"][0]

        # adding start to get the sample number for the complete phase and not just random selected part

        b_points = b_points + start

        q_points = q_points + start
        c_points = c_points + start

        heartbeats["start_sample"] = heartbeats["start_sample"] + start
        heartbeats["end_sample"] = heartbeats["end_sample"] + start
        return b_points, q_points, heartbeats, c_points, pep_results

    def load_annotations(self):
        # load annotations of label borders of random selected part of the phase
        participant = self.index["participant"][0]
        phase = self.index["phase"][0]
        annotations_path = self.base_path.joinpath(
            f"Data_raw/{participant}/TFM-Daten/manual_labeling/labeling_borders_{participant}.csv"
        )
        if not annotations_path.exists():
            return None
        data = pd.read_csv(annotations_path)[["pos", "description"]]
        data["description"] = data["description"].apply(lambda s: ast.literal_eval(s))
        rows = data[data["description"].apply(lambda x: phase in x.keys())]
        rows = rows.reset_index()
        return rows
