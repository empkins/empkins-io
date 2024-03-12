from itertools import product
from typing import Optional, Sequence, Union, Dict

from functools import cached_property, lru_cache

import pandas as pd
from tpcp import Dataset

from empkins_io.utils._types import path_t
from empkins_io.datasets.d05.guardian._tfm_tilt_table_loader import TFMTiltTableLoader
from empkins_io.datasets.d05.guardian._helper import _load_tfm_data

_cached_get_tfm_data = lru_cache(maxsize=4)(_load_tfm_data)


class GuardianTiltTableDataset(Dataset):
    """Dataset for the Task Force Monitor (TFM) data from the Guardian dataset."""

    base_path: path_t
    use_cache: bool

    SAMPLING_RATES = {"ecg_1": 500, "ecg_2": 500, "icg_der": 500}
    PHASES = ["BeginRecording", "Pause", "Valsalva", "HoldingBreath", "TiltUp", "TiltDown"]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: bool = True,
    ):
        self.base_path = base_path
        self.use_cache = use_cache
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        overview_df = pd.read_csv(self.base_path.joinpath("dataset_overview.csv"), sep=";")
        pids = list(overview_df["participant"])
        index = list(product(pids, self.PHASES))
        return pd.DataFrame(index, columns=["participant", "phase"])

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

        tfm_path = self.base_path.joinpath(f"Data_raw/{participant}/TFM-Daten/{lower_letters}_no01.mat")
        if self.use_cache:
            tfm_data = _cached_get_tfm_data(tfm_path)
        else:
            tfm_data = _load_tfm_data(tfm_path)

        if self.is_single(None):
            return tfm_data[phases[0]]
        print(tfm_data)
        return tfm_data
