from functools import lru_cache
from typing import Dict, Optional, Sequence, Union, Tuple
import pandas as pd
from tpcp import Dataset
from pathlib import Path
from itertools import product
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.utils._types import path_t

from tpcp import Dataset


class RadarCardiaStenosisTest(Dataset):

    base_path: path_t
    use_cache: bool
    bp_tl_shift: Union[pd.Timedelta, None]

    _SAMPLING_RATES: Dict[str, float] = {
        "radar_original": 8000000 / 4096 / 2,
        "biopac_original": 2000,
        "resampled": 1000
    }

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
    }

    def __init__(
            self,
            base_path: path_t,
            groupby_cols: Optional[Sequence[str]] = None,
            subset_index: Optional[Sequence[str]] = None,
            use_cache: Optional[bool] = False,
    ):
        """
        Creates new instance of the RadarCardiaStudyDataset class
        Args:
            base_path: path_t (path to the base directory of the dataset)
            groupby_cols: Optional[Sequence[str]]
            subset_index: Optional[Sequence[str]]
            use_cache: Optional[bool] (currently not implemented)
        Returns:
        """
        self.base_path = base_path
        self.use_cache = use_cache

        self.bp_tl_shift = None

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        measurements = [
            "rest_elbow_post", "elbow_post", "rest_elbow_pre", "elbow_pre",
            "rest_wrist_post", "wrist_post", "rest_knee_pre", "knee_pre"
        ]

        index = list(product(subject_ids, measurements))
        index = pd.DataFrame(index, columns=["subject", "measurement"])

        return index

    @property
    def measurement(self) -> str:
        if not self.is_single(["measurement"]):
            raise ValueError("Measurement can only be accessed for a single measurement at once")
        return self.index["measurement"][0]

    @property
    def biopac_raw(self) -> pd.DataFrame:
        # biopac data unsynchronisiert
        pass

    def emrad_raw(self) -> pd.DataFrame:
        # radar data unsynchronisiert
        pass

    def biopac_synced(self) -> pd.DataFrame:
        # biopac data synchronisiert
        pass

    def emrad_synced(self) -> pd.DataFrame:
        # radar data synchronisiert
        pass

    def biopac_data(self) -> pd.DataFrame:
        # biopac data syncronisiert, zugeschnitten auf eine Messung
        pass

    def emrad_data(self) -> pd.DataFrame:
        # radar data syncronisiert, zugeschnitten auf eine Messung
        pass

    def timelog(self) -> pd.DataFrame:
        pass

