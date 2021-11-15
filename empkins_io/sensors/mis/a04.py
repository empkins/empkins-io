import datetime
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
from biopsykit.utils._types import path_t

from empkins_io.sensors.mis import mis

__all__ = ["load_data", "load_data_raw", "load_data_folder"]


def load_data_folder(
    folder_path: path_t,
    phase_names: Optional[Sequence[str]] = None,
    datastreams: Optional[Union[mis.DATASTREAMS, Sequence[mis.DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], float]:
    warnings.warn("The naming 'a04' is deprecated! Consider using 'mis' instead.", category=DeprecationWarning)
    return mis.load_data_folder(folder_path, phase_names=phase_names, datastreams=datastreams, timezone=timezone)


# TODO define MIS data datatypes
def load_data(
    path: path_t,
    datastreams: Optional[Union[mis.DATASTREAMS, Sequence[mis.DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], float]:
    warnings.warn("The naming 'a04' is deprecated! Consider using 'mis' instead.", category=DeprecationWarning)
    return mis.load_data(path, datastreams=datastreams, timezone=timezone)


def load_data_raw(path: path_t, timezone: Optional[Union[datetime.tzinfo, str]] = None) -> Tuple[pd.DataFrame, float]:
    warnings.warn("The naming 'a04' is deprecated! Consider using 'mis' instead.", category=DeprecationWarning)
    return mis.load_data_raw(path, timezone=timezone)
