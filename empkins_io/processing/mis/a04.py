import warnings
from typing import Dict, Optional, Sequence, Union

import pandas as pd
from biopsykit.utils.datatype_helper import RPeakDataFrame

from empkins_io.processing.mis import MISProcessor


class A04Processor(MISProcessor):
    def __init__(
        self,
        rpeaks: Union[RPeakDataFrame, Dict[str, RPeakDataFrame]],
        sampling_rate: Optional[float] = None,
        time_intervals: Optional[Union[pd.Series, Dict[str, Sequence[str]]]] = None,
        include_start: Optional[bool] = False,
    ):
        super().__init__(rpeaks, sampling_rate, time_intervals, include_start)
        warnings.warn("The naming 'a04' is deprecated! Consider using 'mis' instead.", category=DeprecationWarning)
