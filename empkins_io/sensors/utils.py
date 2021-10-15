from typing import Tuple

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.time import get_time_from_date, timedelta_to_time

__all__ = ["align_to_reference_data"]


def align_to_reference_data(
    radar_data: pd.DataFrame, ref_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align radar data to simultaneously collected reference data.

    This function aligns both dataframes according to their time axis and cuts data to the overlapping region.

    .. note::
        Both dataframes are expected to have an :class:`~pandas.DatetimeIndex` with absolute time information.


    Parameters
    ----------
    radar_data : :class:`~pandas.DataFrame`
        dataframe with radar heart rate data, as returned by :func:`~empkins-io.sensors.sensors.a04.load_data`
    ref_data : :class:`~pandas.DataFrame`
        dataframe with reference data

    Returns
    -------
    radar_data : :class:`~pandas.DataFrame`
        aligned radar data
    ref_data : :class:`~pandas.DataFrame`
        aligned reference data

    """
    _assert_is_dtype(radar_data.index, pd.DatetimeIndex)
    _assert_is_dtype(ref_data.index, pd.DatetimeIndex)
    radar_time = timedelta_to_time(get_time_from_date(radar_data.index.to_series()))
    ref_time = timedelta_to_time(get_time_from_date(ref_data.index.to_series()))
    start_idx = np.max([radar_time[0], ref_time[0]])
    end_idx = np.min([radar_time[-1], ref_time[-1]])

    return radar_data.between_time(start_idx, end_idx), ref_data.between_time(
        start_idx, end_idx
    )
