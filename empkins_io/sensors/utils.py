from typing import Tuple

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.time import get_time_from_date, timedelta_to_time

__all__ = ["cut_data_to_overlap"]


def cut_data_to_overlap(
    reference: pd.DataFrame,
    target: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cut target data and simultaneously collected reference data to overlapping regions according to their time axis.

    .. note::
        Both dataframes are expected to have an :class:`~pandas.DatetimeIndex` with absolute time information.
        If the dataframes do not overlap, two empty dataframes will be returned.


    Parameters
    ----------
    reference : :class:`~pandas.DataFrame`
        dataframe with reference data
    target : :class:`~pandas.DataFrame`
        dataframe with target data, typically radar data

    Returns
    -------
    reference : :class:`~pandas.DataFrame`
        reference data cut to overlapping region
    target : :class:`~pandas.DataFrame`
        aligned target cut to overlapping region

    """
    _assert_is_dtype(reference.index, pd.DatetimeIndex)
    _assert_is_dtype(target.index, pd.DatetimeIndex)
    reference_time = timedelta_to_time(get_time_from_date(reference.index.to_series()))
    target_time = timedelta_to_time(get_time_from_date(target.index.to_series()))
    start_idx = np.max([target_time[0], reference_time[0]])
    end_idx = np.min([target_time[-1], reference_time[-1]])

    if start_idx > end_idx: 
        # no overlap
        return pd.DataFrame(columns=reference.columns.copy()), pd.DataFrame(columns=target.columns.copy())
    else:
        # extract overlapping regions
        return reference.between_time(start_idx, end_idx), target.between_time(start_idx, end_idx)
