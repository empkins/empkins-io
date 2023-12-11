import pandas as pd
from numpy.linalg import norm
import numpy as np


def find_peak_around_timestamp(time_stamp: pd.Timestamp, series: pd.Series, roi_minutes: float = 1.0):
    start_roi = time_stamp - pd.Timedelta(roi_minutes, "min")
    end_roi = time_stamp + pd.Timedelta(roi_minutes, "min")

    series_roi = series.loc[start_roi:end_roi]
    peak = series_roi.idxmax()
    return peak


def get_claps_from_board_and_timelog(board_data: pd.DataFrame, timelog: pd.DataFrame, phase_fine=False):
    """Returns the timestamps of the norm maximum in the timespan 1min before and after the (f)TSST beginning and end, each.

    Parameters
    ----------
    board_data: pd.DataFrame
        contains the gyro data from the "board" NilsPod
    timelog: pd.DataFrame
        (f)TSST timelog, as returned e.g. by MacroBaseDataset.timelog_test

    Returns
    -------
    tuple
        the pd.Timestamp of the first clap and the second clap

    """
    board_data = board_data.copy()
    board_data["norm"] = norm(board_data, axis=1)
    first_timestamp = timelog.Prep.end.iloc[0]
    second_timestamp = timelog.Math.end.iloc[0] if not phase_fine else timelog.Math_2.end.iloc[0]
    if second_timestamp - first_timestamp > pd.Timedelta(15, "min"):
        # TODO does this need more advanced error handling?
        raise Warning("Preparation end and math end timestamps are more than 15min apart. This is unusual.")

    first_clap = find_peak_around_timestamp(first_timestamp, board_data.norm)
    second_clap = find_peak_around_timestamp(second_timestamp, board_data.norm)
    return first_clap, second_clap


def get_xsens_start_and_end(xsens_sync_data: pd.DataFrame, timelog: pd.DataFrame, phase_fine=False):
    sync_signal = xsens_sync_data["analog_1"]
    first_timestamp = timelog.Prep.start.iloc[0]
    second_timestamp = timelog.Math.end.iloc[0] if not phase_fine else timelog.Math_2.end.iloc[0]
    if second_timestamp - first_timestamp > pd.Timedelta(25, "min"):
        # TODO does this need more advanced error handling?
        raise Warning("Preparation start and math end timestamps are more than 25min apart. This is unusual.")

    xsens_start = find_peak_around_timestamp(first_timestamp, sync_signal)
    xsens_end = find_peak_around_timestamp(second_timestamp, sync_signal)
    return xsens_start, xsens_end

def get_biopac_start(biopac_data:pd.DataFrame, timelog:pd.DataFrame):
    biopac_signal = biopac_data["sync"]
    first_timestamp = timelog.Prep.start.iloc[0]
    biopac_start = find_peak_around_timestamp(first_timestamp, biopac_signal)
    return biopac_start

def sync_biopac_with_nilspod(biopac_data:pd.DataFrame, xsens_sync_data: pd.DataFrame, timelog:pd.DataFrame, phase_fine=False):
    # get timestamp of nilspod peak
    xsens_start, _ = get_xsens_start_and_end(xsens_sync_data, timelog, phase_fine=phase_fine)
    # get timestamp of biopac peak
    biopac_start = get_biopac_start(biopac_data, timelog)
    # get difference between timelogs
    time_diff = xsens_start - biopac_start
    biopac_sync = biopac_data.copy()
    # synchronize biopac with nilspod
    biopac_sync.index = biopac_sync.index + time_diff

    if xsens_start != get_biopac_start(biopac_sync, timelog):
        raise Warning("Biopac and nilspod have different peak timestamps after synchronization.")
    return biopac_sync

