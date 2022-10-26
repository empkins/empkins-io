import pandas as pd
from numpy.linalg import norm


def find_peak_around_timestamp(time_stamp: pd.Timestamp, series: pd.Series, roi_minutes: float=1.):
    start_roi = time_stamp - pd.Timedelta(roi_minutes, 'min')
    end_roi = time_stamp + pd.Timedelta(roi_minutes, 'min')

    series_roi = series.loc[start_roi:end_roi]
    peak = series_roi.idxmax()
    return peak


def get_claps_from_board_and_timelog(board_data: pd.DataFrame, timelog: pd.DataFrame):
    """ Returns the timestamps of the norm maximum in the timespan 1min before and after the (f)TSST beginning and end, each.

    Parameters
    ---------
    board_data: pd.DataFrame
        contains the gyro data from the "board" NilsPod
    timelog: pd.DataFrame
        (f)TSST timelog, as returned e.g. by MacroBaseDataset.timelog_test

    Returns
    -------
    tuple
        the pd.Timestamp of the first clap and the second clap

    """
    board_data["norm"] = norm(board_data, axis=1)
    first_timestamp = timelog.prep.end.iloc[0]
    second_timestamp = timelog.math.end.iloc[0]
    if second_timestamp - first_timestamp > pd.Timedelta(15, 'min'):
        # TODO does this need more advanced error handling?
        raise Warning("Preparation end and math end timestamps are more than 15min apart. This is unusual.")

    first_clap = find_peak_around_timestamp(first_timestamp, board_data.norm)
    second_clap = find_peak_around_timestamp(second_timestamp, board_data.norm)
    return first_clap, second_clap


