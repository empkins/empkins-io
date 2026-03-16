import numpy as np
import pandas as pd
import scipy


class SyncedData:
    def __init__(self, subset):
        self.subset = subset

    def sync(self, search_window_min=0.8) -> pd.Timedelta:
        mocap_resampled = pd.DataFrame(
            scipy.signal.resample_poly(self.subset.mocap_data, up=256, down=60, axis=0),
            columns=self.subset.mocap_data.columns,
        )

        mocap_channels = mocap_resampled["mvnx_segment"]["T8"].filter(like="gyr")

        start = self.subset.timelog_test["prep", "start"].iloc[0] - pd.Timedelta(
            minutes=search_window_min
        )
        end = self.subset.timelog_test["math", "end"].iloc[0] + pd.Timedelta(
            minutes=search_window_min
        )

        nilspod_channels = self.subset.nilspod[
            self.subset.NILSPOD_MAPPING["chest"]
        ].filter(like="gyr")[start:end]

        if len(nilspod_channels) == 0:
            raise ValueError("No nilspod data found.")

        mocap_norm = self.normalize(mocap_channels)
        nilspod_norm = self.normalize(nilspod_channels)

        corr = np.correlate(nilspod_norm, mocap_norm, mode="full")
        lags = scipy.signal.correlation_lags(
            nilspod_norm.size, mocap_norm.size, mode="full"
        )

        # Find the lag that maximizes correlation
        best_lag = lags[np.argmax(corr)]

        id = (self.subset.subject, self.subset.condition)

        print(id)

        # Manually correct failed
        if id == ("VP_18", "tsst"):
            best_lag = lags[271182]
        elif id == ("VP_23", "tsst"):
            best_lag = lags[260182]
        elif id == ("VP_25", "ftsst"):
            best_lag = lags[257232]
        elif id == ("VP_25", "tsst"):
            best_lag = lags[253665]
        elif id == ("VP_29", "ftsst"):
            best_lag = lags[255368]
        elif id == ("VP_29", "tsst"):
            best_lag = lags[254313]
        elif id == ("VP_34", "tsst"):
            best_lag = lags[263027]
        elif id == ("VP_39", "tsst"):
            best_lag = lags[348236]

        # get new start times for both signals
        start_time_nilspod = nilspod_channels.iloc[best_lag].name

        shift = start_time_nilspod - (
            pd.Timestamp(self.subset.start_mocap_timestamp)
            + pd.Timedelta(seconds=self.subset.mocap_data.index[0])
        )
        print(f"Shift in seconds: {shift.total_seconds()}")

        self.subset.nilspod.index = self.subset.nilspod.index - shift

        return shift

    def normalize(self, data):
        data = np.linalg.norm(data, axis=1)
        return (data - np.mean(data)) / np.std(data)
