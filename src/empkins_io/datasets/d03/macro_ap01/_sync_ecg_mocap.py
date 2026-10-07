import warnings

import numpy as np
import pandas as pd
import scipy.signal

METHOD_NORMALIZED = "normalized"
METHOD_RAW = "raw"

# Manual corrections for the raw method (index into the full cross-correlation, only valid for search_window_min=0.8
# and the T8 segment). Not needed for the normalized method, which finds all of these on its own.
_RAW_MANUAL_CORRECTIONS = {
    ("VP_18", "tsst"): 271182,
    ("VP_23", "tsst"): 260182,
    ("VP_25", "ftsst"): 257232,
    ("VP_25", "tsst"): 253665,
    ("VP_29", "ftsst"): 255368,
    ("VP_29", "tsst"): 254313,
    ("VP_34", "tsst"): 263027,
    ("VP_39", "tsst"): 348236,
}


class SyncedData:
    """Synchronize the chest NilsPod with the Xsens mocap of a single subject and condition.

    The gyroscope norm of the NilsPod (between prep start and math end, extended by ``search_window_min``) is
    cross-correlated with the gyroscope norm of a mocap segment (``mocap_data``, i.e. talk start - math end,
    resampled to the NilsPod sampling rate). The resulting ``shift`` is the offset of the NilsPod clock relative to
    the mocap clock::

        shift = t_nilspod - t_mocap      i.e.      t_mocap = t_nilspod - shift

    The subset's data is not modified; use :meth:`synced_nilspod` to get NilsPod data on the mocap clock.

    Two methods are available:

    - ``"normalized"`` (default): Pearson correlation at every lag at which the mocap signal lies completely within
      the NilsPod search window. Robust against stretches with a lot of NilsPod movement.
    - ``"raw"``: unnormalized cross-correlation over all lags, as originally implemented, including the manual
      corrections for recordings where it picks the wrong peak. Reproduces ``data_tabular/_extras/sync_shifts.csv``.

    After :meth:`sync`, the attributes ``shift``, ``lag`` (samples), ``correlation`` (Pearson r of the aligned
    gyroscope norms; below ~0.3 the sync is unreliable), ``lags`` and ``scores`` are available.
    """

    FS_NILSPOD = 256

    def __init__(self, subset, *, nilspod_sensor: str = "chest", mocap_segment: str = "T8"):
        self.subset = subset
        self.nilspod_sensor = nilspod_sensor
        self.mocap_segment = mocap_segment

        self.shift: pd.Timedelta | None = None
        self.lag: int | None = None
        self.correlation: float | None = None
        self.lags: np.ndarray | None = None
        self.scores: np.ndarray | None = None

    def sync(self, search_window_min: float = 0.8, method: str = METHOD_NORMALIZED) -> pd.Timedelta:
        if method not in (METHOD_NORMALIZED, METHOD_RAW):
            raise ValueError(f"Unknown method '{method}', use '{METHOD_NORMALIZED}' or '{METHOD_RAW}'.")

        mocap_data = self.subset.mocap_data
        mocap_gyr = mocap_data["mvnx_segment"][self.mocap_segment].filter(like="gyr")
        mocap_resampled = scipy.signal.resample_poly(
            mocap_gyr, up=self.FS_NILSPOD, down=int(self.subset.sampling_rate), axis=0
        )

        timelog = self.subset.timelog_test
        start = timelog["prep", "start"].iloc[0] - pd.Timedelta(minutes=search_window_min)
        end = timelog["math", "end"].iloc[0] + pd.Timedelta(minutes=search_window_min)
        nilspod = self.subset.nilspod[self.subset.NILSPOD_MAPPING[self.nilspod_sensor]]
        nilspod_gyr = nilspod.filter(like="gyr")[start:end]
        if len(nilspod_gyr) == 0:
            raise ValueError("No NilsPod data found in the search window.")

        nilspod_norm = self.normalize(nilspod_gyr)
        mocap_norm = self.normalize(mocap_resampled)

        # FFT-based, identical to np.correlate(nilspod_norm, mocap_norm, mode="full")
        corr = scipy.signal.correlate(nilspod_norm, mocap_norm, mode="full", method="fft")
        lags = scipy.signal.correlation_lags(nilspod_norm.size, mocap_norm.size, mode="full")
        ncc = self._normalized_correlation(nilspod_norm, mocap_norm.size, corr, lags)

        if method == METHOD_NORMALIZED:
            if np.isnan(ncc).all():
                raise ValueError("NilsPod search window is shorter than the mocap recording.")
            scores = np.nan_to_num(ncc, nan=-np.inf)
            best_idx = int(np.argmax(scores))
        else:
            scores = corr
            key = (self.subset.subject, self.subset.condition)
            best_idx = _RAW_MANUAL_CORRECTIONS.get(key, int(np.argmax(corr)))

        lag = int(lags[best_idx])
        if 0 <= lag < len(nilspod_gyr):
            start_time_nilspod = nilspod_gyr.index[lag]
        else:  # mocap starts outside the search window (raw method only)
            start_time_nilspod = nilspod_gyr.index[0] + pd.Timedelta(seconds=lag / self.FS_NILSPOD)
        start_time_mocap = pd.Timestamp(self.subset.start_mocap_timestamp) + pd.Timedelta(
            seconds=mocap_data.index[0]
        )

        self.shift = start_time_nilspod - start_time_mocap
        self.lag = lag
        self.correlation = float(ncc[best_idx])
        self.lags = lags
        self.scores = scores

        if not self.correlation > 0.3:
            warnings.warn(
                f"Weak synchronization for {self.subset.subject} {self.subset.condition} "
                f"(r={self.correlation:.2f}); check the NilsPod and mocap gyroscope data.",
                stacklevel=2,
            )
        return self.shift

    def synced_nilspod(self) -> pd.DataFrame:
        """NilsPod data (all sensors) with the index shifted onto the mocap clock (``t_nilspod - shift``)."""
        if self.shift is None:
            raise ValueError("Call sync() first.")
        data = self.subset.nilspod.copy()
        data.index = data.index - self.shift
        return data

    @staticmethod
    def normalize(data) -> np.ndarray:
        data = np.linalg.norm(data, axis=1)
        return (data - np.mean(data)) / np.std(data)

    @staticmethod
    def _normalized_correlation(
        nilspod_norm: np.ndarray, mocap_size: int, corr: np.ndarray, lags: np.ndarray
    ) -> np.ndarray:
        """Pearson correlation at every full-overlap lag (NaN elsewhere).

        The mocap norm has mean 0 and std 1, so dividing the dot product by the standard deviation of the
        overlapping NilsPod window gives the Pearson correlation.
        """
        ncc = np.full(len(corr), np.nan)
        if nilspod_norm.size < mocap_size:
            return ncc
        csum = np.concatenate([[0.0], np.cumsum(nilspod_norm)])
        csum2 = np.concatenate([[0.0], np.cumsum(nilspod_norm**2)])
        local_mean = (csum[mocap_size:] - csum[:-mocap_size]) / mocap_size
        local_var = (csum2[mocap_size:] - csum2[:-mocap_size]) / mocap_size - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 1e-12))
        full_overlap = (lags >= 0) & (lags <= nilspod_norm.size - mocap_size)
        ncc[full_overlap] = corr[full_overlap] / (mocap_size * local_std)
        return ncc
