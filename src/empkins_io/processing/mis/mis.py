from typing import Dict, Optional, Sequence, Union

import pandas as pd
from biopsykit.signals.ecg import EcgProcessor
from biopsykit.utils.data_processing import split_data
from biopsykit.utils.dtypes import HeartRatePhaseDict, RPeakDataFrame, is_r_peak_dataframe
from tqdm.auto import tqdm

__all__ = ["MISProcessor"]


class MISProcessor:
    def __init__(
        self,
        rpeaks: Union[RPeakDataFrame, Dict[str, RPeakDataFrame]],
        sampling_rate: Optional[float] = None,
        time_intervals: Optional[Union[pd.Series, Dict[str, Sequence[str]]]] = None,
        include_start: Optional[bool] = False,
    ):
        if sampling_rate is None:
            sampling_rate = 2000
        self.sampling_rate: float = sampling_rate
        """Sampling rate of recorded data."""

        if isinstance(rpeaks, dict):
            for _, df in rpeaks.items():
                is_r_peak_dataframe(df)
            rpeaks_dict = rpeaks
        else:
            is_r_peak_dataframe(rpeaks)
            if time_intervals is not None:
                # split data into subphases if time_intervals are passed
                rpeaks_dict = split_data(
                    data=rpeaks,
                    time_intervals=time_intervals,
                    include_start=include_start,
                )
            else:
                rpeaks_dict = {"Data": rpeaks}

        self.rpeaks: Dict[str, RPeakDataFrame] = rpeaks_dict
        """Dictionary with R peak location indices, split into different phases.

        See Also
        --------
        :obj:`~biopsykit.utils.datatype_helper.RPeakDataFrame`
            dictionary format

        """

        self.heart_rate: HeartRatePhaseDict = {key: df[["Heart_Rate"]] for key, df in self.rpeaks.items()}
        """Dictionary with time-series heart rate data, split into different phases.

        See Also
        --------
        :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
            dictionary format

        """

    def process(
        self,
        outlier_correction: Optional[Union[str, Sequence[str]]] = "all",
        outlier_params: Optional[Dict[str, Union[float, Sequence[float]]]] = None,
        title: Optional[str] = None,
    ):
        """Process Radar signal.

        The Radar processing pipeline consists of the following steps:

        * ``Outlier correction`` (optional): Uses :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_outlier`
          to check detected R peaks for outlier and impute removed outlier by linear interpolation.


        Parameters
        ----------
        outlier_correction : list, ``all`` or ``None``, optional
            List containing outlier correction methods to be applied. Alternatively, pass ``all`` to apply all
            available outlier correction methods, or ``None`` to not apply any outlier correction.
            See :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections` to get a list of possible
            outlier correction methods. Default: ``all``
        outlier_params : dict
            Dictionary of outlier correction parameters or ``None`` for default parameters.
            See :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default` for the default parameters.
            Default: ``None``


        See Also
        --------
        :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_outlier`
            function to perform R peak outlier correction
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections`
            list of all available outlier correction methods
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default`
            dictionary with default parameters for outlier correction


        Examples
        --------
        >>> from empkins_io import MISProcessor
        >>> # initialize MISProcessor instance
        >>> mis_processor = MISProcessor(...)

        >>> # Option 1: don't apply any outlier correction
        >>> mis_processor.process(outlier_correction=None)

        >>> # Option 2: use default outlier correction pipeline
        >>> mis_processor.process()

        >>> # Option 3: use custom outlier correction pipeline: only physiological and statistical outlier with custom
        >>> # thresholds
        >>> methods = ["physiological", "statistical"]
        >>> params = {
        >>>    'physiological': (50, 150),
        >>>    'statistical': 2.576
        >>>}
        >>> mis_processor.process(outlier_correction=methods, outlier_params=params)

        >>> # Print available results from ECG processing
        >>> print(mis_processor.rpeaks)

        """
        for name, rpeaks in tqdm(self.rpeaks.items(), desc=title):
            rpeaks_corr = self.correct_outlier(
                rpeaks=rpeaks,
                outlier_correction=outlier_correction,
                outlier_params=outlier_params,
                sampling_rate=self.sampling_rate,
            )
            self.rpeaks[name] = rpeaks_corr
            self.heart_rate[name] = rpeaks_corr[["Heart_Rate"]]

    @classmethod
    def correct_outlier(
        cls,
        mis_processor: Optional["MISProcessor"] = None,
        key: Optional[str] = None,
        rpeaks: Optional[RPeakDataFrame] = None,
        outlier_correction: Optional[Union[str, Sequence[str]]] = "all",
        outlier_params: Optional[Dict[str, Union[float, Sequence[float]]]] = None,
        sampling_rate: Optional[float] = 2000.0,
        **kwargs,
    ):
        _assert_input(mis_processor, key, rpeaks)
        if mis_processor is not None:
            rpeaks = mis_processor.rpeaks[key]
            sampling_rate = mis_processor.sampling_rate

        if outlier_correction == "all":
            outlier_correction = EcgProcessor.outlier_corrections()
        if isinstance(outlier_correction, str):
            outlier_correction = [outlier_correction]
        if "correlation" in outlier_correction:
            # We can't apply correlation outlier correction to MIS data because we don't have ECG waveform
            # to compute correlation from
            outlier_correction.remove("correlation")
        if "quality" in outlier_correction:
            # We can't apply quality outlier correction to MIS data because we don't have ECG signal quality
            # (we have another quality indicator but that's something different)
            outlier_correction.remove("quality")

        rpeaks = rpeaks.copy()
        _, rpeaks = EcgProcessor.correct_outlier(
            rpeaks=rpeaks,
            outlier_correction=outlier_correction,
            outlier_params=outlier_params,
            sampling_rate=sampling_rate,
            **kwargs,
        )
        return rpeaks


def _assert_input(mis_processor: MISProcessor, key: str, rpeaks: RPeakDataFrame):
    if all(x is None for x in [mis_processor, key, rpeaks]):
        raise ValueError("Either 'mis_processor' and 'key', or 'rpeaks' must be passed as arguments!")
    if mis_processor is not None and key is None:
        raise ValueError("Both of 'mis_processor' and 'key' must be passed as arguments!")
