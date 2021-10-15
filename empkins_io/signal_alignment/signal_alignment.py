from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.array_handling import sanitize_input_1d
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import ccovf


def chisqr_align(reference, target, roi=None, order=1, init=0.1, bound=1):
    """
    Align a target signal to a reference signal within a region of interest (ROI)
    by minimizing the chi-squared between the two signals. Depending on the shape
    of your signals providing a highly constrained prior is necessary when using a
    gradient based optimization technique in order to avoid local solutions.

    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        order (int): order of spline interpolation for shifting target signal
        init (int):  initial guess to offset between the two signals
        bound (int): symmetric bounds for constraining the shift search around initial guess

    Returns:
        shift (float): offset between target and reference signal

    Todo:
        * include uncertainties on spectra
        * update chi-squared metric for uncertainties
        * include loss function on chi-sqr

    """
    _assert_is_dtype(reference, np.ndarray)
    _assert_is_dtype(target, np.ndarray)
    if len(reference) != len(target):
        raise ValueError(
            "Both input signals need to have equal length! Got target {}, reference {}".format(
                len(target), len(reference)
            )
        )
    if roi is None:
        roi = [0, len(reference) - 1]

    # convert to int to avoid indexing issues
    roi = slice(int(roi[0]), int(roi[1]), 1)

    # normalize ref within ROI
    reference = reference / np.mean(reference[roi])

    # define objective function: returns the array to be minimized
    def fcn2min(x):
        shifted = shift(target, x, order=order)
        shifted = shifted / np.mean(shifted[roi])
        return np.sum(((reference - shifted) ** 2)[roi])

    # set up bounds for pos/neg shifts
    minb = min([(init - bound), (init + bound)])
    maxb = max([(init - bound), (init + bound)])

    # minimize chi-squared between the two signals
    result = minimize(fcn2min, init, method="L-BFGS-B", bounds=[(minb, maxb)])

    return result.x[0]


def phase_align(reference: np.ndarray, target: np.ndarray, roi: Tuple[int, int], upsample_rate: Optional[int] = 100):
    """Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision

    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation

    Returns:
        shift (float): offset between target and reference signal

    Parameters
    ----------
    reference : :class:`np.ndarray`
        reference signal that won't be shifted
    target : :class:`np.ndarray`
        target signal to be shifted to reference
    roi : tuple of int
        region of interest to compute chi-squared
    upsample_rate : int, optional
        factor to increase resolution of data via linear interpolation

    Returns
    -------

    """
    # convert to int to avoid indexing issues
    roi = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid
    x, r1 = upsample(reference[roi], upsample_rate=upsample_rate, interpol_type="linear")
    x, r2 = upsample(target[roi], upsample_rate=upsample_rate, interpol_type="linear")

    # subtract mean
    r1 -= r1.mean()
    r2 -= r2.mean()

    # compute cross covariance
    cc = ccovf(r1, r2, demean=False, adjusted=False)

    # determine if shift if positive/negative
    if np.argmax(cc) == 0:
        cc = ccovf(r2, r1, demean=False, adjusted=False)
        mod = -1
    else:
        mod = 1

    # often found this method to be more accurate then the way below
    return np.argmax(cc) * mod * (1.0 / upsample_rate)


def upsample(
    data: Union[pd.DataFrame, pd.Series], upsample_rate: Optional[int] = 10, interpol_type: Optional[str] = "cubic"
) -> pd.DataFrame:
    """Upsample input data to a frequency of 1 Hz.

    .. note::
        For resampling the index of ``data`` either be has to be a :class:`~pandas.DatetimeIndex`
        or a :class:`~pandas.Index` with time information in seconds.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        data to resample
    upsample_rate : int, optional
        factor to increase resolution of data via interpolation.
        Default: 10
    interpol_type : str, optional
        interpolation type passed to :func:`~scipy.interpolate.interp1d`.
        Default: "cubic"

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with upsampled data


    Raises
    ------
    ValueError
        If ``data`` is not a :class:`~pandas.DataFrame` or :class:`~pandas.Series`

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))

    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    else:
        column_name = [data.name]

    if isinstance(data.index, pd.DatetimeIndex):
        x_old = np.array((data.index - data.index[0]).total_seconds())
    else:
        x_old = np.array(data.index - data.index[0])

    step = 1 / upsample_rate
    x_new = np.arange(1, np.ceil(x_old[-1]) + step, step=step)

    data = sanitize_input_1d(data)

    interpol_f = interp1d(x=x_old, y=data, kind=interpol_type, fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=pd.Index(x_new, name="time"), columns=column_name)
