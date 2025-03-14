"""Module providing functions for signal alignment.

This is useful for aligning a target signal to a reference signal.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.array_handling import sanitize_input_1d
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from scipy.optimize import minimize
from statsmodels.tsa.stattools import ccovf

from empkins_io.utils._types import arr_t

__all__ = ["chisqr_align", "phase_align", "upsample", "signal_align"]


def chisqr_align(
    reference: arr_t,
    target: arr_t,
    roi: Optional[Tuple[int, int]] = None,
    order: Optional[int] = 1,
    init: Optional[float] = 0.1,
    bound: Optional[int] = 1,
    **kwargs,  # pylint:disable=unused-argument
):
    """Align a target signal to a reference signal by minimizing the chi-squared between the two signals.

    This function aligns a target signal to a reference signal within a region of interest (ROI) by minimizing the
    chi-squared between the two signals.

    .. note::
        Depending on the shape of your signals providing a highly constrained prior is necessary when using a
        gradient based optimization technique in order to avoid local solutions.


    Parameters
    ----------
    reference : array-like
        reference signal that won't be shifted
    target : array-like
        target signal to be shifted to reference
    roi : tuple of int
        region of interest to compute chi-squared
    order : int, optional
        order of spline interpolation for shifting target signal.
        Default: 1
    init: int, optional
        initial guess to offset between the two signals.
        Default: 0.1
    bound: int, optional
        symmetric bounds for constraining the shift search around initial guess.
        Default: 1

    Returns
    -------
    float
        shift (offset) between target and reference signal

    """
    reference = sanitize_input_1d(reference)
    target = sanitize_input_1d(target)

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


def phase_align(
    reference: arr_t,
    target: arr_t,
    roi: Optional[Tuple[int, int]] = None,
    upsample_rate: Optional[int] = 100,
    **kwargs,  # pylint:disable=unused-argument
):
    """Align a target signal to a reference signal using cross-correlation.

    Cross-correlation is computed on the signals within a region of interest at a precision of 1./res.

    .. note::
        If data is cross-correlated at native resolution (i.e. ``upsample_rate`` = 1) this function can only
        achieve integer precision.

    Parameters
    ----------
    reference : array-like
        reference signal that won't be shifted
    target : array-like
        target signal to be shifted to reference
    roi : tuple of int, optional
        region of interest to compute chi-squared or ``None`` to use the whole signal as region of interest.
        Default: ``None``
    upsample_rate : int, optional
        factor to increase resolution of data via linear interpolation

    Returns
    -------
    float
        shift (offset) between target and reference signal

    """
    if roi is None:
        roi = [0, len(reference)]
    # convert to int to avoid indexing issues
    roi = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid
    r1 = upsample(reference[roi], upsample_rate=upsample_rate, interpol_type="linear")
    r2 = upsample(target[roi], upsample_rate=upsample_rate, interpol_type="linear")

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

    return np.argmax(cc) * mod * (1.0 / upsample_rate)


def upsample(
    data: Union[pd.DataFrame, pd.Series],
    upsample_rate: Optional[int] = 10,
    interpol_type: Optional[str] = "cubic",
) -> pd.DataFrame:
    """Upsample input data.

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

    column_name = data.columns if isinstance(data, pd.DataFrame) else [data.name]

    if isinstance(data.index, pd.DatetimeIndex):
        x_old = np.array((data.index - data.index[0]).total_seconds())
    else:
        x_old = np.array(data.index - data.index[0])

    step = 1 / upsample_rate
    x_new = np.arange(1, np.ceil(x_old[-1]) + step, step=step)

    data = sanitize_input_1d(data)

    interpol_f = interp1d(x=x_old, y=data, kind=interpol_type, fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=pd.Index(x_new, name="time"), columns=column_name)


def signal_align(
    reference: arr_t, target: arr_t, align_func: Optional[Callable] = None, **kwargs
) -> Tuple[arr_t, arr_t, float]:
    """Align a target signal to a reference signal.

    Parameters
    ----------
    reference : array-like
        reference signal that won't be shifted
    target : array-like
        target signal to be shifted to reference
    align_func : function, optional
        function to use for alignment. Choose one of
        {:func:`empkins_io.signal_alignment.phase_align`, :func:`empkins_io.signal_alignment.chisqr_align`}.
        Default: ``None`` (corresponds to default alignment function :func:`empkins_io.signal_alignment.phase_align`)
    kwargs : dict
        additional arguments passed to the alignment function. See the function documentations for more
        information about the required arguments.

    Returns
    -------
    reference : array-like
        reference signal
    target_shift : array-like
        shifted target signal
    float
        shift (offset) between target and reference signal

    """
    if align_func is None:
        align_func = phase_align
    shift_idx = align_func(reference, target, **kwargs)

    target_shift = shift(target, shift=shift_idx, mode="nearest")
    if isinstance(target, pd.DataFrame):
        target_shift = pd.DataFrame(target_shift, columns=target.columns, index=target.index)

    return reference, target_shift, shift_idx
