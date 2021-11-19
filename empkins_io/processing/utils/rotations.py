from typing import Optional, Sequence

import pandas as pd
from scipy.spatial.transform.rotation import Rotation
from pyquaternion import Quaternion
from biopsykit.utils._datatype_validation_helper import _assert_has_column_multiindex

from tqdm.auto import tqdm


def euler_to_quat_hierarchical(data: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert rotation data in a hierarchical dataframe from euler angles to quaternions.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in euler angles. Dataframe must be hierarchical, i.e., must have columns as
        :class:`~pandas.MultiIndex` where the lowest level contains the euler angles.
    columns : list of str, optional
        list of columns if only specific columns should be transformed to quaternions or ``None`` to
        transform all columns.
        Default: ``None``

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to quaternions. Either all columns of ``data`` or only the columns
        specified by ``columns``.

    """
    _assert_has_column_multiindex(data, nlevels=2, nlevels_atleast=True)
    if columns is None:
        columns = data.columns

    rot_quat_total = {}
    for col in columns:
        rot_quat_total[col] = euler_to_quat(data[col])

    return pd.concat(rot_quat_total, axis=1, names=[data.columns.names[0]])


def euler_to_quat(data: pd.DataFrame):
    """Convert rotation data from euler angles to quaternions.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in euler angles.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to quaternions.

    """
    rot_euler = Rotation.from_euler("yxz", data, degrees=True)
    rot_quat = rot_euler.as_quat()
    rot_quat = pd.DataFrame(data=rot_quat, columns=["w", "x", "y", "z"])
    rot_quat.columns.name = data.columns.name
    return rot_quat


def quat_to_euler_hierarchical(data: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert rotation data from quaternions to euler angles.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in quaternions. Dataframe must be hierarchical, i.e., must have columns as
        :class:`~pandas.MultiIndex` where the lowest level contains quaternion axes.
    columns : list of str, optional
        list of columns if only specific columns should be transformed to euler angles or ``None`` to
        transform all columns.
        Default: ``None``

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to euler angles. Either all columns of ``data`` or only the columns
        specified by ``columns``.

    """
    if columns is None:
        columns = data.columns

    rot_quat_total = {}
    for col in columns:
        euler_to_quat(data[col])

    return pd.concat(rot_quat_total, axis=1, names=["column", "axis"])


def euler_to_quat(data: pd.DataFrame):
    """Convert rotation data from euler angles to quaternions.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in euler angles.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to quaternions.

    """
    rot_euler = Rotation.from_euler("yxz", data, degrees=True)
    rot_quat = rot_euler.as_quat()
    rot_quat = pd.DataFrame(data=rot_quat, columns=["w", "x", "y", "z"])
    rot_quat.columns.name = data.columns.name

    return rot_quat


def rotate_quat(data: pd.DataFrame, data_rot: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Rotate a series of quaternions with rotation quaternion data.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in quaternions.
    data_rot : :class:`~pandas.DataFrame`
        dataframe with quaternions used to rotate the quaternions in ``data``.
    columns : list of str, optional
        list of columns if only specific columns should be rotated or ``None`` to rotate all columns.
        Default: ``None``

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotated quaternion data. Either all columns of ``data`` or only the columns specified by
        ``columns``.

    """
    if columns is None:
        columns = data.columns
    data_out = {}

    # loop over body parts
    for col in tqdm(columns):
        # save the initial quaternion
        start_quat = Quaternion(data_rot[col].loc[0]).unit

        data_tmp = pd.concat({"data": data[col], "data_rot": data_rot[col]}, axis=1)
        result = data_tmp.apply(lambda row: _apply_quat(row, start_quat), axis=1, result_type="expand")
        data_out[col] = pd.DataFrame(result.values, columns=data[col].columns, index=data[col].index)

    return pd.concat(data_out, names=[data.columns.names[0]], axis=1)


def rotate_quat_loop(
    data: pd.DataFrame, data_rot: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Rotate a series of quaternions with rotation quaternion data.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in quaternions.
    data_rot : :class:`~pandas.DataFrame`
        dataframe with quaternions used to rotate the quaternions in ``data``.
    columns : list of str, optional
        list of columns if only specific columns should be rotated or ``None`` to rotate all columns.
        Default: ``None``

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotated quaternion data. Either all columns of ``data`` or only the columns specified by
        ``columns``.

    """
    if columns is None:
        columns = data.columns
    data_out = {}

    # loop over body parts
    for col in tqdm(columns):
        # save the initial quaternion
        start_quat = Quaternion(data_rot[col].loc[0]).unit

        result = []
        for (i_data, row_data), (i_rot, row_rot) in zip(data[col].iterrows(), data_rot[col].iterrows()):
            # get unit quaternions of data quaternion and rotation quaternion
            data_quat = Quaternion(row_data).unit
            rot_quat = Quaternion(row_rot).unit
            # inverse rotation quaternion
            rot_quat_inverse = rot_quat.inverse
            # multiply the original data by the inverse of the rotation quaternion
            corr_quat = data_quat * rot_quat_inverse
            # multiply by initial quaternion
            corr_quat = corr_quat * start_quat
            result.append(list(corr_quat))

        data_out[col] = pd.DataFrame(result, index=data.index, columns=data[col].columns)

    return pd.concat(data_out, names=[data.columns.names[0]], axis=1)


def _apply_quat(row: pd.Series, start_quat: Quaternion):
    # get unit quaternions of data quaternion and rotation quaternion
    data_quat = Quaternion(row["data"]).unit
    rot_quat = Quaternion(row["data_rot"]).unit
    # inverse rotation quaternion
    rot_quat_inverse = rot_quat.inverse
    # multiply the original data by the inverse of the rotation quaternion
    corr_quat = data_quat * rot_quat_inverse
    # multiply by initial quaternion
    corr_quat = corr_quat * start_quat
    return list(corr_quat)
