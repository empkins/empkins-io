from typing import Optional, Sequence

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_has_column_multiindex
from scipy.spatial.transform.rotation import Rotation


def euler_to_quat_hierarchical(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    seq: Optional[str] = "xyz",
    degrees: Optional[bool] = False,
) -> pd.DataFrame:
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
    seq : str
        Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
        {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
        rotations cannot be mixed in one function call.
        Default: "xyz"
    degrees : bool, optional
        If ``True``, then the given angles are assumed to be in degrees.
        Default: ``False``.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to quaternions. Either all columns of ``data`` or only the columns
        specified by ``columns``.

    """
    _assert_has_column_multiindex(data, nlevels=2, nlevels_atleast=True)
    if columns is None:
        columns = data.columns.get_level_values(level=0).unique()

    rot_quat_total = {}
    for col in columns:
        rot_quat_total[col] = euler_to_quat(data[col], seq=seq, degrees=degrees)
    return pd.concat(rot_quat_total, axis=1, names=[data.columns.names[0]])


def euler_to_quat(data: pd.DataFrame, seq: Optional[str] = "xyz", degrees: Optional[bool] = False):
    """Convert rotation data from euler angles to quaternions.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in euler angles.
    seq : str
        Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
        {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
        rotations cannot be mixed in one function call.
        Default: "xyz"
    degrees : bool, optional
        If ``True``, then the given angles are assumed to be in degrees.
        Default: ``False``.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to quaternions.

    """
    # print(data)
    rot_euler = Rotation.from_euler(seq, data, degrees=degrees)
    rot_quat = rot_euler.as_quat()
    columns = _get_cols_multiindex(data, list("wxyz"))
    return pd.DataFrame(data=rot_quat, columns=columns, index=data.index)


def quat_to_euler_hierarchical(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    seq: Optional[str] = "xyz",
    degrees: Optional[bool] = False,
) -> pd.DataFrame:
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
    seq : str
        Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
        {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
        rotations cannot be mixed in one function call.
        Default: "xyz"
    degrees : bool, optional
        Returned angles are in degrees if this flag is ``True``, otherwise they are in radians.
        Default: ``False``.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to euler angles. Either all columns of ``data`` or only the columns
        specified by ``columns``.

    """
    if columns is None:
        columns = data.columns.get_level_values(level=0).unique()

    rot_euler_total = {}
    for col in columns:
        rot_euler_total[col] = quat_to_euler(data[col], seq=seq, degrees=degrees)
    return pd.concat(rot_euler_total, axis=1, names=[data.columns.names[0]])


def quat_to_euler(data: pd.DataFrame, seq: Optional[str] = "xyz", degrees: Optional[bool] = False):
    """Convert rotation data from quaternions to euler angles.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in quaternions.
    seq : str
        Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
        {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
        rotations cannot be mixed in one function call.
        Default: "xyz"
    degrees : bool, optional
        Returned angles are in degrees if this flag is ``True``, otherwise they are in radians.
        Default: ``False``.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotation data transformed to euler angles.

    """
    # print(data)
    rot_quat = Rotation.from_quat(data)
    rot_euler = rot_quat.as_euler(seq, degrees=degrees)
    columns = _get_cols_multiindex(data, list(seq))
    rot_euler = pd.DataFrame(data=rot_euler, columns=columns, index=data.index)
    rot_euler = rot_euler.reindex(columns=sorted(seq), level="axis")
    return rot_euler


def rotate_quat_hierarchical(
    data: pd.DataFrame, data_rot: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Rotate a series of quaternions with rotation quaternion data in a hierarchical dataframe.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in quaternions. Dataframe must be hierarchical, i.e., must have columns as
        :class:`~pandas.MultiIndex` where the lowest level contains the quaternions.
    data_rot : :class:`~pandas.DataFrame`
        dataframe with quaternions used to rotate the quaternions in ``data``. Dataframe must be hierarchical, i.e.,
        must have columns as :class:`~pandas.MultiIndex` where the lowest level contains the quaternions.
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

    for col in columns:
        corr_quat = rotate_quat(data[col], data_rot[col])
        data_out[col] = pd.DataFrame(corr_quat, index=data.index, columns=data[col].columns)
    return pd.concat(data_out, names=[data.columns.names[0]], axis=1)


def rotate_quat(data: pd.DataFrame, data_rot: pd.DataFrame) -> pd.DataFrame:
    """Rotate a series of quaternions with rotation quaternion data.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with rotation data in quaternions.
    data_rot : :class:`~pandas.DataFrame`
        dataframe with quaternions used to rotate the quaternions in ``data``.

    Returns
    -------
    :class:`~pd.DataFrame`
        dataframe with rotated quaternion data.

    """
    # get quaternions of data and rotation quaternions
    data_quat = Rotation.from_quat(data)
    rot_quat = Rotation.from_quat(data_rot)
    # save the initial quaternion
    start_quat = rot_quat[0]
    # get inverse rotation quaternion
    rot_quat_inv = rot_quat.inv()
    # multiply the original data by the inverse of the rotation quaternion
    corr_quat = data_quat * rot_quat_inv
    # multiply by initial quaternion
    corr_quat_out = corr_quat * start_quat
    return corr_quat_out.as_quat()


def _get_cols_multiindex(data, axis_cols):
    if data.columns.nlevels > 1:
        col_vals = data.columns.droplevel(-1)
        col_list = [list(col_vals.get_level_values(col).unique()) for col in col_vals.names]
        return pd.MultiIndex.from_product([*col_list, axis_cols], names=[*list(col_vals.names), "axis"])
    return pd.Index(axis_cols, name=["axis"])
