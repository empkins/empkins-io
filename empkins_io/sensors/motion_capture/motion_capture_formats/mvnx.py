import gzip
from pathlib import Path
from typing import List

import mvnx
import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.utils._types import path_t, _check_file_exists


class MvnxData(_BaseMotionCaptureDataFormat):
    """Class for handling data from mvnx files."""

    num_frames: int = 0
    sampling_rate: float = 0.0
    segments: List[str] = None
    joints: List[str] = None
    sensors: List[str] = None
    data: pd.DataFrame = None
    joint_data: pd.DataFrame = None
    sensor_data: pd.DataFrame = None
    _index = None
    _types = {"segment": "body_part", "joint": "joint", "sensor": "sensor"}
    _quat = ("q0", "q1", "q2", "q3")
    _xyz = ("x", "y", "z")

    def __init__(self, file_path: path_t):
        file_path = Path(file_path)
        _assert_file_extension(file_path, [".mvnx", ".gz"])
        _check_file_exists(file_path)

        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                _raw_data = mvnx.MVNX(f)
        else:
            with open(file_path, "r") as f:
                _raw_data = mvnx.MVNX(f)

        sampling_rate = _raw_data.frameRate
        self.num_frames = len(_raw_data.acceleration)
        self._index = np.float_(_raw_data.time) / 1000

        self.joints = list(_raw_data.joints)
        self.sensors = _raw_data.sensors
        self.segments = list(_raw_data.segments.values())

        data = self._parse_segment_df(_raw_data)
        self.joint_data = self._parse_joint_df(_raw_data)
        self.sensor_data = self._parse_sensor_df(_raw_data)

        super().__init__(data=data, sampling_rate=sampling_rate, system="xsens")

    def _parse_segment_df(self, _raw_data: mvnx.MVNX) -> pd.DataFrame:
        type = "segment"

        position_df = self._parse_df_for_value("pos", _raw_data.position, type)
        velocity_df = self._parse_df_for_value("vel", _raw_data.velocity, type)
        orientation_df = self._parse_df_for_value("ori", _raw_data.orientation, type)
        acceleration_df = self._parse_df_for_value("acc", _raw_data.acceleration, type)
        ang_acceleration_df = self._parse_df_for_value("ang_acc", _raw_data.angularAcceleration, type)
        ang_velocity_df = self._parse_df_for_value("ang_vel", _raw_data.angularVelocity, type)

        data = position_df.join([velocity_df, orientation_df, acceleration_df, ang_velocity_df, ang_acceleration_df])
        data.sort_index(axis=1, level=self._types[type], inplace=True)

        return data

    def _parse_joint_df(self, _raw_data: mvnx.MVNX) -> pd.DataFrame:
        type = "joint"

        joint_angle_df = self._parse_df_for_value("ang", _raw_data.jointAngle, type)
        joint_angle_xzy_df = self._parse_df_for_value("ang_xzy", _raw_data.jointAngleXZY, type)

        joint_data = joint_angle_df.join(joint_angle_xzy_df)
        joint_data.sort_index(axis=1, level=self._types[type], inplace=True)

        return joint_data

    def _parse_sensor_df(self, _raw_data: mvnx.MVNX) -> pd.DataFrame:
        type = "sensor"

        sensor_ori_df = self._parse_df_for_value("ori", _raw_data.sensorOrientation, type)
        sensor_acc_df = self._parse_df_for_value("acc", _raw_data.sensorFreeAcceleration, type)
        sensor_mag_df = self._parse_df_for_value("mag", _raw_data.sensorMagneticField, type)

        sensor_data = sensor_acc_df.join([sensor_ori_df, sensor_mag_df])
        sensor_data.sort_index(axis=1, level=self._types[type], inplace=True)

        return sensor_data

    def _parse_df_for_value(self, name: str, data: np.ndarray, type: str) -> pd.DataFrame:
        if type not in self._types.keys():
            raise ValueError(f"Expected on of {self._types.keys()}, got {type} instead.")

        if name == "ori":
            axis = self._quat
        else:
            axis = self._xyz

        if type == "segment":
            index_type = self.segments
        elif type == "joint":
            index_type = self.joints
        else:
            index_type = self.sensors

        multi_index = pd.MultiIndex.from_product(
            [index_type, [name], axis], names=[self._types[type], "channel", "axis"]
        )

        data = pd.DataFrame(data)
        data.columns = multi_index
        data.index = self._index
        data.index.name = "time"

        return data

    def data_to_gzip_csv(self, file_path: path_t):
        """Export segment data to gzip-compressed csv file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".gz")
        self.data.to_csv(file_path, float_format="%.4f", compression="gzip")

    def load_data(self, path: path_t):
        path = Path(path)
        _assert_file_extension(path, [".csv", ".gz"])
        self.data = pd.read_csv(path, header=[0, 1, 2], index_col=0)

    def joint_data_to_gzip_csv(self, file_path: path_t):
        """Export joint data to gzip-compressed csv file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".gz")
        self.joint_data.to_csv(file_path, float_format="%.4f", compression="gzip")

    def load_segment_data(self, path: path_t):
        path = Path(path)
        _assert_file_extension(path, [".csv", ".gz"])
        self.joint_data = pd.read_csv(path, header=[0, 1, 2], index_col=0)

    def sensor_data_to_gzip_csv(self, file_path: path_t):
        """Export sensor data to gzip-compressed csv file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".gz")
        self.sensor_data.to_csv(file_path, float_format="%.4f", compression="gzip")

    def load_sensor_data(self, path: path_t):
        path = Path(path)
        _assert_file_extension(path, [".csv", ".gz"])
        self.sensor_data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
