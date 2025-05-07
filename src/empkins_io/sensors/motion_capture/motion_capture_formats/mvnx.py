import gzip
import locale
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats._utils.mvnx_parser import _MvnxParser
from empkins_io.utils._types import check_file_exists, path_t

_RAD_TO_DEG = 57.29578


class MvnxData(_BaseMotionCaptureDataFormat):
    """Class for handling data from mvnx files."""

    start: datetime = None
    num_frames: int = 0
    sampling_rate_hz: float = 0.0
    segments: list[str] = None
    joints: list[str] = None
    sensors: list[str] = None
    center_mass: ClassVar[list[str]] = ["CenterMass"]
    data: pd.DataFrame = None
    sensor_data: pd.DataFrame = None
    joint_data: pd.DataFrame = None
    _index: float = None
    _types: ClassVar[dict[str, str]] = {
        "segment": "body_part",
        "joint": "body_part",
        "sensor": "body_part",
        "center_mass": "body_part",
    }
    _quat: tuple[str] = ("q0", "q1", "q2", "q3")
    _xyz: tuple[str] = ("x", "y", "z")
    _foot_contacts: tuple[str] = ("heel", "toe")
    _tz: str = None

    def __init__(
        self, file_path: path_t, load_sensor_data: bool = False, tz: str = "Europe/Berlin", *, verbose: bool = True
    ):
        file_path = Path(file_path)
        _assert_file_extension(file_path, [".mvnx", ".gz"])
        check_file_exists(file_path)

        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                _raw_data = _MvnxParser(f, verbose=verbose)
        else:
            with file_path.open() as f:
                _raw_data = _MvnxParser(f, verbose=verbose)

        sampling_rate = _raw_data.frameRate
        self.num_frames = len(_raw_data.acceleration)
        self._index = np.float64(_raw_data.time) / 1000
        self._tz = tz

        self._parse_start_time(_raw_data)

        self.joints = list(_raw_data.joints)
        self.sensors = _raw_data.sensors
        self.segments = list(_raw_data.segments.values())

        data = self._parse_segment_df(_raw_data)
        data = data.join(self._parse_joint_df(_raw_data))
        data = data.join(self._parse_foot_contact_df(_raw_data))
        data = data.join(self._parse_center_mass(_raw_data))

        if load_sensor_data:
            self.sensor_data = self._parse_sensor_df(_raw_data)

        super().__init__(data=data, sampling_rate=sampling_rate, system="xsens")

    def _parse_start_time(self, _raw_data: _MvnxParser):
        locale.setlocale(locale.LC_TIME, "en_US.UTF-8")

        start_time = pd.to_datetime(_raw_data.recordingDate, unit="ms").tz_localize("UTC").tz_convert(self._tz)
        self.start = start_time.to_pydatetime()

        # try:  # english date format
        #     self.start = datetime.strptime(_raw_data.recordingDate, "%a %b %d %H:%M:%S.%f %Y")
        # except ValueError:  # if not its probably german
        #     locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
        #     self.start = datetime.strptime(_raw_data.recordingDate, "%a %b %d %H:%M:%S.%f %Y")

    def _parse_segment_df(self, _raw_data: _MvnxParser) -> pd.DataFrame:
        data_type = "segment"

        position_df = self._parse_df_for_value("pos", _raw_data.position, data_type)
        velocity_df = self._parse_df_for_value("vel", _raw_data.velocity, data_type)
        orientation_df = self._parse_df_for_value("ori", _raw_data.orientation, data_type)
        acceleration_df = self._parse_df_for_value("acc", _raw_data.acceleration, data_type)
        ang_acceleration_df = (
            self._parse_df_for_value("ang_acc", _raw_data.angularAcceleration, data_type) * _RAD_TO_DEG
        )
        ang_velocity_df = self._parse_df_for_value("gyr", _raw_data.angularVelocity, data_type) * _RAD_TO_DEG
        # foot_contact_df = self._parse_foot_contacts(_raw_data.footContacts, data_type)

        data = position_df.join(
            [
                velocity_df,
                orientation_df,
                acceleration_df,
                ang_velocity_df,
                ang_acceleration_df,
            ]
        )

        data = pd.concat(
            [data],
            keys=["mvnx_segment"],
            names=["data_format"],
            axis=1,
        )

        data = data.sort_index(axis=1, level=self._types[data_type])
        return data

    def _parse_center_mass(self, _raw_data: _MvnxParser) -> pd.DataFrame:
        center_mass_df = self._parse_df_for_value(["pos", "vel", "acc"], _raw_data.centerOfMass, "center_mass")
        center_mass_df = center_mass_df.sort_index(axis=1, level="body_part")
        return pd.concat([center_mass_df], keys=["center_mass"], names=["data_format"], axis=1)

    def _parse_foot_contact_df(self, _raw_data: _MvnxParser) -> pd.DataFrame:
        foot_contact_df = self._parse_foot_contacts(_raw_data.footContacts, fc_type="segment")
        foot_contact_df = foot_contact_df.sort_index(axis=1, level="body_part")
        return pd.concat([foot_contact_df], keys=["foot_contact"], names=["data_format"], axis=1)

    def _parse_joint_df(self, _raw_data: _MvnxParser) -> pd.DataFrame:
        data_type = "joint"

        joint_angle_df = self._parse_df_for_value("ang", _raw_data.jointAngle, data_type)
        joint_angle_xzy_df = self._parse_df_for_value("ang_xzy", _raw_data.jointAngleXZY, data_type)

        joint_data = joint_angle_df.join(joint_angle_xzy_df)
        joint_data = joint_data.sort_index(axis=1, level=self._types[data_type])
        joint_data = pd.concat([joint_data], keys=["mvnx_joint"], names=["data_format"], axis=1)

        return joint_data

    def _parse_sensor_df(self, _raw_data: _MvnxParser) -> pd.DataFrame:
        data_type = "sensor"

        sensor_ori_df = self._parse_df_for_value("ori", _raw_data.sensorOrientation, data_type)
        sensor_acc_df = self._parse_df_for_value("acc", _raw_data.sensorFreeAcceleration, data_type)
        sensor_mag_df = self._parse_df_for_value("mag", _raw_data.sensorMagneticField, data_type)

        sensor_data = pd.concat(
            [sensor_acc_df, sensor_ori_df, sensor_mag_df],
            keys=["mvnx_sensor"],
            names=["data_format"],
            axis=1,
        )
        sensor_data = sensor_data.sort_index(axis=1, level=self._types[data_type])

        return sensor_data

    def _parse_df_for_value(self, name: str | list[str], data: np.ndarray, data_type: str) -> pd.DataFrame:
        if data_type not in self._types:
            raise ValueError(f"Expected on of {self._types.keys()}, got {data_type} instead.")

        axis = self._quat if name == "ori" else self._xyz

        if data_type == "segment":
            index_type = self.segments
        elif data_type == "joint":
            index_type = self.joints
        elif data_type == "sensor":
            index_type = self.sensors
        else:
            index_type = self.center_mass

        if isinstance(name, str):
            name = [name]

        multi_index = pd.MultiIndex.from_product(
            [index_type, name, axis], names=[self._types[data_type], "channel", "axis"]
        )

        data = pd.DataFrame(data)

        data.columns = multi_index
        data.index = self._index
        data.index.name = "t"

        return data

    def _parse_foot_contacts(self, data: np.ndarray, fc_type: str) -> pd.DataFrame:
        if fc_type not in self._types:
            raise ValueError(f"Expected on of {self._types.keys()}, got {fc_type} instead.")

        axis = self._foot_contacts

        multi_index = pd.MultiIndex.from_product(
            [["FootContacts"], ["left", "right"], axis],
            names=[self._types[fc_type], "channel", "axis"],
        )

        foot_df = pd.DataFrame(data)
        foot_df.columns = multi_index
        foot_df.index = self._index

        return foot_df

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
