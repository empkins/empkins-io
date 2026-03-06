"""Module for importing Motion Capture data saved as .bvh file."""

import gzip
import re
import sys
from io import StringIO
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from scipy.spatial.transform.rotation import Rotation
from tqdm.auto import tqdm

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM
from empkins_io.utils._types import T, check_file_exists, path_t


class BvhData(_BaseMotionCaptureDataFormat):
    """Class for handling data from bvh files."""

    root: "BvhJoint" = None
    joints: ClassVar[dict[str, "BvhJoint"]] = {}
    num_frames: int = 0
    sampling_rate_hz: float = 0.0
    data_global: pd.DataFrame = None

    def __init__(self, file_path: path_t, system: MOTION_CAPTURE_SYSTEM | None = "perception_neuron"):
        """Create new ``BvhData`` instance.

        Parameters
        ----------
        file_path : :class:`~pathlib.Path` or str
            path to bvh file

        Raises
        ------
        FileNotFoundError
            if the data path is not valid, or the file does not exist
        :ex:`~biopsykit.utils.exceptions.FileExtensionError`
            if file in ``file_path`` is not a bvh file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, [".bvh", ".gz"])
        check_file_exists(file_path)

        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                _raw_data_str = f.read().decode("utf8")
        else:
            with file_path.open() as f:
                _raw_data_str = f.read()

        hierarchy_str, motion_str = _raw_data_str.split("MOTION")
        frame_info_str = re.findall(r"(Frames: \d+\nFrame Time: \d*[.,]?\d*\n)", motion_str)[0]

        self._hierarchy_str: str = hierarchy_str
        self._frame_info_str: str = frame_info_str
        self._parse_hierarchy(hierarchy_str)

        motion_str = re.sub(r"(Frames: \d+\nFrame Time: \d*[.,]?\d*\n)", "", motion_str)
        num_frames = int(re.findall(r"Frames: (\d+)", frame_info_str)[0])
        frame_time = float(re.findall(r"Frame Time: (\d*[.,]?\d*)", frame_info_str)[0])

        sampling_rate = 1.0 / frame_time

        list(self.joints.keys())
        data = self._parse_df(motion_str, sampling_rate)
        self.num_frames = num_frames

        super().__init__(data=data, system=system, sampling_rate=sampling_rate)

    def _parse_hierarchy(self, hierarchy_str: str):
        lines = re.split("\\s*\\n+\\s*", hierarchy_str)

        joint_stack = []
        joint_dict = {}

        for line in lines:
            words = re.split("\\s+", line)
            instruction = words[0]

            if instruction in ("JOINT", "ROOT"):
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                joint_dict[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                channels = [re.findall(r"(\w)(position|rotation)", channel)[0] for channel in words[2:]]
                channels = [(val[1][0:3], val[0].lower()) for val in channels]
                channel_dict = {key: [] for key in np.unique([ch[0] for ch in channels])}
                channel_dict = {key: val + [ch[1] for ch in channels if key in ch] for key, val in channel_dict.items()}
                joint_stack[-1].channels = channel_dict
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack.append(joint)
                joint_dict[joint.name] = joint
            elif instruction == "}":
                joint_stack.pop()

        self.joints = {key: val for key, val in joint_dict.items() if "_end" not in key}

    def _parse_df(self, motion_str: str, sampling_rate: float):
        motion_str = motion_str.strip()
        motion_str_buf = StringIO(motion_str)

        multi_index = [
            (joint_name, *list(channel))
            for joint_name, joint in self.joints.items()
            for channel in [(i, x) for i in joint.channels for x in joint.channels[i]]
        ]
        multi_index = pd.MultiIndex.from_tuples(multi_index, names=["body_part", "channel", "axis"])
        data = pd.read_csv(motion_str_buf, sep=" ", index_col=False, header=None, skip_blank_lines=False)
        data = data.dropna(how="all", axis=1)
        data.columns = multi_index
        data.index = np.around(data.index / sampling_rate, 5)
        data.index.name = "time"
        return data

    def _extract_position(self, joint: "BvhJoint", frame_data: pd.Series, index_offset: int):
        frame_data_joint = frame_data[joint.name]
        frame_data_joint = frame_data_joint["pos"]
        index_offset += 3
        return frame_data_joint.to_numpy(), index_offset

    def _extract_rotation(self, joint: "BvhJoint", frame_data: pd.Series, index_offset: int):
        frame_data_joint = frame_data[joint.name]
        frame_data_joint = frame_data_joint["rot"]
        local_rotation = frame_data_joint.reindex(sorted(joint.channels["rot"]))
        local_rotation = local_rotation.to_numpy()

        local_rotation = np.deg2rad(local_rotation)
        matrix_rotation = np.eye(3)
        for channel in list(joint.channels["rot"]):
            if channel == "x":
                euler_rot = np.array([local_rotation[0], 0.0, 0.0])
            elif channel == "y":
                euler_rot = np.array([0.0, local_rotation[1], 0.0])
            elif channel == "z":
                euler_rot = np.array([0.0, 0.0, local_rotation[2]])
            else:
                raise ValueError(f"Unknown channel {channel}")
            matrix_channel = Rotation.from_rotvec(euler_rot).as_matrix()
            matrix_rotation = matrix_rotation.dot(matrix_channel)

        return matrix_rotation, index_offset

    def _recursive_apply_frame(
        self,
        joint: "BvhJoint",
        frame_data: pd.Series,
        index_offset: int,
        pos: np.ndarray,
        rot: np.ndarray,
        matrix_parent: np.ndarray,
        pos_parent: np.ndarray,
    ):
        if joint.position_animated():
            position, index_offset = self._extract_position(joint, frame_data, index_offset)
        else:
            position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            pos[joint_index] = pos_parent + matrix_parent.dot(joint.offset)
            rot[joint_index] = Rotation.from_matrix(matrix_parent).as_euler("xyz")
            return index_offset

        if joint.rotation_animated():
            matrix_rotation, index_offset = self._extract_rotation(joint, frame_data, index_offset)
        else:
            matrix_rotation = np.eye(3)
        matrix_new = matrix_parent.dot(matrix_rotation)

        position = pos_parent + matrix_parent.dot(joint.offset) + position
        rotation = np.rad2deg(Rotation.from_matrix(matrix_new).as_euler("xyz"))

        joint_index = list(self.joints.values()).index(joint)
        pos[joint_index] = position
        rot[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_data, index_offset, pos, rot, matrix_new, position)

        return index_offset

    def global_pose_for_frame(self, frame_index: int):
        pos = np.empty((len(self.joints), 3))
        rot = np.empty((len(self.joints), 3))
        frame_data = self.data.iloc[frame_index, :]
        matrix_parent = np.eye(3, 3)
        self._recursive_apply_frame(self.root, frame_data, 0, pos, rot, matrix_parent, np.zeros(3))

        pos = pd.DataFrame(pos, columns=self.root.channels["pos"])
        rot = pd.DataFrame(rot, columns=self.root.channels["rot"])

        frame = pd.concat({"pos_global": pos, "rot_global": rot}, names=["channel", "axis"], axis=1)
        frame.index.name = "body_part"
        frame = pd.DataFrame([frame.unstack()])
        frame = frame.reorder_levels([2, 0, 1], axis=1).sort_index(axis=1)
        body_part_map = dict(enumerate(self.body_parts))
        frame = frame.rename(columns=body_part_map, level="body_part")
        frame.index = [np.around(frame_index / self.sampling_rate, 5)]
        frame.index.name = "time"
        return frame

    def global_poses(self, **kwargs) -> T:
        frame_list = []
        for frame_index in tqdm(
            range(self.num_frames),
            mininterval=1,
            miniters=self.num_frames / self.sampling_rate,
            unit="frame",
            file=kwargs.get("file", sys.stderr),
        ):
            frame_list.append(self.global_pose_for_frame(frame_index))

        self.data_global = pd.concat(frame_list)
        return self

    def to_gzip_bvh(self, file_path: path_t):
        """Export to gzip-compressed bvh file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data


        See Also
        --------
        BvhData.to_bvh
            Export data as (uncompressed) bvh file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".gz")

        with gzip.open(file_path, "w") as fp:
            self._write_bvh_fp(fp, encode=True)

    def to_bvh(self, file_path: path_t):
        """Export to bvh file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data


        See Also
        --------
        BvhData.to_gzip_bvh
            Export data as gzip-compressed bvh file

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".bvh")

        with file_path.open("w") as fp:
            self._write_bvh_fp(fp)

    def _write_bvh_fp(self, fp, encode: bool | None = False):
        data_out = self.data.groupby(["body_part", "channel"], sort=False, group_keys=False, axis=1).apply(
            lambda df: self._reindex_axis(df)
        )

        if encode:
            fp.write(self._hierarchy_str.encode("utf8"))
            # set the empty line after MOTION
            fp.write(b"MOTION\n\n")
            fp.write(self._frame_info_str.encode("utf8"))
            fp.write(
                data_out.to_csv(float_format="%.4f", sep=" ", header=False, index=False, lineterminator=" \n").encode(
                    "utf8"
                )
            )
        else:
            fp.write(self._hierarchy_str)
            # set the empty line after MOTION
            fp.write("MOTION\n\n")
            fp.write(self._frame_info_str)
            fp.write(data_out.to_csv(float_format="%.4f", sep=" ", header=False, index=False, lineterminator=" \n"))

    def global_pose_to_gzip_csv(self, file_path: path_t):
        """Export global pose information to gzip-compressed csv file.

        Parameters
        ----------
        file_path: :class:`~pathlib.Path` or str
            file name for the exported data

        """
        # ensure pathlib
        file_path = Path(file_path)
        _assert_file_extension(file_path, ".gz")
        self.data_global.to_csv(file_path, float_format="%.4f", compression="gzip")

    def load_global_pose(self, path: path_t):
        path = Path(path)
        _assert_file_extension(path, [".csv", ".gz"])
        self.data_global = pd.read_csv(path, header=[0, 1, 2], index_col=0)

    @staticmethod
    def _reindex_axis(data: pd.DataFrame):
        if "rot" in data.name:
            return data.reindex(columns=list("yxz"), level="axis")
        return data


class BvhJoint:
    def __init__(self, name: str, parent: "BvhJoint"):
        self.name: str = name
        self.parent: BvhJoint = parent
        self.offset: np.ndarray = np.zeros(3)
        self.channels: dict[str, list[str]] = {}
        self.children: list[BvhJoint] = []

    def add_child(self, child: "BvhJoint"):
        self.children.append(child)

    def __repr__(self) -> str:
        return f"Name: {self.name}, Channels: {self.channels}"

    def position_animated(self):
        return "pos" in self.channels

    def rotation_animated(self):
        return "rot" in self.channels
