from collections.abc import Sequence
from pathlib import Path
from typing import Any

from biopsykit.utils._datatype_validation_helper import _assert_is_dir

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.mvnx import MvnxData
from empkins_io.utils._types import path_t

SYSTEM = "xsens"


def _get_files(folder_path: path_t, extensions: Sequence[str] | str):
    if isinstance(extensions, str):
        extensions = [extensions]
    file_list = []
    for ext in extensions:
        file_list.extend(sorted(folder_path.glob(f"*{ext}")))
    return file_list


def load_xsens_folder(folder_path: path_t, index_start: int | None = 0, index_end: int | None = -1) -> dict[str, Any]:
    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)
    return_dict: dict[str, _BaseMotionCaptureDataFormat] = dict.fromkeys(["mvnx"])

    mvnx_files = _get_files(folder_path, [".mvnx", ".mvnx.gz"])
    if len(mvnx_files) == 1:
        mvnx_data = MvnxData(mvnx_files[0])
        return_dict["mvnx"] = mvnx_data

    for key in return_dict:
        if return_dict[key] is not None:
            return_dict[key] = return_dict[key].cut_data(index_start, index_end)
    return return_dict
