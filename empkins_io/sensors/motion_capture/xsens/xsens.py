from pathlib import Path
from typing import Dict, Any, Union, Sequence, Optional

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_is_dir

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.bvh import BvhData
from empkins_io.utils._types import path_t

SYSTEM = "xsens"


def _get_files(folder_path: path_t, extensions: Union[Sequence[str], str]):
    if isinstance(extensions, str):
        extensions = [extensions]
    file_list = []
    for ext in extensions:
        file_list.extend(sorted(folder_path.glob(f"*{ext}")))
    return file_list


# TODO add C3D
def load_xsens_folder(
        folder_path: path_t, index_start: Optional[int] = 0, index_end: Optional[int] = -1
) -> Dict[str, Any]:
    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)
    return_dict: Dict[str, _BaseMotionCaptureDataFormat] = dict.fromkeys(["bvh"])

    bvh_files = _get_files(folder_path, [".bvh", ".bvh.gz"])
    if len(bvh_files) == 1:
        bvh_data = BvhData(bvh_files[0], system=SYSTEM)
        global_pose_files = _get_files(folder_path, ["global_pose.csv", "global_pose.csv.gz"])
        if len(global_pose_files) == 1:
            bvh_data.load_global_pose(global_pose_files[0])
        else:
            raise ValueError(
                f"More than one global pose file found in {folder_path}. Please make sure only one global pose "
                f"file is in the folder!"
            )
        return_dict["bvh"] = bvh_data
    elif len(bvh_files) > 1:
        raise ValueError(
            f"More than one bvh file found in {folder_path}. Please make sure only one bvh file is in the folder!"
        )

    for key in return_dict:
        if return_dict[key] is not None:
            return_dict[key] = return_dict[key].cut_data(index_start, index_end)
    return return_dict
