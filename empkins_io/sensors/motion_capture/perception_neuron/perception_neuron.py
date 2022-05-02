from pathlib import Path
from typing import Dict, Any, Union, Sequence, Optional

from biopsykit.utils._datatype_validation_helper import _assert_is_dir

from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.bvh import BvhData
from empkins_io.sensors.motion_capture.motion_capture_formats.calc import CalcData
from empkins_io.sensors.motion_capture.motion_capture_formats.center_mass import CenterOfMassData
from empkins_io.utils._types import path_t


def _get_files(folder_path: path_t, extensions: Union[Sequence[str], str]):
    if isinstance(extensions, str):
        extensions = [extensions]
    file_list = []
    for ext in extensions:
        files = sorted(folder_path.glob(f"*{ext}"))
        files = [file for file in files if not file.name.startswith("._")]
        file_list.extend(files)
    return file_list


def load_perception_neuron_folder(
    folder_path: path_t, index_start: Optional[int] = 0, index_end: Optional[int] = -1
) -> Dict[str, Any]:
    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)
    return_dict: Dict[str, _BaseMotionCaptureDataFormat] = dict.fromkeys(["bvh", "calc", "center_mass"])

    bvh_files = _get_files(folder_path, [".bvh", ".bvh.gz"])
    if len(bvh_files) == 1:
        bvh_data = BvhData(bvh_files[0], system="perception_neuron")
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

    calc_files = _get_files(folder_path, [".calc", ".calc.gz"])
    if len(calc_files) == 1:
        return_dict["calc"] = CalcData(calc_files[0], system="perception_neuron")
    elif len(calc_files) > 1:
        raise ValueError(
            f"More than one calc file found in {folder_path}. Please make sure only one calc file is in the folder!"
        )

    center_mass_files = _get_files(folder_path, ["centerOfMass.txt", "centerOfMass.csv"])
    if len(center_mass_files) == 1:
        return_dict["center_mass"] = CenterOfMassData(center_mass_files[0], system="perception_neuron")
    elif len(center_mass_files) > 1:
        raise ValueError(
            f"More than one center of mass file found in {folder_path}. "
            f"Please make sure only one center of mass file is in the folder!"
        )

    for key in return_dict:
        if return_dict[key] is not None:
            return_dict[key] = return_dict[key].cut_data(index_start, index_end)
    return return_dict
