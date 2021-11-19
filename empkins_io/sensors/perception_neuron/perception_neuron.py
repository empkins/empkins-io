from pathlib import Path
from typing import Dict, Any

from biopsykit.utils._datatype_validation_helper import _assert_is_dir

from empkins_io.sensors.perception_neuron._bvh_data import BvhData
from empkins_io.sensors.perception_neuron._calc_data import CalcData
from empkins_io.sensors.perception_neuron._center_mass_data import CenterOfMassData
from empkins_io.utils._types import path_t


def load_bvh_file(file_path: path_t) -> BvhData:
    return BvhData(file_path)


def load_calc_file(file_path: path_t) -> CalcData:
    return CalcData(file_path)


def load_center_mass_file(file_path: path_t) -> CenterOfMassData:
    return CenterOfMassData(file_path)


def load_perception_neuron_folder(folder_path: path_t) -> Dict[str, Any]:
    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)
    return_dict = dict.fromkeys(["bvh", "calc", "center_mass"])

    bvh_files = sorted(folder_path.glob("*.bvh"))
    if len(bvh_files) == 1:
        return_dict["bvh"] = load_bvh_file(bvh_files[0])
    elif len(bvh_files) > 1:
        raise ValueError(
            f"More than one bvh file found in {folder_path}. Please make sure only one bvh file is in the folder!"
        )

    calc_files = sorted(folder_path.glob("*.calc"))
    if len(calc_files) == 1:
        return_dict["calc"] = load_calc_file(calc_files[0])
    elif len(calc_files) > 1:
        raise ValueError(
            f"More than one calc file found in {folder_path}. Please make sure only one calc file is in the folder!"
        )

    center_mass_files = sorted(folder_path.glob("*.txt"))
    if len(center_mass_files) == 1:
        return_dict["center_mass"] = load_center_mass_file(center_mass_files[0])
    elif len(center_mass_files) > 1:
        raise ValueError(
            f"More than one center of mass file found in {folder_path}. "
            f"Please make sure only one center of mass file is in the folder!"
        )
    return return_dict
