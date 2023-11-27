from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

""" This is a script to check if all expected files are present for each participant of a dataset.

The script can also be used via the command line.
Example usage:
`python check_data_completeness.py path/to/data_per_subject -o path/to/output/data_completeness_check_result.csv`

For more information, run:
`python check_data_completeness.py -h`
"""


def _check_filetype_present(
    subj_dir: Path,
    subfolder: str,
    file_pattern: str,
    expected_number_of_files: int,
    expected_filesize: float = np.nan,
) -> dict:
    """Checks if the number of files that match the given pattern is as expected and if the (mean) file size is
    as expected (i.e. at max. 20% below the given expected file size)."""
    dir_path = subj_dir / subfolder
    existing_files = [f for f in dir_path.glob(file_pattern)]
    existing_files_names = [f.name for f in existing_files]
    real_number_of_files = len(existing_files_names)
    num_files_is_expected = real_number_of_files == expected_number_of_files
    comment = ""
    if len(existing_files) == 0:
        # no files found => file size is 0 bytes
        actual_filesize = 0
    elif file_pattern.endswith(".mp4") and len(existing_files) == 1:
        video = cv2.VideoCapture(existing_files[0].as_posix())
        # duration = video.get(cv2.CAP_PROP_POS_MSEC)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        actual_filesize = frame_count  # use frame count as proxy for file size
    else:
        actual_filesize = np.mean([f.stat().st_size for f in existing_files])

    if np.isnan(expected_filesize):
        # timelog and opendbm file size does not matter
        size_as_expected = True
    else:
        size_as_expected = actual_filesize > 0.8 * expected_filesize  # 20% tolerance (downwards)

    if not num_files_is_expected:
        comment = (
            f"Expected {expected_number_of_files} file(s), got {real_number_of_files} file(s)"
            f" instead: {existing_files_names}"
        )
    return {
        "num_as_expected": num_files_is_expected,
        "comment": comment,
        "path": f"{subfolder}/{file_pattern}",
        "size_as_expected": size_as_expected,
        "mean_size": actual_filesize,
    }


def _check_files_for_subject(subj_dir: Path, expected_files: pd.DataFrame) -> dict:
    """Iterates over the list of expected files for one subject and checks if each one of them is present.
    Results are summarized in a dictionary."""
    files_present = {}
    for subfolder, cols in expected_files.iterrows():
        files_present[(cols.condition, cols.description)] = _check_filetype_present(
            subj_dir,
            subfolder,
            cols.pattern,
            cols.number_of_files,
            cols.expected_filesize_mean,
        )
    return files_present


def check_data_completeness_dict(data_per_subject_folder: Path, expected_files_list: pd.DataFrame) -> dict:
    """Iterates over all subject directories in the data_per_subject folder and
    checks if all expected files are present. Results are summarized in a dictionary."""
    file_overview = {}
    for subject_dir in data_per_subject_folder.glob("VP_*"):
        subject = subject_dir.name
        if "Template" in subject:
            # skip template folder
            continue
        file_overview[subject] = _check_files_for_subject(subject_dir, expected_files_list)
    return file_overview


def check_data_completeness(data_per_subject_folder: Path, expected_files_list: pd.DataFrame) -> pd.DataFrame:
    """Iterates over all subject directories in the data_per_subject folder and
    checks if all expected files are present. Results are summarized in a DataFrame."""
    file_overview_dict = check_data_completeness_dict(data_per_subject_folder, expected_files_list)
    file_overview_df = (
        pd.DataFrame(file_overview_dict)
        .stack()
        .apply(pd.Series)  # create long format DataFrame with MultiIndex
        .swaplevel(0, 2)  # make VP_* the first level of the MultiIndex
        .swaplevel(1, 2)  # make condition the second level of the MultiIndex
        .sort_index()
    )
    file_overview_df.index = file_overview_df.index.set_names(["subject", "condition", "modality"])
    return file_overview_df


if __name__ == "__main__":
    parser = ArgumentParser(description="Checks if all expected files are present for each subject.")

    parser.add_argument(
        "data_per_subject_folder",
        type=Path,
        help="Path to the folder containing the data per subject.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        default=None,
        help="Path to the output csv file, default is data_per_subject_folder/../data_completeness_check_results.csv.",
    )
    parser.add_argument(
        "-e",
        "--expected_files",
        type=Path,
        default=None,
        help="Path to the csv file containing the list of expected files, default is"
        "data_per_subject_folder/../expected_files_per_subject.csv.",
    )

    args = parser.parse_args()

    # check data per subject folder exists
    dps_folder = args.data_per_subject_folder
    if not dps_folder.exists():
        raise FileNotFoundError(f"Data per subject folder {dps_folder} does not exist.")

    # set default output file
    if not args.output_file:
        output_file_path = dps_folder.parent / "data_completeness_check_results.csv"

    else:
        output_file_path = args.output_file

    # read expected files list
    if not args.expected_files:
        expected_files_path = dps_folder.parent / "expected_files_per_subject.csv"
    else:
        expected_files_path = args.expected_files

    if not expected_files_path.exists():
        raise FileNotFoundError(f"Expected files list {expected_files_path} does not exist.")
    expected_files_list = pd.read_csv(expected_files_path, index_col=1)

    # check data completeness
    print(f"Checking for data completeness in {dps_folder}  ...")
    file_overview_df = check_data_completeness(
        data_per_subject_folder=dps_folder, expected_files_list=expected_files_list
    )
    print(f"Writing results to {output_file_path}  ...")
    file_overview_df.to_csv(args.output_file)
    print("Done :)")
