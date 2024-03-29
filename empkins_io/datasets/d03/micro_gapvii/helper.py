import pathlib
import tarfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.io.biopac import BiopacDataset
from biopsykit.io.nilspod import _handle_counter_inconsistencies_session
from nilspodlib.exceptions import InvalidInputFileError, SessionValidationError, SynchronisationError
from pandas import DataFrame

from empkins_io.datasets.d03.micro_gapvii._custom_synced_session import CustomSyncedSession
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import (
    NilsPodDataLoadException,
    NilsPodDataNotFoundException,
    SamplingRateMismatchException,
    TimelogNotFoundException,
)


def _build_data_path(base_path: path_t, participant_id: str, condition: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}/{condition}")
    assert data_path.exists()
    return data_path


def _load_biopac_data(base_path: path_t, participant_id: str, condition: str) -> Tuple[pd.DataFrame, int]:
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "biopac/raw"
    )

    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}_{condition}.acq")

    dataset_biopac = BiopacDataset.from_acq_file(biopac_file_path)
    biopac_df = dataset_biopac.data_as_df(index="local_datetime")
    fs = dataset_biopac._sampling_rate

    # check if biopac sampling rate is the same for each channel
    sampling_rates = set(fs.values())
    if len(sampling_rates) > 1:
        raise SamplingRateMismatchException(
            f"Biopac sampling rates are not the same for every channel! Found sampling rates: {sampling_rates}"
        )

    fs = list(sampling_rates)[0]
    return biopac_df, fs


def _load_radar_data(base_path: path_t, participant_id: str, condition: str) -> tuple[DataFrame, float]:
    radar_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "emrad/raw"
    )
    radar_file_path = radar_dir_path.joinpath(f"emrad_data_{participant_id}_{condition}.h5")

    dataset_radar = EmradDataset.from_hd5_file(radar_file_path)
    radar_df = dataset_radar.data_as_df(index="local_datetime")
    # radar_df.index.name = "time"
    fs = dataset_radar.sampling_rate_hz
    return radar_df, fs


def _load_timelog(base_path: path_t, participant_id: str, condition: str, phase: str, phase_fine: bool) -> pd.DataFrame:
    timelog_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "timelog/cleaned"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"{participant_id}_{condition}_processed_phases_timelog.csv")
    if timelog_file_path.exists():
        timelog = load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")
        if (phase == "all") & phase_fine:
            timelog_fine = timelog.drop("Talk", axis=1, level=0)
            timelog_fine = timelog_fine.drop("Math", axis=1, level=0)
            return timelog_fine
        elif (phase == "all") & (not phase_fine):
            timelog_coarse = timelog.drop("Talk_1", axis=1, level=0)
            timelog_coarse = timelog_coarse.drop("Talk_2", axis=1, level=0)
            timelog_coarse = timelog_coarse.drop("Math_1", axis=1, level=0)
            timelog_coarse = timelog_coarse.drop("Math_2", axis=1, level=0)
            return timelog_coarse
        else:
            timelog = timelog.iloc[:, timelog.columns.get_level_values(0) == phase]
            return timelog
    raise TimelogNotFoundException(
        f"No cleaned timelog file was found for {participant_id}! "
        "Run the 'notebooks/clean_timelog.ipynb' notebook first!"
    )


def _load_nilspod_session(base_path: path_t, participant_id: str, condition: str) -> Tuple[pd.DataFrame, float]:
    data_path = _build_data_path(base_path, participant_id=participant_id, condition=condition)
    data_path = data_path.joinpath("nilspod/raw")

    nilspod_files = sorted(data_path.glob("NilsPodX-*.bin"))
    if len(nilspod_files) == 0:
        raise NilsPodDataNotFoundException("No NilsPod files found in directory!")

    try:
        session = CustomSyncedSession.from_folder_path(data_path)

        # fix for "classical nilspod bug" where last sample counter is corrupted
        session = session.cut(stop=-10)
        session = session.align_to_syncregion()
    except (ZeroDivisionError, SynchronisationError, SessionValidationError, InvalidInputFileError, KeyError) as e:
        raise NilsPodDataLoadException("Cannot load NilsPod data!") from e

    _handle_counter_inconsistencies_session(session, handle_counter_inconsistency="ignore")

    # convert dataset to dataframe and localize timestamp
    df = session.data_as_df(index="local_datetime", concat_df=True)
    df.index.name = "time"
    fs = session.info.sampling_rate_hz[0]
    return df, fs


def build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def build_opendbm_tarfile_path(
    base_path: path_t, subject_id: str, condition: str, suffix: Optional[str] = None
) -> Path:
    path = build_data_path(base_path, subject_id, condition)
    path = path.joinpath("video", "face", "processed")

    if suffix is None:
        path = path.joinpath(f"opendbm_output_video_face_{subject_id}_{condition}.tar.gz")
        if not path.exists():
            print("path to tarfile does not exist")

    else:
        path = path.joinpath(f"opendbm_output_{subject_id}_{condition}_{suffix}.tar.gz")

    return path


def build_opendbm_extracted_tar_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    path = build_data_path(base_path, subject_id, condition)
    path = path.joinpath("video", "processed", "output")
    return path


def build_opendbm_raw_data_path(subject_id: str, condition: str, group: str, subgroup: Optional[str] = None) -> list:
    data_path = []

    if subgroup is None:
        if group == "facial":
            path = f"output/raw_variables/{subject_id}_{condition}/{group}/"
            data_path = [
                    f"{path}face_asymmetry/{subject_id}_{condition}_facasym.csv",
                    f"{path}face_au/{subject_id}_{condition}_facau.csv",
                    f"{path}face_expressivity/{subject_id}_{condition}_facemo.csv",
                    f"{path}face_landmark/{subject_id}_{condition}_faclmk.csv",
                ]
        elif group == "acoustic":
            path = f"output/raw_variables/{subject_id}_{condition}/{group}/"
            data_path = [
                    f"{path}formant_freq/{subject_id}_{condition}_formant.csv",
                    f"{path}harmonic_noise/{subject_id}_{condition}_hnr.csv",
                    f"{path}intensity/{subject_id}_{condition}_intensity.csv",
                    f"{path}mfcc/{subject_id}_{condition}_mfcc.csv",
                    f"{path}pitch/{subject_id}_{condition}_pitch.csv",
                ]
        elif group == "acoustic_seg":
            path = f"output/raw_variables/{subject_id}_{condition}/acoustic/"
            data_path = [
                    f"{path}glottal_noise_recomputed/{subject_id}_{condition}_gne.csv",
                    f"{path}jitter_recomputed/{subject_id}_{condition}_jitter.csv",
                    f"{path}shimmer_recomputed/{subject_id}_{condition}_shimmer.csv",
                ]
        elif group == "audio_seg":
            path = f"output/raw_variables/{subject_id}_{condition}/"
            data_path = [
                    f"{path}acoustic/pause_segment_recomputed/{subject_id}_{condition}_pausechar.csv",
                    f"{path}acoustic/voice_frame_score_recomputed/{subject_id}_{condition}_voiceprev.csv",
                    f"{path}movement/voice_tremor_recomputed/{subject_id}_{condition}_vtremor.csv",
                ]

        elif group == "movement":
            path = f"output/raw_variables/{subject_id}_{condition}/{group}/"
            data_path = [
                    f"{path}gaze/{subject_id}_{condition}_eyegaze.csv",
                    f"{path}head_movement/{subject_id}_{condition}_headmov.csv",
                    f"{path}head_pose/{subject_id}_{condition}_headpose.csv",
                    f"{path}eye_blink_binarized/{subject_id}_{condition}_eyeblinks.csv",
                ]
        elif group == "facial_tremor":
            data_path = [
                f"output/raw_variables/{subject_id}_{condition}/{subject_id}_{condition}_openface_lmk/"
                f"{subject_id}_{condition}_landmark_output.csv"
            ]

    else:
        path = f"output/raw_variables/{subject_id}_{condition}/{group}/"
        if group == "acoustic" and subgroup == "pitch":
            data_path = [f"{path}pitch/{subject_id}_{condition}_pitch.csv"]

        elif group == "acoustic" and subgroup == "jitter_recomputed":
            data_path = [f"{path}jitter_recomputed/{subject_id}_{condition}_jitter.csv"]

        elif group == "acoustic" and subgroup == "shimmer_recomputed":
            data_path = [f"{path}shimmer_recomputed/{subject_id}_{condition}_shimmer.csv"]

        elif group == "acoustic" and subgroup == "gne_recomputed":
            data_path = [f"{path}glottal_noise_recomputed/{subject_id}_{condition}_gne.csv"]

        elif group == "acoustic" and subgroup == "pause_segment_recomputed":
            data_path = [f"{path}pause_segment_recomputed/{subject_id}_{condition}_pausechar.csv"]

        elif group == "acoustic" and subgroup == "voice_frame_score_recomputed":
            data_path = [f"{path}voice_frame_score_recomputed/{subject_id}_{condition}_voiceprev.csv"]

        elif group == "movement" and subgroup == "eyeblink":
            data_path = [f"{path}eye_blink/{subject_id}_{condition}_eyeblinks.csv"]

        elif group == "movement" and subgroup == "eyeblink_binarized":
            data_path = [f"{path}eye_blink_binarized/{subject_id}_{condition}_eyeblinks.csv"]

        elif group == "movement" and subgroup == "voice_tremor_recomputed":
            data_path = [f"{path}voice_tremor_recomputed/{subject_id}_{condition}_vtremor.csv"]

    return data_path


def build_opendbm_derived_data_path() -> Path:
    return Path("output", "derived_variables")


def load_opendbm_facial_data(
    base_path: path_t, subject_id: str, condition: str, sampling_rate: float, suffix: str
) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix=suffix)
    facial_paths = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="facial")
    columns_to_drop = [
        "frame",
        "error_reason",
        "face_id",
        "timestamp",
        "confidence",
        "success",
        "dbm_master_url",
        "s_confidence",
        "neu_exp",
        "cai_exp",
        "neu_exp_full",
        "cai_exp_full",
    ]

    tar = tarfile.open(name=tar_path, mode="r")
    data = []
    for path in facial_paths:
        file = tar.extractfile(path)
        tmp = pd.read_csv(file)
        col_to_be_dropped = [val for val in tmp.columns if val in columns_to_drop]
        tmp = tmp.drop(col_to_be_dropped, axis=1)
        data.append(tmp)

    data = pd.concat(data, axis=1)
    data.index = data.index / sampling_rate
    data.index.name = "time [s]"
    tar.close()
    return data


def load_opendbm_acoustic_data(
    base_path: path_t, subject_id: str, condition: str, sampling_rate: float, suffix: str
) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix=suffix)
    acoustic_paths = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="acoustic")
    columns_to_drop = ["error_reason", "Frames", "dbm_master_url"]
    tar = tarfile.open(name=tar_path, mode="r")

    data = []
    for path in acoustic_paths:
        file = tar.extractfile(path)
        tmp = pd.read_csv(file)
        col_to_be_dropped = [val for val in tmp.columns if val in columns_to_drop]
        tmp = tmp.drop(col_to_be_dropped, axis=1)
        data.append(tmp)

    data = pd.concat(data, axis=1)
    data.index = data.index / sampling_rate
    data.index.name = "time [s]"
    tar.close()
    return data


def load_opendbm_movement_data(
    base_path: path_t, subject_id: str, condition: str, sampling_rate: float, suffix: str
) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix=suffix)
    movement_paths = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="movement")
    columns_to_drop = ["error", "error_reason", "Frames", "dbm_master_url"]
    tar = tarfile.open(name=tar_path, mode="r")

    data = []
    for path in movement_paths:
        file = tar.extractfile(path)
        tmp = pd.read_csv(file)
        col_to_be_dropped = [val for val in tmp.columns if val in columns_to_drop]
        tmp = tmp.drop(col_to_be_dropped, axis=1)
        data.append(tmp)

    data = pd.concat(data, axis=1)
    data.index = data.index / sampling_rate
    data.index.name = "time [s]"
    tar.close()
    return data


def load_opendbm_acoustic_seg_data(base_path: path_t, subject_id: str, condition: str, suffix: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix=suffix)
    acoustic_seg_path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="acoustic_seg")
    columns_to_drop = ["error"]
    tar = tarfile.open(name=tar_path, mode="r")

    data = []
    for path in acoustic_seg_path:
        file = tar.extractfile(path)
        tmp = pd.read_csv(file)
        col_to_be_dropped = [val for val in tmp.columns if val in columns_to_drop]
        tmp = tmp.drop(col_to_be_dropped, axis=1)
        data.append(tmp)

    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    tar.close()
    return data


def load_opendbm_audio_seg_data(base_path: path_t, subject_id: str, condition: str, suffix: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix=suffix)
    audio_seg_path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="audio_seg")
    columns_to_drop = ["error"]
    tar = tarfile.open(name=tar_path, mode="r")

    data = []
    for path in audio_seg_path:
        file = tar.extractfile(path)
        tmp = pd.read_csv(file)
        col_to_be_dropped = [val for val in tmp.columns if val in columns_to_drop]
        tmp = tmp.drop(col_to_be_dropped, axis=1)
        data.append(tmp)

    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    tar.close()
    return data


def load_opendbm_facial_tremor_data(
    base_path: path_t, subject_id: str, condition: str, sampling_rate: float, suffix: str
) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix=suffix)
    facial_tremor_path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="facial_tremor")[
        0
    ]
    tar = tarfile.open(name=tar_path, mode="r")
    file = tar.extractfile(facial_tremor_path)
    data = pd.read_csv(file)
    data.index = data.index / sampling_rate
    data.index.name = "time [s]"
    tar.close()
    return data


def load_speaker_diarization(base_path: path_t, subject_id: str, condition: str):
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    data_path = data_path.joinpath("video", "processed", f"{subject_id}_{condition}_speaker_diarization.csv")
    data = pd.read_csv(data_path, index_col="segment_id")
    return data


def get_opendbm_pitch_data(base_path: path_t, subject_id: str, condition: str, suffix: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix)
    path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="acoustic", subgroup="pitch")
    tar = tarfile.open(name=tar_path, mode="r")
    file = tar.extractfile(path[0])
    data = pd.read_csv(file)
    tar.close()
    return data


def get_opendbm_eyeblink_data(base_path: path_t, subject_id: str, condition: str, suffix: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix)
    path = build_opendbm_raw_data_path(
        subject_id=subject_id, condition=condition, group="movement", subgroup="eyeblink"
    )
    tar = tarfile.open(name=tar_path, mode="r")
    file = tar.extractfile(path[0])
    data = pd.read_csv(file)
    tar.close()
    return data


def get_opendbm_derived_features(base_path: path_t, subject_id: str, condition: str, suffix: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id.lower(), condition, suffix)
    file_path = build_opendbm_derived_data_path().joinpath("derived_output.csv")
    tar = tarfile.open(name=tar_path, mode="r")
    if type(file_path) == pathlib.WindowsPath:
        file_path = str(file_path)
        file_path = file_path.replace("\\", "/")
    else:
        file_path = str(file_path)
    file = tar.extractfile(file_path)
    data = pd.read_csv(file)
    data = data.drop(columns=["Filename"])
    # data.index = pd.MultiIndex.from_arrays([[subject_id], [condition]], names=["subject", "condition"])
    tar.close()
    return data
