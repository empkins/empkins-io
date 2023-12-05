import json
import math
import pathlib
import shutil
import tarfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from empkins_io.datasets.d03._utils.dataset_utils import get_uncleaned_openpose_data
from empkins_io.sensors.motion_capture.perception_neuron import PerceptionNeuronDataset
from empkins_io.utils._types import path_t
from empkins_io.utils.exceptions import TimelogNotFoundException


def build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def build_opendbm_tarfile_path(
    base_path: path_t, subject_id: str, condition: str, suffix: Optional[str] = None
) -> Path:
    path = build_data_path(base_path, subject_id, condition)
    path = path.joinpath("video", "processed")

    if suffix is None:
        path = path.joinpath(f"opendbm_output_{subject_id}_{condition}.tar.gz")
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


def load_mocap_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)

    mocap_path = data_path.joinpath("mocap/filtered")
    mocap_data = PerceptionNeuronDataset.from_folder(mocap_path)
    return mocap_data.data_as_df(index="time")


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


def get_opendbm_derived_features(
    base_path: path_t, subject_id: str, condition: str, phase: str, suffix: str
) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition, suffix)
    file_path = (
        build_opendbm_derived_data_path()
        .joinpath(phase)
        .joinpath(f"{subject_id}_{condition}_{phase}_derived_features_long.csv")
    )
    tar = tarfile.open(name=tar_path, mode="r")
    if type(file_path) == pathlib.WindowsPath:
        file_path = str(file_path)
        file_path = file_path.replace("\\", "/")
    else:
        file_path = str(file_path)
    file = tar.extractfile(file_path)
    data = pd.read_csv(file)
    index_cols = list(data.columns)[:-1]
    data = data.set_index(index_cols)
    tar.close()
    return data


def get_times_for_mocap(
    base_path: path_t, sampling_rate: float, subject_id: str, condition: str, phase: Optional[str] = "total"
) -> Sequence[float]:
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    time_file = data_path.joinpath(f"{subject_id}_times_{condition}.json")
    time_json = json.load(time_file.open(encoding="utf-8"))
    clap_mocap_frame = time_json["mocap"]["clap_frame"]
    clap_video_sec = time_json["video"]["clap_sec"]
    times_video = list(time_json["video"][phase].values())
    # subtract clap time to get relative start/end time
    times_relative = [vid_time - clap_video_sec for vid_time in times_video]
    # convert mocap clap time to seconds and add to start/end times
    times_mocap = [clap_mocap_frame / sampling_rate + time_rel for time_rel in times_relative]
    return times_mocap


def get_times_for_video(base_path: path_t, subject_id: str, condition: str, phase: Optional[str] = "total") -> list:
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    time_file = data_path.joinpath(f"{subject_id}_times_{condition}.json")
    try:
        time_json = json.load(time_file.open(encoding="utf-8"))
    except FileNotFoundError as e:
        raise TimelogNotFoundException(f"Video timelogs not found for {subject_id} {condition}!") from e
    times_video = list(time_json["video"][phase].values())
    return times_video


def get_video_path(base_path: path_t, subject_id: str, condition: str):
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    data_path = data_path.joinpath("video", "raw", f"{subject_id}_{condition}.mp4")
    assert data_path.exists()
    return data_path


def get_audio_path(base_path: path_t, subject_id: str, condition: str):
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    data_path = data_path.joinpath("video", "raw", f"{subject_id}_{condition}.wav")
    assert data_path.exists()
    return data_path


def clean_diarization(diarization) -> pd.DataFrame:
    dia = diarization.iloc[np.where(diarization["speaker"] == "SPEAKER_PANEL_INV")[0]]
    dia = dia.reset_index(drop=True)
    return dia


def identify_test_subject(diarization) -> str:
    panel_inv = "SPEAKER_PANEL_INV"
    diarization = diarization[diarization.speaker != panel_inv].reset_index(drop=True)
    diarization = diarization.set_index("speaker")
    data = diarization[["length"]].groupby("speaker").sum()
    return data.idxmax()[0]


def fix_stop_time(diarization) -> float:
    test_subject = identify_test_subject(diarization)
    last_element = diarization[diarization.speaker == test_subject].tail(1)
    return np.float(last_element["stop"])


def apply_diarization_aco(df, diarization, sampling_rate_audio):
    dia_segments = _prepare_diarization(diarization)
    bin_dia = _binarize_diarization(df.index[-1], dia_segments, sampling_rate_audio)
    df["diarization"] = bin_dia
    return df


def apply_diarization_aco_seg(df, diarization, sampling_rate_audio):
    dia_segments = _prepare_diarization(diarization)
    df = df.round({"start_time": 3, "end_time": 3})
    max_time = np.max([df.tail(1)["end_time"], dia_segments.tail(1)["stop"]])
    bin_dia = _binarize_diarization(max_time, dia_segments, sampling_rate_audio)
    start, stop = (df[["start_time", "end_time"]].to_numpy() * sampling_rate_audio).astype(int).T
    indices = [np.all(bin_dia[t1:t2]) for t1, t2 in zip(start, stop)]

    return df.loc[indices].reset_index(drop=True)


def _prepare_diarization(diarization):
    dia_segments = clean_diarization(diarization)
    if math.isnan(dia_segments.tail(1)["stop"]):
        idx = len(dia_segments.index) - 1
        t_stop = fix_stop_time(diarization)
        dia_segments.loc[idx, "stop"] = t_stop
        dia_segments.loc[idx, "length"] = dia_segments.loc[idx, "stop"] - dia_segments.loc[idx, "start"]
    dia_segments = dia_segments.round({"start": 3, "stop": 3})
    return dia_segments


def _binarize_diarization(max_time, diarization, fs):
    bin_dia = np.zeros(int(max_time * fs + 1), dtype=bool)
    for _, seg in diarization.iterrows():
        bin_dia[int(seg["start"] * fs) : int(seg["stop"] * fs)] = True
    return bin_dia


def extract_opendbm_data(base_path: path_t, subject_id: str, condition: str, suffix: Optional[str] = None):
    tarfile_path = build_opendbm_tarfile_path(
        base_path=base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition, suffix=suffix
    )

    data_path = build_opendbm_extracted_tar_path(
        base_path=base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition
    )
    if data_path.exists():
        shutil.rmtree(data_path)

    data_path = data_path.parent

    with tarfile.open(tarfile_path, "r:gz") as tar:
        tar.extractall(path=data_path)


def write_file_to_opendbm_tar(
    base_path: path_t,
    subject_id: str,
    condition: str,
    data: pd.DataFrame,
    data_type: Optional[str] = None,
    raw: Optional[bool] = False,
    derived: Optional[bool] = False,
    phase: Optional[str] = None,
    group: Optional[str] = None,
    subgroup: Optional[str] = None,
):
    tarfile_path = build_opendbm_tarfile_path(
        base_path=base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition
    )
    data_path = tarfile_path.parent

    if raw:
        path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group=group, subgroup=subgroup)[
            0
        ]
        path = data_path.joinpath(path)
    elif derived:
        path = build_opendbm_derived_data_path()
        path = path.joinpath(phase)
        path = path.joinpath(f"{subject_id}_{condition}_{phase}_derived_features_{data_type}.csv")

    else:
        return

    path = data_path.joinpath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def compress_opendbm_data(base_path: path_t, subject_id: str, condition: str, suffix: Optional[str] = None):
    tarfile_path = build_opendbm_tarfile_path(
        base_path=base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition, suffix=suffix
    )

    data_path = build_opendbm_extracted_tar_path(
        base_path=base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition
    )

    with tarfile.open(tarfile_path, "w:gz") as tar:
        tar.add(data_path, arcname=data_path.name)

    shutil.rmtree(data_path)


def get_uncleaned_openpose_prestudy_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id, condition)
    file_path = data_path.joinpath("video/processed/openpose_output.csv")
    return get_uncleaned_openpose_data(file_path)
