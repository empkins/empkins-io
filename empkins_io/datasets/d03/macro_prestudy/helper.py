import json
from pathlib import Path
from typing import Optional, Sequence
import tarfile

import pandas as pd
import numpy as np
from empkins_io.sensors.motion_capture.perception_neuron import PerceptionNeuronDataset
from empkins_io.utils._types import path_t


def build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def build_opendbm_tarfile_path(base_path: path_t, subject_id: str, condition: str, new: Optional[bool]=False) -> Path:
    path = build_data_path(base_path, subject_id, condition)
    path = path.joinpath("video", "processed")

    if not new:
        path = path.joinpath(f"opendbm_output_{subject_id}_{condition}.tar.gz")
        assert path.exists()

    else:
        path = path.joinpath(f"opendbm_output_{subject_id}_{condition}_new2.tar.gz")

    return path


def build_opendbm_raw_data_path(subject_id: str, condition: str, group: str, subgroup: Optional[str] = None) -> list:
    path = f"output/raw_variables/{subject_id}_{condition}/{group}/"

    data_path = []

    if subgroup is None:
        if group == "facial":
            data_path = list((f"{path}face_asymmetry/{subject_id}_{condition}_facasym.csv",
                              f"{path}face_au/{subject_id}_{condition}_facau.csv",
                              f"{path}face_expressivity/{subject_id}_{condition}_facemo.csv",
                              f"{path}face_landmark/{subject_id}_{condition}_faclmk.csv"))
        elif group == "acoustic":
            data_path = list((f"{path}formant_freq/{subject_id}_{condition}_formant.csv",
                              f"{path}harmonic_noise/{subject_id}_{condition}_hnr.csv",
                              f"{path}intensity/{subject_id}_{condition}_intensity.csv",
                              f"{path}mfcc/{subject_id}_{condition}_mfcc.csv",
                              f"{path}pitch/{subject_id}_{condition}_pitch.csv",
                              # f"{path}glottal_noise/{subject_id}_{condition}_gne.csv",
                              # f"{path}jitter/{subject_id}_{condition}_jitter.csv",
                              # f"{path}shimmer/{subject_id}_{condition}_shimmer.csv",
            ))
        elif group == "movement":
            data_path = list((f"{path}gaze/{subject_id}_{condition}_eyegaze.csv",
                              f"{path}head_movement/{subject_id}_{condition}_headmov.csv",
                              f"{path}head_pose/{subject_id}_{condition}_headpose.csv",
                              # f"{path}eye_blink/{subject_id}_{condition}_eyeblinks.csv",
            ))

    else:
        if group == "acoustic" and subgroup == "pitch":
            data_path = [f"{path}pitch/{subject_id}_{condition}_pitch.csv"]

        if group == "acoustic" and subgroup == "jitter":
            data_path = [f"{path}jitter_recomp/{subject_id}_{condition}_jitter.csv"]

        if group == "acoustic" and subgroup == "shimmer":
            data_path = [f"{path}shimmer/{subject_id}_{condition}_shimmer.csv"]

        if group == "acoustic" and subgroup == "gne":
            data_path = [f"{path}glottal_noise/{subject_id}_{condition}_gne.csv"]

        elif group == "movement" and subgroup == "eyeblink":
            data_path = [f"{path}eye_blink/{subject_id}_{condition}_eyeblinks.csv"]

    return data_path

def load_mocap_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)

    mocap_path = data_path.joinpath("mocap/filtered")
    mocap_data = PerceptionNeuronDataset.from_folder(mocap_path)
    return mocap_data.data_as_df(index="time")

def load_opendbm_facial_data(base_path: path_t, subject_id: str, condition: str, sampling_rate: float) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition)
    facial_paths = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="facial")
    columns_to_drop = ["frame", "error_reason", "face_id", "timestamp", "confidence", "success", "dbm_master_url",
                       "s_confidence", "neu_exp", "cai_exp", "neu_exp_full", "cai_exp_full"]
    tar = tarfile.open(name=tar_path, mode='r')
    print(tar.getmembers())

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

def load_opendbm_acoustic_data(base_path: path_t, subject_id: str, condition: str, sampling_rate: float) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition)
    acoustic_paths = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="acoustic")
    columns_to_drop = ["error_reason", "Frames", "dbm_master_url"]
    tar = tarfile.open(name=tar_path, mode='r')

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

def load_opendbm_movement_data(base_path: path_t, subject_id: str, condition: str, sampling_rate: float) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition)
    movement_paths = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="movement")
    columns_to_drop = ["error_reason", "Frames", 'dbm_master_url']
    tar = tarfile.open(name=tar_path, mode='r')

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

def get_opendbm_pitch_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition)
    path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="acoustic", subgroup="pitch")
    tar = tarfile.open(name=tar_path, mode='r')
    file = tar.extractfile(path[0])
    data = pd.read_csv(file)
    tar.close()
    return data

def get_opendbm_eyeblink_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    tar_path = build_opendbm_tarfile_path(base_path.joinpath("data_per_subject"), subject_id, condition)
    path = build_opendbm_raw_data_path(subject_id=subject_id, condition=condition, group="movement", subgroup="eyeblink")
    tar = tarfile.open(name=tar_path, mode='r')
    file = tar.extractfile(path[0])
    data = pd.read_csv(file)
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

def get_times_for_video(base_path: path_t, subject_id: str, condition: str,
                        phase: Optional[str] = "total") -> list:
    data_path = build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    time_file = data_path.joinpath(f"{subject_id}_times_{condition}.json")
    time_json = json.load(time_file.open(encoding="utf-8"))
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