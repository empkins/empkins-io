from datetime import datetime
from pathlib import Path

import pandas as pd
from biopsykit.io.nilspod import _handle_counter_inconsistencies_session

from empkins_io.datasets.d03._utils.dataset_utils import get_uncleaned_openpose_data
from empkins_io.datasets.d03.macro_ap01._custom_synced_session import CustomSyncedSession
from empkins_io.sensors.motion_capture.motion_capture_formats import mvnx
from empkins_io.sensors.motion_capture.xsens import XSensDataset
from empkins_io.utils._types import path_t, str_t
from empkins_io.utils.exceptions import NilsPodDataNotFoundError


def _build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def _load_nilspod_session(base_path: path_t, subject_id: str, condition: str) -> tuple[pd.DataFrame, float]:
    data_path = _build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    data_path = data_path.joinpath("nilspod/raw")

    nilspod_files = sorted(data_path.glob("NilsPodX-*.bin"))
    if len(nilspod_files) == 0:
        raise NilsPodDataNotFoundError("No NilsPod files found in directory!")

    session = CustomSyncedSession.from_folder_path(data_path)
    # fix for "classical nilspod bug" where last sample counter is corrupted
    session = session.cut(stop=-10)
    session = session.align_to_syncregion()

    _handle_counter_inconsistencies_session(session, handle_counter_inconsistency="ignore")

    # convert dataset to dataframe and localize timestamp
    df = session.data_as_df(index="local_datetime", concat_df=True)
    df.index.name = "time"
    fs = session.info.sampling_rate_hz[0]
    return df, fs


def _load_tsst_mocap_data(
    base_path: path_t, subject_id: str, condition: str, *, verbose: bool = True
) -> (pd.DataFrame, datetime):
    data_path = _build_data_path(
        base_path.joinpath("data_per_subject"),
        subject_id=subject_id,
        condition=condition,
    )

    mocap_path = data_path.joinpath("mocap/processed")
    mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-TEST.mvnx")
    if not mocap_file.exists():
        # look for gzip file
        mocap_file = mocap_file.with_suffix(".mvnx.gz")

    if not mocap_file.exists():
        raise FileNotFoundError(f"File '{mocap_file}' not found!")

    data = XSensDataset.from_mvnx_file(mocap_file, verbose=verbose)
    data = data.data_as_df(index="local_datetime")

    return data


def _load_gait_mocap_data(
    base_path: path_t,
    subject_id: str,
    condition: str,
    test: str,
    trial: int,
    speed: str,
) -> (pd.DataFrame, datetime):
    data_path = _build_data_path(
        base_path.joinpath("data_per_subject"),
        subject_id=subject_id,
        condition=condition,
    )

    mocap_path = data_path.joinpath("mocap/processed")

    if test == "TUG" or trial == 0:
        mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-{test}{trial}.mvnx")
    else:
        mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-{test}{trial}_{speed}.mvnx")

    if not mocap_file.exists():
        mocap_file = mocap_file.with_suffix(".mvnx.gz")

    if not mocap_file.exists():
        raise FileNotFoundError(f"File '{mocap_file}' not found!")

    mvnx_data = mvnx.MvnxData(mocap_file)

    return mvnx_data.data


def _get_times_for_mocap(
    timelog: pd.DataFrame,
    phase: str_t | None = "total",
) -> pd.DataFrame:
    if phase == "total":
        timelog = timelog.drop(columns="prep", level="phase")
        timelog = timelog.loc[:, [("talk", "start"), ("math", "end")]]
        timelog = timelog.rename({"talk": "total", "math": "total"}, level="phase", axis=1)
    else:
        if isinstance(phase, str):
            phase = [phase]
        timelog = timelog.loc[:, phase]

    timelog = timelog.T["time"].unstack("start_end").reindex(["start", "end"], level="start_end", axis=1)
    return timelog


def rearrange_hr_ensemble_data(
    hr_ensemble: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    hr_ensemble = {
        cond: {key: val.xs(cond, level="condition", axis=1) for key, val in hr_ensemble.items()}
        for cond in ["tsst", "ftsst"]
    }

    times = {cond: {phase: len(data) for phase, data in hr_ensemble[cond].items()} for cond in ["tsst", "ftsst"]}

    hr_ensemble_concat = {key: _compute_ensemble(hr_ensemble[key]) for key in ["tsst", "ftsst"]}
    return hr_ensemble_concat, times


def _compute_ensemble(data: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat(data, names=["phase"])
    data = data.reindex(["baseline", "prep", "talk", "math"], level="phase")
    data = data.reset_index(drop=True)
    data.index.name = "time"
    return data


def get_uncleaned_openpose_macro_data(base_path: path_t, subject_id: str, condition: str) -> pd.DataFrame:
    body_video_path = base_path.joinpath("data_per_subject").joinpath(f"{subject_id}/{condition}/video/body")
    file_path = body_video_path.joinpath("processed/openpose_output.csv")
    return get_uncleaned_openpose_data(file_path)
