from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.io.nilspod import _handle_counter_inconsistencies_session

from empkins_io.datasets.d03.macro_ap01._custom_synced_session import CustomSyncedSession
from empkins_io.sensors.motion_capture.motion_capture_formats import mvnx
from empkins_io.utils._types import path_t, str_t
from empkins_io.utils.exceptions import NilsPodDataNotFoundException


def _build_data_path(base_path: path_t, subject_id: str, condition: str) -> Path:
    base_path = Path(base_path)
    path = base_path.joinpath(f"{subject_id}/{condition}")
    assert path.exists()
    return path


def _load_nilspod_session(base_path: path_t, subject_id: str, condition: str) -> Tuple[pd.DataFrame, float]:
    data_path = _build_data_path(base_path.joinpath("data_per_subject"), subject_id=subject_id, condition=condition)
    data_path = data_path.joinpath("nilspod/raw")

    nilspod_files = sorted(data_path.glob("NilsPodX-*.bin"))
    if len(nilspod_files) == 0:
        raise NilsPodDataNotFoundException("No NilsPod files found in directory!")

    session = CustomSyncedSession.from_folder_path(data_path)
    # fix for "classical nilspod bug" where last sample counter is corrupted
    session.cut(stop=-10, inplace=True)
    session.align_to_syncregion(inplace=True)

    _handle_counter_inconsistencies_session(session, handle_counter_inconsistency="ignore")

    # convert dataset to dataframe and localize timestamp
    df = session.data_as_df(index="local_datetime", concat_df=True)
    df.index.name = "time"
    fs = session.info.sampling_rate_hz[0]
    return df, fs


def _load_tsst_mocap_data(base_path: path_t, subject_id: str, condition: str) -> (pd.DataFrame, datetime):
    data_path = _build_data_path(
        base_path.joinpath("data_per_subject"),
        subject_id=subject_id,
        condition=condition,
    )

    mocap_path = data_path.joinpath("mocap/processed")
    mocap_file = mocap_path.joinpath(f"{subject_id}_{condition}-TEST.mvnx")

    if not mocap_file.is_file():
        raise FileNotFoundError

    mvnx_data = mvnx.MvnxData(mocap_file)

    return mvnx_data.data, mvnx_data.start


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

    if not mocap_file.is_file():
        raise FileNotFoundError

    mvnx_data = mvnx.MvnxData(mocap_file)

    return mvnx_data.data


def _get_times_for_mocap(
    timelog: pd.DataFrame,
    start_time: datetime,
    phase: Optional[str_t] = "total",
) -> pd.DataFrame:
    if phase == "total":
        timelog = timelog.drop(columns="Prep", level="phase")
    else:
        if isinstance(phase, str):
            phase = [phase]
        timelog = timelog.loc[:, phase]

    timelog = (timelog - start_time).apply(lambda x: x.dt.total_seconds())
    timelog = timelog.T["time"].unstack("start_end")
    return timelog


def rearrange_hr_ensemble_data(
    hr_ensemble: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    hr_ensemble = {
        cond: {key: val.xs(cond, level="condition", axis=1) for key, val in hr_ensemble.items()}
        for cond in ["tsst", "ftsst"]
    }

    times = {cond: {phase: len(data) for phase, data in hr_ensemble[cond].items()} for cond in ["tsst", "ftsst"]}

    hr_ensemble_concat = {key: _compute_ensemble(hr_ensemble[key]) for key in ["tsst", "ftsst"]}
    return hr_ensemble_concat, times


def _compute_ensemble(data: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat(data, names=["phase"])
    data = data.reindex(["Baseline", "Prep", "Talk", "Math"], level="phase")
    data = data.reset_index(drop=True)
    data.index.name = "time"
    return data
