from collections.abc import Sequence
from datetime import datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from biopsykit.io import load_pandas_dict_excel, load_time_log
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.utils.data_processing import resample_sec

from empkins_io.datasets.d03.micro_prestudy.constants import (
    ARTIFACT_MAPPING,
    STUDY_PART_DICT_RADAR,
    STUDY_PART_DICT_VIDEO,
    STUDY_PARTS,
    VIDEO_FRAME_RATE,
)
from empkins_io.sensors.utils import cut_data_to_overlap
from empkins_io.signal_alignment import signal_align
from empkins_io.utils._types import path_t, str_t

__all__ = [
    "build_data_path",
    "get_video_path",
    "load_ecg_data",
    "load_ecg_hr",
    "load_mis_data",
    "load_timelog_file",
    "load_annotation_file",
    "load_sync_interval",
    "load_emotion_data",
    "load_dominant_emotion_data",
    "load_hr_synced",
    "preprocess_hr_data",
    "extract_annotated_data",
    "get_video_start_time",
    "load_annotations",
    "load_heart_rate_data",
]

MAX_SHIFT = 5


def build_data_path(base_path: path_t, subject_id: str, condition: str | None = None) -> Path:
    if condition is None:
        path = Path(base_path).joinpath(f"data_per_subject/{subject_id}")
    else:
        path = Path(base_path).joinpath(f"data_per_subject/{subject_id}/{condition}")
    assert path.exists()
    return path


def get_video_path(base_path: path_t, subject_id: str, condition: str, study_part: str) -> path_t:
    video_folder = build_data_path(base_path, subject_id, condition)
    video_folder = video_folder.joinpath("video/raw")
    video_files = sorted(video_folder.glob(f"*_{STUDY_PART_DICT_VIDEO[study_part]}.mp4"))
    video_files = [file for file in video_files if not file.name.startswith("._")]

    if len(video_files) == 0:
        raise ValueError(f"No video for study part '{study_part}' in folder {video_folder}!")
    return video_files[0]


def load_ecg_data(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    artifact: str_t | None = None,
    use_cache: bool | None = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath("ecg/raw")
    ecg_files = sorted(file_path.glob("*.bin"))
    ecg_file = ecg_files[STUDY_PARTS.index(study_part)]
    ecg_data = load_dataset_nilspod(ecg_file, datastreams=["ecg"])[0]

    # TODO shouldn't the signal be flipped? (see ECG_Radar_Sync.ipynb):
    # ecg_data["ecg"] = -1 * ecg_data["ecg"]

    timelog = load_timelog_file(base_path, subject_id, condition, study_part, phase, subphase, use_cache)
    annotations = load_annotation_file(base_path, subject_id, condition, study_part, artifact, use_cache)
    sync_interval = load_sync_interval(base_path, subject_id, condition, study_part)
    return _load_and_split_data(ecg_data, timelog, annotations, study_part, phase, subphase, artifact, sync_interval)


def load_ecg_hr(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    artifact: str_t | None = None,
    use_cache: bool | None = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath("ecg/processed")
    ecg_file = _find_study_part_file(file_path, study_part, "*.csv")

    ecg_df = pd.read_csv(ecg_file, index_col="time")
    ecg_df.index = pd.to_datetime(ecg_df.index).tz_convert("CET")  # TODO: do we need tz_convert?

    timelog = load_timelog_file(base_path, subject_id, condition, study_part, phase, subphase, use_cache)
    annotations = load_annotation_file(base_path, subject_id, condition, study_part, artifact, use_cache)
    sync_interval = load_sync_interval(base_path, subject_id, condition, study_part)

    return _load_and_split_data(ecg_df, timelog, annotations, study_part, phase, subphase, artifact, sync_interval)


def load_mis_data(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    artifact: str_t | None = None,
    use_cache: bool | None = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath("radar/processed")
    file_path = sorted(file_path.glob("*.xlsx"))[0]
    mis_data = load_pandas_dict_excel(file_path, timezone="CET")[STUDY_PART_DICT_RADAR[study_part]]
    mis_data = mis_data[["Heart_Rate"]]

    timelog = load_timelog_file(base_path, subject_id, condition, study_part, phase, subphase, use_cache)
    annotations = load_annotation_file(base_path, subject_id, condition, study_part, artifact, use_cache)
    sync_interval = load_sync_interval(base_path, subject_id, condition, study_part)

    return _load_and_split_data(mis_data, timelog, annotations, study_part, phase, subphase, artifact, sync_interval)


def load_hr_synced(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    artifact: str_t | None = None,
    use_cache: bool | None = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath("hr_synced")
    file_path = sorted(file_path.glob("*.xlsx"))[0]
    hr_data = load_pandas_dict_excel(file_path, timezone="CET")[STUDY_PART_DICT_RADAR[study_part]]

    timelog = load_timelog_file(base_path, subject_id, condition, study_part, phase, subphase, use_cache)
    annotations = load_annotation_file(base_path, subject_id, condition, study_part, artifact, use_cache)
    sync_interval = load_sync_interval(base_path, subject_id, condition, study_part)

    return _load_and_split_data(hr_data, timelog, annotations, study_part, phase, subphase, artifact, sync_interval)


def load_emotion_data(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    use_cache: bool | None = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath(
        f"video/processed/emotion_data_{subject_id}_{condition}_{STUDY_PART_DICT_VIDEO[study_part]}.csv"
    )

    timelog = load_timelog_file(base_path, subject_id, condition, study_part, phase, subphase, use_cache)
    annotations = load_annotation_file(base_path, subject_id, condition, study_part, use_cache=use_cache)
    emotion_data = pd.read_csv(file_path, index_col="time")
    emotion_data.index = pd.to_datetime(emotion_data.index).tz_localize("Europe/Berlin")
    return _load_and_split_data(emotion_data, timelog, annotations, study_part, phase, subphase)


def load_dominant_emotion_data(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    use_cache: bool | None = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath(
        f"video/processed/dominant_emotion_{subject_id}_{condition}_{STUDY_PART_DICT_VIDEO[study_part]}.csv"
    )

    timelog = load_timelog_file(base_path, subject_id, condition, study_part, phase, subphase, use_cache)
    annotations = load_annotation_file(base_path, subject_id, condition, study_part, use_cache=use_cache)
    emotion_data = pd.read_csv(file_path, index_col="time")
    emotion_data.index = pd.to_datetime(emotion_data.index).tz_localize("Europe/Berlin")
    return _load_and_split_data(emotion_data, timelog, annotations, study_part, phase, subphase)


def _load_and_split_data(
    data: pd.DataFrame,
    timelog: pd.DataFrame,
    annotations: pd.DataFrame,
    study_part: str_t,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    artifact: str_t | None = None,
    sync_interval: tuple[time, time] | None = None,
) -> Sequence[pd.DataFrame] | pd.DataFrame | dict[str, pd.DataFrame | Sequence[pd.DataFrame]]:
    dict_data = {}
    if annotations is not None and annotations.empty:
        # no annotations of this type exist for these specs -> return empty dataframe
        return pd.DataFrame(columns=data.columns)
    if phase is None:
        # phases == None => no specific phase(s) were selected, i.e. extract data from all phases
        if subphase is None:
            # all phases and subphases were selected => return continuous dataframe from beginning of the
            # first phase until the end of the last phase
            if artifact is None:
                # return Dataframe without further splitting
                end_time_idx = 1 if study_part == "pre" else -1
                return data.between_time(timelog.iloc[0, 0], timelog.iloc[0, end_time_idx])
            relevant_data, relevant_annotations = _cut_data_to_timelog(data, annotations, timelog, study_part)
            return extract_annotated_data(relevant_data, relevant_annotations, sync_interval)
        for key, timelog_phase in timelog.groupby("phase", axis=1):
            # all phases were selected, but only specific subphases were selected => extract these subphases and return
            # a dictionary with data from these subphases
            if artifact is None:
                dict_data[key] = data.between_time(*list(timelog_phase.iloc[0]))
            else:
                relevant_data, relevant_annotations = _cut_data_to_timelog(data, annotations, timelog_phase, study_part)
                dict_data[key] = extract_annotated_data(relevant_data, relevant_annotations, sync_interval)
    else:
        for ph in phase:
            # specific phases were selected => filter these phases and return data from these phases in a dictionary
            # here, we don't need to take care of whether all or only specific subphases were selected because the
            # timelog file was already filtered to only include the selected subphases
            timelog_phase = timelog.filter(like=ph)
            if artifact is None:
                dict_data[ph] = data.between_time(timelog_phase.iloc[0, 0], timelog_phase.iloc[0, -1])
            else:
                relevant_data, relevant_annotations = _cut_data_to_timelog(data, annotations, timelog_phase, study_part)
                dict_data[ph] = extract_annotated_data(relevant_data, relevant_annotations, sync_interval)

    if len(dict_data) == 1:
        # "squeeze" dictionary if it only contains one entry
        return next(iter(dict_data.values()))
    return dict_data


def _get_annotation_start_and_end(annotations: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    start = annotations["start_time"].str.split(" ", expand=True).iloc[:, 1]
    end = annotations["end_time"].str.split(" ", expand=True).iloc[:, 1]
    return start, end


def _cut_data_to_timelog(
    data: pd.DataFrame, annotations: pd.DataFrame, timelog: pd.DataFrame, study_part: str_t
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # split data and annotations into lists of relevant parts
    start, end = _get_annotation_start_and_end(annotations)
    end_time_idx = 1 if study_part == "pre" else -1
    timelog_start = timelog.iloc[0, 0]
    timelog_end = timelog.iloc[0, end_time_idx]
    # exclude all annotations without overlap with timelog interval
    relevant_annotations = annotations[
        ~(((start <= timelog_start) & (end <= timelog_start)) | ((start >= timelog_end) & (end >= timelog_end)))
    ]
    relevant_data = data.between_time(timelog_start, timelog_end)
    return relevant_data, relevant_annotations


def load_timelog_file(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str_t | None = None,
    phase: str_t | None = None,
    subphase: str_t | None = None,
    use_cache: bool | None = True,
):
    if isinstance(phase, str):
        phase = [phase]
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath(f"time_log/processed/time_log_{subject_id}_{condition}.csv")
    if use_cache:
        timelog = _cached_load_time_log(file_path, continuous_time=False)
    else:
        timelog = load_time_log(file_path, continuous_time=False)
    timelog = timelog.filter(like=study_part)
    list_filter = []
    # filter columns in timelog according to the selected phases and subphases
    for level in [phase, subphase]:
        if level is None:
            continue
        for element in level:
            list_filter.append(timelog.filter(like=element))
        timelog = pd.concat(list_filter, axis=1)
        list_filter = []
    return timelog


def load_annotation_file(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str_t | None = None,
    artifact: str_t | None = None,
    use_cache: bool | None = True,
):
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath("annotations")
    file_path = _find_study_part_file(file_path, study_part, "*.csv")
    annotations = _cached_load_annotations(file_path, artifact) if use_cache else load_annotations(file_path, artifact)
    return annotations


def load_annotations(annotation_file: path_t, artifact: str_t | None = None) -> pd.DataFrame | None:
    annot_data = pd.read_csv(annotation_file)
    annot_data = annot_data[["start_time", "end_time", "description"]]
    annot_data = annot_data.sort_values(by=["start_time"])
    if artifact:
        if artifact == "clean":
            last_end = annot_data["end_time"].to_numpy()[-1]
            date = last_end.split(" ")[0]
            annot_data["start_time_shifted"] = annot_data["end_time"].shift(1, fill_value=f"{date} {time.min}")
            annot_data["end_time"] = annot_data["start_time"]
            annot_data["start_time"] = annot_data["start_time_shifted"]
            annot_data = (
                annot_data[["start_time", "end_time"]]
                .append(pd.DataFrame([[last_end, f"{date} {time.max}"]], columns=["start_time", "end_time"]))
                .reset_index(drop=True)
            )
            annot_data = annot_data[annot_data["start_time"] != annot_data["end_time"]]
            return annot_data
        if artifact == "all":
            return annot_data[["start_time", "end_time"]]
        descriptor = ARTIFACT_MAPPING[artifact]
        # only one keyword to search for
        if isinstance(descriptor, str):
            annot_data_filtered = annot_data[annot_data["description"].str.contains(descriptor)]
        # list of possible keywords to search for
        elif isinstance(descriptor, list):
            annot_data_filtered = pd.DataFrame(columns=annot_data.columns)
            for d in descriptor:
                data_to_append = annot_data[annot_data["description"].str.contains(str(d))]
                annot_data_filtered = pd.concat((annot_data_filtered, data_to_append))
            annot_data_filtered = annot_data_filtered.drop_duplicates()
        else:
            raise ValueError("Artifact mapping has invalid data type")
        return annot_data_filtered[["start_time", "end_time"]]
    # if artifact is None, data won't be cut into artifact occurrences
    return None


def _find_study_part_file(path: path_t, study_part: str, file_ending: str):
    files = list(path.glob(file_ending))
    try:
        file = next(f for f in files if study_part in Path(f).stem.lower())
    except IndexError as e:
        raise ValueError(f"No corresponding file for study part '{study_part}' in folder {path}!") from e
    return file


def extract_annotated_data(
    data: pd.DataFrame, annotations: pd.DataFrame, sync_interval: tuple[time, time] | None = None
) -> Sequence[pd.DataFrame]:
    if sync_interval:
        # cut to synced interval that was considered for labeling
        data = data.between_time(sync_interval[0], sync_interval[1])
    # use annotated intervals
    filtered_data = annotations.apply(
        lambda df: data[((data.index >= df["start_time"]) & (data.index <= df["end_time"]))], axis=1
    )

    if filtered_data.empty:
        return []
    return filtered_data.to_list()


def load_heart_rate_data(ecg_file: path_t, radar_file: path_t, study_part: str):
    """Load data from storage and cut to overlapping region."""
    ecg_df = pd.read_csv(ecg_file, index_col="time")
    ecg_df.index = pd.to_datetime(ecg_df.index).tz_convert("CET")

    radar_df = load_pandas_dict_excel(radar_file, timezone="CET")[STUDY_PART_DICT_RADAR[study_part]]

    # align signals
    ecg_df, radar_df = cut_data_to_overlap(ecg_df, radar_df)
    return ecg_df, radar_df


def preprocess_hr_data(ecg_df: pd.DataFrame, radar_df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 1Hz, do fine alignment and concatenate to one Dataframe."""
    hr_ecg = ecg_df["Heart_Rate"]
    hr_radar = radar_df["Heart_Rate"]

    # resample to 1 Hz
    hr_ecg = resample_sec(hr_ecg)
    hr_radar = resample_sec(hr_radar)
    index_intersect = hr_ecg.index.intersection(hr_radar.index)
    hr_ecg = hr_ecg.loc[index_intersect]
    hr_radar = hr_radar.loc[index_intersect]

    # TODO window-wise shift
    hr_ecg, hr_radar_shift, shift_idx = signal_align(hr_ecg, hr_radar)
    print(f"Shift: {shift_idx:.3f} seconds")
    if shift_idx < MAX_SHIFT:
        hr_radar = hr_radar_shift

    # extract heart rates only
    hr_ecg = hr_ecg.rename(columns={"Heart_Rate": "Heart_Rate_Ecg"})
    hr_radar = hr_radar.rename(columns={"Heart_Rate": "Heart_Rate_Radar"})

    combined_data = hr_ecg.join(hr_radar)
    combined_data = combined_data.reset_index(drop=True)
    return combined_data


def load_sync_interval(
    base_path: path_t,
    subject_id: str,
    condition: str,
    study_part: str,
):
    video_start = get_video_start_time(base_path, subject_id, condition, study_part)
    file_path = build_data_path(base_path, subject_id, condition)
    file_path = file_path.joinpath("video/raw")
    file_path = _find_study_part_file(file_path, study_part, "*.xlsx")
    sync_data = pd.read_excel(file_path)
    # calculate start and end of synced part in seconds
    sync_start = sync_data.iloc[0]["Video"] / VIDEO_FRAME_RATE
    sync_end = sync_data.iloc[1]["Video"] / VIDEO_FRAME_RATE
    # calculate time of sync start and end based on video start time
    sync_start = video_start + timedelta(seconds=sync_start)
    sync_end = video_start + timedelta(seconds=sync_end)
    return sync_start.time(), sync_end.time()


def get_video_start_time(base_path: path_t, subject_id: str, condition: str, study_part: str) -> datetime:
    video_path = get_video_path(base_path, subject_id, condition, study_part)
    filename = Path(video_path).name
    start_time = "".join(filename.split("_")[2:6])
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
    return start_time


_cached_load_time_log = lru_cache(maxsize=20)(load_time_log)
_cached_load_annotations = lru_cache(maxsize=20)(load_annotations)
