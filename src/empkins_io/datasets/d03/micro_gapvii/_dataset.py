from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import ClassVar

import pandas as pd
from biopsykit.io.io import load_long_format_csv
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass
from biopsykit.utils.file_handling import get_subject_dirs
from pandas import DataFrame
from tpcp import Dataset

from empkins_io.datasets.d03._utils.check_data_completeness import (
    check_data_completeness,
)
from empkins_io.datasets.d03.micro_gapvii.helper import (
    _build_data_path,
    _load_biopac_data,
    _load_nilspod_session,
    _load_radar_data,
    _load_timelog,
    _load_timelog_video,
    compute_reference_heartbeats,
    compute_reference_pep,
    get_opendbm_derived_features,
    load_labeling_borders,
    load_opendbm_acoustic_data,
    load_opendbm_acoustic_seg_data,
    load_opendbm_audio_seg_data,
    load_opendbm_facial_data,
    load_opendbm_facial_tremor_data,
    load_opendbm_movement_data,
    load_speaker_diarization,
)
from empkins_io.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
_cached_get_radar_data = lru_cache(maxsize=4)(_load_radar_data)
_cached_load_nilspod_data = lru_cache(maxsize=4)(_load_nilspod_session)
_cached_load_opendbm_facial_data = lru_cache(maxsize=4)(load_opendbm_facial_data)
_cached_load_opendbm_acoustic_data = lru_cache(maxsize=4)(load_opendbm_acoustic_data)
_cached_load_opendbm_movement_data = lru_cache(maxsize=4)(load_opendbm_movement_data)
_cached_load_opendbm_acoustic_seg_data = lru_cache(maxsize=4)(load_opendbm_acoustic_seg_data)
_cached_load_opendbm_audio_seg_data = lru_cache(maxsize=4)(load_opendbm_audio_seg_data)
_cached_load_opendbm_facial_tremor_data = lru_cache(maxsize=4)(load_opendbm_facial_tremor_data)
_cached_load_speaker_diarization = lru_cache(maxsize=4)(load_speaker_diarization)


class MicroBaseDataset(Dataset):
    base_path: path_t
    data_tabular_path: path_t

    SAMPLING_RATES_BIOPAC: ClassVar[dict[str, int]] = {"ecg": 1000, "icg": 1000}
    SAMPLING_RATE_EMRAD: ClassVar[float] = 1953.125

    sync_on_load: bool
    use_cache: bool
    phase_fine: bool
    opendbm_suffix: str | None
    _sample_times_saliva: ClassVar[tuple[int]] = (-40, -1, 16, 25, 35, 45, 60, 75)
    _sample_times_bloodspot: ClassVar[tuple[int]] = (-40, 60)

    NILSPOD_MAPPING: ClassVar[dict[str, str]] = {
        "chest": "b0c2",  # ecg
        "sternum": "157e",  # sternum
        "sync": "9e02",  # sync with mocap (MASTER)
        "board_broken": "b012",
        "board": "3d73",
        # sync with video (clapper board)
    }

    PHASES_COARSE: ClassVar[Sequence[str]] = [
        "Prep",
        "Pause_1",
        "Talk",
        "Pause_2",
        "Pause_3",
        "Math",
        "Pause_4",
        "Pause_5",
    ]

    PHASES_FINE: ClassVar[Sequence[str]] = [
        "Prep",
        "Pause_1",
        "Talk_1",
        "Pause_2",
        "Talk_2",
        "Pause_3",
        "Math_1",
        "Pause_4",
        "Math_2",
        "Pause_5",
    ]

    CONDITIONS: ClassVar[Sequence[str]] = ["tsst", "ftsst"]

    MISSING_DATA: ClassVar[Sequence[str]] = [
        "VP_001",
        "VP_002",
        "VP_003",
        "VP_004",
        "VP_005",
        "VP_006",
        "VP_007",
        "VP_008",
        "VP_009",
        "VP_010",
        "VP_011",
        "VP_012",
        "VP_013",
        "VP_014",
        "VP_015",
        "VP_016",
        "VP_017",
        "VP_018",
        "VP_019",
        "VP_020",
        "VP_021",
        "VP_022",
        "VP_023",
        "VP_024",
        "VP_025",
        "VP_026",
        "VP_027",
        "VP_028",
        "VP_029",
        "VP_030",
        "VP_031",
        "VP_032",
        "VP_033",
        "VP_034",
        "VP_035",
        "VP_036",
        "VP_037",
        "VP_038",
        "VP_039",
        "VP_040",
        "VP_041",
        "VP_042",
        "VP_043",
        "VP_044",
        "VP_045",
        "VP_046",
        "VP_047",
        "VP_048",
        "VP_049",
        "VP_050",
        "VP_051",
        "VP_052",
        "VP_053",
        "VP_054",
        "VP_055",
        "VP_056",
        "VP_058",
        "VP_060",
        "VP_066",
        "VP_067",
        "VP_068",
        "VP_071",
        "VP_077",
        "VP_080",
        "VP_082",
        "VP_083",
        "VP_086",
        "VP_093",
        "VP_094",
        "VP_103",
    ]  # Missing data (add participant IDs here)

    MISSING_RADAR_SYNC_IN_PEAKS: ClassVar[Sequence[str]] = ["VP_002", "VP_003"]

    MANUAL_START_TIME_BIOPAC: ClassVar[dict[str, dict[str, str]]] = {
        "VP_096": {"tsst": "2023-11-16 13:23:43"},
        "VP_097": {"tsst": "2023-11-16 14:26:18"},
    }

    GENDER_MAPPING: ClassVar[dict[int, str]] = {1: "Female", 2: "Male"}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        exclude_missing_data: bool = False,
        use_cache: bool = True,
        phase_fine: bool = False,
        opendbm_suffix: str | None = None,
        only_labeled: bool = False,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.data_tabular_path = self.base_path.joinpath("data_tabular")
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.phase_fine = phase_fine
        self.opendbm_suffix = opendbm_suffix
        self.only_labeled = only_labeled
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"

        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_[0-9]{3}")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        if self.phase_fine:
            index = list(product(participant_ids, self.CONDITIONS, self.PHASES_FINE))
            index = pd.DataFrame(index, columns=["participant", "condition", "phase"])
            return index

        index = list(product(participant_ids, self.CONDITIONS, self.PHASES_COARSE))
        index = pd.DataFrame(index, columns=["participant", "condition", "phase"])
        return index

    @property
    def subset_micro1_0(self):
        # return the subset with subject identifiers > VP_045
        self.subset_index = self.index.query("subject >= 'VP_001' and subject <= 'VP_045'")
        return self

    @property
    def subset_micro1_1(self):
        # return the subset from the second part of the study VP >= 050
        self.subset_index = self.index.query("subject >= 'VP_050'")
        return self

    @property
    def sampling_rate_emrad(self) -> float:
        return self.SAMPLING_RATE_EMRAD

    @property
    def sampling_rate_ecg(self) -> int:
        return self.SAMPLING_RATES_BIOPAC["ecg"]

    @property
    def sampling_rate_icg(self) -> int:
        return self.SAMPLING_RATES_BIOPAC["icg"]

    @cached_property
    def biopac(self) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]

        if self.is_single(None):
            phase = self.index["phase"][0]
        elif self.is_single(["participant", "condition"]):
            phase = "all"
        else:
            raise ValueError("Biopac data can only be accessed for one single participant and condition at once!")

        # some recorded biopac data failed to have a correct start time in the acq file, so we manually set it here
        # for most cases, the start time is correct in the acq file
        start_time = self.MANUAL_START_TIME_BIOPAC.get(participant, {}).get(condition, None)
        if start_time is not None:
            start_time = pd.to_datetime(start_time).tz_localize("Europe/Berlin")
        data, fs = self._get_biopac_data(participant, condition, phase, start_time=start_time)

        if self.only_labeled:
            biopac_data_dict = {}
            labeling_borders = self.labeling_borders

            if self.is_single(None):
                biopac_data_dict = self._cut_to_labeling_borders(data, labeling_borders)
            else:
                for phase in self.PHASES_FINE if self.phase_fine else self.PHASES_COARSE:
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x, ph=phase: ph in x)]
                    biopac_data_dict[phase] = self._cut_to_labeling_borders(data, borders)
            return biopac_data_dict

        return data

    @staticmethod
    def _cut_to_labeling_borders(data: pd.DataFrame, labeling_borders: pd.DataFrame) -> pd.DataFrame:
        start_index = labeling_borders["sample_relative"].iloc[0]
        end_index = labeling_borders["sample_relative"].iloc[-1]
        return data.iloc[start_index:end_index]

    @property
    def icg(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "ICG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        return self.biopac[["icg_der"]]

    @property
    def icg_clean(self) -> pd.DataFrame:
        algo = IcgPreprocessingBandpass()
        algo.clean(icg=self.icg, sampling_rate_hz=self.sampling_rate_icg)
        return algo.icg_clean_

    @property
    def ecg(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "ECG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        return self.biopac[["ecg"]]

    @property
    def ecg_clean(self) -> pd.DataFrame:
        algo = EcgPreprocessingNeurokit()
        algo.clean(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        return algo.ecg_clean_

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(None):
            participant = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]
            return self._get_timelog(participant, condition, phase)

        if self.is_single(["participant", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Timelog can only be accessed for all phases or one specific phase!")

            participant = self.index["participant"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant, condition, "all")

        raise ValueError("Timelog can only be accessed for a single participant and a single condition at once!")

    @property
    def labeling_borders(self) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]

        if not self.is_single("participant"):
            raise ValueError("Labeling borders can only be accessed for a single participant.")

        file_path = self.base_path.joinpath(
            f"data_per_subject/{participant}/{condition}/biopac/reference_labels/{participant}_{condition}_labeling_borders.csv"
        )
        data = load_labeling_borders(file_path)

        if self.is_single(None):
            phase = self.index["phase"][0]
            data = data[data["description"].apply(lambda x, ph=phase: ph in x)]

        return data

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        return self._load_reference_heartbeats()

    @property
    def reference_labels_ecg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return self._load_reference_labels("ICG")

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]
        phases = self.index["phase"]

        if not (
            self.is_single(None) or len(phases) == len(self.PHASES_FINE if self.phase_fine else self.PHASES_COARSE)
        ):
            raise ValueError(
                "Reference data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )

        reference_data_dict = {}
        for phase in phases:
            file_path = self.base_path.joinpath(
                f"data_per_subject/{participant}/{condition}/biopac/reference_labels/"
                f"{participant}_{condition}_reference_labels_{phase.lower()}_{channel.lower()}.csv"
            )
            reference_data = pd.read_csv(file_path)
            reference_data = reference_data.set_index(["heartbeat_id", "channel", "label"])

            start_idx = self.get_subset(phase=phase).labeling_borders.iloc[0]
            reference_data = reference_data.assign(
                sample_relative=reference_data["sample_absolute"] - start_idx["sample_absolute"]
            )

            reference_data_dict[phase] = reference_data

        if self.is_single(None):
            return reference_data_dict[phases[0]]
        return pd.concat(reference_data_dict, names=["phase"])

    @property
    def reference_pep(self) -> pd.DataFrame:
        return compute_reference_pep(self)

    @property
    def heartbeats(self) -> pd.DataFrame:
        heartbeat_algo = HeartbeatSegmentationNeurokit(variable_length=True)
        ecg_clean = self.ecg_clean
        heartbeat_algo.extract(ecg=ecg_clean, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

    @property
    def id_mapping(self) -> pd.DataFrame:
        return load_long_format_csv(self.data_tabular_path.joinpath("extras/processed/id_mapping.csv"))

    @property
    def subset_demographics(self):
        # TODO
        return NotImplementedError

    @property
    def cortisol(self) -> pd.DataFrame:
        cortisol_path = self.data_tabular_path.joinpath("saliva/cortisol/cleaned/cortisol_cleaned.csv")
        return load_long_format_csv(cortisol_path)

    @property
    def amylase(self) -> pd.DataFrame:
        amylase_path = self.data_tabular_path.joinpath("saliva/amylase/cleaned/amylase.csv")
        return load_long_format_csv(amylase_path)

    @property
    def amylase_features(self) -> pd.DataFrame:
        amylase_features_path = self.data_tabular_path.joinpath("saliva/amylase/processed/amylase_features.csv")
        return load_long_format_csv(amylase_features_path, index_cols=["subject", "condition"])

    @property
    def gender(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("extras/processed/gender.csv"),
            index_cols=["subject"],
        )

    @property
    def condition_order(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("extras/processed/condition_order.csv"),
            index_cols=["subject"],
        )

    @property
    def sit_stand(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.base_path.joinpath("extras/processed/sit_stand.csv"),
            index_cols=["subject"],
        )

    @property
    def condition_day_mapping(self) -> pd.DataFrame:
        return load_long_format_csv(
            self.data_tabular_path.joinpath("extras/processed/day_condition_mapping.csv"),
            index_cols=["subject", "day"],
        )

    @property
    def timelog_video(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Timelog can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            return self._get_timelog_video(participant_id, condition)

        # TODO allow for multiple participants and conditions in the future (return as concatenated dataframe)
        raise ValueError("Timelog can only be accessed for a single participant and a single condition at once!")

    @property
    def emrad(self) -> pd.DataFrame:
        if self.is_single(None):
            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]

            data, fs = self._get_radar_data(participant_id, condition, phase)
            return data

        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Radar data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data, fs = self._get_radar_data(participant_id, condition, "all")
            return data

        raise ValueError("Radar data can only be accessed for a single participant and a single condition at once!")

    @cached_property
    def nilspod(self) -> pd.DataFrame:
        if self.is_single(None):
            # participant_id = self.index["subject"][0]
            # condition = self.index["condition"][0]
            # phase = self.index["phase"][0]
            # TODO: load nilspod data for phase
            raise NotImplementedError("Loading nilspod data for a specific phase is not implemented yet!")
        if self.is_single(["subject", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("NilsPod data can only be accessed for all phases or one specific phase!")

            participant_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            # load nilspod data for all phases
            data = self._get_nilspod_data(participant_id, condition)
            return data

        raise ValueError("NilsPod data can only be accessed for a single participant in a single condition!")

    @property
    def face_video_path(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/face/raw/video_face_{participant_id.lower()}_{condition}.mp4")
        return path

    @property
    def body_video_path(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/raw/video_body_{participant_id.lower()}_{condition}.mp4")
        return path

    @property
    def diarization(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/diarization_{participant_id.lower()}_{condition}.csv")
        return path

    @property
    def trim_to_talk(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(
            f"video/body/processed/trimmed_talk_section_audio_{participant_id.lower()}_{condition}.wav"
        )
        return path

    @property
    def trim_with_sp_dia(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/trimmed_sp_dia_audio_{participant_id.lower()}_{condition}.wav")
        return path

    @property
    def transcript(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/transcript_{participant_id.lower()}_{condition}.txt")
        return path

    @property
    def speech_features(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/speech_features{participant_id.lower()}_{condition}.csv")
        return path

    @property
    def speech_sentiment(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/speech_sentiment{participant_id.lower()}_{condition}.csv")
        return path

    @property
    def word_count(self) -> Path:
        if not self.is_single(["subject", "condition"]):
            raise ValueError("Video can only be accessed for a single participant in a single condition!")
        if self.is_single("phase"):
            raise ValueError("Video can only be accessed for all phases!")

        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = _build_data_path(self.base_path, participant_id, condition)
        path = path.joinpath(f"video/body/processed/word_count{participant_id.lower()}_{condition}.csv")
        return path

    @property
    def timelog_path(self) -> Path:
        participant_id = self.index["subject"][0]
        condition = self.index["condition"][0]
        path = self.base_path.joinpath(f"data_per_subject/{participant_id}/{condition}")
        path = path.joinpath(f"timelog/cleaned/{participant_id}_{condition}_processed_phases_timelog.csv")
        return path

    @property
    def expected_files_list(self) -> pd.DataFrame:
        file_path = self.base_path / "expected_files_per_subject.csv"
        return pd.read_csv(file_path, index_col=2)

    def check_data_completeness(self, save_output: bool = False, output_path: path_t | None = None):
        """Check if all expected files are present for each subject."""
        file_overview = check_data_completeness(
            data_per_subject_folder=self.base_path.joinpath("data_per_subject"),
            expected_files_list=self.expected_files_list,
        )
        if save_output:
            if output_path is None:
                output_path = self.base_path / "data_completeness_check_result.csv"
            file_overview.to_csv(output_path)
        return file_overview

    def _get_biopac_data(
        self, participant_id: str, condition: str, phase: str, start_time: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition, start_time)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition, start_time)

        if phase == "all":
            return data, fs
        # cut biopac data to specified phase
        timelog = self.timelog
        phase_start = timelog[phase]["start"].iloc[0]
        phase_end = timelog[phase]["end"].iloc[0]
        data = data.loc[phase_start:phase_end]
        return data, fs

    def _get_nilspod_data(self, subject_id: str, condition: str) -> pd.DataFrame:
        if self.use_cache:
            data, fs = _cached_load_nilspod_data(self.base_path, subject_id, condition)
        else:
            data, fs = _load_nilspod_session(self.base_path, subject_id, condition)
        return data

    def _get_radar_data(self, participant_id: str, condition: str, phase: str) -> tuple[DataFrame, float]:
        if self.use_cache:
            data, fs = _cached_get_radar_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_radar_data(self.base_path, participant_id, condition)

        if phase == "all":
            timelog = self.timelog
            phase_start = timelog["Pause_1"]["start"][0]
            phase_end = timelog["Pause_5"]["end"][0]
            data = data.loc[phase_start:phase_end]
            return data, fs
        # cut radar data to specified phase
        timelog = self.timelog
        phase_start = timelog[phase]["start"][0]
        phase_end = timelog[phase]["end"][0]
        data = data.loc[phase_start:phase_end]
        return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        timelog = _load_timelog(self.base_path, participant_id, condition, phase, self.phase_fine)
        return self._validate_timelog_timestamps(timelog)

    def _get_timelog_video(self, participant_id: str, condition: str) -> pd.DataFrame:
        return _load_timelog_video(self.base_path, participant_id, condition)

    def _all_phases_selected(self) -> bool:
        # check if all phases are selected
        all_phases_fine = self.phase_fine and (len(self.index["phase"]) == len(self.PHASES_FINE))
        all_phases_coarse = not self.phase_fine and (len(self.index["phase"]) == len(self.PHASES_COARSE))
        return all_phases_fine or all_phases_coarse

    def _validate_timelog_timestamps(self, timelog):
        timelog = timelog.stack().T.reset_index()
        # Define the order of the phases
        phase_order = self.PHASES_FINE
        # Convert 'phase' to a categorical type with the defined order
        timelog["phase"] = pd.Categorical(timelog["phase"], categories=phase_order, ordered=True)
        # Sort the DataFrame by the 'phase' column
        timelog = timelog.sort_values("phase")

        # Shift the 'start' column to get the 'next_start' value for each phase
        timelog[("time", "next_start")] = timelog[("time", "start")].shift(-1)

        # Where the 'end' of the current phase is greater than the 'next_start', adjust the 'end'
        timelog.loc[timelog[("time", "end")] > timelog[("time", "next_start")], ("time", "end")] = timelog[
            ("time", "next_start")
        ] - pd.Timedelta(seconds=3)
        timelog = timelog.set_index("phase")
        timelog = timelog[[("time", "start"), ("time", "end")]].stack().T

        return timelog

    # def _get_opendbm_facial_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_facial_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #         )
    #     return load_opendbm_facial_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_acoustic_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_acoustic_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_audio, self.opendbm_suffix
    #         )
    #     return load_opendbm_acoustic_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_audio, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_movement_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_movement_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #         )
    #     return load_opendbm_movement_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_acoustic_seg_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_acoustic_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #     return load_opendbm_acoustic_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_opendbm_audio_seg_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_audio_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #     return load_opendbm_audio_seg_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_opendbm_facial_tremor_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_opendbm_facial_tremor_data(
    #             self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #         )
    #     return load_opendbm_facial_tremor_data(
    #         self.base_path, subject_id, condition, self.sampling_rate_video, self.opendbm_suffix
    #     )
    #
    # def _get_opendbm_pitch_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     return get_opendbm_pitch_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_opendbm_eyeblink_data(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     return get_opendbm_eyeblink_data(self.base_path, subject_id, condition, self.opendbm_suffix)
    #
    # def _get_speaker_diarization(self, subject_id: str, condition: str) -> pd.DataFrame:
    #     if self.use_cache:
    #         return _cached_load_speaker_diarization(self.base_path, subject_id, condition)
    #     return load_speaker_diarization(self.base_path, subject_id, condition)

    # @cached_property
    # def opendbm_facial_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_facial_data(subject_id, condition)
    #
    #         # TODO add support for phases
    #         if self.is_single(None):
    #            phase = self.index["phase"].unique()[0]
    #         else:
    #            phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #         data = data.loc[times[0] : times[1]]
    #
    #         if self.normalize_video_time:
    #             data.index -= data.index[0]
    #
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    @property
    def opendbm_derived_features(self) -> pd.DataFrame:
        if self.is_single(["subject", "condition"]):
            subject_id = self.index["subject"][0]
            condition = self.index["condition"][0]

            data = get_opendbm_derived_features(self.base_path, subject_id, condition, self.opendbm_suffix)
            return data

        raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

    # @cached_property
    # def opendbm_acoustic_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_acoustic_data(subject_id, condition)
    #         data = apply_diarization_aco(data, self.speaker_diarization, self.sampling_rate_audio)
    #
    #         if self.is_single(None):
    #             phase = self.index["phase"].unique()[0]
    #         else:
    #             phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #         data = data.loc[times[0] : times[1]]
    #
    #         if self.normalize_video_time:
    #             data.index -= data.index[0]
    #
    #         data = data.loc[data["diarization"], :]
    #         data = data.drop(["diarization"], axis=1)
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_movement_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_movement_data(subject_id, condition)
    #
    #         if self.is_single(None):
    #             phase = self.index["phase"].unique()[0]
    #         else:
    #             phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #         data = data.loc[times[0] : times[1]]
    #
    #         if self.normalize_video_time:
    #             data.index -= data.index[0]
    #
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_acoustic_seg_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         try:
    #             data = self._get_opendbm_acoustic_seg_data(subject_id, condition)
    #             data = apply_diarization_aco_seg(data, self.speaker_diarization, self.sampling_rate_audio)
    #
    #             if self.is_single(None):
    #                 phase = self.index["phase"].unique()[0]
    #             else:
    #                 phase = "total"
    #
    #             times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #             data = data.loc[(data["start_time"] >= times[0]) & (data["end_time"] <= times[1])]
    #
    #             if self.normalize_video_time:
    #                 data.index -= data.index[0]
    #         except Exception as e:
    #             print(f"loading acoustic_seg opendbm data failed for phase {phase}: {e}")
    #
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_audio_seg_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         try:
    #             data = self._get_opendbm_audio_seg_data(subject_id, condition)
    #
    #             if self.is_single(None):
    #                 phase = self.index["phase"].unique()[0]
    #             else:
    #                 phase = "total"
    #
    #             times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #             data = data.loc[(data["start"] > times[0]) & (data["stop"] < times[1])]
    #             if self.normalize_video_time:
    #                 data.index -= data.index[0]
    #         except Exception as e:
    #             print(f"loading audio_seg opendbm data failed for phase {phase}: {e}")
    #
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def opendbm_facial_tremor_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_facial_tremor_data(subject_id, condition)
    #
    #         if self.is_single(None):
    #             phase = self.index["phase"].unique()[0]
    #         else:
    #             phase = "total"
    #
    #         times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #
    #         try:
    #             data = data.loc[times[0] : times[1]]
    #             if self.normalize_video_time:
    #                 data.index -= data.index[0]
    #         except Exception as e:
    #             print(f"warning: cutting facial tremor data failed (data shape: {data.shape}):", e)
    #
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @property
    # def opendbm_pitch_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         return self._get_opendbm_pitch_data(subject_id, condition)
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @property
    # def opendbm_eyeblink_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         return self._get_opendbm_eyeblink_data(subject_id, condition)
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @property
    # def opendbm_eyeblink_ear_data(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_opendbm_eyeblink_data(subject_id, condition)
    #         if len(data.index) != 0:
    #             fps = data.at[0, "fps"]
    #             data.index = data["mov_blinkframes"] / fps
    #             data.index.name = "time [s]"
    #             if self.is_single(None):
    #                 phase = self.index["phase"].unique()[0]
    #             else:
    #                 phase = "total"
    #             times = get_times_for_video(self.base_path, subject_id, condition, phase)
    #             data = data.loc[times[0] : times[1]]
    #
    #         data = data.drop(["vid_dur", "fps", "mov_blinkframes", "mov_blinkdur", "dbm_master_url"], axis=1)
    #         return data
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
    #
    # @cached_property
    # def speaker_diarization(self) -> pd.DataFrame:
    #     if self.is_single(None) or self.is_single(["subject", "condition"]):
    #         subject_id = self.index["subject"][0]
    #         condition = self.index["condition"][0]
    #         data = self._get_speaker_diarization(subject_id, condition)
    #         return data
    #
    #     raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")
