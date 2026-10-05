from collections.abc import Sequence
from itertools import product
from typing import ClassVar

import pandas as pd
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.sensors.zebris._zebris import ZebrisDataset
from empkins_io.utils._types import path_t
from empkins_io.datasets.d03.macro_ap01.helper import _get_times_for_mocap
from biopsykit.io import load_atimelogger_file
from empkins_io.sensors.motion_capture.motion_capture_formats import mvnx
from functools import cached_property, lru_cache
from empkins_io.datasets.d03.macro_ap03.helper import _load_mocap_data

# _cached_load_nilspod_data = lru_cache(maxsize=4)(_load_nilspod_session)
_cached_load_mocap_data = lru_cache(maxsize=4)(_load_mocap_data)

__all__ = ["MacroBaseDataset"]

from empkins_io.utils.exceptions import ZebrisDataNotFoundError, TimelogNotFoundError


class MacroBaseDataset(Dataset):
    base_path: path_t
    use_cache: bool

    use_phases: bool
    verbose: bool
    include_prep: bool

    _sample_times_saliva: tuple[int] = (-40, -1, 15, 25, 35, 45, 60, 75)
    _sample_times_bloodspot: tuple[int] = (-40, 60)

    SAMPLING_RATE_MOCAP = 60

    SAMPLING_RATE_ZEBRIS = 60

    CONDITIONS: ClassVar[Sequence[str]] = ["ftsst", "tsst"]

    PHASES: ClassVar[Sequence[str]] = ["prep", "talk", "math"]

    SUBSET_IRREGULAR_DATA = (
        ("VP_002", "tsst"),
        ("VP_015", "ftsst"),
        ("VP_020", "ftsst"),
        ("VP_021", "tsst"),
        ("VP_022", "tsst"),
        ("VP_102", "ftsst"),

    )

    SUBSET_MISSING_CONDITIONS = (
        ("VP_037", "tsst"),
        ("VP_042", "tsst"),
        ("VP_080", "tsst"),
        ("VP_094", "ftsst"),
        ("VP_097", "tsst"),
        ("VP_130", "ftsst"),
        ("VP_133", "ftsst"),
    )

    SUBSETS_WITHOUT_MOCAP = (
        ("VP_003", "ftsst"),
        #("VP_004", "tsst"),   #TEST1 and TEST2
        ("VP_005", "ftsst"),
        ("VP_010", "tsst"),
        #("VP_013", "tsst"),     #TEST1 and TEST2
        ("VP_016", "ftsst"),
        ("VP_019", "ftsst"), #very short recording
        ("VP_023", "tsst"),
        ("VP_025", "ftsst"),
        ("VP_026", "tsst"), #very short recording
        ("VP_027", "ftsst"),
        ("VP_031", "ftsst"),
        ("VP_033", "ftsst"),
        ("VP_042", "tsst"), #very short recording
        ("VP_042", "ftsst"),
        ("VP_052", "ftsst"), #very short recording
        ("VP_062", "tsst"),
        ("VP_063", "ftsst"),
        ("VP_063", "tsst"),
        ("VP_065", "tsst"), #very short recording
        ("VP_071", "ftsst"),
        ("VP_075", "ftsst"), #very short recording
        ("VP_080", "tsst"),
        ("VP_089", "tsst"), #very short recording
        ("VP_094", "ftsst"),
        ("VP_094", "tsst"),
        ("VP_095", "tsst"),
        ("VP_103", "ftsst"), #maybe just missed while exporting
        ("VP_104", "ftsst"),#maybe just missed while exporting
        ("VP_105", "tsst"),#maybe just missed while exporting
        ("VP_105", "ftsst"),#maybe just missed while exporting
        ("VP_112", "tsst"),#maybe just missed while exporting
        ("VP_114", "ftsst"),#maybe just missed while exporting
        ("VP_115", "ftsst"),#maybe just missed while exporting
        ("VP_122", "ftsst"),#maybe just missed while exporting
        ("VP_134", "tsst"),#maybe just missed while exporting
        ("VP_144", "tsst"),#maybe just missed while exporting
        ("VP_146", "tsst"),#maybe just missed while exporting
        ("VP_146", "ftsst"),#maybe just missed while exporting
        ("VP_147", "tsst"),#maybe just missed while exporting
        ("VP_150", "tsst"),#maybe just missed while exporting
        ("VP_152", "tsst"),#maybe just missed while exporting
        ("VP_153", "ftsst"),#maybe just missed while exporting
        ("VP_154", "ftsst"),#maybe just missed while exporting
    )


    SUBSETS_WITHOUT_ZEBRIS = (
        ("VP_001", "ftsst", "math"),  # short recording 190.10 seconds
        ("VP_015", "ftsst", "math"),  # much shorter # TODO: did not find in short recording check
        ("VP_016", "tsst", "talk"),  # no data
        ("VP_016", "tsst", "math"),  # no data
        ("VP_020", "ftsst", "talk"),  # no data
        ("VP_020", "ftsst", "math"),  # no data
        ("VP_023", "ftsst", "math"),  # no data
        ("VP_024", "tsst", "talk"),  # no data
        ("VP_024", "tsst", "math"),  # no data
        ("VP_027", "tsst", "math"),  # short recording of 0.00 seconds
        ("VP_032", "tsst", "talk"),  # no data
        ("VP_036", "tsst", "talk"),  # no data
        ("VP_037", "tsst", "math"),  # short recording 63.87 seconds
        ("VP_038", "ftsst", "math"),  # short recording of 2.35 seconds
        ("VP_039", "tsst", "talk"),  # no data
        ("VP_039", "tsst", "math"),  # no data
        ("VP_039", "ftsst", "talk"),  # no data
        ("VP_039", "ftsst", "math"),  # no data
        ("VP_041", "ftsst", "talk"),  # short recording of 3.28 seconds
        ("VP_042", "tsst", "talk"),  # no data
        ("VP_042", "tsst", "math"),  # no data
        ("VP_043", "tsst", "math"),  # short recording of 195.60 seconds.
        ("VP_044", "ftsst", "talk"),  # no data
        ("VP_045", "tsst", "math"),  # short recording of 3.17 seconds
        ("VP_051", "tsst", "talk"),  # short recording of 160.72 seconds
        ("VP_059", "tsst", "talk"),  # no data
        ("VP_069", "ftsst", "talk"),  # no data
        ("VP_069", "ftsst", "math"),  # no data
        ("VP_069", "tsst", "talk"),  # no data
        ("VP_069", "tsst", "math"),  # no data
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        use_phases: bool = False,
        include_prep: bool = False,
        exclude_complete_subjects_if_error: bool = True,
        exclude_irregular_data = True,
        exclude_without_mocap: bool = True,
        exclude_without_zebris: bool = True,
        exclude_missing_conditions: bool = True,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.data_path = base_path.joinpath(f"macro2_data")
        self.use_phases = use_phases
        self.include_prep = include_prep
        self.exclude_complete_subjects_if_error = exclude_complete_subjects_if_error
        self.exclude_irregular_data = exclude_irregular_data
        self.exclude_without_mocap = exclude_without_mocap
        self.exclude_without_zebris = exclude_without_zebris
        self.exclude_missing_conditions = exclude_missing_conditions

        self.data_to_exclude = self._find_data_to_exclude(exclude_complete_subjects_if_error)
        self.use_cache = use_cache
        self.verbose = verbose
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
        overview_dataset = pd.read_csv(self.base_path.joinpath("overview_dataset.csv"))

        subject_ids = overview_dataset["participant"].tolist()
        if self.use_phases:
            index_cols = ["participant", "condition", "phase"]
            phases = self.PHASES if self.include_prep else self.PHASES[1:]
            index = list(product(subject_ids, self.CONDITIONS, phases))
        else:
            index_cols = ["participant", "condition"]
            index = list(product(subject_ids,self.CONDITIONS))

        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()

        return index
    """def create_index(self):
        subject_ids = [
            subject_dir.name
            for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_participant"), "VP_*")
        ]
        index_cols = ["participant", "condition", "phase"]
        phases = self.PHASES if self.include_prep else self.PHASES[1:]
        index = list(product(subject_ids, self.CONDITIONS, phases))

        index = pd.DataFrame(index, columns=index_cols)
        index = index.set_index(index_cols)
        index = index.drop(index=self.data_to_exclude).reset_index()

        return index"""

    def _find_data_to_exclude(self, exclude_complete_subjects_if_error: bool):
        data_to_exclude = []
        if self.exclude_without_mocap:
            data_to_exclude += self.SUBSETS_WITHOUT_MOCAP
        if self.exclude_without_zebris:
            data_to_exclude += self.SUBSETS_WITHOUT_ZEBRIS
        if self.exclude_missing_conditions:
            data_to_exclude += self.SUBSET_MISSING_CONDITIONS
        if self.exclude_irregular_data:
            data_to_exclude += self.SUBSET_IRREGULAR_DATA
        if exclude_complete_subjects_if_error:
            data_to_exclude = [x[0] for x in data_to_exclude]

        return data_to_exclude

    @property
    def participant(self) -> str:
        if not self.is_single("participant"):
            raise ValueError("Subject data can only be accessed for a single participant!")
        return self.index["participant"][0]

    @property
    def condition(self) -> str:
        if not self.is_single("condition"):
            raise ValueError("Condition data can only be accessed for a single condition!")
        return self.index["condition"][0]

    @property
    def phase(self) -> str:
        if not self.is_single("phase"):
            raise ValueError("Phase data can only be accessed for a single phase!")
        return self.index["phase"][0]

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the MoCap system."""
        return self.SAMPLING_RATE_MOCAP

    @property
    def sample_times_saliva(self) -> Sequence[int]:
        return self._sample_times_saliva

    @property
    def sample_times_bloodspot(self) -> Sequence[int]:
        return self._sample_times_bloodspot


    @property
    def participant_information(self) -> pd.DataFrame:
        overview = pd.read_csv(self.base_path.joinpath("overview_dataset.csv"))
        overview_participant = overview[overview["participant"].isin(self.index["participant"])]
        return overview_participant

    @property
    def gender(self) -> str:
        overview = self.participant_information
        gender = overview["gender"]
        gender = gender.replace({
            "m": "male",
            "w": "female"
        })
        return gender.tolist()


    @property
    def condition_order(self) -> pd.DataFrame:
        overview = self.participant_information
        data = overview.set_index("participant")[["condition_order"]]
        participant_ids = self.index["participant"].unique()
        return data.loc[participant_ids]
        """print(overview)
        #return overview["condition_order"].tolist()
        data = pd.DataFrame(overview["participant", "condition_order"])
        #data = pd.read_csv(self.base_path.joinpath("_extras/condition_order.csv"))
        data = data.set_index("participant")[["condition_order"]]
        subject_ids = self.index["participant"].unique()
        return data.loc[subject_ids]"""

    @property
    def language(self) -> str:
        overview = self.participant_information
        return overview["language"].tolist()

    @property
    def panel(self) -> str:
        overview = self.participant_information
        return overview["panel"].tolist()

    @property
    def date(self) -> pd.DataFrame:
        overview = self.participant_information
        return overview["date"].tolist()

    @property
    def time(self) -> pd.DataFrame:
        overview = self.participant_information
        return overview["time"].tolist()
    @property
    def timelog(self) -> pd.DataFrame:
        participant = self.participant
        condition = self.condition
        data_path = self.data_path.joinpath(f"timelogs/cleaned")
        file_path = data_path.joinpath(f"{participant.lower()}_{condition}_timelog.csv")
        if not file_path.exists():
            raise TimelogNotFoundError(
                f"No time log data was found for {condition} condition of {participant}!"
            )
        timelog = pd.read_csv(file_path)

        timelog = load_atimelogger_file(file_path, timezone="Europe/Berlin")
        #timelog = load_atimelogger_file(file_path)
        # convert all column names of the multi-level column index to lower case
        timelog.columns = timelog.columns.set_levels([level.str.lower() for level in timelog.columns.levels])

        if self.use_phases:
            phase = self.index["phase"][0] if self.is_single(None) else list(self.index["phase"])
            timelog = timelog[phase]
        return timelog


    @cached_property
    def get_mocap_data(self) -> pd.DataFrame:
        if not self.is_single(["participant", "condition"]):
            raise ValueError("Data can only be accessed for a single recording of a single participant in the subset")

        participant = self.index["participant"][0]
        condition = self.index["condition"][0]
        data, start = self._get_mocap_data(participant, condition, verbose=self.verbose)
        t = data.index.tolist()[-1]
        if self.use_phases and self.is_single(None):
            timelog = self.timelog
            phase = self.index["phase"][0]
            timelog.columns = pd.MultiIndex.from_product([[phase], timelog.columns])
            times = _get_times_for_mocap(timelog, start, phase)
            times = times.loc[phase]


        else:
            times = _get_times_for_mocap(self.timelog, start, phase="total")
            times = times.loc["total"]

        if t < times["end"]:
            print("Mocap recording shorter than phase")
        data_total = data.loc[times["start"] : times["end"]]

        return data_total

    def _get_mocap_data(self, participant: str, condition: str, *, verbose: bool = True) -> pd.DataFrame:
        if self.use_cache:
            return _cached_load_mocap_data(self.data_path, participant, condition, verbose=verbose)
        return _load_mocap_data(self.data_path, participant, condition, verbose=verbose)

    @property
    def zebris(self) -> pd.DataFrame | None:
        if not self.is_single(None):
            raise ValueError("Data can only be accessed for a single recording (participant, condition, phase).")
        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase

        folder_path = self.data_path.joinpath("zebris", "processed", p_id, condition, phase)
        try:
            zebris_dataset = ZebrisDataset.from_folder(folder_path)
            return zebris_dataset.data_as_df()
        except FileNotFoundError as e:
            raise ZebrisDataNotFoundError(
                f"No Zebris data found for participant {p_id}, condition {condition}, phase {phase}."
            ) from e

    """def zebris(self) -> pd.DataFrame | None:
        if not self.is_single(None):
            raise ValueError("Data can only be accessed for a single recording (participant, condition, phase).")
        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase

        folder_path = self.base_path.joinpath("data_per_participant", p_id, condition, "zebris", "export", phase)
        try:
            # TODO: cut the data according to timelogs once they are available
            zebris_dataset = ZebrisDataset.from_folder(folder_path)
            return zebris_dataset.data_as_df()
        except FileNotFoundError as e:
            raise ZebrisDataNotFoundError(
                f"No Zebris data found for participant {p_id}, condition {condition}, phase {phase}."
            ) from e"""

    @property
    def zebris_cut(self):
        """Return the Zebris data cut to 300 seconds (center of the recording).

        Returns
        -------
            :class:`pd.DataFrame`
                The cut Zebris data with time index in seconds.
        """
        data = self.zebris
        duration = data.index[-1]
        max_duration = min([300, duration])
        slice_start = 0.5 * (duration - max_duration)
        slice_end = slice_start + max_duration
        data = data.loc[slice_start:slice_end].reset_index(drop=True)
        data.index /= self.SAMPLING_RATE_ZEBRIS
        return data

    @property
    def zebris_aggregated(self) -> pd.DataFrame | None:
        if not self.is_single(None):
            raise ValueError(
                "Zebris aggregated data can only be accessed for a single recording (participant, condition, phase)."
            )

        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase
        folder_path = self.data_path.joinpath("zebris", "processed", p_id, condition, phase)
        try:
            zebris_dataset = ZebrisDataset.from_folder(folder_path)
            return zebris_dataset.aggregated_data
        except FileNotFoundError as e:
            raise ZebrisDataNotFoundError(
                f"No aggregated Zebris data found for participant {p_id}, condition {condition}, phase {phase}."
            ) from e

    """def zebris_aggregated(self) -> pd.DataFrame | None:
        if not self.is_single(None):
            raise ValueError(
                "Zebris aggregated data can only be accessed for a single recording (participant, condition, phase)."
            )

        p_id = self.group_label.participant
        condition = self.group_label.condition
        phase = self.group_label.phase
        folder_path = self.base_path.joinpath("data_per_participant", p_id, condition, "zebris", "export", phase)
        try:
            zebris_dataset = ZebrisDataset.from_folder(folder_path)
            return zebris_dataset.aggregated_data
        except FileNotFoundError as e:
            raise ZebrisDataNotFoundError(
                f"No aggregated Zebris data found for participant {p_id}, condition {condition}, phase {phase}."
            ) from e
"""