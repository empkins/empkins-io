# Note: this script briefly demonstrates the postprocessing steps to synchronize data recorded with the OpenSyncBoard.
# As exemplary recording devices, two different sensors measuring ECG data are used: Biopac and NilsPod.
# For further visualization and more detailed explanations,
# have a look at the corresponding jupyter notebook `synchronization_demo.ipynb`.


# Setup and Helper Functions

from pathlib import Path
import pandas as pd

from biopsykit.io.biopac import BiopacDataset
from empkins_io.sync import SyncedDataset

data_path = Path("data/sync_demo")
nilspod_sync = "9e02"
nilspod_ecg = "b0c2"


# Helper to load data

from typing import Optional, Tuple

from nilspodlib import Dataset, SyncedSession
from packaging.version import Version


class CustomSyncedSession(SyncedSession):
    """Custom SyncedSession to handle different firmware versions of the SyncPod.

    The SyncPod was using a custom dev version of the NilsPod firmware and needs to be loaded differently.
    """

    @classmethod
    def from_file_paths(cls, paths, legacy_support="resolve", force_version=None, tz="Europe/Berlin"):
        ds = []
        for p in paths:
            if "NilsPodX-9E02" in p.name:
                dataset = Dataset.from_bin_file(
                    p, legacy_support=legacy_support, force_version=Version("0.17.0"), tz=tz
                )
            else:
                dataset = Dataset.from_bin_file(p, legacy_support=legacy_support, force_version=force_version, tz=tz)

            ds.append(dataset)

        return cls(ds)

    @classmethod
    def from_folder_path(
        cls,
        base_path,
        filter_pattern: Optional[str] = "NilsPodX-*.bin",
        legacy_support: Optional[str] = "resolve",
        force_version=None,
        tz: Optional[str] = "Europe/Berlin",
    ):
        return cls.from_file_paths(
            sorted(base_path.glob(filter_pattern)), legacy_support=legacy_support, force_version=force_version, tz=tz
        )


def load_nilspod_data(type: str) -> Tuple[pd.Series, pd.Series]:
    """Load and return synchronization channel and ECG data from NilsPod."""
    nilspod_path = data_path.joinpath(type)
    session = CustomSyncedSession.from_folder_path(nilspod_path)
    session = session.align_to_syncregion()
    fs = session.ecg[1].sampling_rate_hz
    data = session.data_as_df(index="local_datetime", concat_df=True)
    # artifact in recording, TODO: rerecord data
    if type == "falling-trigger":
        data = data.loc[:"2023-11-20 11:07:18.8+01:00"]
    # constant filter delay in ECG sensor
    delay = 0.11261
    delay_periods = int(-delay / (1 / fs))
    # see https://www.analog.com/media/en/technical-documentation/data-sheets/max30003.pdf, p. 19 for details
    ecg_data = data[nilspod_ecg].shift(periods=delay_periods)["ecg"]
    data = pd.concat([data[nilspod_sync]["analog_1"], ecg_data], axis=1)
    return data, fs


def load_biopac_data(type: str) -> Tuple[pd.Series, float]:
    """Load and return synchronization channel and ECG data from Biopac."""
    biopac_path = data_path.joinpath(type).joinpath(f"Sync_Board_Demo_{type}.acq")
    dataset = BiopacDataset.from_acq_file(biopac_path)
    data = dataset.data_as_df(index="local_datetime")
    if type == "falling-trigger":
        data = data.loc[:"2023-11-20 11:07:29+01:00"]
    return data, dataset.sampling_rate_hz_sync


# Demo Trigger Synchronization

np_data, np_fs = load_nilspod_data("falling-trigger")
bp_data, bp_fs = load_biopac_data("falling-trigger")

synced_dataset = SyncedDataset(sync_type="falling-trigger")
synced_dataset.add_dataset("nilspod", np_data, sync_channel_name="analog_1", sampling_rate=np_fs)
synced_dataset.add_dataset("biopac", bp_data, sync_channel_name="sync", sampling_rate=bp_fs)
synced_dataset.cut_to_sync_region()
synced_dataset.align_datasets("biopac", cut_to_shortest=True)

# final synchronized output
print(synced_dataset.nilspod_aligned_)
print(synced_dataset.biopac_aligned_)


# Demo Edge Synchronization

np_data, np_fs = load_nilspod_data("falling-edge")
bp_data, bp_fs = load_biopac_data("falling-edge")
synced_dataset = SyncedDataset(sync_type="falling-edge")
synced_dataset.add_dataset("nilspod", np_data, sync_channel_name="analog_1", sampling_rate=np_fs)
synced_dataset.add_dataset("biopac", bp_data, sync_channel_name="sync", sampling_rate=bp_fs)
synced_dataset.cut_to_sync_region()
synced_dataset.align_datasets("biopac")

# final synchronized output
print(synced_dataset.nilspod_aligned_)
print(synced_dataset.biopac_aligned_)


# Demo Clock Synchronization

np_data, np_fs = load_nilspod_data("falling-clock")
bp_data, bp_fs = load_biopac_data("falling-clock")

synced_dataset = SyncedDataset(sync_type="falling-clock")
synced_dataset.add_dataset("nilspod", np_data, sync_channel_name="analog_1", sampling_rate=np_fs)
synced_dataset.add_dataset("biopac", bp_data, sync_channel_name="sync", sampling_rate=bp_fs)
synced_dataset.cut_to_sync_region(sync_params={"wave_frequency": 50})
synced_dataset.align_datasets("biopac", cut_to_shortest=False)

# final synchronized output
print(synced_dataset.nilspod_aligned_)
print(synced_dataset.biopac_aligned_)


# Demo M-Sequence Synchronization

np_data, np_fs = load_nilspod_data("m-sequence")
bp_data, bp_fs = load_biopac_data("m-sequence")

synced_dataset = SyncedDataset(sync_type="m-sequence")
synced_dataset.add_dataset("nilspod", np_data, sync_channel_name="analog_1", sampling_rate=np_fs)
synced_dataset.add_dataset("biopac", bp_data, sync_channel_name="sync", sampling_rate=bp_fs)
synced_dataset.resample_datasets(np_fs, method="static", wave_frequency=50)
synced_dataset.align_and_cut_m_sequence(primary="biopac", cut_to_shortest=True, reset_time_axis=True)

# final synchronized output
print(synced_dataset.nilspod_aligned_)
print(synced_dataset.biopac_aligned_)
