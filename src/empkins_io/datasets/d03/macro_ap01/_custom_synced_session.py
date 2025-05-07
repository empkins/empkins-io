import pandas as pd
from nilspodlib import Dataset, SyncedSession
from packaging.version import Version


class CustomSyncedSession(SyncedSession):
    """Custom SyncedSession to handle counter issues of the ClapperBoardPod and different firmware versions.

    Additionally, the SyncPod was using a custom dev version of the NilsPod firmware and needs to be loaded differently.
    Further this fixes the start and end datetime for the sensors that were recorded with the phone that had the wrong
    date.
    """

    VALIDATE_ON_INIT = False  # We need to disable this, as one of the phones recorded at the wrong date time.

    timestamp = pd.Timestamp("2022-01-01", tz="Europe/Berlin")

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
            if "NilsPodX-E76B" in p.name and dataset.info.utc_datetime_start_day_midnight < cls.timestamp:
                # This fixes the date of the sensor
                dataset.info.utc_start = ds[0].info.utc_start
                dataset.info.utc_stop = ds[0].info.utc_stop

            ds.append(dataset)

        return cls(ds)

    @classmethod
    def from_folder_path(
        cls,
        base_path,
        filter_pattern: str | None = "NilsPodX-*.bin",
        legacy_support: str | None = "resolve",
        force_version=None,
        tz: str | None = "Europe/Berlin",
    ):
        return cls.from_file_paths(
            sorted(base_path.glob(filter_pattern)), legacy_support=legacy_support, force_version=force_version, tz=tz
        )
