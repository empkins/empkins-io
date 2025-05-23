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
        filter_pattern: str | None = "NilsPodX-*.bin",
        legacy_support: str | None = "resolve",
        force_version=None,
        tz: str | None = "Europe/Berlin",
    ):
        return cls.from_file_paths(
            sorted(base_path.glob(filter_pattern)), legacy_support=legacy_support, force_version=force_version, tz=tz
        )
