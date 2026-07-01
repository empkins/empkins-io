from empkins_io.datasets.d05.guardian._base_dataset import GuardianBaseDataset


class TFMGuardianDataset(GuardianBaseDataset):
    """Dataset for the Task Force Monitor (TFM) data from the Guardian dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
