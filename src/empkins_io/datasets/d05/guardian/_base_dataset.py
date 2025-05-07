from tpcp import Dataset

from empkins_io.utils._types import path_t


class GuardianBaseDataset(Dataset):
    """Base class for Guardian datasets."""

    base_path: path_t
    use_cache: bool

    PHASES = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = None
