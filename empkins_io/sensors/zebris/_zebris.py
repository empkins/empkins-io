__all__ = ["ZebrisDataset"]

from pathlib import Path
from typing import Self

from empkins_io.utils._types import path_t


class ZebrisDataset:

    def __init__(self):
        pass

    @classmethod
    def from_folder(cls, folder_path: path_t) -> Self:

        # ensure pathlib
        folder_path = Path(folder_path)

        list(sorted(folder_path.glob("*.csv")))
