# -*- coding: utf-8 -*-
<<<<<<< HEAD

=======
>>>>>>> ebff522f1c08fbd4afca6c8a135a3ef1fdba4eda
from empkins_io.datasets.d03.micro_gapvii._dataset import MicroBaseDataset

__version__ = "0.1.0"

__all__ = ["MicroBaseDataset"]
<<<<<<< HEAD

def conf_rel_path():
    """Configure relative path imports for the experiments folder."""

    from pathlib import Path
    import sys

    parent_folder = str(Path("..").resolve())
    if parent_folder not in sys.path:
        sys.path.append(parent_folder)
=======
>>>>>>> ebff522f1c08fbd4afca6c8a135a3ef1fdba4eda
