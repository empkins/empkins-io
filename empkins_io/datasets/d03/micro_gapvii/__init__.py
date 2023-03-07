# -*- coding: utf-8 -*-

from empkins_io.datasets.d03.micro_gapvii._dataset import PepEstimationDataset

__version__ = "0.1.0"

__all__ = ["PepEstimationDataset"]

def conf_rel_path():
    """Configure relative path imports for the experiments folder."""

    from pathlib import Path
    import sys

    parent_folder = str(Path("..").resolve())
    if parent_folder not in sys.path:
        sys.path.append(parent_folder)
