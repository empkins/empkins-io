import datetime
from typing import Optional, Union, Sequence

import empkins_io.sensors.a04 as a04
from empkins_io.utils._types import path_t

__all__ = ["load_data", "load_data_raw", "load_data_folder"]


def load_data_folder(
    folder_path: path_t,
    phase_names: Optional[Sequence[str]] = None,
    datastreams: Optional[Union[a04.DATASTREAMS, Sequence[a04.DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    return a04.load_data_folder(folder_path, phase_names=phase_names, datastreams=datastreams, timezone=timezone)


def load_data(
    path: path_t,
    datastreams: Optional[Union[a04.DATASTREAMS, Sequence[a04.DATASTREAMS]]] = None,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    return a04.load_data(path, datastreams=datastreams, timezone=timezone)


def load_data_raw(
    path: path_t,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
):
    return a04.load_data_raw(path, timezone=timezone)
