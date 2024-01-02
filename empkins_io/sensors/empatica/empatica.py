from functools import lru_cache
from typing import Optional

import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

from empkins_io.utils._types import path_t


class EmpaticaDataset:

    _path: path_t
    _raw_data: dict

    _index_type: str
    _tz: str

    def __init__(
        self,
        path: path_t,
        index_type: Optional[str] = None,
        tz: Optional[str] = None,
    ):
        self.path = path
        self._raw_data = _from_folder(path)
        self._index_type = index_type
        self._tz = tz

    @property
    def timezone(self):
        """Timezone of the recording."""
        return self._tz

    @property
    def acc(self) -> pd.DataFrame:

        acc = self._raw_data["rawData"]["accelerometer"]

        df = pd.DataFrame(
            {
                "acc_x": acc["x"],
                "acc_y": acc["y"],
                "acc_z": acc["z"],
            }
        )

        return self._add_index(
            df, self._index_type, acc["samplingFrequency"], acc["timestampStart"]
        )

    def _add_index(
        self,
        data: pd.DataFrame,
        index: str,
        sampling_rate_hz: float,
        start_time_unix: int,
    ) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }
        if index and index not in index_names:
            raise ValueError(
                f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}"
            )
        index_name = index_names[index]
        data.index.name = index_name

        if index is None:
            return data
        if index == "time":
            data.index -= data.index[0]
            data.index /= sampling_rate_hz
            return data

        data.index = [
            round(start_time_unix + i * (1e6 / sampling_rate_hz))
            for i in range(len(data.index))
        ]

        if index == "utc_datetime":
            # convert unix timestamps to datetime
            data.index = pd.to_datetime(data.index, unit="us")
            data.index = data.index.tz_localize("UTC")
        if index == "local_datetime":
            data.index = pd.to_datetime(data.index, unit="us")
            data.index = data.index.tz_localize("UTC").tz_convert(self.timezone)

        return data


@lru_cache(maxsize=2)
def _from_folder(path: path_t) -> dict:
    raw_data_path = path.joinpath("raw_data", "v6")  # should be fix for now

    # list all .avro files
    files = sorted(raw_data_path.glob("*.avro"))

    # only read the first file for now
    reader = DataFileReader(open(files[0], "rb"), DatumReader())
    data = next(reader)

    return data
