__all__ = ["ZebrisDataset"]

import pandas as pd
from pathlib import Path

from src.empkins_io.utils._types import path_t


class ZebrisDataset:
    base_path: path_t

    _sensor_dict = {
        "force curve": ["values"],  # force-curve.csv
        "force curve L": ["values"],  # force-curve-L.csv
        "force curve R": ["values"],  # force-curve-R.csv
        "force forefoot L %": ["values"],  # parameters.csv
        "force forefoot R %": ["values"],  # parameters.csv
        "force backfoot L %": ["values"],  # parameters.csv
        "force backfoot R %": ["values"],  # parameters.csv
        "total force %": ["values"],  # parameters.csv
        "total force R %": ["values"],  # parameters.csv
        "total force L %": ["values"],  # parameters.csv
        "force curve backfoot L": ["values"],  # force-curve_backfoot-L.csv
        "force curve backfoot R": ["values"],  # force-curve_backfoot-R.csv
        "force curve forefoot L": ["values"],  # force-curve_forefoot-L.csv
        "force curve forefoot R": ["values"],  # force-curve_forefoot-R.csv
        "gait line": ["x", "y"],  # gait-line.csv
        "gait line L": ["x", "y"],  # gait-line-L.csv
        "gait line R": ["x", "y"],  # gait-line-R.csv
        "COP": ["x", "y"],  # parameters.csv in [mm]
        "COP L": ["x", "y"],  # parameters.csv in [mm]
        "COP R": ["x", "y"],  # parameters.csv in [mm]
        "COP forefoot L": ["x", "y"],  # parameters.csv [mm]
        "COP forefoot R": ["x", "y"],  # parameters.csv [mm]
        "COP backfoot L": ["x", "y"],  # parameters.csv [mm]
        "COP backfoot R": ["x", "y"],  # parameters.csv [mm]
        "COP trajectory length": ["values"],  # parameters.csv in [mm]
        "total time": ["values"],  # parameters.csv
        "Area of the 95% confidence ellipse": ["values"],  # parameters.csv [mm2]
        "average velocity": ["values"],  # parameters.csv [mm/s]
    }

    def __init__(self, path: path_t):
        path = Path(path)  # Convert to Path object
        self.path = path
        if path.is_dir():
            self._raw_data = self.from_folder(path)
        elif path.is_file():
            self._raw_data = self.from_file(path)
        else:
            raise FileNotFoundError(f"Invalid path: '{path}'. Not a file or directory.")

    @classmethod
    def from_folder(cls, folder_path: Path) -> list[Path]:
        # Ensure folder_path is a Path object
        folder_path = Path(folder_path)

        # Get a sorted list of all CSV files in the folder
        return sorted(folder_path.glob("*.csv"))

    @classmethod
    def from_file(cls, file_path: Path) -> list[Path]:
        file_path = Path(file_path)  # Ensure it's a Path object
        if file_path.suffix == ".csv":
            return [file_path]
        return []

    def data_as_df(self):
        sensor_data = {}

        for sensor, columns in self._sensor_dict.items():
            matching_files = [file for file in self._raw_data if sensor in file.stem]

            if matching_files:
                df_list = [pd.read_csv(file) for file in matching_files]
                sensor_data[sensor] = pd.concat(df_list, axis=0, ignore_index=True)
            else:
                sensor_data[sensor] = pd.DataFrame(columns=columns)
        # Create a MultiIndex for the columns (sensor names + values or x/y)
        multi_index = pd.MultiIndex.from_tuples(
            [(sensor, col) for sensor, cols in self._sensor_dict.items() for col in cols],
            names=["Sensor", "Metric"]
        )

        # Combine all the sensor data into a single DataFrame
        zebris_df = pd.DataFrame(sensor_data)

        # Reindex to apply the MultiIndex to the columns
        zebris_df.columns = multi_index

        return zebris_df