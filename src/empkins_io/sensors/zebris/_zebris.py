__all__ = ["ZebrisDataset"]

import pandas as pd
from pathlib import Path

from src.empkins_io.utils._types import path_t


class ZebrisDataset:
    base_path: path_t

    _sensor_dict = {
        "time_series_data": ["time", "values"],  # For time-series CSVs
        "parameter_data": ["type", "Vorfuß COP x, links, mm", "Vorfuß COP y, links, mm",
                           "Rückfuß COP x, links, mm", "Rückfuß COP y, links, mm",
                           "Vorfuß COP x, rechts, mm", "Vorfuß COP y, rechts, mm",
                           "Rückfuß COP x, rechts, mm", "Rückfuß COP y, rechts, mm",
                           "COP x, links, mm", "COP y, links, mm", "COP x, rechts, mm",
                           "COP y, rechts, mm", "COP x, mm", "COP y, mm",
                           "Kraft Vorfuß L, %", "Kraft Rückfuß L, %", "Gesamtkraft L, %",
                           "Kraft Vorfuß R, %", "Kraft Rückfuß R, %", "Gesamtkraft R, %",
                           "Messdauer [Sek]", "Fläche der 95% Vertrauensellipse [mm²]",
                           "Länge der COP-Spur [mm]", "Gemittelte Geschwindigkeit [mm/Sek]"]
    }

    def __init__(self, path: Path):
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
        folder_path = Path(folder_path)
        return sorted(folder_path.glob("*.csv"))

    @classmethod
    def from_file(cls, file_path: Path) -> list[Path]:
        file_path = Path(file_path)  # Ensure it's a Path object
        if file_path.suffix == ".csv":
            return [file_path]
        return []

    def data_as_df(self):
        time_series_data = {}
        parameter_data = {}

        for file in self._raw_data:
            # Read the file
            df = pd.read_csv(file)

            # Check if the file contains metadata section (looking for 'type' column)
            if "type" in df.columns:
                # Read metadata and extract time-series data
                metadata = df.iloc[0, :]
                signal_type = metadata["type"]
                signal_name = metadata["name"]
                units = metadata["units"]

                # Read actual time-series data (skip first row)
                df_data = pd.read_csv(file, skiprows=2)  # Skip metadata rows
                df_data.columns = ["time", "value"]  # Ensure columns are named correctly

                # Store the time-series data with signal metadata
                time_series_data[file.stem] = {
                    "metadata": {"type": signal_type, "name": signal_name, "units": units},
                    "data": df_data
                }
            else:
                # Add parameter data to the dictionary (single-row data)
                parameter_data[file.stem] = df

            # Combine time-series data into DataFrames (multiple signals may exist)
        time_series_df = pd.concat(
            [data["data"] for data in time_series_data.values()],
            axis=0, ignore_index=True
        ) if time_series_data else pd.DataFrame()

        # Combine parameter data into a DataFrame (one row per file)
        parameter_df = pd.concat(parameter_data.values(), axis=0,
                                 ignore_index=True) if parameter_data else pd.DataFrame()

        # Return a dictionary with the data frames for each type
        return {
            "time_series_data": time_series_df,
            "parameter_data": parameter_df,
            "time_series_metadata": time_series_data  # Additional metadata information
        }