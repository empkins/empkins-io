import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union

from src.empkins_io.utils._types import path_t


class ZebrisDataset:
    def __init__(self, path: Union[str, Path]):
        """
        Initialize ZebrisDataset with a path to a folder or specific CSV file.

        Args:
            path (str or Path): Path to a directory containing CSV files or a specific CSV file
        """
        path = Path(path)  # Convert to Path object
        self.path = path

        if path.is_dir():
            self._raw_data = sorted(path.glob("*.csv"))
        elif path.is_file() and path.suffix == ".csv":
            self._raw_data = [path]
        else:
            raise FileNotFoundError(f"Invalid path: '{path}'. Not a CSV file or directory.")

    def _read_csv_with_metadata(self, file_path: Path) -> pd.DataFrame:
        """
        Read CSV files with potential metadata rows.

        Args:
            file_path (Path): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            # First, try to read the first two rows to understand the file structure
            with open(file_path, 'r') as f:
                first_row = f.readline().strip().strip('"').split('","')
                second_row = f.readline().strip().strip('"').split('","')

            # Special handling for pressure matrix files with many columns
            if len(first_row) > 10 and 'signal_matrix' in first_row:
                # Read the file with manual parsing to handle wide data
                df = pd.read_csv(file_path,
                                 header=[0, 1],  # Use first two rows as multi-level header
                                 skiprows=[2],  # Skip the third row if it's continuation of metadata
                                 engine='python')  # Use Python engine for more flexible parsing

                # Clean up column names
                df.columns = [col[1] if col[1] else col[0] for col in df.columns]

                # Attach metadata
                df.attrs['metadata'] = dict(zip(first_row[:-1], second_row[:-1]))
                df.attrs['filename'] = file_path.stem

                return df

            # Default handling for other files
            df = pd.read_csv(file_path,
                             header=None,  # Read all rows as data
                             engine='python')  # More flexible parsing

            # Check if the CSV has at least two rows
            if len(df) < 2:
                print(f"Warning: {file_path.name} has fewer than two rows.")
                return pd.DataFrame()

            # Extract metadata from first row
            metadata_row = df.iloc[0].tolist()
            data_type_row = df.iloc[1].tolist()

            # Use the second row as column names, or the first row if second is empty
            if len(set(data_type_row)) <= 1:  # If second row is mostly identical
                columns = metadata_row
                data_rows = df.iloc[2:]
            else:
                columns = data_type_row
                data_rows = df.iloc[2:]

            # Create a new DataFrame with correct columns
            df_cleaned = pd.DataFrame(data_rows.values, columns=columns)

            # Attach metadata as attributes
            df_cleaned.attrs['metadata'] = dict(zip(metadata_row, data_type_row))
            df_cleaned.attrs['filename'] = file_path.stem

            return df_cleaned

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def data_as_df(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files and organize them by type.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames by data type
        """
        # Organize dataframes by type
        time_series_dfs = {}
        parameter_dfs = {}
        pressure_matrix_dfs = {}
        patient_info_dfs = {}

        for file_path in self._raw_data:
            df = self._read_csv_with_metadata(file_path)

            if df.empty:
                continue

            # Determine DataFrame type based on columns and metadata
            metadata = df.attrs.get('metadata', {})
            data_type = metadata.get('type', '')

            # Categorize based on content
            if len(df.columns) == 2 and 'time' in df.columns:
                # Two-column time series (possibly force curves)
                if 'value' not in df.columns:
                    df.columns = ['time', 'value']
                time_series_dfs[df.attrs['filename']] = df

            elif len(df.columns) == 3 and all(col in df.columns for col in ['time', 'x', 'y']):
                # Three-column time series (possibly COP or 2D signals)
                time_series_dfs[df.attrs['filename']] = df

            elif data_type == 'signal_matrix':
                # Pressure matrix data
                pressure_matrix_dfs[df.attrs['filename']] = df

            elif 'Vorname' in df.columns or 'type' in df.columns and 'patient and record info' in df.iloc[0].values:
                # Patient and record info
                patient_info_dfs[df.attrs['filename']] = df

            elif 'type' in df.columns and 'parameter values' in df.iloc[0].values:
                # Parameter data
                parameter_dfs[df.attrs['filename']] = df

            else:
                # Default to parameter data if no specific type matches
                parameter_dfs[df.attrs['filename']] = df

        # Combine DataFrames if multiple exist
        result = {'time_series_data': pd.concat(time_series_dfs.values(),
                                                ignore_index=True) if time_series_dfs else pd.DataFrame(),
                  'parameter_data': pd.concat(parameter_dfs.values(),
                                              ignore_index=True) if parameter_dfs else pd.DataFrame(),
                  'pressure_matrix_data': pd.concat(pressure_matrix_dfs.values(),
                                                    ignore_index=True) if pressure_matrix_dfs else pd.DataFrame(),
                  'patient_info': pd.concat(patient_info_dfs.values(),
                                            ignore_index=True) if patient_info_dfs else pd.DataFrame(),
                  '_file_details': {
                      'time_series_files': list(time_series_dfs.keys()),
                      'parameter_files': list(parameter_dfs.keys()),
                      'pressure_matrix_files': list(pressure_matrix_dfs.keys()),
                      'patient_info_files': list(patient_info_dfs.keys())
                  }}

        # Attach original file details for reference

        return result