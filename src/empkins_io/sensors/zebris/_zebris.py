from pathlib import Path
from typing import Dict, Union

import pandas as pd


class ZebrisDataset:
    def __init__(self, path: Union[str, Path], explain: bool = True):
        """
        Initialize ZebrisDataset with a path to a folder or specific CSV file.

        Args:
            path (str or Path): Path to data directory containing CSV files or a specific CSV file
            explain (bool): Enable detailed logging
        """
        path = Path(path)  # Convert to Path object
        self.path = path
        self.explain = explain

        if path.is_dir():
            self._raw_data = sorted(path.glob("*.csv"))
        elif path.is_file() and path.suffix == ".csv":
            self._raw_data = [path]
        else:
            raise FileNotFoundError(f"Invalid path: '{path}'. Not a CSV file or directory.")

        if self.explain:
            print(f"Found {len(self._raw_data)} CSV files")
            for file in self._raw_data:
                print(f"  - {file.name}")

        # Store processed data after initialization
        self._processed_data = self.separate_data(explain=self.explain)

    def _read_csv_with_metadata(self, file_path: Path) -> pd.DataFrame:
        """
        Method to read CSV files with parsing for Zebris datasets.
        """
        try:
            # Read the first two lines to determine file structure
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip().strip('"').split('","')
                second_line = f.readline().strip().strip('"').split('","')

            if self.explain:
                print(f"\nReading file: {file_path.name}")
                print(f"First line: {first_line}")
                print(f"Second line: {second_line}")

            # Determine file type
            file_type = second_line[0] if len(second_line) > 0 else first_line[0]

            # Read CSV, skipping the first two rows and using robust parsing
            df = pd.read_csv(file_path,
                             encoding='utf-8-sig',
                             header=None,  # No header
                             skiprows=2,  # Skip metadata rows
                             names=['time', 'value'])  # Simple initial column names

            # Clean and convert time column
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # Rename column based on file content
            if len(first_line) > 1:
                value_column_name = first_line[1].split(',')[0].strip('"')
                df.rename(columns={'value': value_column_name}, inplace=True)

            # Set file type and filename as attributes
            df.attrs['type'] = file_type
            df.attrs['filename'] = file_path.stem

            if self.explain:
                print(f"Recognized as {file_type} file")
                print(f"Columns: {df.columns}")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def separate_data(self, explain: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Separate CSV files into time-dependent and single-value DataFrames.
        """
        # Dictionaries to store DataFrames by type
        time_dependent_dfs = {}
        single_value_dfs = {}

        # Error tracking
        processing_errors = []

        if explain:
            print(f"Processing {len(self._raw_data)} CSV files...")

        for file_path in self._raw_data:
            try:
                # Read the DataFrame
                df = self._read_csv_with_metadata(file_path)

                if df.empty:
                    if explain:
                        print(f"Skipping empty DataFrame: {file_path}")
                    continue

                # Determine file type from attributes
                file_type = df.attrs.get('type', 'unknown')
                filename = df.attrs.get('filename', file_path.stem)

                if explain:
                    print(f"Processing file: {filename} (Type: {file_type})")

                # Categorize time-dependent vs single-value data
                if file_type in ['signal', 'signal_2d', 'signal_matrix']:
                    # Time series and matrix data
                    time_dependent_dfs[filename] = df
                elif file_type in ['parameter values', 'patient and record info']:
                    # Single-value data
                    single_value_dfs[filename] = df
                else:
                    if explain:
                        print(f"Unrecognized file type for {filename}: {file_type}")

            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                processing_errors.append(error_msg)
                if explain:
                    print(error_msg)

        # Combine DataFrames
        result = {
            'time_dependent_data': pd.concat(list(time_dependent_dfs.values()),
                                             ignore_index=True) if time_dependent_dfs else pd.DataFrame(),
            'single_value_data': pd.concat(list(single_value_dfs.values()),
                                           ignore_index=True) if single_value_dfs else pd.DataFrame()
        }

        # Verbose logging
        if explain:
            print("\n--- Processing Summary ---")
            print(f"Time-Dependent Files: {len(time_dependent_dfs)}")
            print(f"Single-Value Files: {len(single_value_dfs)}")

            # Log any processing errors
            if processing_errors:
                print("\n--- Processing Errors ---")
                for error in processing_errors:
                    print(error)

        return result

    def force_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        """
        Extract force curve data from the dataset.
        """
        # Retrieve time-dependent data
        time_dependent_data = self._processed_data['time_dependent_data']

        if self.explain:
            print("\nForce Data Extraction:")
            print(f"Time-dependent data columns: {time_dependent_data.columns}")

        # Updated column detection for force data
        force_columns = [
            col for col in time_dependent_data.columns
            if any(keyword in str(col).lower() for keyword in ['kraft', 'force', 'value'])
        ]

        if self.explain:
            print(f"Detected force columns: {force_columns}")

        if not force_columns:
            raise ValueError("No force curve data found in the dataset.")

        # Side filtering with more robust matching
        if side == 'left':
            force_columns = [col for col in force_columns if any(marker in col.lower() for marker in ['l', 'links'])]
        elif side == 'right':
            force_columns = [col for col in force_columns if any(marker in col.lower() for marker in ['r', 'rechts'])]

        # Ensure time column is present
        columns_to_extract = ['time'] + force_columns if 'time' in time_dependent_data.columns else force_columns

        # Extract relevant columns
        force_df = time_dependent_data[columns_to_extract].copy()

        return force_df

    def gait_line_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        """
        Extract gait line data from the dataset.
        """
        # Retrieve time-dependent data
        time_dependent_data = self._processed_data['time_dependent_data']

        # More flexible column detection for gait line files
        gait_columns = [
            col for col in time_dependent_data.columns
            if any(keyword in str(col).lower() for keyword in ['cop', 'center of pressure', 'x', 'y', 'position'])
        ]

        if self.explain:
            print(f"Detected gait columns: {gait_columns}")

        if not gait_columns:
            print("Warning: No explicit gait line columns found. Attempting alternative detection.")
            gait_columns = time_dependent_data.columns.tolist()

        # Select data based on side specification with more robust matching
        if side == 'left':
            gait_columns = [col for col in gait_columns if any(marker in col.lower() for marker in ['l', 'links'])]
        elif side == 'right':
            gait_columns = [col for col in gait_columns if any(marker in col.lower() for marker in ['r', 'rechts'])]

        # Ensure 'time' column is included if it exists
        columns_to_extract = ['time'] + gait_columns if 'time' in time_dependent_data.columns else gait_columns

        # Extract relevant columns
        gait_df = time_dependent_data[columns_to_extract].copy()

        return gait_df

    def pressure_data_as_df(self, average: bool = True) -> pd.DataFrame:
        """
        Extract pressure distribution data from the dataset.

        Args:
            average (bool, optional): If True, returns averaged pressure data.
                                      If False, returns full pressure matrix.
                                      Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing pressure distribution data
        """
        # Retrieve time-dependent data
        time_dependent_data = self._processed_data['time_dependent_data']

        # Filter for pressure distribution files
        pressure_columns = [col for col in time_dependent_data.columns if col not in ['time']]

        if not pressure_columns:
            raise ValueError("No pressure distribution data found in the dataset.")

        # Extract relevant columns
        pressure_df = time_dependent_data[pressure_columns].copy()

        return pressure_df

    def patient_info_as_df(self) -> pd.DataFrame:
        """
        Extract patient and record information.

        Returns:
            pd.DataFrame: DataFrame containing patient information
        """
        return self._processed_data['single_value_data']
