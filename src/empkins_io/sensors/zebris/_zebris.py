import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, List, Any


class ZebrisDataset:
    def __init__(self, path: Union[str, Path], verbose: bool = True):
        """
        Initialize ZebrisDataset with a path to a folder or specific CSV file.

        Args:
            path (str or Path): Path to a directory containing CSV files or a specific CSV file
            verbose (bool): Enable detailed logging
        """
        path = Path(path)  # Convert to Path object
        self.path = path
        self.verbose = verbose

        if path.is_dir():
            self._raw_data = sorted(path.glob("*.csv"))
        elif path.is_file() and path.suffix == ".csv":
            self._raw_data = [path]
        else:
            raise FileNotFoundError(f"Invalid path: '{path}'. Not a CSV file or directory.")

        if self.verbose:
            print(f"Found {len(self._raw_data)} CSV files")
            for file in self._raw_data:
                print(f"  - {file.name}")

        # Store processed data after initialization
        self._processed_data = self.separate_data(verbose=self.verbose)

    def _read_csv_with_metadata(self, file_path: Path) -> pd.DataFrame:
        """
        Enhanced method to read CSV files with flexible parsing for Zebris datasets.
        """
        try:
            # Read the first few lines to determine file structure
            with open(file_path, 'r', encoding='utf-8') as f:
                # First line: headers or type
                first_line = f.readline().strip().strip('"').split('","')
                # Second line: type or first data row
                second_line = f.readline().strip().strip('"').split('","')

            # Debug print
            if self.verbose:
                print(f"\nReading file: {file_path.name}")
                print(f"First line: {first_line}")
                print(f"Second line: {second_line}")

            # Determine file type
            file_type = second_line[0] if len(second_line) > 0 else first_line[0]

            # Specific handling for different file types
            if file_type == 'parameter values':
                # Parameters file
                df = pd.read_csv(file_path, header=0, encoding='utf-8')
                df.columns = [col.strip('"') for col in df.columns]
                df.attrs['type'] = 'parameter values'

                if self.verbose:
                    print("Recognized as parameter values file")

            elif file_type == 'patient and record info':
                # Patient info file
                df = pd.read_csv(file_path, header=0, encoding='utf-8')
                df.columns = [col.strip('"') for col in df.columns]
                df.attrs['type'] = 'patient and record info'

                if self.verbose:
                    print("Recognized as patient and record info file")

            elif file_type in ['signal', 'signal_2d', 'signal_matrix']:
                # Time series files
                # Skip first two rows, use third row as header
                df = pd.read_csv(file_path, header=1, encoding='utf-8')

                # Add metadata from first line
                metadata = {}
                for i in range(0, len(first_line) - 1, 2):
                    metadata[first_line[i]] = first_line[i + 1]

                df.attrs['metadata'] = metadata
                df.attrs['type'] = file_type

                if self.verbose:
                    print(f"Recognized as {file_type} file")
                    print(f"Columns: {df.columns}")

                # Rename columns if needed
                if 'value' in df.columns and 'time' not in df.columns:
                    df = df.rename(columns={'index': 'time'})

            else:
                # Fallback to default reading
                df = pd.read_csv(file_path, header=0, encoding='utf-8')
                print(f"Unrecognized file type for {file_path}")

            # Add filename and path to attributes
            df.attrs['filename'] = file_path.stem
            df.attrs['file_path'] = str(file_path)

            return df

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def separate_data(self, verbose: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Separate CSV files into time-dependent and single-value DataFrames.

        Args:
            verbose (bool, optional): If True, provides detailed logging about file processing.
                                      Defaults to False.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with two DataFrames:
            - 'time_dependent_data': DataFrame with time series and signal data
            - 'single_value_data': DataFrame with parameters and patient info
        """
        # Dictionaries to store DataFrames by type
        time_dependent_dfs = {}
        single_value_dfs = {}

        # Logging method
        def log(message: str) -> None:
            """Internal logging method that prints only if verbose is True."""
            if verbose:
                print(message)

        # Error tracking
        processing_errors = []

        log(f"Processing {len(self._raw_data)} CSV files...")

        for file_path in self._raw_data:
            try:
                # Read the DataFrame
                df = self._read_csv_with_metadata(file_path)

                if df.empty:
                    log(f"Skipping empty DataFrame: {file_path}")
                    continue

                # Determine file type from attributes
                file_type = df.attrs.get('type', 'unknown')
                filename = df.attrs.get('filename', file_path.stem)

                log(f"Processing file: {filename} (Type: {file_type})")

                # Categorize time-dependent vs single-value data
                if file_type in ['signal', 'signal_2d', 'signal_matrix']:
                    # Time series and matrix data
                    time_dependent_dfs[filename] = df
                elif file_type in ['parameter values', 'patient and record info']:
                    # Single-value data
                    single_value_dfs[filename] = df
                else:
                    log(f"Unrecognized file type for {filename}: {file_type}")

            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                processing_errors.append(error_msg)
                log(error_msg)

        # Combine DataFrames
        result = {
            'time_dependent_data': pd.concat(list(time_dependent_dfs.values()),
                                             ignore_index=True) if time_dependent_dfs else pd.DataFrame(),
            'single_value_data': pd.concat(list(single_value_dfs.values()),
                                           ignore_index=True) if single_value_dfs else pd.DataFrame()
        }

        # Log processing summary
        log("\n--- Processing Summary ---")
        log(f"Time-Dependent Files: {len(time_dependent_dfs)}")
        log(f"Single-Value Files: {len(single_value_dfs)}")

        # Log any processing errors
        if processing_errors:
            log("\n--- Processing Errors ---")
            for error in processing_errors:
                log(error)

        return result

    def force_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        """
        Extract force curve data from the dataset.
        """
        # Retrieve time-dependent data
        time_dependent_data = self._processed_data['time_dependent_data']

        if self.verbose:
            print("\nForce Data Extraction:")
            print(f"Time-dependent data columns: {time_dependent_data.columns}")

        # Updated column detection for force data
        force_columns = [
            col for col in time_dependent_data.columns
            if any(keyword in col.lower() for keyword in ['kraft', 'force', 'value'])
        ]

        if self.verbose:
            print(f"Detected force columns: {force_columns}")

        if not force_columns:
            raise ValueError("No force curve data found in the dataset.")

        # Side filtering
        if side == 'left':
            force_columns = [col for col in force_columns if 'l' in col.lower()]
        elif side == 'right':
            force_columns = [col for col in force_columns if 'r' in col.lower()]

        # Ensure time column is present
        columns_to_extract = ['time'] + force_columns if 'time' in time_dependent_data.columns else force_columns

        # Extract relevant columns
        force_df = time_dependent_data[columns_to_extract].copy()

        return force_df


    def gait_line_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        """
        Extract gait line data from the dataset.

        Args:
            side (str, optional): Specify which side to extract.
                                  Options: 'left', 'right', 'both'.
                                  Defaults to 'both'.

        Returns:
            pd.DataFrame: DataFrame containing gait line data
        """
        # Retrieve time-dependent data
        time_dependent_data = self._processed_data['time_dependent_data']

        # Filter for gait line files
        gait_columns = [col for col in time_dependent_data.columns if 'x' in col.lower() or 'y' in col.lower()]

        if not gait_columns:
            raise ValueError("No gait line data found in the dataset.")

        # Select data based on side specification
        if side == 'left':
            gait_columns = [col for col in gait_columns if 'l' in col.lower()]
        elif side == 'right':
            gait_columns = [col for col in gait_columns if 'r' in col.lower()]

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