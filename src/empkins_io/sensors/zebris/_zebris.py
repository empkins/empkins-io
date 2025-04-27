from pathlib import Path
from typing import Dict, Union

import pandas as pd


class ZebrisDataset:
    """_sensor_dict = {
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
    """

    def __init__(self, path: Union[str, Path], explain: bool = True):
        """
        Initialize Zebris Dataset with a path to a folder or specific CSV file.

        Args:
            path (str or Path): Path to a data directory containing CSV files or a specific CSV file
            explain (bool): Enable detailed logging
        """
        path = Path(path)  # Convert to a Path object
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

    def _read_parameters_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read a parameters CSV file and return a DataFrame.
        """
        try:
            # 1) Load the CSV, keep the header row as column names
            df = pd.read_csv(
                file_path,
                quotechar='"',
                skipinitialspace=True,
                encoding='utf-8-sig',  # strip BOM if present
                header=0
            )

            # 2) Drop the 'type' column (it just holds 'parameter values')
            if 'type' in df.columns:
                df = df.drop(columns=['type'])  # :contentReference[oaicite:0]{index=0}

            # 3) (Optional) Rename the lone row index if you like:
            #    e.g. df.index = [file_path.stem]

            if self.explain:
                print(f"\nReading parameters file: {file_path.name}")
                print(f"Columns (parameters): {list(df.columns)}")
                print(f"Values:\n{df.iloc[0]}")

            return df

        except Exception as e:
            print(f"Error reading parameters file {file_path}: {e}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error reading parameters file {file_path}: {e}")
            return pd.DataFrame()

    def _read_csv_with_metadata(self, file_path: Path) -> pd.DataFrame:
        """
        Determine and read a CSV file based on its content and structure.
        """
        try:
            # Read the first line to check for metadata
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()

            # Specific handling for patient and record info files
            if '"type","' in first_line and '"patient and record info"' in second_line:
                return self._read_patient_info_csv(file_path)

            # Specific handling for parameter files
            if '"type","' in first_line and 'parameter' in first_line.lower():
                return self._read_parameters_csv(file_path)

            if '"type","name"' in first_line:
                # Read metadata row
                metadata_df = pd.read_csv(file_path, nrows=1)

                # Read data rows, skipping the first two rows (metadata)
                df = pd.read_csv(file_path, skiprows=2)

                # Add metadata as attributes
                df.attrs['type'] = metadata_df.iloc[0]['type']
                df.attrs['name'] = metadata_df.iloc[0]['name']
                df.attrs['filename'] = file_path.stem

                return df

            # Existing methods for other file types
            if 'signal"' in first_line or 'signal_2d' in first_line:
                # Force curve or gait line files
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    second_line = f.readline().strip()
                    third_line = f.readline().strip()

                if 'x' in second_line or 'x' in third_line:
                    return self._read_gait_line_csv(file_path)
                else:
                    return self._read_force_curve_csv(file_path)

            # unrecognized file
            print(f"Unrecognized file type for {file_path}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def _read_patient_info_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read a patient and record info CSV file.
        """
        try:
            # Read the CSV with explicit handling of quotes
            df = pd.read_csv(file_path,
                             quotechar='"',
                             skipinitialspace=True,
                             encoding='utf-8-sig',
                             header=0)

            # Remove the first 'type' column if it exists
            if 'type' in df.columns:
                df = df.iloc[:, 1:]

            # Add metadata attributes
            df.attrs['type'] = 'patient and record info'
            df.attrs['filename'] = file_path.stem

            if self.explain:
                print(f"\nReading patient info file: {file_path.name}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading patient info file {file_path}: {e}")
            return pd.DataFrame()

    def _read_force_curve_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read force curve CSV files with multiple rows of time-series data.
        """
        try:
            # Read metadata first
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                metadata_type = f.readline().strip().split(',')[1].strip('"')
                metadata_name = f.readline().strip().split(',')[1].strip('"')

            # Read the actual data
            df = pd.read_csv(file_path,
                             skiprows=2,  # Skip metadata rows
                             names=['time', 'value'])  # Ensure consistent column names

            # Add metadata as attributes
            df.attrs['type'] = metadata_type
            df.attrs['name'] = metadata_name

            if self.explain:
                print(f"\nReading force curve file: {file_path.name}")
                print(f"Metadata type: {metadata_type}")
                print(f"Metadata name: {metadata_name}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading force curve file {file_path}: {e}")
            return pd.DataFrame()

    def _read_gait_line_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read gait line CSV files with multiple rows of 2D coordinate data.
        """
        try:
            # Read metadata first
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                metadata_type = f.readline().strip().split(',')[1].strip('"')
                metadata_name = f.readline().strip().split(',')[1].strip('"')

            # Read the actual data
            df = pd.read_csv(file_path,
                             skiprows=2,  # Skip metadata rows
                             names=['time', 'x', 'y'])  # Ensure consistent column names

            # Add metadata as attributes
            df.attrs['type'] = metadata_type
            df.attrs['name'] = metadata_name

            if self.explain:
                print(f"\nReading gait line file: {file_path.name}")
                print(f"Metadata type: {metadata_type}")
                print(f"Metadata name: {metadata_name}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading gait line file {file_path}: {e}")
            return pd.DataFrame()

    def _read_pressure_matrix_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read pressure matrix CSV files with multiple pressure cells.
        """
        try:
            # Read metadata first
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                metadata_type = f.readline().strip().split(',')[1].strip('"')
                metadata_name = f.readline().strip().split(',')[1].strip('"')

            # Read the actual data
            df = pd.read_csv(file_path,
                             skiprows=2,  # Skip metadata rows
                             names=['time'] + [f'x{i + 1}' for i in range(54)])  # Dynamic column names

            # Add metadata as attributes
            df.attrs['type'] = metadata_type
            df.attrs['name'] = metadata_name

            if self.explain:
                print(f"\nReading pressure matrix file: {file_path.name}")
                print(f"Metadata type: {metadata_type}")
                print(f"Metadata name: {metadata_name}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading pressure matrix file {file_path}: {e}")
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

                # Determine the file type from attributes
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

    def pressure_data_as_df(self) -> pd.DataFrame:
        """
        Extract pressure distribution data from the dataset.

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


    def parameters_as_df(self) -> pd.DataFrame:
        """
        Extract parameters from the dataset.

        Returns:
            pd.DataFrame: DataFrame containing parameter values
        """
        # Find parameter CSV file
        parameters_files = [f for f in self._raw_data if 'parameters' in f.name.lower()]

        if not parameters_files:
            raise ValueError("No parameters file found in the dataset.")

        # Read the parameter file
        parameters_df = self._read_parameters_csv(parameters_files[0])

        return parameters_df