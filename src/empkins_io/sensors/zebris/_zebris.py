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

    def _read_csv_with_metadata(self, file_path: Path) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip().lower()
                second_line = f.readline().strip().lower()
                third_line = f.readline().strip().lower()

            if '"type","' in first_line and '"patient and record info"' in second_line:
                return self._read_patient_info_csv(file_path)

            if '"type","' in first_line and ('parameter' in second_line or 'parameter' in first_line):
                return self._read_parameters_csv(file_path)

            # --- New logic: smarter detection ---
            if 'signal_2d' in second_line and 'cop' in second_line:
                return self._read_gait_line_csv(file_path)
            elif 'signal"' in first_line or 'signal_2d"' in first_line or 'signal_matrix' in first_line:
                if 'x' in second_line or 'x' in third_line:
                    return self._read_gait_line_csv(file_path)
                elif 'time' in second_line:
                    return self._read_force_curve_csv(file_path)
                else:
                    return self._read_pressure_matrix_csv(file_path)

            if '"type","name"' in first_line:
                return self._read_generic_signal_csv(file_path)

            print(f"Unrecognized file type for {file_path.name}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_generic_signal_csv(self, file_path: Path) -> pd.DataFrame:
        try:
            metadata_df = pd.read_csv(file_path, nrows=1)
            df = pd.read_csv(file_path, skiprows=2)

            df.attrs['type'] = metadata_df.iloc[0].get('type', 'unknown')
            df.attrs['name'] = metadata_df.iloc[0].get('name', 'unknown')
            df.attrs['filename'] = file_path.stem

            return df

        except Exception as e:
            print(f"Error reading generic CSV {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_parameters_csv(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                file_path,
                quotechar='"',
                skipinitialspace=True,
                encoding='utf-8-sig',
                header=0
            )
            if 'type' in df.columns:
                df = df.drop(columns=['type'])

            df.attrs['type'] = 'parameter values'
            df.attrs['filename'] = file_path.stem

            if self.explain:
                print(f"\nReading parameters file: {file_path.name}")
                print(f"Columns (parameters): {list(df.columns)}")
                print(f"Values:\n{df.iloc[0]}")

            return df

        except Exception as e:
            print(f"Error reading parameters file {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_patient_info_csv(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                file_path,
                quotechar='"',
                skipinitialspace=True,
                encoding='utf-8-sig',
                header=0
            )
            if 'type' in df.columns:
                df = df.iloc[:, 1:]

            df.attrs['type'] = 'patient and record info'
            df.attrs['filename'] = file_path.stem

            if self.explain:
                print(f"\nReading patient info file: {file_path.name}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading patient info file {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_force_curve_csv(self, file_path: Path) -> pd.DataFrame:
        return self._read_generic_data_file(file_path, names=['time', 'value'])

    def _read_gait_line_csv(self, file_path: Path) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip().lower()
                second_line = f.readline().strip().lower()
                third_line = f.readline().strip().lower()
                fourth_line = f.readline().strip().lower()  # NEW: read fourth line

            metadata_type = first_line.split(',')[1].strip('"') if ',' in first_line else "unknown"
            metadata_name = second_line.split(',')[1].strip('"') if ',' in second_line else "unknown"

            # Now smart detect based on real header line
            if all(keyword in fourth_line for keyword in ["time", "x", "y"]):
                skiprows = 3
                header = 0
                names = None
            else:
                skiprows = 3
                header = None
                names = ["time", "x", "y"]

            df = pd.read_csv(file_path, skiprows=skiprows, header=header, names=names)

            df.attrs['type'] = metadata_type
            df.attrs['name'] = metadata_name
            df.attrs['filename'] = file_path.stem

            if self.explain:
                print(f"\nReading {file_path.name} (type: {metadata_type})")
                print(f"Detected shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_pressure_matrix_csv(self, file_path: Path) -> pd.DataFrame:
        dynamic_names = ['time'] + [f'x{i + 1}' for i in range(54)]
        return self._read_generic_data_file(file_path, names=dynamic_names)

    def _read_generic_data_file(self, file_path: Path, names: list) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                metadata_type = f.readline().strip().split(',')[1].strip('"')
                metadata_name = f.readline().strip().split(',')[1].strip('"')

            df = pd.read_csv(file_path, skiprows=2, names=names)

            df.attrs['type'] = metadata_type
            df.attrs['name'] = metadata_name
            df.attrs['filename'] = file_path.stem

            if self.explain:
                print(f"\nReading {file_path.name} (type: {metadata_type})")
                print(f"Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def separate_data(self, explain: bool = False) -> Dict[str, pd.DataFrame]:
        time_dependent_dfs = {}
        single_value_dfs = {}
        processing_errors = []

        if explain:
            print(f"Processing {len(self._raw_data)} CSV files...")

        for file_path in self._raw_data:
            try:
                df = self._read_csv_with_metadata(file_path)

                if df.empty:
                    if explain:
                        print(f"Skipping empty DataFrame: {file_path}")
                    continue

                file_type = df.attrs.get('type', 'unknown')
                filename = df.attrs.get('filename', file_path.stem)

                if explain:
                    print(f"Processing file: {filename} (Type: {file_type})")

                if file_type in ['signal', 'signal_2d', 'signal_matrix']:
                    time_dependent_dfs[filename] = df
                elif file_type in ['parameter values', 'patient and record info']:
                    single_value_dfs[filename] = df
                else:
                    if explain:
                        print(f"Unrecognized file type for {filename}: {file_type}")

            except Exception as e:
                processing_errors.append(f"Error processing {file_path}: {str(e)}")
                if explain:
                    print(f"Error processing {file_path}: {e}")

        result = {
            'time_dependent_data': pd.concat(list(time_dependent_dfs.values()),
                                             ignore_index=True) if time_dependent_dfs else pd.DataFrame(),
            'single_value_data': pd.concat(list(single_value_dfs.values()),
                                           ignore_index=True) if single_value_dfs else pd.DataFrame()
        }

        if explain:
            print("\n--- Processing Summary ---")
            print(f"Time-Dependent Files: {len(time_dependent_dfs)}")
            print(f"Single-Value Files: {len(single_value_dfs)}")
            if processing_errors:
                print("\n--- Processing Errors ---")
                for error in processing_errors:
                    print(error)

        return result

    def force_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        df = self._processed_data['time_dependent_data']

        force_columns = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['kraft', 'force', 'value'])]
        if not force_columns:
            raise ValueError("No force curve data found in the dataset.")

        if side == 'left':
            force_columns = [col for col in force_columns if any(marker in col.lower() for marker in ['l', 'links'])]
        elif side == 'right':
            force_columns = [col for col in force_columns if any(marker in col.lower() for marker in ['r', 'rechts'])]

        columns_to_extract = ['time'] + force_columns if 'time' in df.columns else force_columns

        return df[columns_to_extract].copy()

    def gait_line_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        """
        Load gait-line data (Center of Pressure) for 'left', 'right' or 'both' sides.
        """
        gait_dfs = []

        # Search for matching gait-line files
        for file_path in self._raw_data:
            name = file_path.stem.lower()
            if "gait-line" in name:
                # Side filtering
                if side == 'left' and not name.endswith("-l"):
                    continue
                if side == 'right' and not name.endswith("-r"):
                    continue
                # If side == 'both', take all

                # Read the file cleanly
                df = self._read_gait_line_csv(file_path)
                df['source'] = name  # add where it came from (optional)
                gait_dfs.append(df)

        if not gait_dfs:
            raise ValueError(f"No gait-line data found for side='{side}'.")

        # Merge all matching gait-line files
        df = pd.concat(gait_dfs, ignore_index=True)

        return df.copy()

    def pressure_data_as_df(self) -> pd.DataFrame:
        df = self._processed_data['time_dependent_data']
        pressure_columns = [col for col in df.columns if col != 'time']

        if not pressure_columns:
            raise ValueError("No pressure distribution data found in the dataset.")

        return df[pressure_columns].copy()

    def patient_info_as_df(self) -> pd.DataFrame:
        return self._processed_data['single_value_data']

    def parameters_as_df(self) -> pd.DataFrame:
        parameters_files = [f for f in self._raw_data if 'parameters' in f.name.lower()]
        if not parameters_files:
            raise ValueError("No parameters file found in the dataset.")
        return self._read_parameters_csv(parameters_files[0])
