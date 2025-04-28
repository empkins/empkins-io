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
        Initializes an object that processes CSV files from a given file path or directory.
        If the path represents a directory, it collects all CSV files within the directory.
        If the path represents a single CSV file, it processes that file.
        The initialization optionally prints explanations/logging about the discovered files.

        Attributes:
        path: Union[str, Path]
            The path to a CSV file or a directory containing CSV files.
        explain: bool
            Flag indicating whether to print information about the discovered CSV files.
        _raw_data: List[Path]
            A list of paths to the discovered CSV files, sorted by name.
        _processed_data: Any
            Processed data resulting from the `separate_data` method.

        Parameters:
        path: Union[str, Path]
            A file path to a CSV file or a directory containing multiple CSV files. If a directory is
            provided, all files with the ".csv" suffix are collected. If a single file is provided, it
            must have a ".csv" suffix.
        explain: bool, optional
            If set to True, logs information about the discovered CSV files during initialization.

        Raises:
        FileNotFoundError
            If the provided path is invalid, is not a CSV file, or is a directory containing no CSV files.
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
        """
        Reads a CSV file and determines the appropriate parsing function based on the file's metadata.

        This method inspects the initial lines of the CSV file to infer the kind of data it contains.
        Depending on the detected metadata, it delegates parsing to a specific handler function.
        If the file type cannot be recognized, an empty DataFrame is returned, and the issue is logged.

        Parameters:
        file_path (Path): The file path to the CSV file that needs to be parsed.

        Returns:
        pd.DataFrame: The parsed data as a pandas DataFrame. An empty DataFrame is returned if the file
                      type is unrecognized or if an error occurs during parsing.

        Raises:
        Exception: Catches and logs unexpected errors encountered while accessing or reading the file,
                   returning an empty DataFrame in such cases.
        """
        try:
            # Special fallback by filename first
            filename = file_path.stem.lower()
            if "parameters" in filename:
                return self._read_parameters_csv(file_path)
            if "patient" in filename:
                return self._read_patient_info_csv(file_path)

            # Read the first few lines manually
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = [next(f).strip().lower() for _ in range(4)]

            # Parse metadata cleanly
            if len(lines) >= 2 and "type" in lines[0] and "name" in lines[0]:
                meta_parts = lines[1].split(",")
                if len(meta_parts) >= 2:
                    metadata_type = meta_parts[0].strip('"').strip().lower()
                    metadata_name = meta_parts[1].strip('"').strip().lower()
                else:
                    metadata_type = "unknown"
                    metadata_name = "unknown"
            else:
                metadata_type = "unknown"
                metadata_name = "unknown"

            # Now decide reader
            if metadata_type == "signal_2d" and "gait line" in metadata_name:
                return self._read_gait_line_csv(file_path)
            elif metadata_type in ["signal", "signal_2d", "signal_matrix"]:
                return self._read_generic_signal_csv(file_path)

            print(f"Unrecognized file type for {file_path.name}: {metadata_type} / {metadata_name}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_generic_signal_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read and parse a generic signal CSV file, extracting both data and metadata.

        This method reads a CSV file containing signal data. The file is expected
        to have a specific format: a metadata section in the first row and
        actual data from the third row onwards. The metadata is stored as
        attributes of the returned DataFrame, including the type, name, and
        filename.

        Parameters:
            file_path (Path): Path to the CSV file to be loaded.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed signal data. Attributes
            of the DataFrame include metadata fields like 'type', 'name', and
            'filename'.

        Raises:
            None
        """
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
        """
        Reads a CSV file containing parameter values and processes its data into a pandas DataFrame.
        The function removes the 'type' column if it exists, adds metadata as attributes to the resulting
        DataFrame, and optionally prints details of the loaded data for explanation purposes.

        Arguments:
        file_path (Path): The file path to the CSV file to be read.

        Returns:
        pd.DataFrame: A DataFrame containing the processed parameter values from the CSV file. The resulting
        DataFrame also has additional attributes for 'type' and 'filename'.

        Raises:
        None explicitly, but any exceptions arising from reading the CSV file or manipulating
        the DataFrame are printed as error messages.
        """
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

            return df

        except Exception as e:
            print(f"Error reading parameters file {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_patient_info_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads a patient information CSV file into a DataFrame.

        This method processes a specified CSV file containing patient and record
        information. It adjusts the structure of the DataFrame by removing the "type"
        column if it exists and assigns custom attributes to the resulting DataFrame.
        The method can also provide additional explanations about the processed file
        if enabled.

        Parameters
        ----------
        file_path : Path
            The file path to the CSV file being read.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the processed patient and record
            information.

        Raises
        ------
        Exception
            If there is an error reading the specified CSV file, it handles the
            exception internally, prints an error message, and returns an empty
            DataFrame.
        """
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

            return df

        except Exception as e:
            print(f"Error reading patient info file {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_force_curve_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads force curve data from a CSV file.

        This method reads a force curve CSV file and loads its contents into a Pandas DataFrame.
        It uses the `_read_generic_data_file` method internally to process the file.

        Args:
            file_path (Path): The path to the CSV file containing the force curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the force curve data with columns 'time' and 'value'.
        """
        return self._read_generic_data_file(file_path, names=['time', 'value'])

    def _read_gait_line_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads and parses a CSV file containing gait line data and returns the data as a pandas DataFrame.
        This method reads metadata from the file and uses a smart detection mechanism to identify
        the appropriate header and row structure for the data. Metadata such as type and name are
        extracted from the first two lines of the file and stored as attributes in the resulting DataFrame.

        Attributes:
            explain (bool): Determines whether additional information about the parser's actions
                and the results are printed to the console.

        Parameters:
            file_path (Path): The path to the CSV file to be read.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the parsed gait line data. The DataFrame
            includes metadata attributes like `type`, `name`, and `filename` extracted from the CSV file.

        Raises:
            Exception: Catches and handles any exception that occurs during file reading or CSV parsing
            and returns an empty DataFrame in such cases.
        """
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip().lower()
                second_line = f.readline().strip().lower()
                third_line = f.readline().strip().lower()  # empty line
                fourth_line = f.readline().strip().lower()

            metadata_type = first_line.split(',')[1].strip('"') if ',' in first_line else "unknown"
            metadata_name = second_line.split(',')[1].strip('"') if ',' in second_line else "unknown"

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

            return df

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_pressure_matrix_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads a pressure matrix data file in CSV format and returns it as a DataFrame.

        This method processes a CSV file containing pressure matrix data. It generates
        column names dynamically before proceeding to read the file using a generic
        data file reader method.

        Args:
            file_path (Path): The path to the CSV file to read.

        Returns:
            pd.DataFrame: A DataFrame containing the pressure matrix data with
            dynamically generated column names.
        """
        dynamic_names = ['time'] + [f'x{i + 1}' for i in range(54)]
        return self._read_generic_data_file(file_path, names=dynamic_names)

    def _read_generic_data_file(self, file_path: Path, names: list) -> pd.DataFrame:
        """
        Reads a generic data file, extracts metadata, and loads the data into a
        pandas DataFrame with predefined column names. Supports optional output
        of metadata and DataFrame shape for debugging.

        Parameters
        ----------
        file_path : Path
            The path to the data file to be read.
        names : list
            A list of column names to assign to the loaded DataFrame.

        Returns
        -------
        pd.DataFrame
            Loaded data with its associated metadata stored in attributes. Returns
            an empty DataFrame if there is an exception during the file reading process.

        Attributes
        ----------
        type : str
            Extracted type metadata from the file, stored in the DataFrame's attributes.
        name : str
            Extracted name metadata from the file, stored in the DataFrame's attributes.
        filename : str
            The stem of the file name (without extension), stored in the DataFrame's attributes.

        Raises
        ------
        Exception
            Logs an error message if reading the file fails and returns an empty DataFrame.
        """
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                metadata_type = f.readline().strip().split(',')[1].strip('"')
                metadata_name = f.readline().strip().split(',')[1].strip('"')

            df = pd.read_csv(file_path, skiprows=2, names=names)

            df.attrs['type'] = metadata_type
            df.attrs['name'] = metadata_name
            df.attrs['filename'] = file_path.stem

            return df

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def separate_data(self, explain: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Processes raw data files to separate them into time-dependent data and single-value data. This method
        parses and categorizes a collection of CSV files based on their metadata attributes, organizing them
        into two separate groups. The method supports optional verbose output for debugging and
        error handling during processing.

        Parameters
        ----------
        explain : bool, optional
            If set to True, detailed logs are printed during processing.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary containing two keys:
            - 'time_dependent_data': DataFrame combining all time-dependent data.
            - 'single_value_data': DataFrame combining all single-value data.
        """
        time_dependent_dfs = {}
        single_value_dfs = {}
        processing_errors = []

        for file_path in self._raw_data:
            try:
                df = self._read_csv_with_metadata(file_path)

                if df.empty:
                    if explain:
                        print(f"Skipping empty DataFrame: {file_path}")
                    continue

                file_type = df.attrs.get('type', 'unknown')
                filename = df.attrs.get('filename', file_path.stem)

                # if explain:
                # print(f"Processing file: {filename} (Type: {file_type})")

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

    def force_data_as_df(self) -> pd.DataFrame:
        force_dfs = {}

        def read_force_csv(file):
            try:
                df = pd.read_csv(file, skiprows=2, names=["time", "value"])
            except Exception:
                df = pd.read_csv(file)
                df.columns = ["time", "value"]
            return df

        for file_path in self._raw_data:
            name = file_path.stem.lower()
            if "force-curve" in name:
                df = read_force_csv(file_path)

                if "backfoot" in name:
                    channel = "backfoot"
                elif "forefoot" in name:
                    channel = "forefoot"
                else:
                    channel = "foot"

                if name.endswith("-l"):
                    axis = "left"
                elif name.endswith("-r"):
                    axis = "right"
                else:
                    axis = "both"

                df = df.set_index("time")
                df.columns = pd.MultiIndex.from_tuples([(channel, axis)])
                force_dfs[(channel, axis)] = df

        if not force_dfs:
            raise ValueError("No force curve files found.")

        # Separate foot_both_df
        foot_both_df = force_dfs.pop(("foot", "both"), None)

        # Concatenate all DataFrames side by side
        merged = pd.concat(force_dfs.values(), axis=1)

        # Add foot_both_df if available
        if foot_both_df is not None:
            merged = pd.concat([merged, foot_both_df], axis=1)

        # Set MultiIndex column names
        merged.columns.names = ["channel", "axis"]

        # Sort columns for a clean layout
        if not merged.empty:
            merged = merged.sort_index(axis=1, level=["channel", "axis"], sort_remaining=True)

        # Set index name correctly
        merged.index.name = "time"

        # Drop rows where all values are NaN
        merged = merged.dropna(how="all")

        return merged

    """def gait_line_data_as_df(self, side: str = 'both') -> pd.DataFrame:
        for file_path in self._raw_data:
            name = file_path.stem.lower()

            if "gait-line" in name:
                # Side filtering
                if side == 'left' and not name.endswith("-l"):
                    continue
                if side == 'right' and not name.endswith("-r"):
                    continue

                # Read the file
                df = self._read_gait_line_csv(file_path)

                # Correct and strict source labeling
                if name == "gait-line-l":
                    gait_dfs["left"] = df
                elif name == "gait-line-r":
                    gait_dfs["right"] = df
                elif name == "gait-line":
                    gait_dfs["both"] = df

        if not gait_dfs:
            raise ValueError(f"No gait-line data found for side='{side}'.")

        # Concatenate all gait line DataFrames
        df = pd.concat(gait_dfs, axis=1)  # , ignore_index=True)
        df = df.set_index(df.columns[0])
        df.index.name = "time"
        df = df.drop(columns=[col for col in df.columns if "time" in col])
        df.columns = df.columns.set_names(["channel", "axis"])
        df = df.dropna(axis=0, how="all")
        return df

    def pressure_data_as_df(self) -> pd.DataFrame:
        df = self._processed_data['time_dependent_data']
        pressure_columns = [col for col in df.columns if col != 'time']

        if not pressure_columns:
            raise ValueError("No pressure distribution data found in the dataset.")

        return df[pressure_columns].copy()
        """

    """def patient_info_as_df(self) -> pd.DataFrame:
        df = self._processed_data['single_value_data']

        if df.empty:
            raise ValueError("No patient info found.")
        combined = df.iloc[0].combine_first(df.iloc[1])
        combined_df = pd.DataFrame([combined])

        return combined_df

    def parameters_as_df(self) -> pd.DataFrame:
        parameters_files = [f for f in self._raw_data if 'parameters' in f.name.lower()]
        if not parameters_files:
            raise ValueError("No parameters file found in the dataset.")
        return self._read_parameters_csv(parameters_files[0])
        """

    def patient_info_and_parameters_as_df(self) -> pd.DataFrame:
        """
        Creates and returns a DataFrame containing merged and translated patient information
        and corresponding parameters.

        This method processes single-value patient data and additional parameter files, merging
        them into a single DataFrame. The method translates German column headers into English
        using a predefined dictionary. It also categorizes and re-indexes the columns into a
        hierarchical multi-index structure based on attributes such as category, subgroup, side,
        and dimension. Duplicate dimensions are handled by appending a unique identifier to avoid
        conflicts.

        The method assumes the existence of previously loaded data in the `self._processed_data`
        and `self._raw_data` attributes and requires at least one parameters file that contains
        relevant data.

        Raises:
            ValueError: If either single-value patient data does not exist or no parameters
            file is available in the provided data.

        Returns:
            pd.DataFrame: A multi-indexed DataFrame containing processed patient information and
            parameters, sorted by the multi-index keys.
        """
        translation_dict = {
            "vorfuß": "forefoot",
            "rückfuß": "rearfoot",
            "kraft": "force",
            "gesamtkraft": "total force",
            "fläche der 95 vertrauensellipse": "area of 95% confidence ellipse",
            "länge der cop-spur": "cop path length",
            "gemittelte geschwindigkeit": "average velocity",
            "messdauer": "measurement duration",
            "geschwindigkeit": "velocity",
            "aufnahmedatum": "recording date",
            "typ der aufnahme": "recording type",
            "vorname": "first name",
            "nachname": "last name",
            "geburtsdatum": "date of birth",
            "geschlecht": "gender",
            "cop": "cop",
        }

        df = self._processed_data['single_value_data']
        if df.empty:
            raise ValueError("No patient info found.")

        patient_row = df.iloc[0].combine_first(df.iloc[1])
        patient_row = pd.DataFrame([patient_row])

        parameters_files = [f for f in self._raw_data if 'parameters' in f.name.lower()]
        if not parameters_files:
            raise ValueError("No parameters file found.")

        parameters_df = self._read_parameters_csv(parameters_files[0])

        merged = patient_row.copy()
        for col in parameters_df.columns:
            if col not in merged.columns:
                merged[col] = parameters_df.at[0, col]

        merged = merged.dropna(axis=1, how='all')

        multiindex_columns = []
        units_to_remove = ["mm", "mm2", "mm/s", "sek", "%", "[", "]", "/", "²"]
        seen_dimensions = {}

        for col in merged.columns:
            clean_col = col.lower()
            for unit in units_to_remove:
                clean_col = clean_col.replace(unit, "")

            clean_col = (clean_col
                         .replace(",", " ")
                         .replace("  ", " ")
                         .strip())

            side = "both"
            if "links" in clean_col:
                side = "left"
                clean_col = clean_col.replace("links", "").strip()
            elif "rechts" in clean_col:
                side = "right"
                clean_col = clean_col.replace("rechts", "").strip()

            dimension = "value"
            if " x" in clean_col:
                dimension = "x"
                clean_col = clean_col.replace("x", "").strip()
            elif " y" in clean_col:
                dimension = "y"
                clean_col = clean_col.replace("y", "").strip()

            translated = clean_col
            for german, english in translation_dict.items():
                translated = translated.replace(german, english)

            translated = " ".join(translated.split())

            if "cop" in translated.lower() or "average velocity" in translated.lower():
                category = "cop"
                if "forefoot" in translated.lower():
                    subgroup = "forefoot"
                elif "rearfoot" in translated.lower():
                    subgroup = "rearfoot"
                else:
                    subgroup = "total"
            elif "force" in translated.lower():
                category = "force"
                if "forefoot" in translated.lower():
                    subgroup = "forefoot"
                elif "rearfoot" in translated.lower():
                    subgroup = "rearfoot"
                else:
                    subgroup = "total"
            else:
                category = "measurement info"
                subgroup = ""
                dimension = translated.lower()

            dimension = " ".join(dimension.split())

            # Handle duplicate dimension names
            key = (category, subgroup, side, dimension)
            if key in seen_dimensions:
                count = seen_dimensions[key] + 1
                dimension = f"{dimension}_{count}"
                key = (category, subgroup, side, dimension)
                seen_dimensions[key] = count
            else:
                seen_dimensions[key] = 0

            multiindex_columns.append((category, subgroup, side, dimension))

        merged.columns = pd.MultiIndex.from_tuples(multiindex_columns, names=["category", "subgroup", "side", "dimension"])
        merged = merged.sort_index(axis=1)

        flattened_headers = ["-".join(map(str, col)).strip() for col in merged.columns]
        print(flattened_headers)

        return merged

    def time_dependent_data_as_df(self) -> pd.DataFrame:
        """
        Merge force-curve and gait-line data into one MultiIndex DataFrame,
        using the dataset's loaded CSV files.
        """
        dfs = []

        for file in self._raw_data:
            filename = file.stem.lower()

            # Decide type
            if "gait-line" in filename:
                type_ = "gait"
            elif "force-curve" in filename:
                type_ = "force"
            else:
                continue

            # Decide channel
            if "backfoot" in filename:
                channel = "backfoot"
            elif "forefoot" in filename:
                channel = "forefoot"
            elif "gait-line-l" in filename:
                channel = "left"
            elif "gait-line-r" in filename:
                channel = "right"
            else:
                channel = "both"

            # Decide axis
            if filename.endswith("-l"):
                axis = "left"
            elif filename.endswith("-r"):
                axis = "right"
            else:
                axis = "both"

            # Load and process the CSV
            if type_ == "gait":
                df = pd.read_csv(file, skiprows=3)
                df.columns = [col.strip().lower() for col in df.columns]

                if "time" not in df.columns:
                    raise ValueError(f"No 'time' column found in {file.name}")

                df = df.set_index("time")
                df.index = pd.to_numeric(df.index, errors="coerce")
                df = df.dropna()

                for coord in ["x", "y"]:
                    if coord in df.columns:
                        sub_df = df[[coord]].copy()
                        sub_df.columns = pd.MultiIndex.from_tuples([(type_, channel, coord)])
                        dfs.append(sub_df)
            else:  # force curve
                df = pd.read_csv(file, skiprows=2)
                df.columns = [col.strip().lower() for col in df.columns]

                if "time" not in df.columns:
                    raise ValueError(f"No 'time' column found in {file.name}")

                df = df.set_index("time")
                df.index = pd.to_numeric(df.index, errors="coerce")
                df = df.dropna()

                sub_df = df[["value"]].copy()
                sub_df.columns = pd.MultiIndex.from_tuples([(type_, channel, axis)])
                dfs.append(sub_df)

        if not dfs:
            raise ValueError("No valid gait-line or force-curve data found.")

        merged = pd.concat(dfs, axis=1)
        merged.index.name = "time"
        merged = merged.sort_index()
        merged = merged.sort_index(axis=1)

        return merged

