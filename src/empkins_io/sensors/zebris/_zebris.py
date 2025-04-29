import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")


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

        The constructor verifies whether the provided path is valid. If a directory is
        provided, it searches for all CSV files within the directory. If it is a single
        file, it checks whether it is a valid CSV file. The object allows optional
        logging to indicate the number of files identified. The data are prepared for
        further processing or analysis.

        Attributes:
            path (Path): The path to the CSV file or directory containing CSV files.
            explain (bool): Determines whether to log processing details.
            _raw_data (List[Path]): A list of valid CSV file paths found in the path.
            _processed_data (Any): Placeholder for data after processing (implementation dependent).

        Parameters:
            path (Union[str, Path]): The file path or directory containing CSV files.
            explain (bool): Optional; Defaults to True. If set to True, logs details
                            about the files located and data processing.

        Raises:
            FileNotFoundError: Raised if the provided path is invalid, i.e., not pointing
                               to a valid CSV file or directory.
        """
        self.path = Path(path)
        self.explain = explain

        if self.path.is_dir():
            self._raw_data = sorted(self.path.glob("*.csv"))
        elif self.path.is_file() and self.path.suffix == ".csv":
            self._raw_data = [self.path]
        else:
            raise FileNotFoundError(f"Invalid path: '{self.path}'. Not a CSV file or directory.")

        if self.explain:
            logging.info(f"Found {len(self._raw_data)} CSV files")
            for file in self._raw_data:
                logging.info(f"  - {file.name}")

        self._processed_data = self.separate_data(explain=self.explain)

    def _read_zebris_csv(self, *args, **kwargs) -> pd.DataFrame:
        """
        Reads a Zebris CSV file and returns its content as a pandas DataFrame. If the
        file cannot be read, a warning is logged, and an empty DataFrame is returned.

        Args:
            *args: Positional arguments passed to pandas.read_csv.
            **kwargs: Keyword arguments passed to pandas.read_csv.

        Returns:
            pd.DataFrame: The content of the Zebris CSV file or an empty DataFrame if
            an error occurs.
        """
        try:
            return pd.read_csv(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Failed to read CSV: {args[0]} - {e}")
            return pd.DataFrame()

    def _read_csv_with_metadata(self, file_path: Path) -> pd.DataFrame:
        """
        Reads a CSV file with potential metadata and processes it accordingly.

        This method identifies the type of data in the CSV file based on its filename
        or the metadata present in its first few lines. Depending on the identified
        type, it delegates the processing to the appropriate handler method. If the file
        cannot be recognized or processed, it returns an empty pandas DataFrame.

        Attributes
        ----------
        None

        Parameters
        ----------
        file_path : Path
            The path to the CSV file that needs to be read and processed.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the processed data. If the file type is
            unrecognized or reading fails, an empty DataFrame is returned.

        Raises
        ------
        This method does not re-raise exceptions; instead, it logs them as warnings.
        """
        try:
            filename = file_path.stem.lower()
            if "parameters" in filename:
                return self._read_parameters_csv(file_path)
            if "patient" in filename:
                return self._read_patient_info_csv(file_path)

            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = [next(f).strip().lower() for _ in range(4)]

            if len(lines) >= 2 and "type" in lines[0] and "name" in lines[0]:
                meta_parts = lines[1].split(",")
                metadata_type = meta_parts[0].strip('"').strip().lower() if len(meta_parts) >= 2 else "unknown"
                metadata_name = meta_parts[1].strip('"').strip().lower() if len(meta_parts) >= 2 else "unknown"
            else:
                metadata_type = metadata_name = "unknown"

            if metadata_type == "signal_2d" and "gait line" in metadata_name:
                return self._read_gait_line_csv(file_path)
            elif metadata_type in ["signal", "signal_2d", "signal_matrix"]:
                return self._process_zebris_csv(file_path)

            logging.warning(f"Unrecognized file type for {file_path.name}: {metadata_type} / {metadata_name}")
            return pd.DataFrame()

        except Exception as e:
            logging.warning(f"Error reading {file_path.name}: {e}")
            return pd.DataFrame()

    def _process_zebris_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Processes a Zebris CSV file and retrieves its data and metadata.

        This method reads a CSV file, parses the primary data from it, and extracts
        metadata such as type and name based on the file's content. The extracted
        metadata is stored as attributes of the returned DataFrame.

        Parameters:
        file_path : Path
            Path to the Zebris CSV file to be processed.

        Returns:
        pd.DataFrame
            A DataFrame containing the parsed data from the file.

        Attributes:
            type : str
                Indicates the type of the Zebris data retrieved from the file, extracted
                from the metadata.
            name : str
                Describes the name of the Zebris data, extracted from the file's metadata.
            filename : str
                The stem (name without extension) of the CSV file provided in the path.
        """
        df = self._read_zebris_csv(file_path, skiprows=2)
        if not df.empty:
            meta = self._read_zebris_csv(file_path, nrows=1)
            if not meta.empty:
                df.attrs.update({
                    'type': meta.iloc[0].get('type', 'unknown'),
                    'name': meta.iloc[0].get('name', 'unknown'),
                    'filename': file_path.stem
                })
        return df

    def _read_parameters_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads and processes a parameters CSV file.

        This method reads a CSV file containing parameter data, removes the 'type'
        column if it exists, and then returns the resulting DataFrame with updated
        attributes for metadata.

        Arguments:
            file_path (Path): Path to the parameters CSV file.

        Returns:
            pd.DataFrame: The processed DataFrame with parameter data.

        Raises:
            None
        """
        df = self._read_zebris_csv(file_path, quotechar='"', skipinitialspace=True, encoding='utf-8-sig', header=0)
        if not df.empty and 'type' in df.columns:
            df = df.drop(columns=['type'])
        df.attrs.update({'type': 'parameter values', 'filename': file_path.stem})
        return df

    def _read_patient_info_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads patient information from a CSV file and processes the data into a pandas DataFrame. This method
        utilizes an internal helper function to handle the CSV reading with specific parameters and then
        performs additional processing to clean and structure the data.

        Parameters:
        file_path (Path): The path to the CSV file containing patient information data.

        Returns:
        pd.DataFrame: A DataFrame containing structured patient information data. The DataFrame includes
                      additional metadata in its `attrs` attribute, such as the type of data and the filename.
        """
        df = self._read_zebris_csv(file_path, quotechar='"', skipinitialspace=True, encoding='utf-8-sig', header=0)
        if not df.empty and 'type' in df.columns:
            df = df.iloc[:, 1:]
        df.attrs.update({'type': 'patient and record info', 'filename': file_path.stem})
        return df

    def _read_force_curve_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Reads a force curve data CSV file and converts it into a pandas DataFrame.

        This method specifically reads a CSV file containing force curve data,
        identified by the given file path, and maps its content into a pandas
        DataFrame with predefined column names. The method relies on a generic
        data reading helper function for file processing.

        Parameters:
            file_path (Path): The path to the CSV file containing the force
                curve data.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the processed force
            curve data with column names 'time' and 'value'.
        """
        return self._read_generic_data_file(file_path, names=['time', 'value'])

    def _read_gait_line_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Parses gait data from a CSV file and returns it as a pandas DataFrame. The function reads file metadata
        from the first and second lines of the file, including type and name, and validates the structure of the
        data against expected column names. Metadata is attached as attributes to the resulting DataFrame.

        Arguments:
        file_path (Path): The path to the CSV file to be read.

        Returns:
        pd.DataFrame: A DataFrame object containing the parsed data from the CSV file.
        Metadata attributes 'type', 'name', and 'filename' are also attached to the DataFrame.

        Raises:
        Exception: Handles any exceptions that occur while reading the file or processing its contents and
        returns an empty DataFrame in the event of an error.
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
        Reads a pressure matrix CSV file and returns its data as a pandas DataFrame.

        The file is expected to have pressure sensor readings. If a 'time' column exists
        (typically just zeros or NaNs), it will be dropped automatically.
        Column names are dynamically assigned as x1, x2, ..., xN.

        Parameters:
        file_path: Path
            The path to the CSV file containing pressure matrix data.

        Returns:
        pd.DataFrame
            A pandas DataFrame containing pressure sensor data.
        """
        try:
            # Skip metadata
            df = pd.read_csv(file_path, skiprows=2)
            if 'time' in df.columns:
                df = df.drop(columns=['time'])
            df.columns = [f'x{i + 1}' for i in range(df.shape[1])]
            # Save metadata
            df.attrs.update({
                'type': 'pressure matrix',
                'filename': file_path.stem
            })

            return df

        except Exception as e:
            print(f"Error reading pressure matrix {file_path.name}: {e}")
            return pd.DataFrame()

    def _read_generic_data_file(self, file_path: Path, names: list) -> pd.DataFrame:
        """
        Reads a generic data file and converts it into a pandas DataFrame with
        additional metadata attributes extracted from the first two rows of the file.

        Attributes from the file include type and name, which are stored in the
        DataFrame's attributes. The file's stem name is also added to the attributes.

        Parameters:
        file_path: Path
            The path to the data file to be read.
        names: list
            A list of column names to use for the resulting DataFrame.

        Returns:
        pd.DataFrame
            A pandas DataFrame object containing the data from the file, with the
            metadata extracted as attributes. Returns an empty DataFrame if an
            error occurs during reading.

        Raises:
        Exception
            If there is an issue while reading the file or processing its contents.
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
            if processing_errors:
                print("\n--- Processing Errors ---")
                for error in processing_errors:
                    print(error)

        return result

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
            "gesamt": "total",
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

        for col in merged.columns:
            original_col = col.lower()

            # Detect side
            if "links" in original_col or "l," in original_col or " l" in original_col:
                side = "left"
            elif "rechts" in original_col or "r," in original_col or " r" in original_col:
                side = "right"
            else:
                side = "both"

            # clean column
            clean_col = original_col
            for unit in units_to_remove:
                clean_col = clean_col.replace(unit, "")
            clean_col = clean_col.replace(",", " ").replace("  ", " ").strip()

            # Detect category
            if "cop" in clean_col:
                category = "cop"
            elif "kraft" in clean_col:
                category = "force"
            else:
                category = "measurement info"

            # Detect subgroup
            if "forefoot" in clean_col or "vorfuß" in clean_col:
                subgroup = "forefoot"
            elif "rearfoot" in clean_col or "rückfuß" in clean_col:
                subgroup = "rearfoot"
            elif "total" in clean_col or "gesamt" in clean_col:
                subgroup = "total"
            else:
                subgroup = ""

            # Translate
            translated = clean_col
            for german, english in translation_dict.items():
                translated = translated.replace(german, english)
            translated = " ".join(translated.split())

            # Detect dimension
            if category == "measurement info":
                dimension = translated.lower()
            else:
                if " x" in clean_col:
                    dimension = "x"
                elif " y" in clean_col:
                    dimension = "y"
                elif "spur" in clean_col or "trajectory" in clean_col:
                    dimension = "path length"
                elif any(k in clean_col for k in ["kraft", "force", "gesamt"]):
                    dimension = "percent"
                elif "messdauer" in clean_col:
                    dimension = "duration"
                else:
                    dimension = "value"

            multiindex_columns.append((category, subgroup, side, dimension))

        merged.columns = pd.MultiIndex.from_tuples(multiindex_columns,
                                                   names=["category", "subgroup", "side", "dimension"])
        merged = merged.sort_index(axis=1)

        return merged

    def time_dependent_data_as_df(self) -> pd.DataFrame:
        # TODO remove 0 values at the end of fore- & backfoot force data?
        """
        Generates a consolidated pandas DataFrame containing time-dependent data from raw CSV files.

        This method processes multiple raw data files, extracts relevant features, and organizes them into a formatted DataFrame.
        Each file is identified as containing either gait-line or force-curve data, and additional metadata such as channel and axis
        is determined based on the filename. Files not containing recognizable formats or required time columns are skipped or raise errors.

        Raises errors if no valid files are provided or if required columns are missing. Ensures that the final DataFrame is
        sorted by time index and column metadata.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing time-dependent data, with multi-index column labels structured as
            (type, channel, axis/coordinate). The DataFrame is indexed by the time column.

        Raises
        ------
        ValueError
            If no valid gait-line or force-curve data is found among the processed files.
        ValueError
            If the CSV file being processed does not include a 'time' column.
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

    def pressure_data_as_df(self) -> pd.DataFrame:
        """
        Reads pressure matrix data from available files and combines them into a single
        DataFrame. This method only processes files that include the term "pressure" in
        their file name (case-insensitive). The resulting data from all matching files
        is concatenated for unified access.

        Raises:
            ValueError: If no files containing pressure matrix data are found.

        Returns:
            pd.DataFrame: A combined DataFrame containing data from all applicable
            pressure matrix files.
        """
        pressure_files = [f for f in self._raw_data if "pressure" in f.name.lower()]

        if not pressure_files:
            raise ValueError("No pressure matrix files found.")

        pressure_dfs = [self._read_pressure_matrix_csv(f) for f in pressure_files]

        combined = pd.concat(pressure_dfs, ignore_index=True)
        return combined
