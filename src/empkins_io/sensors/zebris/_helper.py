from itertools import product
from pathlib import Path

import pandas as pd

from empkins_io.utils._types import path_t


def _read_gait_line_csv(file_path: path_t) -> pd.DataFrame:
    """
    Parses gait data from a CSV file and returns it as a pandas DataFrame. The function reads file metadata
    from the first and second lines of the file, including type and name, and validates the structure of the
    data against expected column names. Metadata is attached as attributes to the resulting DataFrame.

    Arguments:
    file_path (Path): The path to the CSV file to be read.

    Returns
    -------
    pd.DataFrame: A DataFrame object containing the parsed data from the CSV file.
    Metadata attributes 'type', 'name', and 'filename' are also attached to the DataFrame.

    Raises
    ------
    Exception: Handles any exceptions that occur while reading the file or processing its contents and
    returns an empty DataFrame in the event of an error.
    """
    file_path = Path(file_path)
    gait_line_mapping = {
        "gait-line": "both",
        "gait-line-L": "left",
        "gait-line-R": "right",
    }
    file_path = Path(file_path)
    data = pd.read_csv(file_path, skiprows=3)
    data = data.set_index("time")
    column_header = gait_line_mapping[file_path.stem]
    columns = list(product(["cop"], [column_header], ["total"], data.columns))

    # flatten nested tuples if there are any
    data.columns = pd.MultiIndex.from_tuples(columns, names=["channel", "foot", "foot_region", "axis"])

    return data


def _read_force_curve_csv(file_path: path_t) -> pd.DataFrame:
    """
    Reads a force curve data CSV file and converts it into a pandas DataFrame.

    This method specifically reads a CSV file containing force curve data,
    identified by the given file path, and maps its content into a pandas
    DataFrame with predefined column names. The method relies on a generic
    data reading helper function for file processing.

    Parameters
    ----------
        file_path (Path): The path to the CSV file containing the force
            curve data.

    Returns
    -------
        pd.DataFrame: A pandas DataFrame containing the processed force
        curve data with column names 'time' and 'value'.
    """
    file_path = Path(file_path)
    force_curve_mapping_foot = {
        "force-curve": "both",
        "force-curve-L": "left",
        "force-curve_forefoot-L": "left",
        "force-curve_backfoot-L": "left",
        "force-curve-R": "right",
        "force-curve_forefoot-R": "right",
        "force-curve_backfoot-R": "right",
    }

    force_curve_mapping_foot_region = {
        "force-curve": "total",
        "force-curve-L": "total",
        "force-curve_forefoot-L": "forefoot",
        "force-curve_backfoot-L": "backfoot",
        "force-curve-R": "total",
        "force-curve_forefoot-R": "forefoot",
        "force-curve_backfoot-R": "backfoot",
    }

    data = pd.read_csv(file_path, skiprows=3)
    data = data.set_index("time")
    foot = force_curve_mapping_foot[file_path.stem]
    foot_region = force_curve_mapping_foot_region[file_path.stem]

    data.columns = pd.MultiIndex.from_tuples(
        [("force", foot, foot_region, "z")], names=["channel", "foot", "foot_region", "axis"]
    )
    return data


def _read_pressure_matrix_csv(file_path: path_t) -> pd.DataFrame:
    """
    Reads a pressure matrix CSV file and returns its data as a pandas DataFrame.

    The file is expected to have pressure sensor readings. If a 'time' column exists
    (typically just zeros or NaNs), it will be dropped automatically.
    Column names are dynamically assigned as x1, x2, ..., xN.

    Parameters
    ----------
    file_path: Path
        The path to the CSV file containing pressure matrix data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing pressure sensor data.
    """
    file_path = Path(file_path)
    try:
        # Skip metadata
        df = pd.read_csv(file_path, skiprows=2)
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        df.columns = [f"x{i + 1}" for i in range(df.shape[1])]
        # Save metadata
        df.attrs.update({"type": "pressure matrix", "filename": file_path.stem})

        return df

    except Exception as e:
        print(f"Error reading pressure matrix {file_path.name}: {e}")
        return pd.DataFrame()


def _read_parameters_csv(file_path: path_t) -> pd.DataFrame:
    """
    Reads and processes a parameters CSV file.

    This method reads a CSV file containing parameter data, removes the 'type'
    column if it exists, and then returns the resulting DataFrame with updated
    attributes for metadata.

    Arguments:
        file_path (Path): Path to the parameters CSV file.

    Returns
    -------
        pd.DataFrame: The processed DataFrame with parameter data.

    Raises
    ------
        None
    """
    file_path = Path(file_path)
    translation_dict = {
        "l": "left",
        "r": "right",
        "links": "left",
        "rechts": "right",
        "vorfuß": "forefoot",
        "rückfuß": "backfoot",
        "kraft": "force",
        "gesamtkraft": "total force",
        "gesamt": "total",
        "fläche": "area",
        "länge": "length",
        "geschwindigkeit": "velocity",
        "messdauer": "measurement_duration_s",
    }
    data = pd.read_csv(file_path)
    data = data.set_index("type")

    multiindex_columns = []
    units_to_remove = ["mm", "mm2", "mm/s", "mm/sek", "mm²", "sek", "%", "[", "]", "/", "²"]

    data.columns = data.columns.str.lower()
    # skip the Messdauer column
    data = data.drop(columns=["messdauer [sek]", "measurement_duration"], errors="ignore")

    for col in data.columns:
        col = col.split(" ")
        col = [col.replace(",", "").replace("[", "").replace("]", "").replace("%", "") for col in col]
        # drop units
        col = [c for c in col if c not in units_to_remove]
        # drop empty strings
        col = [c for c in col if c != ""]
        # translate
        col = [translation_dict.get(c, c) for c in col]
        # handle special cases
        # average velocity
        if "velocity" in col:
            col = ("average_velocity", "both", "total", "norm_mm_s")
        # cop path length
        elif "length" in col:
            col = ("cop_path_length", "both", "total", "path_length_mm")
        # conf ellipse area
        elif "area" in col:
            col = ("conf_ellipse_area", "both", "total", "area_mm2")

        # handle cop
        if "cop" in col:
            # if "cop" is the first element, it's from the total foot
            if "cop" in col[0]:
                col = ["total", *col]
            # if the last element is NOT "left" or "right", it's from the total foot => add this as the last element
            if col[-1] not in ["left", "right"]:
                col = [*col, "both"]
            col = [col[1], col[-1], col[0], col[2]]
            col[0] = "average_cop"
        # handle force
        elif any("force" in c for c in col):
            # print(col)
            if "total" in col[0]:
                col = [c.split(" ") for c in col]
                # flatten the nested list
                col = [c for sublist in col for c in sublist]
                # swap the first and second elements
                col = [col[1], col[0]] + col[2:]

            # reorder the columns
            col = [col[0], col[-1], col[1], "z"]
            col[0] = "average_force"

        multiindex_columns.append(tuple(col))

    data.columns = pd.MultiIndex.from_tuples(multiindex_columns, names=["channel", "foot", "foot_region", "metric"])
    data = data.T
    data = data.sort_index()
    data.columns = ["value"]

    return data


def _read_patient_info_csv(file_path: path_t) -> pd.DataFrame:
    """
    Reads patient information from a CSV file and processes the data into a pandas DataFrame. This method
    utilizes an internal helper function to handle the CSV reading with specific parameters and then
    performs additional processing to clean and structure the data.

    Parameters
    ----------
    file_path (Path): The path to the CSV file containing patient information data.

    Returns
    -------
    pd.DataFrame: A DataFrame containing structured patient information data. The DataFrame includes
                  additional metadata in its `attrs` attribute, such as the type of data and the filename.
    """
    translation_dict = {
        "aufnahmedatum": "recording_date",
        "typ der aufnahme": "recording_type",
        "vorname": "first_name",
        "nachname": "last_name",
        "geburtsdatum": "date_of_birth",
        "geschlecht": "gender",
        "körpergröße [cm]": "height_cm",
        "körpergewicht [kg]": "weight_kg",
        "beinlänge links [cm]": "leg_length_left_cm",
        "beinlänge rechts [cm]": "leg_length_right_cm",
        "schuhgröße": "shoe_size",
        "schuhgröße links": "shoe_size_left",
        "schuhgröße rechts": "shoe_size_right",
        "fußlänge links": "foot_length_left_cm",
        "fußlänge rechts": "foot_length_right_cm",
    }
    file_path = Path(file_path)
    data = pd.read_csv(file_path)
    data = data.set_index("type")
    data.columns = data.columns.str.lower()
    data = data.rename(columns=translation_dict)
    data = data[translation_dict.values()]
    data = data.T
    data.columns = ["value"]
    return data


def _read_stance_average_csv(file_path: path_t) -> pd.DataFrame:
    file_path = Path(file_path)

    data = pd.read_csv(file_path, skiprows=2)
    data = data.drop(columns="time")
    data.index.name = "y"
    data.columns = list(range(data.shape[1]))
    data.columns.name = "x"

    return data
