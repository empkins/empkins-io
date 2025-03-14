"""A set of custom exceptions."""

__all__ = [
    "InvalidFileFormatError",
    "SynchronizationError",
    "ValidationError",
    "TimelogNotFoundException",
    "NilsPodDataNotFoundException",
    "HeartRateDataNotFoundException",
]


class InvalidFileFormatError(Exception):
    """Exception indicating an invalid file format."""


class SynchronizationError(Exception):
    """Exception when an error occurred during synchronization."""


class ValidationError(Exception):
    """Exception when validating the provided dataframe failed."""


class TimelogNotFoundException(Exception):
    """An error indicating that no time log data are available."""


class NilsPodDataNotFoundException(Exception):
    """An error indicating that no NilsPod sensor data are available."""


class NilsPodDataLoadException(Exception):
    """An error indicating that an error occurred while attempting to load the NilsPod data."""


class HeartRateDataNotFoundException(Exception):
    """An error indicating that no processed heart rate data are available."""


class SamplingRateMismatchException(Exception):
    """An error indicating that no timelog file is available."""


class SyncDataNotFoundException(Exception):
    """An error indicating that no sync.json file is available."""


class OpenposeDataNotFoundException(Exception):
    """An error indicating that no openpose_output.csv file is available."""


class CleanedOpenposeDataNotFoundException(Exception):
    """An error indicating that no openpose.csv file is available."""


class TimestampDataNotFoundException(Exception):
    """An error indicating that no timestamps.csv file is available."""
