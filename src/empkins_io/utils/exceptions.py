"""A set of custom exceptions."""

__all__ = [
    "InvalidFileFormatError",
    "SynchronizationError",
    "ValidationError",
    "TimelogNotFoundError",
    "NilsPodDataNotFoundError",
    "HeartRateDataNotFoundError",
    "NilsPodDataLoadError",
    "SamplingRateMismatchError",
    "SyncDataNotFoundError",
    "OpenposeDataNotFoundError",
    "CleanedOpenposeDataNotFoundError",
    "TimestampDataNotFoundError",
    "ZebrisDataNotFoundError",
]


class InvalidFileFormatError(Exception):
    """Exception indicating an invalid file format."""


class SynchronizationError(Exception):
    """Exception when an error occurred during synchronization."""


class ValidationError(Exception):
    """Exception when validating the provided dataframe failed."""


class TimelogNotFoundError(Exception):
    """An error indicating that no time log data are available."""


class NilsPodDataNotFoundError(Exception):
    """An error indicating that no NilsPod sensor data are available."""


class NilsPodDataLoadError(Exception):
    """An error indicating that an error occurred while attempting to load the NilsPod data."""


class HeartRateDataNotFoundError(Exception):
    """An error indicating that no processed heart rate data are available."""


class SamplingRateMismatchError(Exception):
    """An error indicating that the sampling rates of sensors do not match."""


class SyncDataNotFoundError(Exception):
    """An error indicating that no sync.json file is available."""


class OpenposeDataNotFoundError(Exception):
    """An error indicating that no openpose_output.csv file is available."""


class CleanedOpenposeDataNotFoundError(Exception):
    """An error indicating that no openpose.csv file is available."""


class TimestampDataNotFoundError(Exception):
    """An error indicating that no timestamps.csv file is available."""


class ZebrisDataNotFoundError(Exception):
    """An error indicating that no Zebris data are available."""
