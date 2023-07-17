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

    pass


class SynchronizationError(Exception):
    """Exception when an error occurred during synchronization."""

    pass


class ValidationError(Exception):
    """Exception when validating the provided dataframe failed."""

    pass


class TimelogNotFoundException(Exception):
    """An error indicating that no time log data are available."""

    pass


class NilsPodDataNotFoundException(Exception):
    """An error indicating that no NilsPod sensor data are available."""

    pass


class NilsPodDataLoadException(Exception):
    """An error indicating that an error occurred while attempting to load the NilsPod data."""

    pass


class HeartRateDataNotFoundException(Exception):
    """An error indicating that no processed heart rate data are available."""

    pass


class SamplingRateMismatchException(Exception):
    """An error indicating that no timelog file is available."""

    pass
