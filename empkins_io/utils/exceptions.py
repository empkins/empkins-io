"""A set of custom exceptions."""


class InvalidFileFormatError(Exception):
    """Exception indicating an invalid file format."""

    pass


class SynchronizationError(Exception):
    """Exception when an error occurred during synchronization."""

    pass


class ValidationError(Exception):
    """Exception when validating the provided dataframe failed."""

    pass
