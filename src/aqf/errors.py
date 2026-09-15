"""Exception hierarchy.

Every failure raised by library code derives from :class:`AqfError`, so callers
can tell an expected domain failure from a genuine bug.
"""

from __future__ import annotations


class AqfError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(AqfError):
    """Configuration is missing, malformed or internally inconsistent."""


class DataError(AqfError):
    """A dataset could not be loaded, or does not have the expected shape."""


class DataQualityError(AqfError):
    """The data is present but not fit to model.

    Raised where continuing would produce a number that looks like a result and
    is not - too little of a column observed, a target that is mostly
    interpolation, a split with no data in it.
    """


class ModelError(AqfError):
    """A model could not be built, trained or used to predict."""


class EvaluationError(AqfError):
    """An evaluation was given inconsistent inputs."""
