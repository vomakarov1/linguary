"""Base application layer errors."""


class ApplicationError(Exception):
    """Application use case execution error.

    Base class for use case errors: invalid user commands, illegal state transitions,
    or rejected outputs from external systems. Adapters map infrastructure failures
    to subclasses of this class at their boundaries.
    """
