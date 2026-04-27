"""Custom exceptions with student-readable failure messages."""


class RLAgentError(Exception):
    """Base class for project-owned agent errors."""


class OptionalDependencyMissing(RLAgentError):
    """Raised when an optional training dependency is not installed."""


class IllegalActionError(RLAgentError):
    """Raised when a selected action is not legal in the current state."""


class InvalidStateError(RLAgentError):
    """Raised when a board or game state is internally inconsistent."""


class CheckpointError(RLAgentError):
    """Raised when a candidate checkpoint cannot be loaded or saved."""
