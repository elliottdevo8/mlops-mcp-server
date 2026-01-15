"""Custom exceptions for MLOps MCP Server."""

from typing import Any


class MLOpsMCPError(Exception):
    """Base exception for MLOps MCP Server."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for MCP error response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConnectionError(MLOpsMCPError):
    """Raised when connection to MLOps platform fails."""

    def __init__(self, platform: str, message: str) -> None:
        super().__init__(
            message=f"Failed to connect to {platform}: {message}",
            details={"platform": platform},
        )


class AuthenticationError(MLOpsMCPError):
    """Raised when authentication fails."""

    def __init__(self, platform: str, message: str = "Authentication failed") -> None:
        super().__init__(
            message=f"{platform} authentication failed: {message}",
            details={"platform": platform},
        )


class ExperimentNotFoundError(MLOpsMCPError):
    """Raised when an experiment is not found."""

    def __init__(self, experiment_id: str | None = None, experiment_name: str | None = None) -> None:
        identifier = experiment_id or experiment_name or "unknown"
        super().__init__(
            message=f"Experiment not found: {identifier}",
            details={
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
            },
        )


class RunNotFoundError(MLOpsMCPError):
    """Raised when a run is not found."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            message=f"Run not found: {run_id}",
            details={"run_id": run_id},
        )


class ModelNotFoundError(MLOpsMCPError):
    """Raised when a model is not found."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"Model not found: {model_name}",
            details={"model_name": model_name},
        )
