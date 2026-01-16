"""Pydantic schemas for MLOps MCP Server."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ExperimentInfo(BaseModel):
    """MLflow experiment information."""

    experiment_id: str
    name: str
    artifact_location: str | None = None
    lifecycle_stage: str = "active"
    tags: dict[str, str] = Field(default_factory=dict)
    creation_time: datetime | None = None
    last_update_time: datetime | None = None


class RunInfo(BaseModel):
    """MLflow run information."""

    run_id: str
    experiment_id: str
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    artifact_uri: str | None = None
    lifecycle_stage: str = "active"


class RunMetrics(BaseModel):
    """Metrics from an ML run."""

    run_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class RunComparison(BaseModel):
    """Comparison of multiple runs."""

    runs: list[RunMetrics]
    metric_names: list[str]
    best_run_id: str | None = None
    best_metric_value: float | None = None
    comparison_metric: str | None = None


class ModelVersion(BaseModel):
    """Registered model version information."""

    name: str
    version: str
    creation_timestamp: datetime | None = None
    last_updated_timestamp: datetime | None = None
    current_stage: str = "None"
    description: str | None = None
    source: str | None = None
    run_id: str | None = None
    status: str = "READY"
    tags: dict[str, str] = Field(default_factory=dict)


class RegisteredModel(BaseModel):
    """Registered model information."""

    name: str
    creation_timestamp: datetime | None = None
    last_updated_timestamp: datetime | None = None
    description: str | None = None
    latest_versions: list[ModelVersion] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Standard result wrapper for tool responses."""

    success: bool
    data: Any = None
    error: str | None = None
    message: str | None = None
