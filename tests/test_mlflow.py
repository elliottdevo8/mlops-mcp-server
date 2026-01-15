"""Tests for MLflow tools."""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class MockExperiment:
    """Mock MLflow experiment."""

    def __init__(self, experiment_id: str, name: str):
        self.experiment_id = experiment_id
        self.name = name
        self.artifact_location = f"./mlruns/{experiment_id}"
        self.lifecycle_stage = "active"
        self.creation_time = 1704067200000  # 2024-01-01
        self.last_update_time = 1704153600000
        self.tags = {}


class MockRunInfo:
    """Mock MLflow run info."""

    def __init__(self, run_id: str, experiment_id: str):
        self.run_id = run_id
        self.run_name = f"run-{run_id[:8]}"
        self.experiment_id = experiment_id
        self.status = "FINISHED"
        self.start_time = 1704067200000
        self.end_time = 1704070800000
        self.artifact_uri = f"./mlruns/{experiment_id}/{run_id}/artifacts"


class MockRunData:
    """Mock MLflow run data."""

    def __init__(self, metrics: dict, params: dict):
        self.metrics = metrics
        self.params = params
        self.tags = {"mlflow.user": "test"}


class MockRun:
    """Mock MLflow run."""

    def __init__(self, run_id: str, experiment_id: str, metrics: dict, params: dict):
        self.info = MockRunInfo(run_id, experiment_id)
        self.data = MockRunData(metrics, params)


class MockRegisteredModel:
    """Mock MLflow registered model."""

    def __init__(self, name: str):
        self.name = name
        self.description = f"Model {name}"
        self.creation_timestamp = 1704067200000
        self.last_updated_timestamp = 1704153600000
        self.latest_versions = []
        self.tags = {}


class MockModelVersion:
    """Mock MLflow model version."""

    def __init__(self, name: str, version: str, stage: str):
        self.name = name
        self.version = version
        self.current_stage = stage
        self.status = "READY"
        self.description = ""
        self.source = f"runs:/{name}/model"
        self.run_id = "abc123"
        self.creation_timestamp = 1704067200000
        self.last_updated_timestamp = 1704153600000
        self.tags = {}


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client."""
    # Clear the lru_cache before each test
    from mlops_mcp.utils.auth import get_mlflow_client
    get_mlflow_client.cache_clear()

    with patch("mlops_mcp.tools.mlflow_tools.get_mlflow_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.mark.asyncio
async def test_mlflow_list_experiments(mock_mlflow_client):
    """Test listing experiments."""
    from mlops_mcp.tools.mlflow_tools import mlflow_list_experiments

    # Setup mock
    mock_mlflow_client.search_experiments.return_value = [
        MockExperiment("1", "fraud-detection"),
        MockExperiment("2", "recommendation"),
    ]

    # Execute
    result = await mlflow_list_experiments()
    data = json.loads(result)

    # Verify
    assert data["count"] == 2
    assert len(data["experiments"]) == 2
    assert data["experiments"][0]["name"] == "fraud-detection"
    assert data["experiments"][1]["name"] == "recommendation"


@pytest.mark.asyncio
async def test_mlflow_get_runs(mock_mlflow_client):
    """Test getting runs for an experiment."""
    from mlops_mcp.tools.mlflow_tools import mlflow_get_runs

    # Setup mock
    mock_mlflow_client.search_runs.return_value = [
        MockRun("run1", "1", {"accuracy": 0.95}, {"model": "xgboost"}),
        MockRun("run2", "1", {"accuracy": 0.92}, {"model": "random_forest"}),
    ]

    # Execute
    result = await mlflow_get_runs(experiment_id="1")
    data = json.loads(result)

    # Verify
    assert data["count"] == 2
    assert data["experiment_id"] == "1"
    assert data["runs"][0]["metrics"]["accuracy"] == 0.95
    assert data["runs"][1]["params"]["model"] == "random_forest"


@pytest.mark.asyncio
async def test_mlflow_compare_runs(mock_mlflow_client):
    """Test comparing multiple runs."""
    from mlops_mcp.tools.mlflow_tools import mlflow_compare_runs

    # Setup mock
    mock_mlflow_client.get_run.side_effect = [
        MockRun("run1", "1", {"accuracy": 0.95, "f1": 0.93}, {}),
        MockRun("run2", "1", {"accuracy": 0.92, "f1": 0.90}, {}),
    ]

    # Execute
    result = await mlflow_compare_runs(
        run_ids=["run1", "run2"],
        metric_names=["accuracy", "f1"]
    )
    data = json.loads(result)

    # Verify
    assert len(data["runs"]) == 2
    assert len(data["comparison_table"]) == 2
    assert data["metrics_compared"] == ["accuracy", "f1"]


@pytest.mark.asyncio
async def test_mlflow_get_best_run(mock_mlflow_client):
    """Test finding best run by metric."""
    from mlops_mcp.tools.mlflow_tools import mlflow_get_best_run

    # Setup mock
    best_run = MockRun("best_run", "1", {"accuracy": 0.98}, {"model": "xgboost"})
    mock_mlflow_client.search_runs.return_value = [best_run]

    # Execute
    result = await mlflow_get_best_run(
        experiment_id="1",
        metric="accuracy",
        maximize=True
    )
    data = json.loads(result)

    # Verify
    assert data["run_id"] == "best_run"
    assert data["metric_name"] == "accuracy"
    assert data["metric_value"] == 0.98
    assert data["optimization"] == "maximize"


@pytest.mark.asyncio
async def test_mlflow_search_runs(mock_mlflow_client):
    """Test searching runs with filter."""
    from mlops_mcp.tools.mlflow_tools import mlflow_search_runs

    # Setup mock - need experiments for the search
    mock_mlflow_client.search_experiments.return_value = [
        MockExperiment("1", "test-exp"),
    ]
    mock_mlflow_client.search_runs.return_value = [
        MockRun("run1", "1", {"accuracy": 0.95}, {"model": "xgboost"}),
    ]

    # Execute
    result = await mlflow_search_runs(
        filter_string="metrics.accuracy > 0.9"
    )
    data = json.loads(result)

    # Verify
    assert data["count"] == 1
    assert data["filter_applied"] == "metrics.accuracy > 0.9"


@pytest.mark.asyncio
async def test_mlflow_list_models(mock_mlflow_client):
    """Test listing registered models."""
    from mlops_mcp.tools.mlflow_tools import mlflow_list_models

    # Setup mock
    mock_mlflow_client.search_registered_models.return_value = [
        MockRegisteredModel("fraud-model"),
        MockRegisteredModel("recommendation-model"),
    ]

    # Execute
    result = await mlflow_list_models()
    data = json.loads(result)

    # Verify
    assert data["count"] == 2
    assert data["models"][0]["name"] == "fraud-model"
    assert data["models"][1]["name"] == "recommendation-model"


@pytest.mark.asyncio
async def test_mlflow_get_model_versions(mock_mlflow_client):
    """Test getting model versions."""
    from mlops_mcp.tools.mlflow_tools import mlflow_get_model_versions

    # Setup mock
    mock_model = MockRegisteredModel("fraud-model")
    mock_mlflow_client.get_registered_model.return_value = mock_model
    mock_mlflow_client.search_model_versions.return_value = [
        MockModelVersion("fraud-model", "1", "Archived"),
        MockModelVersion("fraud-model", "2", "Staging"),
        MockModelVersion("fraud-model", "3", "Production"),
    ]

    # Execute
    result = await mlflow_get_model_versions(model_name="fraud-model")
    data = json.loads(result)

    # Verify
    assert data["model_name"] == "fraud-model"
    assert data["count"] == 3
    assert data["versions"][2]["current_stage"] == "Production"


@pytest.mark.asyncio
async def test_mlflow_get_model_versions_filtered(mock_mlflow_client):
    """Test getting model versions with stage filter."""
    from mlops_mcp.tools.mlflow_tools import mlflow_get_model_versions

    # Setup mock
    mock_model = MockRegisteredModel("fraud-model")
    mock_mlflow_client.get_registered_model.return_value = mock_model
    mock_mlflow_client.search_model_versions.return_value = [
        MockModelVersion("fraud-model", "1", "Archived"),
        MockModelVersion("fraud-model", "2", "Staging"),
        MockModelVersion("fraud-model", "3", "Production"),
    ]

    # Execute with filter
    result = await mlflow_get_model_versions(
        model_name="fraud-model",
        stages=["Production", "Staging"]
    )
    data = json.loads(result)

    # Verify - should only have Production and Staging versions
    assert data["count"] == 2
    stages = [v["current_stage"] for v in data["versions"]]
    assert "Archived" not in stages
    assert "Production" in stages
    assert "Staging" in stages
