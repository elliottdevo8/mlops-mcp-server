"""MLflow integration tools for MLOps MCP Server."""

import json
from datetime import datetime
from typing import Any

import structlog

from mlops_mcp.utils.auth import get_mlflow_client
from mlops_mcp.utils.errors import (
    ExperimentNotFoundError,
    ModelNotFoundError,
    RunNotFoundError,
)

logger = structlog.get_logger()


def _serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _format_result(data: Any) -> str:
    """Format result data as JSON string."""
    return json.dumps(data, default=_serialize_datetime, indent=2)


async def mlflow_list_experiments(
    view_type: str = "ACTIVE_ONLY",
    max_results: int = 100,
) -> str:
    """List all MLflow experiments.

    Args:
        view_type: Filter by lifecycle stage ('ACTIVE_ONLY', 'DELETED_ONLY', 'ALL')
        max_results: Maximum number of experiments to return

    Returns:
        JSON string with list of experiments
    """
    from mlflow.entities import ViewType

    client = get_mlflow_client()

    # Map string to ViewType enum
    view_type_map = {
        "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
        "DELETED_ONLY": ViewType.DELETED_ONLY,
        "ALL": ViewType.ALL,
    }
    view = view_type_map.get(view_type, ViewType.ACTIVE_ONLY)

    experiments = client.search_experiments(
        view_type=view,
        max_results=max_results,
    )

    result = []
    for exp in experiments:
        result.append({
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "creation_time": datetime.fromtimestamp(exp.creation_time / 1000) if exp.creation_time else None,
            "last_update_time": datetime.fromtimestamp(exp.last_update_time / 1000) if exp.last_update_time else None,
            "tags": dict(exp.tags) if exp.tags else {},
        })

    logger.info("mlflow_list_experiments", count=len(result))
    return _format_result({
        "experiments": result,
        "count": len(result),
    })


async def mlflow_get_runs(
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    max_results: int = 50,
    order_by: list[str] | None = None,
) -> str:
    """Get runs for an MLflow experiment.

    Args:
        experiment_id: The experiment ID
        experiment_name: The experiment name (alternative to experiment_id)
        max_results: Maximum number of runs to return
        order_by: List of columns to order by

    Returns:
        JSON string with list of runs and their metrics
    """
    client = get_mlflow_client()

    # Resolve experiment ID from name if needed
    if not experiment_id and experiment_name:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            raise ExperimentNotFoundError(experiment_name=experiment_name)
        experiment_id = exp.experiment_id
    elif not experiment_id:
        raise ValueError("Either experiment_id or experiment_name must be provided")

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_results,
        order_by=order_by or ["start_time DESC"],
    )

    result = []
    for run in runs:
        result.append({
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
            "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            "artifact_uri": run.info.artifact_uri,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        })

    logger.info("mlflow_get_runs", experiment_id=experiment_id, count=len(result))
    return _format_result({
        "experiment_id": experiment_id,
        "runs": result,
        "count": len(result),
    })


async def mlflow_compare_runs(
    run_ids: list[str],
    metric_names: list[str] | None = None,
) -> str:
    """Compare metrics across multiple runs.

    Args:
        run_ids: List of run IDs to compare
        metric_names: List of metric names to compare

    Returns:
        JSON string with comparison data
    """
    client = get_mlflow_client()

    runs_data = []
    all_metrics: set[str] = set()

    for run_id in run_ids:
        run = client.get_run(run_id)
        if not run:
            raise RunNotFoundError(run_id)

        metrics = dict(run.data.metrics)
        all_metrics.update(metrics.keys())

        runs_data.append({
            "run_id": run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "metrics": metrics,
            "params": dict(run.data.params),
        })

    # Filter to requested metrics if specified
    compare_metrics = metric_names or list(all_metrics)

    # Build comparison table
    comparison_table = []
    for metric in compare_metrics:
        row = {"metric": metric}
        for run in runs_data:
            row[run["run_id"]] = run["metrics"].get(metric)
        comparison_table.append(row)

    logger.info("mlflow_compare_runs", run_count=len(run_ids), metric_count=len(compare_metrics))
    return _format_result({
        "runs": runs_data,
        "comparison_table": comparison_table,
        "metrics_compared": compare_metrics,
    })


async def mlflow_get_best_run(
    metric: str,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    maximize: bool = True,
) -> str:
    """Find the best performing run based on a metric.

    Args:
        metric: The metric to optimize
        experiment_id: The experiment ID
        experiment_name: The experiment name (alternative to experiment_id)
        maximize: If True, find highest value; if False, find lowest

    Returns:
        JSON string with best run details
    """
    client = get_mlflow_client()

    # Resolve experiment ID
    if not experiment_id and experiment_name:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            raise ExperimentNotFoundError(experiment_name=experiment_name)
        experiment_id = exp.experiment_id
    elif not experiment_id:
        raise ValueError("Either experiment_id or experiment_name must be provided")

    # Search with ordering
    order = "DESC" if maximize else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"metrics.{metric} IS NOT NULL",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        return _format_result({
            "error": f"No runs found with metric '{metric}'",
            "experiment_id": experiment_id,
        })

    best_run = runs[0]
    result = {
        "run_id": best_run.info.run_id,
        "run_name": best_run.info.run_name,
        "status": best_run.info.status,
        "metric_name": metric,
        "metric_value": best_run.data.metrics.get(metric),
        "optimization": "maximize" if maximize else "minimize",
        "all_metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
        "start_time": datetime.fromtimestamp(best_run.info.start_time / 1000) if best_run.info.start_time else None,
        "artifact_uri": best_run.info.artifact_uri,
    }

    logger.info("mlflow_get_best_run", metric=metric, run_id=result["run_id"], value=result["metric_value"])
    return _format_result(result)


async def mlflow_search_runs(
    experiment_ids: list[str] | None = None,
    filter_string: str | None = None,
    max_results: int = 50,
    order_by: list[str] | None = None,
) -> str:
    """Search MLflow runs with filters.

    Args:
        experiment_ids: List of experiment IDs to search
        filter_string: SQL-like filter expression
        max_results: Maximum number of runs to return
        order_by: List of columns to order by

    Returns:
        JSON string with matching runs
    """
    client = get_mlflow_client()

    # If no experiment IDs provided, search all
    if not experiment_ids:
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string or "",
        max_results=max_results,
        order_by=order_by or ["start_time DESC"],
    )

    result = []
    for run in runs:
        result.append({
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        })

    logger.info("mlflow_search_runs", filter=filter_string, count=len(result))
    return _format_result({
        "runs": result,
        "count": len(result),
        "filter_applied": filter_string,
    })


async def mlflow_list_models(
    max_results: int = 100,
    filter_string: str | None = None,
) -> str:
    """List all registered models in Model Registry.

    Args:
        max_results: Maximum number of models to return
        filter_string: Filter expression

    Returns:
        JSON string with list of registered models
    """
    client = get_mlflow_client()

    models = client.search_registered_models(
        max_results=max_results,
        filter_string=filter_string,
    )

    result = []
    for model in models:
        versions = []
        for v in model.latest_versions or []:
            versions.append({
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id,
            })

        result.append({
            "name": model.name,
            "description": model.description,
            "creation_timestamp": datetime.fromtimestamp(model.creation_timestamp / 1000) if model.creation_timestamp else None,
            "last_updated_timestamp": datetime.fromtimestamp(model.last_updated_timestamp / 1000) if model.last_updated_timestamp else None,
            "latest_versions": versions,
            "tags": dict(model.tags) if model.tags else {},
        })

    logger.info("mlflow_list_models", count=len(result))
    return _format_result({
        "models": result,
        "count": len(result),
    })


async def mlflow_get_model_versions(
    model_name: str,
    stages: list[str] | None = None,
) -> str:
    """Get all versions of a registered model.

    Args:
        model_name: Name of the registered model
        stages: Filter by stages (e.g., ['Production', 'Staging'])

    Returns:
        JSON string with model versions
    """
    client = get_mlflow_client()

    # Get the model
    try:
        model = client.get_registered_model(model_name)
    except Exception:
        raise ModelNotFoundError(model_name) from None

    # Get all versions
    filter_str = f"name='{model_name}'"
    versions = client.search_model_versions(filter_str)

    result = []
    for v in versions:
        # Filter by stage if specified
        if stages and v.current_stage not in stages:
            continue

        result.append({
            "version": v.version,
            "name": v.name,
            "current_stage": v.current_stage,
            "status": v.status,
            "description": v.description,
            "source": v.source,
            "run_id": v.run_id,
            "creation_timestamp": datetime.fromtimestamp(v.creation_timestamp / 1000) if v.creation_timestamp else None,
            "last_updated_timestamp": datetime.fromtimestamp(v.last_updated_timestamp / 1000) if v.last_updated_timestamp else None,
            "tags": dict(v.tags) if v.tags else {},
        })

    logger.info("mlflow_get_model_versions", model_name=model_name, count=len(result))
    return _format_result({
        "model_name": model_name,
        "model_description": model.description,
        "versions": result,
        "count": len(result),
        "stages_filtered": stages,
    })
