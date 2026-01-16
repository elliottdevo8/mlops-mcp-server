"""Authentication and client utilities for MLOps platforms."""

import os
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger()


def get_tracking_uri() -> str:
    """Get MLflow tracking URI from environment or default.

    Priority:
    1. MLFLOW_TRACKING_URI environment variable
    2. Default to local ./mlruns directory
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    logger.debug("mlflow_tracking_uri", uri=uri)
    return uri


@lru_cache(maxsize=1)
def get_mlflow_client() -> Any:
    """Get cached MLflow tracking client.

    Returns:
        MlflowClient instance configured with tracking URI.

    Raises:
        ConnectionError: If MLflow connection fails.
    """
    from mlflow.tracking import MlflowClient

    from mlops_mcp.utils.errors import ConnectionError

    tracking_uri = get_tracking_uri()

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        logger.info("mlflow_client_initialized", tracking_uri=tracking_uri)
        return client
    except Exception as e:
        logger.error("mlflow_connection_failed", error=str(e))
        raise ConnectionError("MLflow", str(e)) from e


def get_wandb_api() -> Any:
    """Get Weights & Biases API client.

    Requires WANDB_API_KEY environment variable.

    Returns:
        wandb.Api instance.

    Raises:
        AuthenticationError: If WANDB_API_KEY is not set.
        ConnectionError: If W&B connection fails.
    """
    from mlops_mcp.utils.errors import AuthenticationError, ConnectionError

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise AuthenticationError("Weights & Biases", "WANDB_API_KEY not set")

    try:
        import wandb  # type: ignore[import-not-found]
        api = wandb.Api()
        logger.info("wandb_api_initialized")
        return api
    except ImportError:
        raise ConnectionError("Weights & Biases", "wandb package not installed") from None
    except Exception as e:
        logger.error("wandb_connection_failed", error=str(e))
        raise ConnectionError("Weights & Biases", str(e)) from e


def get_sagemaker_client() -> Any:
    """Get AWS SageMaker client.

    Requires AWS credentials configured via:
    - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or
    - AWS credentials file, or
    - IAM role (when running on AWS)

    Returns:
        boto3 SageMaker client.

    Raises:
        ConnectionError: If SageMaker connection fails.
    """
    from mlops_mcp.utils.errors import ConnectionError

    try:
        import boto3  # type: ignore[import-not-found]
        client = boto3.client("sagemaker")
        logger.info("sagemaker_client_initialized")
        return client
    except ImportError:
        raise ConnectionError("SageMaker", "boto3 package not installed") from None
    except Exception as e:
        logger.error("sagemaker_connection_failed", error=str(e))
        raise ConnectionError("SageMaker", str(e)) from e
