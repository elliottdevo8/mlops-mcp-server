"""MLOps MCP Server - Main server implementation."""

import asyncio
import os
from typing import Any

import structlog
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mlops_mcp.tools.mlflow_tools import (
    _format_result,
    mlflow_compare_runs,
    mlflow_get_best_run,
    mlflow_get_model_versions,
    mlflow_get_runs,
    mlflow_list_experiments,
    mlflow_list_models,
    mlflow_search_runs,
)

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer() if os.getenv("DEBUG") else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_logger().level if hasattr(structlog.get_logger(), "level") else 20
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize MCP server
server = Server("mlops-mcp-server")

# Tool definitions
TOOLS: list[Tool] = [
    Tool(
        name="mlflow_list_experiments",
        description="List all MLflow experiments. Returns experiment names, IDs, and metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "view_type": {
                    "type": "string",
                    "description": "Filter by lifecycle stage: 'ACTIVE_ONLY', 'DELETED_ONLY', or 'ALL'",
                    "enum": ["ACTIVE_ONLY", "DELETED_ONLY", "ALL"],
                    "default": "ACTIVE_ONLY",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of experiments to return",
                    "default": 100,
                },
            },
        },
    ),
    Tool(
        name="mlflow_get_runs",
        description="Get runs for an MLflow experiment with their metrics, parameters, and status.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "The experiment ID to get runs from",
                },
                "experiment_name": {
                    "type": "string",
                    "description": "The experiment name (alternative to experiment_id)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of runs to return",
                    "default": 50,
                },
                "order_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of columns to order by (e.g., ['metrics.accuracy DESC'])",
                },
            },
        },
    ),
    Tool(
        name="mlflow_compare_runs",
        description="Compare metrics across multiple MLflow runs. Useful for finding the best performing model.",
        inputSchema={
            "type": "object",
            "properties": {
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of run IDs to compare",
                },
                "metric_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metric names to compare (e.g., ['accuracy', 'loss'])",
                },
            },
            "required": ["run_ids"],
        },
    ),
    Tool(
        name="mlflow_get_best_run",
        description="Find the best performing run in an experiment based on a specific metric.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "The experiment ID to search",
                },
                "experiment_name": {
                    "type": "string",
                    "description": "The experiment name (alternative to experiment_id)",
                },
                "metric": {
                    "type": "string",
                    "description": "The metric to optimize (e.g., 'accuracy', 'loss')",
                },
                "maximize": {
                    "type": "boolean",
                    "description": "If True, find run with highest metric value; if False, find lowest",
                    "default": True,
                },
            },
            "required": ["metric"],
        },
    ),
    Tool(
        name="mlflow_search_runs",
        description="Search MLflow runs with filters. Supports SQL-like filter expressions.",
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of experiment IDs to search",
                },
                "filter_string": {
                    "type": "string",
                    "description": "SQL-like filter (e.g., \"metrics.accuracy > 0.9 AND params.model_type = 'xgboost'\")",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of runs to return",
                    "default": 50,
                },
                "order_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of columns to order by",
                },
            },
        },
    ),
    Tool(
        name="mlflow_list_models",
        description="List all registered models in the MLflow Model Registry.",
        inputSchema={
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of models to return",
                    "default": 100,
                },
                "filter_string": {
                    "type": "string",
                    "description": "Filter expression (e.g., \"name LIKE 'fraud%'\")",
                },
            },
        },
    ),
    Tool(
        name="mlflow_get_model_versions",
        description="Get all versions of a registered model with their stage and metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the registered model",
                },
                "stages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by stages (e.g., ['Production', 'Staging'])",
                },
            },
            "required": ["model_name"],
        },
    ),
]


@server.list_tools()  # type: ignore[untyped-decorator, no-untyped-call]
async def list_tools() -> list[Tool]:
    """Return list of available tools."""
    return TOOLS


@server.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info("tool_called", tool=name, arguments=arguments)

    try:
        # Route to appropriate handler
        if name == "mlflow_list_experiments":
            result = await mlflow_list_experiments(**arguments)
        elif name == "mlflow_get_runs":
            result = await mlflow_get_runs(**arguments)
        elif name == "mlflow_compare_runs":
            result = await mlflow_compare_runs(**arguments)
        elif name == "mlflow_get_best_run":
            result = await mlflow_get_best_run(**arguments)
        elif name == "mlflow_search_runs":
            result = await mlflow_search_runs(**arguments)
        elif name == "mlflow_list_models":
            result = await mlflow_list_models(**arguments)
        elif name == "mlflow_get_model_versions":
            result = await mlflow_get_model_versions(**arguments)
        else:
            result = _format_result({"error": f"Unknown tool: {name}"})

        logger.info("tool_completed", tool=name, success=True)
        return [TextContent(type="text", text=str(result))]

    except Exception as e:
        logger.error("tool_failed", tool=name, error=str(e))
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server() -> None:
    """Run the MCP server."""
    logger.info("starting_mlops_mcp_server", version="0.1.0")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for the MLOps MCP Server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
