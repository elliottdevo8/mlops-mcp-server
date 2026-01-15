# MLOps MCP Server

[![PyPI version](https://badge.fury.io/py/mlops-mcp-server.svg)](https://badge.fury.io/py/mlops-mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**AI-powered MLOps workflows through Claude Code**

An MCP (Model Context Protocol) server that enables Claude to interact with ML experiment tracking, model registries, and deployment pipelines across popular MLOps platforms.

## Features

- **MLflow Integration** - List experiments, compare runs, find best models, search with filters
- **Model Registry** - Browse registered models, track versions, check deployment stages
- **Cross-Platform** - Unified interface for MLflow, Weights & Biases, and SageMaker (coming soon)

## Quick Start

### Installation

```bash
# Install from PyPI
pip install mlops-mcp-server

# Or install with all optional dependencies
pip install mlops-mcp-server[all]
```

### Configuration

Add to your Claude Code MCP configuration (`~/.claude.json`):

```json
{
  "mcpServers": {
    "mlops": {
      "command": "mlops-mcp-server",
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `./mlruns` |
| `WANDB_API_KEY` | Weights & Biases API key | - |
| `AWS_REGION` | AWS region for SageMaker | `us-east-1` |

## Available Tools

### Experiment Tracking

| Tool | Description |
|------|-------------|
| `mlflow_list_experiments` | List all MLflow experiments |
| `mlflow_get_runs` | Get runs for an experiment with metrics |
| `mlflow_compare_runs` | Compare metrics across multiple runs |
| `mlflow_get_best_run` | Find best run by metric |
| `mlflow_search_runs` | Search runs with SQL-like filters |

### Model Registry

| Tool | Description |
|------|-------------|
| `mlflow_list_models` | List registered models |
| `mlflow_get_model_versions` | Get model version history |

## Usage Examples

### List Experiments

```
User: Show me all my MLflow experiments

Claude: [Uses mlflow_list_experiments]
Found 5 experiments:
1. fraud-detection (ID: 1) - 23 runs
2. recommendation-engine (ID: 2) - 45 runs
...
```

### Find Best Model

```
User: Which model has the highest accuracy in the fraud-detection experiment?

Claude: [Uses mlflow_get_best_run]
Best run: run_abc123
- Accuracy: 0.956
- Model: XGBoost
- Parameters: max_depth=6, learning_rate=0.1
```

### Compare Runs

```
User: Compare the last 3 runs in terms of accuracy and F1 score

Claude: [Uses mlflow_compare_runs]
| Run ID | Accuracy | F1 Score |
|--------|----------|----------|
| abc123 | 0.956    | 0.943    |
| def456 | 0.948    | 0.935    |
| ghi789 | 0.951    | 0.940    |
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/jeru/mlops-mcp-server.git
cd mlops-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Locally

```bash
# Start the server
python -m mlops_mcp.server

# Or use the CLI entry point
mlops-mcp-server
```

## Roadmap

- [x] MLflow experiment tracking
- [x] MLflow model registry
- [ ] Weights & Biases integration
- [ ] SageMaker model registry
- [ ] SageMaker endpoint management
- [ ] Model drift monitoring
- [ ] Cost analysis tools

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with the [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic.
