# Lovelace

AI-Powered Monolith-to-Microservice Migration Framework

## Overview

Lovelace is a developer-native framework designed to assist in decomposing Java monoliths into microservices. It employs a Graph-Augmented Agentic Workflow that parses source code into a semantic graph, identifies logical boundaries using community detection algorithms, and employs Large Language Models (LLMs) to plan, document, and execute the migration.

## Installation

```bash
poetry install
```

## Usage

### CLI Runner

Run the V2 pipeline from the package CLI:

```bash
poetry run lovelace --project example-monolith --auto-approve
```

If Poetry is not installed, run the module directly:

```bash
python -m lovelace --project example-monolith --auto-approve
```

Run with explicit paths:

```bash
poetry run lovelace --source-dir /path/to/src/main/java --output /path/to/output
```

Common options:

- `--cost-limit 0.50`
- `--no-resume`
- `--clear` (alias: `--clear-checkpoints`)
- `--auto-approve`
- `--skip-build-verify`
- `--config /path/to/lovelace.yaml`
- `--visualize html` (or `--visualize png`)
- `--graph-json /path/to/dependency_graph.json`

Visualization-only command:

```bash
# Generate visualization without running the migration pipeline
poetry run lovelace visualize --project example-monolith --format html
```

```python
from lovelace.core import LovelaceAnalyzer

analyzer = LovelaceAnalyzer()
graph = analyzer.analyze(source_dir=Path("./src"))

# Export graph
analyzer.export_graph(Path("graph.json"), format="json")

# Get summary
summary = analyzer.get_summary()

# Visualize graph (programmatically)
graph.visualize(output_path="graph.png")  # Static PNG
graph.visualize_interactive(output_path="graph.html")  # Interactive HTML
```

## Visualization

Generate dependency graph visualization directly from the Lovelace CLI:

```bash
# Visualization only (no pipeline execution)
poetry run lovelace visualize --project example-monolith --format html

# Visualization only with static PNG output
poetry run lovelace visualize --project example-monolith --format png
```

Or, generate visualization as part of a full pipeline run:

```bash
# Interactive HTML visualization (recommended for large graphs)
poetry run lovelace --project example-monolith --visualize html

# Static PNG visualization
poetry run lovelace --project example-monolith --visualize png

# Optional: custom graph JSON output path
poetry run lovelace --project example-monolith --visualize html --graph-json ./artifacts/dependency_graph.json
```

The interactive HTML visualization allows you to:

- Drag and zoom nodes
- Click nodes to see details
- Filter by node type
- Explore dependencies interactively

The PNG visualization provides a static overview with color-coded node types.

## Configuration

Create a `lovelace.yaml` file in your project root:

```yaml
project:
  name: "MyMonolith"
  java_version: 11

analysis:
  ignore_paths:
    - "**/test/**"
    - "**/generated/**"

llm:
  model: "gpt-4o"
  cost_limit_usd: 5.00
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"
```

### API Key Configuration

For Phase 3 (LLM Integration), you need to provide your OpenAI API key. You can do this in two ways:

**Option 1: Environment Variable**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option 2: .env File (Recommended)**
Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your-api-key-here
```

The `.env` file will be automatically loaded if it exists. Environment variables take precedence over `.env` file values.

**Note:** The `.env` file is automatically ignored by git (via `.gitignore`), so your API key won't be committed.
