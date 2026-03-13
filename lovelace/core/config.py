"""Configuration loader for Lovelace."""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    """Project configuration section."""

    name: str
    java_version: int = Field(ge=8, le=21)


class ConstraintConfig(BaseModel):
    """Constraint configuration for grouping classes."""

    group: str
    classes: List[str]


class ClusteringConfig(BaseModel):
    """Clustering configuration section."""

    algorithm: str = "louvain"  # or "girvan_newman"
    resolution: float = Field(default=1.0, ge=0.1, le=10.0)  # Higher = more clusters
    weights: dict = Field(
        default_factory=lambda: {"structural": 0.5, "semantic": 0.2, "data_gravity": 0.3}
    )


class EmbeddingConfig(BaseModel):
    """Embedding configuration section."""

    model: str = "local"  # or "openai"
    openai_model: str = "text-embedding-3-small"  # if model: "openai"


class AnalysisConfig(BaseModel):
    """Analysis configuration section."""

    ignore_paths: List[str] = Field(default_factory=lambda: ["**/test/**", "**/generated/**"])
    constraints: List[ConstraintConfig] = Field(default_factory=list)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class LLMConfig(BaseModel):
    """LLM configuration section."""

    model: str = "gpt-4o"
    base_url: Optional[str] = Field(default=None, description="Custom API endpoint URL (e.g., https://api.deepseek.com)")
    cost_limit_usd: float = Field(default=5.00, ge=0.0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable name for API key")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate LLM model name (warning only for unknown models)."""
        import logging
        known_models = [
            # OpenAI
            "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo",
            # Anthropic (via OpenRouter or compatible API)
            "claude-3-5-sonnet", "claude-3-opus",
            # X.AI (Grok) - uses hyphens, not dots
            "grok-4-1", "grok-4-1-fast", "grok-4-1-fast-reasoning",
            # DeepSeek
            "deepseek-chat", "deepseek-reasoner",
        ]
        if v not in known_models:
            logging.getLogger(__name__).warning(
                f"Unknown model '{v}', pricing and context limits may be estimated"
            )
        return v


class LovelaceConfig(BaseModel):
    """Root configuration model."""

    project: ProjectConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(config_path: Optional[Path] = None) -> LovelaceConfig:
    """
    Load and validate Lovelace configuration from YAML file.

    Args:
        config_path: Path to lovelace.yaml file. If None, searches for it
                     in current directory and parent directories.

    Returns:
        Validated LovelaceConfig instance.

    Raises:
        FileNotFoundError: If config file cannot be found.
        ValueError: If config file is invalid.
    """
    if config_path is None:
        # Search for lovelace.yaml in current directory and parents
        current = Path.cwd()
        for path in [current] + list(current.parents):
            candidate = path / "lovelace.yaml"
            if candidate.exists():
                config_path = candidate
                break

        if config_path is None:
            raise FileNotFoundError(
                "Could not find lovelace.yaml in current directory or parents"
            )

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError("Configuration file is empty")

    try:
        return LovelaceConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def get_project_root(config_path: Optional[Path] = None) -> Path:
    """
    Get the project root directory (where lovelace.yaml is located).

    Args:
        config_path: Path to config file. If None, searches for it.

    Returns:
        Path to project root directory.
    """
    if config_path is None:
        current = Path.cwd()
        for path in [current] + list(current.parents):
            candidate = path / "lovelace.yaml"
            if candidate.exists():
                return path
        return Path.cwd()

    return config_path.parent

