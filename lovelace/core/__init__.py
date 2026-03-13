"""Core engine modules for Lovelace."""

# Import order matters to avoid circular imports
# Import base modules first
from lovelace.core.config import load_config, LovelaceConfig
from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient, LLMResponse
from lovelace.core.parser import ClassMetadata, JavaParser
from lovelace.core.vector import VectorEngine

# Then modules that depend on base modules
from lovelace.core.clustering import ClusterEngine, ClusterInfo
from lovelace.core.reporter import MigrationReporter
from lovelace.core.token_budget import TokenBudget

# Analyzer imports agents, so import last
from lovelace.core.analyzer import LovelaceAnalyzer

# Pipeline orchestrator (depends on analyzer)
from lovelace.core.pipeline import run_llm_first_pipeline_v2
from lovelace.core.service_spec import ServiceSpec, ServiceResult
from lovelace.core.spec_builder import SpecBuilder

__all__ = [
    "load_config",
    "LovelaceConfig",
    "JavaParser",
    "ClassMetadata",
    "DependencyGraph",
    "LovelaceAnalyzer",
    "VectorEngine",
    "ClusterEngine",
    "ClusterInfo",
    "MigrationReporter",
    "LLMClient",
    "LLMResponse",
    "TokenBudget",
    "run_llm_first_pipeline_v2",
]
