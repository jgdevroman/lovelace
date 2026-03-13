"""Base agent class for all Lovelace agents."""

from typing import Any, Dict

from lovelace.core.graph import DependencyGraph
from lovelace.core.llm import LLMClient


class BaseAgent:
    """
    Base class for all agents (future LangGraph-compatible).

    This base class provides a common interface for all agents and is designed
    to be easily migrated to LangGraph nodes in the future.
    """

    def __init__(self, llm_client: LLMClient, graph: DependencyGraph):
        """
        Initialize the base agent.

        Args:
            llm_client: LLM client for making API calls
            graph: Dependency graph from Phase 1 analysis
        """
        self.llm = llm_client
        self.graph = graph

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent workflow. Override in subclasses.

        Args:
            input_data: Input data for the agent

        Returns:
            Output data from the agent

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement run() method")

