"""Token budget management for LLM context windows."""

import logging
from typing import Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)

# Model context limits (conservative estimates)
MODEL_CONTEXT_LIMITS = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4": 8_192,
    "gpt-4-turbo": 128_000,
    # Anthropic
    "claude-3-5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    # X.AI (Grok) - uses hyphens, not dots
    "grok-4-1": 128_000,
    "grok-4-1-fast": 2_000_000,  # 2M tokens
    "grok-4-1-fast-reasoning": 2_000_000,  # 2M tokens
    # DeepSeek
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 128_000,
}


class TokenBudget:
    """Manage token budget for LLM calls."""

    def __init__(self, model: str = "gpt-4o", max_context: Optional[int] = None, reserve_output: int = 4_000):
        """
        Initialize token budget manager.

        Args:
            model: LLM model name (for encoding selection)
            max_context: Maximum context window size. If None, uses model default.
            reserve_output: Tokens to reserve for LLM response.
        """
        self.model = model
        self.max_context = max_context or MODEL_CONTEXT_LIMITS.get(model, 128_000)
        self.reserve_output = reserve_output

        # Get encoding for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (GPT-4 encoding)
            logger.warning(f"Unknown model {model}, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self.encoding.encode(text))

    def available_input_tokens(self) -> int:
        """
        Get available input tokens after reserving space for output.

        Returns:
            Available input tokens.
        """
        return self.max_context - self.reserve_output

    def check_fits(self, text: str, buffer_percent: float = 0.1) -> bool:
        """
        Check if text fits within available token budget.

        Args:
            text: Text to check.
            buffer_percent: Percentage of budget to keep as buffer.

        Returns:
            True if text fits, False otherwise.
        """
        tokens = self.count_tokens(text)
        max_allowed = int(self.available_input_tokens() * (1 - buffer_percent))
        return tokens <= max_allowed

    def get_usage_info(self, text: str) -> Dict[str, int]:
        """
        Get token usage information for text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with token counts and limits.
        """
        tokens = self.count_tokens(text)
        return {
            "tokens": tokens,
            "max_context": self.max_context,
            "available_input": self.available_input_tokens(),
            "reserve_output": self.reserve_output,
            "fits": tokens <= self.available_input_tokens(),
            "utilization_percent": round((tokens / self.available_input_tokens()) * 100, 1),
        }

