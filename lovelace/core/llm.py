"""LLM client with cost tracking and OpenAI integration."""

import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import openai
from dotenv import load_dotenv
from openai import OpenAI

from lovelace.core.config import LLMConfig

logger = logging.getLogger(__name__)

# Model pricing (per token, as of 2024)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 5.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4": {"input": 30.0 / 1_000_000, "output": 60.0 / 1_000_000},
    "gpt-4-turbo": {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
    # Anthropic (direct or via OpenRouter)
    "claude-3-5-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-3-opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    # X.AI (Grok) - per official docs: $0.20/1M input, $0.50/1M output
    "grok-4-1": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "grok-4-1-fast": {"input": 0.20 / 1_000_000, "output": 0.50 / 1_000_000},
    "grok-4-1-fast-reasoning": {"input": 0.20 / 1_000_000, "output": 0.50 / 1_000_000},
    # DeepSeek - 2025 pricing: $0.28/1M input (cache miss), $0.42/1M output
    "deepseek-chat": {"input": 0.28 / 1_000_000, "output": 0.42 / 1_000_000},
    "deepseek-reasoner": {"input": 0.28 / 1_000_000, "output": 0.42 / 1_000_000},
}


@dataclass
class LLMResponse:
    """Response from LLM API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""

    pass


class LLMClient:
    """OpenAI-compatible API client with cost tracking."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration from lovelace.yaml
        """
        self.config = config
        self.model = config.model
        self.cost_limit = config.cost_limit_usd
        self.temperature = getattr(config, "temperature", 0.7)
        self.api_key_env = getattr(config, "api_key_env", "OPENAI_API_KEY")
        self.base_url = getattr(config, "base_url", None)

        # Try to load .env file from project root or current directory
        self._load_env_file()

        # Get API key from environment (now includes .env file)
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{self.api_key_env}'. "
                f"Please either:\n"
                f"  1. Set it: export {self.api_key_env}=your-api-key\n"
                f"  2. Create a .env file with: {self.api_key_env}=your-api-key"
            )

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        # Cost tracking
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    def _load_env_file(self) -> None:
        """Load .env file from project root or current directory."""
        # Try to find .env file in common locations
        env_paths = [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path(__file__).parent.parent.parent / ".env",  # Project root
        ]

        for env_path in env_paths:
            if env_path.exists():
                logger.debug(f"Loading .env file from {env_path}")
                load_dotenv(env_path, override=False)  # Don't override existing env vars
                return

        # If no .env file found, try loading from current directory anyway
        # (dotenv will silently fail if file doesn't exist)
        load_dotenv(override=False)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = MODEL_PRICING.get(self.model)
        if not pricing:
            logger.warning(f"Unknown pricing for model {self.model}, using gpt-4o pricing")
            pricing = MODEL_PRICING["gpt-4o"]

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    def _check_cost_limit(self, additional_cost: float) -> None:
        """
        Check if adding additional cost would exceed the limit.

        Args:
            additional_cost: Cost to add

        Raises:
            CostLimitExceeded: If cost limit would be exceeded
        """
        if self.total_cost + additional_cost > self.cost_limit:
            raise CostLimitExceeded(
                f"Cost limit exceeded: ${self.total_cost:.4f} + ${additional_cost:.4f} "
                f"> ${self.cost_limit:.2f}"
            )

    def chat(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send chat completion request with cost tracking.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Override default temperature
            max_retries: Override default max retries

        Returns:
            LLMResponse with content, tokens, and cost

        Raises:
            CostLimitExceeded: If cost limit would be exceeded
        """
        if temperature is None:
            temperature = self.temperature

        if max_retries is None:
            max_retries = self.max_retries

        # Estimate cost before making request (rough estimate: 1 token ≈ 4 chars)
        # This is approximate, but helps prevent obvious overruns
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_input_tokens = total_chars // 4
        estimated_output_tokens = 500  # Conservative estimate
        estimated_cost = self._calculate_cost(estimated_input_tokens, estimated_output_tokens)
        self._check_cost_limit(estimated_cost)

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(max_retries):
            try:
                logger.debug(f"Sending request to {self.model} (attempt {attempt + 1}/{max_retries})")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

                # Extract response data
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                # Final cost check with actual tokens
                self._check_cost_limit(cost)

                # Update tracking
                self.total_cost += cost
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.call_count += 1

                logger.debug(
                    f"LLM call completed: {input_tokens} input, {output_tokens} output tokens, "
                    f"${cost:.4f} cost (total: ${self.total_cost:.4f})"
                )

                return LLMResponse(
                    content=content,
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )

            except openai.RateLimitError as e:
                last_exception = e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)

            except openai.APIError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API error: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                # Don't retry on non-API errors
                logger.error(f"Unexpected error in LLM call: {e}")
                raise

        # If we exhausted retries, raise the last exception
        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected: retry loop completed without returning or raising")

    def get_cost_report(self) -> dict:
        """
        Get usage statistics and costs.

        Returns:
            Dictionary with cost and usage statistics
        """
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": self.call_count,
            "cost_limit_usd": self.cost_limit,
            "remaining_budget_usd": round(max(0, self.cost_limit - self.total_cost), 4),
            "model": self.model,
        }

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking counters (useful for testing)."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

