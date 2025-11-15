"""Chat model integrations (e.g., OpenAI)."""

import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

from .pipeline import ChatModel


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load environment variables from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip().strip('"').strip("'")

    return env_vars


class OpenAIChatModel(ChatModel):
    """Chat backend powered by OpenAI's Chat Completions API with automatic rate limit handling."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        organization: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 5,
        base_retry_delay: float = 3.0,
    ) -> None:
        """Initialize OpenAI chat model with automatic rate limit retry.

        Args:
            model: OpenAI model identifier (e.g., "gpt-4o-mini")
            api_key: OpenAI API key (reads from env if not provided)
            organization: OpenAI organization ID (optional)
            system_prompt: Default system message for all completions
            max_retries: Maximum retry attempts on rate limit errors
            base_retry_delay: Base delay for exponential backoff (seconds)
        """
        # Try multiple sources for API key
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            dotenv_vars = _load_dotenv()
            key = dotenv_vars.get("OPENAI_API_KEY")

        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChatModel.")

        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._client = OpenAI(api_key=key, organization=organization)
        self._model = model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

    def _parse_retry_after(self, error_message: str) -> float | None:
        """Extract wait time from rate limit error message.

        Handles formats like:
        - "Please try again in 23ms"
        - "Please try again in 1.5s"
        - "Please try again in 2m"
        """
        patterns = [
            (r"try again in (\d+(?:\.\d+)?)ms", 0.001),  # milliseconds
            (r"try again in (\d+(?:\.\d+)?)s", 1.0),  # seconds
            (r"try again in (\d+(?:\.\d+)?)m", 60.0),  # minutes
        ]
        for pattern, multiplier in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value * multiplier
        return None

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Execute chat completion with automatic rate limit retry.

        Uses intelligent retry strategy:
        1. Parse server-recommended delay from error message
        2. Fall back to exponential backoff if no delay specified
        3. Apply jitter to avoid thundering herd

        Returns:
            Model's text response
        """
        system = system_prompt or self._system_prompt

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                choice = response.choices[0]
                return choice.message.content or ""

            except Exception as e:
                # Check if it's a rate limit error (by type or message)
                is_rate_limit = isinstance(e, RateLimitError) or (
                    "rate" in str(e).lower() and "limit" in str(e).lower()
                )

                if not is_rate_limit or attempt >= self._max_retries:
                    raise  # Not a rate limit or exhausted retries

                # Calculate wait time: server-recommended or exponential backoff
                error_msg = str(e)
                retry_after = self._parse_retry_after(error_msg)

                if retry_after is not None:
                    # Server told us exactly how long to wait
                    wait_time = retry_after + 1  # Add 1s buffer
                else:
                    # Exponential backoff: 3s, 6s, 12s, 24s, 48s...
                    wait_time = self._base_retry_delay * (2**attempt)

                # Add jitter to prevent thundering herd (75-125% of wait_time)
                jitter_factor = random.random() * 0.5 + 0.75
                wait_time = wait_time * jitter_factor

                print(
                    f"Rate limit hit (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Waiting {wait_time:.2f}s before retry..."
                )
                time.sleep(wait_time)

        raise RuntimeError("Unexpected end of retry loop")
