"""Chat model integrations (e.g., OpenAI)."""

import os
import re
import time
from pathlib import Path

from .pipeline import ChatModel

try:  # pragma: no cover
    from openai import OpenAI, RateLimitError
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    RateLimitError = None  # type: ignore[assignment]


def _load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    env_vars: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
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
        if OpenAI is None:  # pragma: no cover
            raise ImportError("Install openai>=1.0.0 to use OpenAIChatModel.")
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
                # Check if it's a rate limit error
                is_rate_limit = (
                    RateLimitError is not None and isinstance(e, RateLimitError)
                ) or ("rate" in str(e).lower() and "limit" in str(e).lower())

                if not is_rate_limit or attempt >= self._max_retries:
                    # Not a rate limit error or out of retries
                    raise

                # Parse the error message to get the recommended wait time
                error_msg = str(e)
                retry_after = self._parse_retry_after(error_msg)

                if retry_after is not None:
                    # Use the server-recommended wait time with a small buffer
                    wait_time = retry_after + 0.1
                else:
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s...
                    wait_time = self._base_retry_delay * (2**attempt)

                print(
                    f"Rate limit hit (attempt {attempt + 1}/{self._max_retries + 1}). "
                    f"Waiting {wait_time:.2f}s before retry..."
                )
                time.sleep(wait_time)

        # This should never be reached due to the raise in the loop
        raise RuntimeError("Unexpected end of retry loop")
