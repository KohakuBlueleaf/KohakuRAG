"""Chat model integrations (e.g., OpenAI)."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import ChatModel

try:  # pragma: no cover
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


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
    """Chat backend powered by OpenAI's Chat Completions API."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        system_prompt: str | None = None,
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

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        system = system_prompt or self._system_prompt
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        choice = response.choices[0]
        return choice.message.content or ""
