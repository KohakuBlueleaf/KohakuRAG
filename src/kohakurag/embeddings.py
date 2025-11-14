"""Embedding utilities used across KohakuRAG."""

from typing import Any, Protocol, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency until used.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover
    from transformers import AutoModel
except ImportError:  # pragma: no cover
    AutoModel = None  # type: ignore[assignment]


class EmbeddingModel(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:  # pragma: no cover - interface only
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return a 2D numpy array of shape (len(texts), dimension)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def average_embeddings(child_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the normalized mean vector for parent nodes."""
    if not child_vectors:
        raise ValueError("average_embeddings requires at least one child vector.")
    stacked = np.vstack(child_vectors)
    return _normalize(np.mean(stacked, axis=0, keepdims=True))[0]


def _detect_device() -> Any:
    if torch is None:
        raise ImportError(
            "PyTorch is required for this embedding model. Install torch>=2.1.0."
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


class JinaEmbeddingModel:
    """Wrapper around jinaai/jina-embeddings-v3 using HuggingFace AutoModel."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        *,
        pooling: str = "cls",
        normalize: bool = True,
        batch_size: int = 8,
        device: Any | None = None,
    ) -> None:
        if AutoModel is None or torch is None:  # pragma: no cover
            raise ImportError(
                "Install torch and transformers to use the JinaEmbeddingModel."
            )
        resolved_device = _detect_device() if device is None else torch.device(device)
        self._model_name = model_name
        self._pooling = pooling
        self._normalize = normalize
        self._batch_size = batch_size
        self._device = resolved_device
        self._dtype = (
            torch.float16 if resolved_device.type in {"cuda", "mps"} else torch.float32
        )
        self._model: Any | None = None
        self._dimension: int | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        model = model.to(self._device, dtype=self._dtype)
        model.eval().requires_grad_(False)
        self._model = model
        dim = getattr(model, "embedding_size", None)
        if dim is None:
            dim = getattr(getattr(model, "config", None), "hidden_size", None)
        if dim is None:
            with torch.no_grad():
                probe = model.encode(["dimension probe"])
            if isinstance(probe, torch.Tensor):
                dim = probe.shape[-1]
            else:
                probe_arr = np.asarray(probe)
                dim = probe_arr.shape[-1]
        if dim is None:
            raise RuntimeError("Unable to infer embedding dimension for the model.")
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        str_texts = [str(t) for t in texts]
        with torch.no_grad():
            embeddings = self._model.encode(str_texts)
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)
        return arr.astype(np.float32, copy=False)
