"""Embedding utilities used across KohakuRAG."""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


class EmbeddingModel(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:  # pragma: no cover - interface only
        ...

    async def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return a 2D numpy array of shape (len(texts), dimension)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length, handling zero vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    return vectors / norms


def average_embeddings(child_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the normalized mean vector for parent nodes."""
    if not child_vectors:
        raise ValueError("average_embeddings requires at least one child vector.")

    stacked = np.vstack(child_vectors)
    return _normalize(np.mean(stacked, axis=0, keepdims=True))[0]


def _detect_device() -> Any:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
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
        """Initialize Jina embedding model with lazy loading.

        Args:
            model_name: HuggingFace model identifier
            pooling: Pooling strategy (unused, kept for compatibility)
            normalize: Whether to normalize embeddings (unused, kept for compatibility)
            batch_size: Batch size for encoding (unused, kept for compatibility)
            device: Target device (auto-detected if None)
        """
        resolved_device = _detect_device() if device is None else torch.device(device)

        self._model_name = model_name
        self._pooling = pooling
        self._normalize = normalize
        self._batch_size = batch_size
        self._device = resolved_device

        # Use FP16 on GPU for 2x speedup
        self._dtype = (
            torch.float16 if resolved_device.type in {"cuda", "mps"} else torch.float32
        )

        # Lazy initialization - model loaded on first use
        self._model: Any | None = None
        self._dimension: int | None = None

        # Single-worker executor for thread-safe async embedding
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        # Load model from HuggingFace
        model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        model = model.to(self._device, dtype=self._dtype)
        model.eval().requires_grad_(False)
        self._model = model

        # Infer embedding dimension from model attributes
        dim = getattr(model, "embedding_size", None)
        if dim is None:
            dim = getattr(getattr(model, "config", None), "hidden_size", None)

        # Fall back to probing with a test input
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

    def _sync_encode(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronous encoding logic (called via executor).

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Ensure all inputs are strings
        str_texts = [str(t) for t in texts]

        # Run inference without gradients
        with torch.no_grad():
            embeddings = self._model.encode(str_texts)

        # Convert to numpy, handling both Tensor and array outputs
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        return arr.astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors (async).

        Uses single-worker executor to ensure thread safety for GPU operations.

        Returns:
            Array of shape (len(texts), dimension) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_encode, texts)


class JinaV4EmbeddingModel:
    """Wrapper around jinaai/jina-embeddings-v4 with multimodal support.

    Supports:
    - Text embedding with configurable Matryoshka dimensions
    - Image embedding (unified embedding space with text)
    - Task-specific adapters (retrieval, text-matching, code)
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        *,
        task: str = "retrieval",
        truncate_dim: int = 1024,
        device: Any | None = None,
    ) -> None:
        """Initialize Jina V4 embedding model with lazy loading.

        Args:
            model_name: HuggingFace model identifier
            task: Task mode - "retrieval", "text-matching", or "code"
            truncate_dim: Matryoshka dimension (128, 256, 512, 1024, 2048)
            device: Target device (auto-detected if None)
        """
        # Validate truncate_dim
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            raise ValueError(
                f"truncate_dim must be one of {valid_dims}, got {truncate_dim}"
            )

        resolved_device = _detect_device() if device is None else torch.device(device)

        self._model_name = model_name
        self._task = task
        self._truncate_dim = truncate_dim
        self._device = resolved_device

        # Use FP16 on GPU for faster inference
        self._dtype = (
            torch.float16 if resolved_device.type in {"cuda", "mps"} else torch.float32
        )

        # Lazy initialization
        self._model: Any | None = None

        # Single-worker executor for thread-safe async operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        # Load JinaV4 model
        model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        model = model.to(self._device, dtype=self._dtype)
        model.eval().requires_grad_(False)
        self._model = model

    @property
    def dimension(self) -> int:
        """Return the configured Matryoshka dimension."""
        return self._truncate_dim

    def _sync_encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Synchronous text encoding (called via executor).

        Returns:
            Array of shape (len(texts), truncate_dim) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not texts:
            return np.zeros((0, self._truncate_dim), dtype=np.float32)

        # Ensure all inputs are strings
        str_texts = [str(t) for t in texts]

        # Run inference with JinaV4's encode_text method
        with torch.no_grad():
            embeddings = self._model.encode_text(
                texts=str_texts,
                task=self._task,
                prompt_name="query",  # Use "document" for documents
                truncate_dim=self._truncate_dim,
                max_length=8192,
            )

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        elif isinstance(embeddings, list):
            arr = torch.stack(embeddings).detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        return arr.astype(np.float32, copy=False)

    def _sync_encode_images(self, image_bytes_list: Sequence[bytes]) -> np.ndarray:
        """Synchronous image encoding (called via executor).

        Args:
            image_bytes_list: List of image data as bytes

        Returns:
            Array of shape (len(images), truncate_dim) with float32 dtype
        """
        self._ensure_model()
        assert self._model is not None

        if not image_bytes_list:
            return np.zeros((0, self._truncate_dim), dtype=np.float32)

        # Convert bytes to PIL Images
        pil_images = []
        for img_bytes in image_bytes_list:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}")

        # Run inference with JinaV4's encode_image method
        with torch.no_grad():
            embeddings = self._model.encode_image(
                images=pil_images,
                task=self._task,
                truncate_dim=self._truncate_dim,
            )

        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            arr = embeddings.detach().float().cpu().numpy()
        elif isinstance(embeddings, list):
            arr = torch.stack(embeddings).detach().float().cpu().numpy()
        else:
            arr = np.asarray(embeddings)

        return arr.astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into embedding vectors (async).

        Args:
            texts: List of text strings to embed

        Returns:
            Array of shape (len(texts), truncate_dim) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_encode_text, texts)

    async def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        """Encode images into embedding vectors (async).

        Args:
            images: List of image data as bytes

        Returns:
            Array of shape (len(images), truncate_dim) with float32 dtype
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_encode_images, images
        )
