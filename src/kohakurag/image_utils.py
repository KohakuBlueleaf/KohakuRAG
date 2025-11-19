"""Utilities for image processing and compression."""

import io
from typing import Literal

from PIL import Image


def get_image_format(data: bytes) -> str | None:
    """Detect image format from bytes.

    Args:
        data: Raw image bytes

    Returns:
        Image format (e.g., 'JPEG', 'PNG', 'WEBP') or None if not detected
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.format
    except Exception:
        return None


def compress_image(
    data: bytes,
    *,
    max_size: int = 1024,
    format: Literal["webp", "jpeg", "png"] = "webp",
    quality: int = 95,
) -> bytes:
    """Compress and resize image to reduce storage size.

    Args:
        data: Raw image bytes
        max_size: Maximum dimension for shorter edge (default: 1024px)
        format: Output format (default: 'webp')
        quality: Compression quality 1-100 (default: 95)

    Returns:
        Compressed image bytes

    Raises:
        ValueError: If image cannot be processed
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(data))

        # Convert to RGB if necessary (handles RGBA, CMYK, etc.)
        if img.mode not in ("RGB", "L"):
            if img.mode == "RGBA" and format == "webp":
                # WebP supports transparency
                pass
            else:
                img = img.convert("RGB")

        # Resize if needed (maintain aspect ratio)
        width, height = img.size
        if width > max_size or height > max_size:
            # Find shorter edge
            if width < height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Compress and save to bytes
        output = io.BytesIO()
        save_kwargs = {"format": format.upper(), "quality": quality}

        # WebP-specific options
        if format == "webp":
            save_kwargs["method"] = 6  # Best quality/compression balance

        img.save(output, **save_kwargs)
        return output.getvalue()

    except Exception as e:
        raise ValueError(f"Failed to process image: {e}") from e


def get_image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Get image dimensions without full decoding.

    Args:
        data: Raw image bytes

    Returns:
        (width, height) tuple or None if cannot determine
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:
        return None
