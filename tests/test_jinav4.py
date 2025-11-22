"""Test JinaV4EmbeddingModel to verify it works before running full workflow.

Usage:
    python tests/test_jinav4.py
"""

import asyncio
import sys

import numpy as np

from kohakurag.embeddings import JinaV4EmbeddingModel


async def test_text_embedding():
    """Test basic text embedding."""
    print("=" * 70)
    print("Test 1: Text Embedding")
    print("=" * 70)

    try:
        model = JinaV4EmbeddingModel(
            truncate_dim=1024,
            task="retrieval",
        )
        print(f"✓ Model initialized")
        print(f"✓ Dimension: {model.dimension}")

        print("\nEmbedding test texts...")
        texts = ["What is machine learning?", "How does AI work?"]
        embeddings = await model.embed(texts)

        print(f"✓ Embeddings generated: shape={embeddings.shape}")
        print(f"✓ Expected: (2, 1024)")
        print(f"✓ Dtype: {embeddings.dtype}")

        assert embeddings.shape == (2, 1024), f"Wrong shape: {embeddings.shape}"
        assert embeddings.dtype == np.float32, f"Wrong dtype: {embeddings.dtype}"

        print("✓ Test passed!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)
        return False


async def test_different_dimensions():
    """Test different Matryoshka dimensions."""
    print("\n" + "=" * 70)
    print("Test 2: Matryoshka Dimensions")
    print("=" * 70)

    results = []
    for dim in [128, 256, 512, 1024, 2048]:
        try:
            print(f"\nTesting dimension: {dim}")
            model = JinaV4EmbeddingModel(truncate_dim=dim, task="retrieval")

            embeddings = await model.embed(["test"])
            print(f"✓ {dim}D: shape={embeddings.shape}")

            assert embeddings.shape == (1, dim), f"Wrong shape for {dim}D"
            results.append(True)

        except Exception as e:
            print(f"✗ {dim}D FAILED: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    if all(results):
        print("✓ All dimensions passed!")
    else:
        print(f"✗ {sum(results)}/{len(results)} dimensions passed")
    print("=" * 70)

    return all(results)


async def test_image_embedding():
    """Test image embedding with a simple test image."""
    print("\n" + "=" * 70)
    print("Test 3: Image Embedding")
    print("=" * 70)

    try:
        from PIL import Image
        import io

        # Create a simple test image (100x100 red square)
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        print("✓ Created test image (100x100 red square)")

        model = JinaV4EmbeddingModel(truncate_dim=1024, task="retrieval")
        print("✓ Model initialized")

        print("\nEmbedding image...")
        embeddings = await model.embed_images([img_bytes])

        print(f"✓ Embeddings generated: shape={embeddings.shape}")
        print(f"✓ Expected: (1, 1024)")

        assert embeddings.shape == (1, 1024), f"Wrong shape: {embeddings.shape}"

        print("✓ Test passed!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)
        return False


async def test_batch_embedding():
    """Test batch text embedding."""
    print("\n" + "=" * 70)
    print("Test 4: Batch Text Embedding")
    print("=" * 70)

    try:
        model = JinaV4EmbeddingModel(truncate_dim=512, task="retrieval")

        texts = [f"Test sentence number {i}" for i in range(10)]
        print(f"Embedding {len(texts)} texts...")

        embeddings = await model.embed(texts)

        print(f"✓ Embeddings: shape={embeddings.shape}")
        assert embeddings.shape == (10, 512)

        print("✓ Test passed!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        print("=" * 70)
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("JinaV4 Embedding Model Test Suite")
    print("=" * 70)
    print("This will test if JinaV4 is properly working.\n")

    results = []

    # Test 1: Basic text embedding
    results.append(await test_text_embedding())

    # Test 2: Different dimensions
    results.append(await test_different_dimensions())

    # Test 3: Image embedding
    results.append(await test_image_embedding())

    # Test 4: Batch processing
    results.append(await test_batch_embedding())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! JinaV4 is working correctly.")
    else:
        print("✗ Some tests failed. Check the errors above.")
        print("\nCommon issues:")
        print("  1. Model not downloaded: Will download on first run")
        print("  2. GPU memory: Try smaller dimension (512 instead of 1024)")
        print("  3. Missing dependencies: pip install -e .")

    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
