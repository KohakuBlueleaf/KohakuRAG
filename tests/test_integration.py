"""Integration test for OpenRouter + JinaV4 pipeline.

This tests the complete integration without running full workflow.

Usage:
    python tests/test_integration.py
"""

import asyncio
import sys

from kohakurag.llm import OpenRouterChatModel
from kohakurag.embeddings import JinaV4EmbeddingModel


async def test_embedder_and_llm():
    """Test that embedder and LLM work together."""
    print("=" * 70)
    print("Integration Test: JinaV4 + OpenRouter")
    print("=" * 70)

    try:
        # Initialize components
        print("\n1. Initializing JinaV4 embedder...")
        embedder = JinaV4EmbeddingModel(truncate_dim=512, task="retrieval")
        print(f"   ✓ JinaV4 ready (dimension: {embedder.dimension})")

        print("\n2. Initializing OpenRouter LLM...")
        llm = OpenRouterChatModel(model="openai/gpt-5-nano")
        print(f"   ✓ OpenRouter ready (model: {llm._model})")

        # Test embeddings
        print("\n3. Testing embeddings...")
        texts = ["What is AI?", "Machine learning basics"]
        embeddings = await embedder.embed(texts)
        print(f"   ✓ Embedded {len(texts)} texts: shape={embeddings.shape}")

        # Test LLM
        print("\n4. Testing LLM completion...")
        response = await llm.complete("What is 2+2? Answer in one word.")
        print(f"   ✓ LLM response: {response}")

        print("\n" + "=" * 70)
        print("✓ Integration test PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)
        return False


async def test_factory_functions():
    """Test factory functions from wattbot_answer.py."""
    print("\n" + "=" * 70)
    print("Factory Functions Test")
    print("=" * 70)

    try:
        from types import SimpleNamespace
        from scripts.wattbot_answer import create_embedder, create_chat_model

        # Test embedder factory
        print("\n1. Testing create_embedder()...")

        # JinaV3
        config_v3 = SimpleNamespace(
            embedding_model="jina", embedding_dim=None, embedding_task="retrieval"
        )
        embedder_v3 = create_embedder(config_v3)
        print(f"   ✓ JinaV3 created: {type(embedder_v3).__name__}")

        # JinaV4
        config_v4 = SimpleNamespace(
            embedding_model="jinav4", embedding_dim=1024, embedding_task="retrieval"
        )
        embedder_v4 = create_embedder(config_v4)
        print(
            f"   ✓ JinaV4 created: {type(embedder_v4).__name__} (dim={embedder_v4.dimension})"
        )

        # Test LLM factory
        print("\n2. Testing create_chat_model()...")

        # OpenRouter
        config_or = SimpleNamespace(
            llm_provider="openrouter",
            model="openai/gpt-5-nano",
            openrouter_api_key=None,
            site_url=None,
            app_name=None,
            max_concurrent=10,
        )
        llm_or = create_chat_model(config_or, "You are helpful.")
        print(f"   ✓ OpenRouter created: {type(llm_or).__name__}")

        # OpenAI
        config_oai = SimpleNamespace(
            llm_provider="openai",
            model="gpt-4o-mini",
            max_concurrent=10,
        )
        llm_oai = create_chat_model(config_oai, "You are helpful.")
        print(f"   ✓ OpenAI created: {type(llm_oai).__name__}")

        print("\n" + "=" * 70)
        print("✓ Factory functions PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Factory test FAILED: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)
        return False


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("Integration Test Suite")
    print("=" * 70)
    print("Tests the complete OpenRouter + JinaV4 integration.\n")

    results = []

    # Test 1: Basic integration
    results.append(await test_embedder_and_llm())

    # Test 2: Factory functions
    results.append(await test_factory_functions())

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All integration tests passed!")
        print("You can now run the full workflow:")
        print("  python workflows/jinav4_pipeline_nocaption.py")
    else:
        print("\n✗ Some tests failed.")
        print("Fix the issues before running the full workflow.")

    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
