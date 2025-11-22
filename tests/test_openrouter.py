"""Test OpenRouterChatModel to verify it works before running full workflow.

Usage:
    python tests/test_openrouter.py
"""

import asyncio
import sys

from kohakurag.llm import OpenRouterChatModel


async def test_text_completion():
    """Test basic text completion."""
    print("=" * 70)
    print("Test 1: Text Completion")
    print("=" * 70)

    try:
        model = OpenRouterChatModel(
            model="openai/gpt-5-nano",
            max_concurrent=1,
        )
        print(f"✓ Model initialized: {model._model}")
        print(f"✓ API key loaded: {model._api_key[:10]}...")

        print("\nSending request: 'What is 2+2?'")
        response = await model.complete("What is 2+2?")

        print(f"✓ Response received: {response}")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        print("=" * 70)
        return False


async def test_with_system_prompt():
    """Test with system prompt."""
    print("\n" + "=" * 70)
    print("Test 2: With System Prompt")
    print("=" * 70)

    try:
        model = OpenRouterChatModel(
            model="openai/gpt-5-nano",
            system_prompt="You are a helpful math tutor.",
        )

        print("Sending request with system prompt...")
        response = await model.complete("Explain what addition means")

        print(f"✓ Response: {response[:100]}...")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        print("=" * 70)
        return False


async def test_different_model():
    """Test with different model."""
    print("\n" + "=" * 70)
    print("Test 3: Different Model (Claude)")
    print("=" * 70)

    try:
        model = OpenRouterChatModel(
            model="anthropic/claude-3.5-haiku",
        )

        print(f"✓ Model: {model._model}")
        print("Sending request...")
        response = await model.complete("Say hello in one sentence.")

        print(f"✓ Response: {response}")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        print("=" * 70)
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("OpenRouter SDK Test Suite")
    print("=" * 70)
    print("This will test if OpenRouter is properly configured.\n")

    results = []

    # Test 1: Basic completion
    results.append(await test_text_completion())

    # Test 2: System prompt
    results.append(await test_with_system_prompt())

    # Test 3: Different model
    results.append(await test_different_model())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! OpenRouter is working correctly.")
    else:
        print("✗ Some tests failed. Check the errors above.")
        print("\nCommon issues:")
        print("  1. Missing API key: export OPENROUTER_API_KEY='your-key'")
        print("  2. SDK not installed: pip install openrouter")
        print("  3. Invalid model name")

    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
