# Test Suite

Quick tests to verify components are working before running full workflows.

## Prerequisites

```bash
# Install the package
pip install -e .

# Set API key
export OPENROUTER_API_KEY="your-key-here"
```

## Tests

### 1. Test OpenRouter LLM

```bash
python tests/test_openrouter.py
```

**What it tests:**
- Basic text completion
- System prompts
- Different models (GPT5-nano, Claude)
- Authentication

**Expected output:**
```
✓ All tests passed! OpenRouter is working correctly.
```

---

### 2. Test JinaV4 Embeddings

```bash
python tests/test_jinav4.py
```

**What it tests:**
- Text embedding
- All Matryoshka dimensions (128, 256, 512, 1024, 2048)
- Image embedding
- Batch processing

**Expected output:**
```
✓ All tests passed! JinaV4 is working correctly.
```

**Note:** First run will download the model (~8GB). This may take a few minutes.

---

### 3. Test Integration

```bash
python tests/test_integration.py
```

**What it tests:**
- JinaV4 + OpenRouter together
- Factory functions
- Complete pipeline integration

**Expected output:**
```
✓ All integration tests passed!
You can now run the full workflow:
  python workflows/jinav4_pipeline_nocaption.py
```

---

## Troubleshooting

### OpenRouter Tests Fail

**Error:** `No auth credentials found`

**Solution:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
# Verify it's set
echo $OPENROUTER_API_KEY
```

---

### JinaV4 Tests Fail

**Error:** `CUDA out of memory`

**Solution:** Use smaller dimension:
```python
# In test, change:
truncate_dim=512  # Instead of 1024
```

**Error:** `Model not found`

**Solution:** First run downloads the model automatically. Wait for it to complete.

---

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'kohakurag'`

**Solution:**
```bash
pip install -e .
```

---

## Quick Test All

```bash
python tests/test_openrouter.py && \
python tests/test_jinav4.py && \
python tests/test_integration.py
```

If all pass, you're ready to run the full workflow!
