# LLaVA Local Model Setup Guide

SceneScriber AI now supports **LLaVA** for offline, privacy-focused video scene descriptions. No API keys needed!

## What is LLaVA?

LLaVA (Large Language and Vision Assistant) is an open-source multimodal model that can analyze images and generate descriptions locally on your machine. It's perfect for:
- Privacy: No data sent to external APIs
- Cost: Free - no API charges
- Speed: Runs offline after initial setup
- Flexibility: Works without internet connection

## System Requirements

- **GPU recommended** (NVIDIA with CUDA 11.8+ or compatible GPU)
- **RAM**: Minimum 8GB (16GB+ recommended for smooth operation)
- **Disk space**: 15-20GB for model weights
- **Python**: 3.9 to 3.13 (3.12+ works fine with latest transformers)

## Installation Steps

### 1. Create a Python Virtual Environment (Optional but Recommended)

```bash
# Using Python 3.13 (or 3.12, 3.10, etc.)
python3.13 -m venv llava_env
source llava_env/bin/activate  # On Windows: llava_env\Scripts\activate

# Or use default python3
python3 -m venv llava_env
source llava_env/bin/activate
```

### 2. Install LLaVA Dependencies

**Option A: GPU Support (NVIDIA - Recommended)**

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and dependencies
pip install transformers>=4.36.0 pillow
```

**Option B: CPU Only (Mac, Linux, Windows without GPU)**

```bash
pip install torch torchvision torchaudio
pip install transformers>=4.36.0 pillow
```

**Option C: Apple Silicon (M1/M2/M3)**

```bash
# Install with Metal acceleration
pip install torch torchvision torchaudio
pip install transformers>=4.36.0 pillow
```

**Verify Installation:**
```bash
python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; print('✓ LLaVA ready!')"
```

### 3. First-Time Model Download

When you first select LLaVA in SceneScriber AI:
1. The model (~9-10GB) will be automatically downloaded from Hugging Face
2. This happens in the background on first use
3. Subsequent analyses will be much faster
4. Models are cached in: `~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/`

**Expected download time**: 5-15 minutes depending on internet speed

**To clear the cache and free disk space:**
```bash
rm -rf ~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/
```

### 4. Select LLaVA in SceneScriber AI

1. Upload a video
2. Go to "Configure Analysis Settings"
3. Under "AI Configuration", select: **"Local LLaVA (Privacy-focused)"**
4. Click "Start Analysis"
5. Wait for first-time model download (if needed)

## Performance Tips

### Optimize Memory Usage

```python
# Reduce precision to save memory (less accurate but faster)
torch_dtype = torch.float16  # Already configured in code
```

### Enable GPU Acceleration

LLaVA automatically detects your GPU:
- **NVIDIA GPU**: Uses CUDA (fastest)
- **Apple Silicon**: Uses Metal Performance Shaders
- **CPU only**: Falls back to CPU (slower)

Check GPU usage:
```bash
nvidia-smi  # For NVIDIA GPUs
```

### Batch Processing

For multiple videos, LLaVA gets faster after the first analysis since the model stays in memory.

## Troubleshooting

### Issue: "LLaVA dependencies not installed"

**Solution**: Run the installation commands above

```bash
pip install torch transformers llava-rlhf
```

### Issue: Out of Memory Error

**Solution**: 
- Use GPU instead of CPU
- Reduce batch size (already optimized in code)
- Close other applications
- Use `torch.float16` instead of `float32`

### Issue: Model download stuck

**Solution**:
- Check internet connection
- Manually download from: https://huggingface.co/liuhaotian/llava-v1.5-7b-hf
- Check available disk space (needs ~20GB free)

### Issue: Slow inference on CPU

**Expected behavior**: CPU inference is 10-50x slower than GPU. Consider:
- Using GPU if available
- Using OpenAI/Claude for time-critical projects
- Batch processing at off-peak hours

## Model Variants

Current implementation uses: **LLaVA v1.5 7B**

Other variants available (not yet integrated):
- **7B** (current): Fast, 7-15 seconds per scene
- **13B**: Slower, more accurate (requires more VRAM)
- **34B**: Most accurate, requires 24GB+ VRAM

## Privacy & Data

- ✅ **No data sent to external servers**
- ✅ **Runs completely offline**
- ✅ **Model weights stored locally**
- ✅ **No API keys required**
- ✅ **Free to use**

## Disable LLaVA

If you want to remove LLaVA support:

```bash
# Uninstall
pip uninstall torch transformers llava-rlhf -y

# Or just don't select it in the UI
```

## Performance Comparison

| Model | Speed | Accuracy | Cost | Privacy |
|-------|-------|----------|------|---------|
| **LLaVA (GPU)** | 7-15s | Good | Free | 100% |
| **LLaVA (CPU)** | 60-120s | Good | Free | 100% |
| **GPT-4o** | 5-10s | Excellent | $$ | No |
| **Claude 3** | 8-15s | Excellent | $$ | No |
| **Gemini** | 6-12s | Good | $ | No |

## Next Steps

1. Install dependencies: `pip install torch transformers llava-rlhf`
2. Upload a video in SceneScriber AI
3. Select "Local LLaVA" as AI model
4. Enjoy offline video analysis!

## Support

For issues with:
- **LLaVA**: https://github.com/haotian-liu/LLaVA
- **PyTorch**: https://pytorch.org
- **SceneScriber AI**: Check AGENTS.md and API_KEYS.md

## Future Improvements

Planned enhancements:
- Support for LLaVA 13B and 34B models
- Quantized models for reduced memory usage
- Model caching and optimization
- GPU memory management
