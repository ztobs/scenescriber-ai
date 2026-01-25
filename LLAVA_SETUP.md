# Local AI Model Setup Guide (LLaVA & Ollama)

SceneScriber AI supports multiple local AI options for offline, privacy-focused video scene descriptions. No API keys needed!

## Available Local AI Options

### 1. LLaVA (Transformers)
Direct integration with Hugging Face Transformers library. Runs LLaVA model locally.

### 2. Ollama Models ✨ NEW!
Integration with Ollama server for any vision model (llava, bakllava, etc.). Automatically detects all available models.

## What are Local AI Models?

Local AI models run entirely on your machine, perfect for:
- Privacy: No data sent to external APIs
- Cost: Free - no API charges
- Speed: Runs offline after initial setup
- Flexibility: Works without internet connection
- Choice: Multiple model options via Ollama

## System Requirements

- **GPU recommended** (NVIDIA with CUDA 11.8+ or compatible GPU)
- **RAM**: Minimum 8GB (16GB+ recommended for smooth operation)
- **Disk space**: 15-20GB for model weights
- **Python**: 3.9 to 3.13 (3.12+ works fine with latest transformers)

## Installation Steps

### Option 1: LLaVA (Transformers) Setup

#### 1. Create a Python Virtual Environment (Optional but Recommended)

```bash
# Using Python 3.13 (or 3.12, 3.10, etc.)
python3.13 -m venv llava_env
source llava_env/bin/activate  # On Windows: llava_env\Scripts\activate

# Or use default python3
python3 -m venv llava_env
source llava_env/bin/activate
```

#### 2. Install LLaVA Dependencies

**GPU Support (NVIDIA - Recommended)**
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and dependencies
pip install transformers>=4.36.0 pillow bitsandbytes accelerate
```

**CPU Only (Mac, Linux, Windows without GPU)**
```bash
pip install torch torchvision torchaudio
pip install transformers>=4.36.0 pillow bitsandbytes accelerate
```

**Apple Silicon (M1/M2/M3)**
```bash
# Install with Metal acceleration
pip install torch torchvision torchaudio
pip install transformers>=4.36.0 pillow bitsandbytes accelerate
```

**Verify Installation:**
```bash
python -c "from transformers import AutoProcessor, LlavaForConditionalGeneration; print('✓ LLaVA ready!')"
```

#### 3. First-Time Model Download

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

#### 4. Select LLaVA in SceneScriber AI

1. Upload a video
2. Go to "Configure Analysis Settings"
3. Under "AI Configuration", select: **"Local LLaVA (Transformers)"**
4. Click "Start Analysis"
5. Wait for first-time model download (if needed)

### Option 2: Ollama Setup ✨ NEW!

#### 1. Install Ollama

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download installer from https://ollama.ai/download

#### 2. Start Ollama Service

```bash
# Start Ollama in the background
ollama serve

# Or run as a service (Linux)
sudo systemctl enable ollama
sudo systemctl start ollama
```

#### 3. Pull Vision Models

```bash
# Pull LLaVA (7B model - recommended)
ollama pull llava:latest

# Alternative models
ollama pull bakllava:latest    # BakLLaVA (7B)
ollama pull llava:13b          # LLaVA 1.5 (13B)
ollama pull llava:34b          # LLaVA 1.5 (34B) - requires 24GB+ VRAM
```

#### 4. Verify Ollama is Running

```bash
# Check service status
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

#### 5. Configure SceneScriber AI (Optional)

Add to `backend/.env`:
```env
OLLAMA_HOST=http://localhost:11434  # Default
OLLAMA_MODEL=llava:latest           # Default model
```

#### 6. Use Ollama Models in SceneScriber AI

1. Upload a video
2. Go to "Configure Analysis Settings"
3. Under "AI Configuration", all available Ollama models will appear automatically
4. Select any model (e.g., "llava:latest (ollama)", "bakllava:latest (ollama)")
5. Click "Start Analysis"

## Performance Tips

### Optimize Memory Usage

```python
# Reduce precision to save memory (less accurate but faster)
torch_dtype = torch.float16  # Already configured in code
```

### Enable GPU Acceleration

Local models automatically detect your GPU:
- **NVIDIA GPU**: Uses CUDA (fastest)
- **Apple Silicon**: Uses Metal Performance Shaders
- **CPU only**: Falls back to CPU (slower)

Check GPU usage:
```bash
nvidia-smi  # For NVIDIA GPUs
```

### Batch Processing

For multiple videos, local models get faster after the first analysis since the model stays in memory.

### Ollama-Specific Tips

1. **Model Management**:
   ```bash
   # List all models
   ollama list
   
   # Remove unused models
   ollama rm model_name
   
   # Copy models between systems
   ollama pull --from model_name
   ```

2. **Memory Management**:
   ```bash
   # Set GPU layers (for limited VRAM)
   OLLAMA_NUM_GPU=1 ollama run llava:latest
   
   # Force CPU mode
   OLLAMA_NUM_GPU=0 ollama run llava:latest
   ```

3. **Performance Tuning**:
   ```bash
   # Use quantized models for less VRAM
   ollama pull llava:7b-q4_0
   
   # Monitor Ollama performance
   ollama ps
   ```

## Troubleshooting

### Issue: "LLaVA dependencies not installed"

**Solution**: Run the installation commands above

```bash
pip install torch transformers pillow bitsandbytes accelerate
```

### Issue: Out of Memory Error

**Solution**: 
- Use GPU instead of CPU
- Reduce batch size (already optimized in code)
- Close other applications
- Use `torch.float16` instead of `float32`
- For Ollama: Use quantized models or reduce GPU layers

### Issue: Model download stuck

**Solution**:
- Check internet connection
- For LLaVA: Manually download from: https://huggingface.co/liuhaotian/llava-v1.5-7b-hf
- For Ollama: Check `ollama pull` output for errors
- Check available disk space (needs ~20GB free)

### Issue: Slow inference on CPU

**Expected behavior**: CPU inference is 10-50x slower than GPU. Consider:
- Using GPU if available
- Using OpenAI/Claude for time-critical projects
- Batch processing at off-peak hours

### Issue: Ollama models not appearing in dropdown

**Solution**:
1. Ensure Ollama is running: `ollama serve`
2. Check connection: `curl http://localhost:11434/api/tags`
3. Verify models are pulled: `ollama list`
4. Restart SceneScriber AI backend

### Issue: "Ollama API error"

**Solution**:
1. Check Ollama service status
2. Verify OLLAMA_HOST in backend/.env (default: http://localhost:11434)
3. Check firewall/network settings
4. Try pulling model again: `ollama pull llava:latest`

## Model Variants

### LLaVA (Transformers)
Current implementation uses: **LLaVA v1.5 7B**

Other variants available:
- **7B** (current): Fast, 7-15 seconds per scene
- **13B**: Slower, more accurate (requires more VRAM)
- **34B**: Most accurate, requires 24GB+ VRAM

### Ollama Models
All available vision models appear automatically in the dropdown:

**Recommended Models:**
- `llava:latest` - LLaVA 1.5 7B (balanced speed/quality)
- `bakllava:latest` - BakLLaVA 7B (alternative vision model)
- `llava:13b` - LLaVA 1.5 13B (higher quality)
- `llava:34b` - LLaVA 1.5 34B (best quality, needs 24GB+ VRAM)

**Quantized Models (less VRAM):**
- `llava:7b-q4_0` - 4-bit quantized (4-6GB VRAM)
- `llava:7b-q8_0` - 8-bit quantized (6-8GB VRAM)

## Privacy & Data

- ✅ **No data sent to external servers**
- ✅ **Runs completely offline**
- ✅ **Model weights stored locally**
- ✅ **No API keys required**
- ✅ **Free to use**
- ✅ **Multiple model options** (via Ollama)

## Disable Local Models

### Disable LLaVA (Transformers):
```bash
# Uninstall
pip uninstall torch transformers bitsandbytes accelerate -y

# Or just don't select it in the UI
```

### Disable Ollama:
```bash
# Stop Ollama service
ollama stop

# Or disable auto-start
sudo systemctl disable ollama  # Linux
```

## Automatic Model Detection

SceneScriber AI automatically:
1. Detects if Ollama is running
2. Fetches all available vision models
3. Lists them in the dropdown with "(ollama)" suffix
4. Updates the list when new models are pulled

No manual configuration needed!

## Performance Comparison

| Model | Speed (GPU) | Speed (CPU) | Accuracy | VRAM | Privacy |
|-------|-------------|-------------|----------|------|---------|
| **LLaVA 7B (Transformers)** | 7-15s/scene | 60-120s/scene | Good | 6-8GB | 100% |
| **LLaVA via Ollama** | 5-12s/scene | 50-100s/scene | Good | 6-8GB | 100% |
| **BakLLaVA via Ollama** | 8-18s/scene | 70-130s/scene | Good | 6-8GB | 100% |
| **LLaVA 13B via Ollama** | 15-30s/scene | 120-240s/scene | Very Good | 12-14GB | 100% |
| **LLaVA 34B via Ollama** | 30-60s/scene | 240-480s/scene | Excellent | 24GB+ | 100% |
| **GPT-4o** | 5-10s/scene | N/A | Excellent | N/A | No |
| **Claude 3** | 8-15s/scene | N/A | Excellent | N/A | No |
| **Gemini** | 6-12s/scene | N/A | Good | N/A | No |

**Note**: Ollama models generally have better memory management than direct Transformers integration.

## Next Steps

### For LLaVA (Transformers):
1. Install dependencies: `pip install torch transformers pillow bitsandbytes accelerate`
2. Upload a video in SceneScriber AI
3. Select "Local LLaVA (Transformers)" as AI model
4. Enjoy offline video analysis!

### For Ollama Models:
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Start service: `ollama serve`
3. Pull model: `ollama pull llava:latest`
4. Upload a video in SceneScriber AI
5. Select any model with "(ollama)" suffix
6. Enjoy offline video analysis with multiple model options!

## Support

For issues with:
- **LLaVA**: https://github.com/haotian-liu/LLaVA
- **Ollama**: https://github.com/ollama/ollama
- **PyTorch**: https://pytorch.org
- **SceneScriber AI**: Check AGENTS.md and API_KEYS.md

## Future Improvements

Planned enhancements:
- Support for more vision models (Qwen-VL, CogVLM, etc.)
- Model comparison and benchmarking
- Batch processing optimization
- GPU memory management improvements
- Model fine-tuning capabilities
