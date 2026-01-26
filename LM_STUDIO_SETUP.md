# LM Studio Setup Guide

This guide helps you set up LM Studio as an OpenAI-compatible AI provider for Video Scene AI Analyzer.

## What is LM Studio?

LM Studio is a desktop application that lets you run large language models locally on your computer. It provides an OpenAI-compatible API, which means it can be used as a drop-in replacement for OpenAI's API.

## Prerequisites

- **RAM**: 8GB minimum (16GB+ recommended for vision models)
- **Storage**: 10-50GB depending on model size
- **GPU**: Optional but recommended (NVIDIA, AMD, or Apple Silicon)

## Installation Steps

### 1. Download and Install LM Studio

1. Visit [https://lmstudio.ai](https://lmstudio.ai)
2. Download the installer for your OS (Windows, macOS, or Linux)
3. Install the application

### 2. Download a Vision-Capable Model

LM Studio needs a vision model to analyze video frames. Recommended models:

- **llava-1.5-7b** - Good balance of speed and quality, ~4GB VRAM
- **minicpm-v** - Fast and efficient, ~3GB VRAM
- **llava-1.5-13b** - Better quality, ~8-10GB VRAM

**Steps:**

1. Open LM Studio
2. Click on the search/explore icon (left sidebar)
3. Search for "llava" or "minicpm-v"
4. Click the download button next to your chosen model
5. Wait for the download to complete

### 3. Start the Local Server

1. In LM Studio, go to the **Local Server** tab (left sidebar)
2. Select the model you just downloaded from the dropdown
3. Click **"Start Server"**
4. Wait for the message "Server is listening on http://localhost:1234"

The local server is now running and ready to use!

### 4. Configure Video Scene AI Analyzer

The `.env` file in the `backend/` directory is already pre-configured for LM Studio:

```
OPENAI_COMPATIBLE_API_BASE=http://localhost:1234/v1
OPENAI_COMPATIBLE_API_KEY=not-needed
```

**No additional configuration needed!** The app will automatically detect LM Studio and list available models.

## Using LM Studio with the App

1. **Start LM Studio** and ensure the Local Server is running
2. **Start the Video Scene AI Analyzer** backend and frontend
3. **Upload a video** and go to Configuration
4. **Select AI Model** - You'll see your LM Studio model listed under "OPENAI_COMPATIBLE"
5. **Start analysis** - The app will use your local model to analyze the video

## Troubleshooting

### Models not appearing in dropdown?

- **Check LM Studio is running**: The server should show "listening on http://localhost:1234"
- **Check the backend logs**: Look for messages about fetching models
- **Verify the endpoint**: Try visiting `http://localhost:1234/v1/models` in your browser

### Slow or hanging inference?

- **Check VRAM usage**: Models need enough VRAM to run
- **Check CPU usage**: On CPU-only systems, inference can be slow (minutes per image)
- **Reduce description length**: Set to "short" for faster processing
- **Use smaller models**: llava-1.5-7b is faster than 13b

### "Connection refused" error?

- Make sure LM Studio is running and the Local Server is started
- Check that no other service is using port 1234
- Restart LM Studio if needed

### Model crashes during inference?

- Model may be too large for your available VRAM
- Try a smaller model (e.g., 7B instead of 13B)
- Allocate more VRAM to GPU if possible
- Close other applications to free up system memory

## Model Recommendations by VRAM

| VRAM | Recommended Model | Speed | Quality |
|------|-------------------|-------|---------|
| 4GB  | llava-1.5-7b-q4   | Fast  | Good    |
| 8GB  | llava-1.5-7b      | Fast  | Good    |
| 12GB | llava-1.5-13b-q4  | Med   | Better  |
| 16GB+| llava-1.5-13b     | Med   | Better  |

## Advanced Configuration

If LM Studio is running on a different machine or port:

1. Edit `backend/.env`
2. Change `OPENAI_COMPATIBLE_API_BASE` to your LM Studio endpoint:
   ```
   OPENAI_COMPATIBLE_API_BASE=http://your-machine-ip:1234/v1
   ```
3. Restart the backend

## Performance Tips

- **Use GPU acceleration**: LM Studio will use GPU if available (NVIDIA, AMD, Apple Silicon)
- **Reduce image size**: Smaller keyframes process faster
- **Use shorter descriptions**: "short" length is much faster than "detailed"
- **Close background apps**: Free up RAM and CPU

## Switching Between Providers

You can use LM Studio alongside other AI providers:

```
# Use both OpenAI AND LM Studio
OPENAI_API_KEY=sk-...your-key...
OPENAI_COMPATIBLE_API_BASE=http://localhost:1234/v1
```

In the Configuration screen, you'll see models from both providers and can switch between them.

## More Information

- LM Studio Documentation: [https://lmstudio.ai/docs](https://lmstudio.ai/docs)
- OpenAI API Reference: [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
- Vision Models: [https://huggingface.co/models?task=image-text-to-text](https://huggingface.co/models?task=image-text-to-text)
