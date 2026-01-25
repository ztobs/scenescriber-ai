# 🎬 SceneScriber AI - Video Scene Analyzer

An intelligent video analysis tool that automatically detects scene cuts, generates AI-powered descriptions based on user-defined themes, and exports SRT caption files for video editing workflows.

## ✨ Features

- **Automatic Scene Detection**: Detects scene boundaries using FFmpeg-based histogram comparison
- **AI-Powered Descriptions**: Multiple AI options:
  - **GPT-4o** (OpenAI) - Fastest, most accurate
  - **Claude 3** (Anthropic) - Excellent quality
  - **Gemini** (Google) - Cost-effective
  - **LLaVA** (Local) - Privacy-focused, offline, free
  - **Ollama Models** - Local inference with any vision model (llava, bakllava, etc.) ✨ NEW!
- **Video Range Selection**: Choose specific time ranges to analyze (e.g., 5s - 30s of the video)
- **Theme Integration**: Tailor descriptions to specific project themes (DIY, cooking, gaming, tutorials, etc.)
- **SRT Export**: Exports to standard SubRip subtitle format compatible with DaVinci Resolve, Premiere Pro, etc.
- **Interactive Review**: Edit and refine AI-generated descriptions before export
- **Video Preview**: Built-in player with scene markers and timeline navigation
- **Modern Web Interface**: Clean, responsive React frontend with Material-UI

## 🏗️ Architecture

```
video-scene-tool/
├── backend/                    # Python FastAPI backend
│   ├── src/
│   │   ├── scene_detector.py  # Scene detection using PySceneDetect
│   │   ├── ai_describer.py    # AI description generation
│   │   ├── srt_exporter.py    # SRT format export
│   │   ├── models.py          # Data models
│   │   └── main.py            # FastAPI application
│   ├── tests/                 # Unit tests
│   └── requirements.txt       # Python dependencies
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── store/            # Zustand state management
│   │   ├── utils/            # Utilities and API client
│   │   └── types.ts          # TypeScript definitions
│   └── package.json          # Node.js dependencies
└── app-description.md        # Complete technical specification
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (tested with 3.12)
- Node.js 18+
- FFmpeg (for video processing) - **REQUIRED**
- AI API keys (OpenAI, Claude, or Gemini) - optional for testing

### Easy Setup (Recommended)

```bash
# Using Makefile (most comprehensive)
make setup    # Full setup
make start    # Start both services

# OR using setup script
chmod +x setup.sh
./setup.sh

# Then start with:
./start.sh    # Or use: make start
```

### Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies (no OpenCV/PySceneDetect)
pip install --upgrade pip
pip install -r requirements-minimal.txt

# Configure API keys (optional but recommended)
cp .env.example .env
# Edit .env and add your API keys (OpenAI, Claude, or Gemini)

# Create necessary directories
mkdir -p uploads exports

# Run the backend server
uvicorn src.main:app --reload --port 8000
```

#### Advanced Backend Setup (with OpenCV)

If you want advanced scene detection with OpenCV:

```bash
cd backend
source venv/bin/activate

# Try installing OpenCV (may not work on all systems)
pip install opencv-python scikit-image scikit-video

# Or use the full requirements (may have compatibility issues)
# pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Verify FFmpeg Installation

```bash
# Check if ffmpeg is installed
ffmpeg -version

# Check if ffprobe is installed
ffprobe -version
```

If FFmpeg is not installed:

- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Start the Application

#### Option 1: Comprehensive Makefile (Recommended)
```bash
# Show all available commands
make help

# Full setup (first time only)
make setup

# Start both services
make start

# Check service status
make status

# Stop services
make stop

# Clear GPU VRAM (for LLaVA users)
make vram-cleanup
```

#### Option 2: Simple One-Command Startup Scripts
```bash
# Make sure scripts are executable
chmod +x start.sh quick-start.sh

# Start both services with detailed output
./start.sh

# OR for a simpler version
./quick-start.sh
```

#### Option 3: Manual Startup (Two Terminals)
1. **Start Backend** (in one terminal):
```bash
cd backend
source venv/bin/activate
uvicorn src.main:app --reload --port 8000
```

2. **Start Frontend** (in another terminal):
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

#### Key Makefile Commands:
| Command | Description |
|---------|-------------|
| `make setup` | Full project setup |
| `make start` | Start both services |
| `make stop` | Stop all services |
| `make restart` | Restart services |
| `make status` | Check service status |
| `make vram-cleanup` | Clear GPU memory |
| `make logs` | View service logs |
| `make clean` | Clean temporary files |
| `make test` | Run all tests |
| `make update` | Update dependencies |

## 🤖 Using Local AI Models (LLaVA & Ollama)

SceneScriber AI supports multiple local AI options for privacy-focused, offline processing:

### Option 1: LLaVA (Transformers)
Local LLaVA model running directly via Hugging Face Transformers.

**Setup LLaVA:**

**For GPU (NVIDIA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.36.0 pillow bitsandbytes accelerate
```

**For CPU or Mac:**
```bash
pip install torch transformers>=4.36.0 pillow bitsandbytes accelerate
```

### Option 2: Ollama Models ✨ NEW!
Use any vision model supported by Ollama (llava, bakllava, etc.) with automatic model detection.

**Setup Ollama:**

1. **Install Ollama**:
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull a vision model**:
   ```bash
   ollama pull llava:latest      # LLaVA 1.5 (7B)
   ollama pull bakllava:latest   # BakLLaVA (7B)
   ollama pull llava:13b         # LLaVA 1.5 (13B)
   ```

4. **Configure environment (optional)**:
   ```bash
   # In backend/.env file
   OLLAMA_HOST=http://localhost:11434  # Default
   OLLAMA_MODEL=llava:latest           # Default model
   ```

### Using Local AI Models

1. Upload a video
2. In "Configure Analysis Settings", select:
   - **"Local LLaVA (Transformers)"** - For direct LLaVA integration
   - **"llava:latest (ollama)"** - For Ollama-based LLaVA
   - **"bakllava:latest (ollama)"** - For BakLLaVA via Ollama
   - Any other Ollama vision model detected automatically
3. Click "Start Analysis"
4. All processing happens locally - no data sent to external services

**For detailed setup instructions, see [LLAVA_SETUP.md](LLAVA_SETUP.md)**

### Benefits

✅ **Privacy**: 100% offline, no data sent anywhere
✅ **Free**: No API keys or subscription needed
✅ **Offline**: Works without internet after setup
✅ **Flexible**: Multiple model options via Ollama
✅ **Automatic Detection**: All available Ollama models appear in dropdown

### Performance Comparison

| Model | GPU Speed | CPU Speed | VRAM Required | Notes |
|-------|-----------|-----------|---------------|-------|
| **LLaVA 7B (Transformers)** | 7-15s/scene | 60-120s/scene | ~6-8GB | Direct integration |
| **LLaVA via Ollama** | 5-12s/scene | 50-100s/scene | ~6-8GB | Better memory management |
| **BakLLaVA via Ollama** | 8-18s/scene | 70-130s/scene | ~6-8GB | Alternative vision model |
| **LLaVA 13B via Ollama** | 15-30s/scene | 120-240s/scene | ~12-14GB | Higher quality |

See [LLAVA_SETUP.md](LLAVA_SETUP.md#performance-comparison) for detailed benchmarks.

## 📋 Usage Workflow

1. **Upload Video**: Drag and drop or select a video file (MP4, MOV, MKV, AVI, WebM)
2. **Configure Analysis**: Set theme, detection sensitivity, AI model, and description length
3. **Process Video**: AI detects scenes and generates descriptions (progress tracked in real-time)
4. **Review & Edit**: View scenes, edit descriptions, adjust timing if needed
5. **Export SRT**: Download SRT file for use in video editing software

## 🔧 Configuration

### Backend Environment Variables

Create a `.env` file in the `backend` directory:

```env
# AI Provider API Keys (choose one or more)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key
GEMINI_API_KEY=your_gemini_key

# Ollama Configuration (for local models)
OLLAMA_HOST=http://localhost:11434  # Ollama server URL
OLLAMA_MODEL=llava:latest           # Default Ollama model

# Application Settings
MAX_UPLOAD_SIZE=2147483648  # 2GB in bytes
UPLOAD_DIR=uploads
EXPORT_DIR=exports
LOG_LEVEL=INFO
```

### Frontend Configuration

Edit `frontend/vite.config.ts` to adjust:
- API proxy settings
- Development server port
- Build options

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_srt_exporter.py

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing
```

### Frontend Tests

```bash
cd frontend

# Run tests (when implemented)
npm test

# Type checking
npm run type-check

# Linting
npm run lint
```

## 📁 Project Structure Details

### Backend Modules

- **`scene_detector.py`**: Uses PySceneDetect for scene boundary detection with configurable sensitivity
- **`ai_describer.py`**: Integrates with multiple AI providers (OpenAI, Claude, Gemini, LLaVA)
- **`srt_exporter.py`**: Generates SRT files with proper formatting and validation
- **`main.py`**: FastAPI application with RESTful endpoints for video processing

### Frontend Components

- **`UploadScreen`**: Video upload with drag-and-drop support
- **`ConfigurationScreen`**: Analysis settings with theme examples
- **`ProcessingScreen`**: Real-time progress tracking with step visualization
- **`ReviewScreen`**: Scene review with editable descriptions
- **`ExportScreen`**: SRT download and video editor integration guides

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload video file |
| POST | `/api/analyze` | Start video analysis |
| GET | `/api/status/{job_id}` | Check processing status |
| GET | `/api/scenes/{job_id}` | Get scene data |
| GET | `/api/export/srt/{job_id}` | Export SRT file |
| PUT | `/api/scenes/{scene_id}` | Update scene description |

## 🎯 Key Technical Features

1. **Scene Detection**: Multiple sensitivity levels, minimum scene duration control
2. **AI Integration**: Support for multiple providers with fallback options
3. **Theme Awareness**: Dynamic prompt engineering based on user themes
4. **SRT Compliance**: Strict adherence to SubRip specification
5. **Progress Tracking**: Real-time updates with estimated time remaining
6. **Error Handling**: Comprehensive error handling with user-friendly messages

## 🔒 Security & Privacy

- **Local Processing Option**: Use LLaVA for offline processing
- **API Key Management**: Secure storage of user API keys
- **Temporary Files**: Auto-cleanup of uploaded videos after processing
- **HTTPS**: All API communications should be encrypted in production

## 📈 Performance

- **Processing Time**: ~3 minutes for 7-minute 1080p video
- **Accuracy**: >90% scene detection, <10% false positives
- **Memory Usage**: <4GB RAM for 1080p video processing
- **Concurrency**: Supports multiple simultaneous processing jobs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

See `AGENTS.md` for detailed development guidelines.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection
- [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Google AI](https://ai.google/) for AI models
- [FastAPI](https://fastapi.tiangolo.com/) for backend framework
- [React](https://reactjs.org/) and [Material-UI](https://mui.com/) for frontend

## 🔧 Troubleshooting

### Common Issues

1. **"No matching distribution found for pyscenedetect"**
   - Use `requirements-minimal.txt` instead of `requirements.txt`
   - The simple scene detector uses FFmpeg instead of PySceneDetect

2. **FFmpeg not found**
   - Install FFmpeg using your package manager
   - Ensure `ffmpeg` and `ffprobe` are in your PATH

3. **OpenCV installation fails**
   - Use the simple scene detector (default)
   - Or try: `pip install opencv-python-headless`

4. **Python version issues**
   - The project works with Python 3.9+
   - For Python 3.12+, use `requirements-minimal.txt`

### Scene Detection Modes

The application has two scene detection modes:

1. **Simple Mode** (default): Uses FFmpeg's built-in scene detection
   - Works without OpenCV
   - Requires FFmpeg installed
   - Good for most use cases

2. **Advanced Mode**: Uses OpenCV and scikit-image
   - More accurate scene detection
   - Requires OpenCV installation
   - Enable by installing: `pip install opencv-python scikit-image`

## 🔑 API Key Configuration

### Getting API Keys

1. **OpenAI GPT-4 Vision** (Recommended):
   - Visit: https://platform.openai.com/api-keys
   - Create new API key
   - Add to `.env`: `OPENAI_API_KEY=your_key_here`

2. **Anthropic Claude 3** (Alternative):
   - Visit: https://console.anthropic.com/settings/keys
   - Create new API key
   - Add to `.env`: `ANTHROPIC_API_KEY=your_key_here`

3. **Google Gemini** (Cost-effective):
   - Visit: https://makersuite.google.com/app/apikey
   - Create new API key
   - Add to `.env`: `GOOGLE_API_KEY=your_key_here`

### Using Local Models (No API Keys Needed)

For complete privacy and offline operation:

1. **LLaVA (Transformers)**:
   - Install: `pip install torch transformers pillow bitsandbytes accelerate`
   - Select "Local LLaVA (Transformers)" in the UI

2. **Ollama Models**:
   - Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Start service: `ollama serve`
   - Pull model: `ollama pull llava:latest`
   - All available models appear automatically in the dropdown

### Without API Keys

The application will still work without API keys:
- Scene detection using FFmpeg ✓
- SRT export ✓
- Local AI descriptions (LLaVA/Ollama) ✓
- Mock AI descriptions (if no local AI) ✓ (editable)
- Full workflow ✓

### Testing the Installation

```bash
# Test backend
cd backend
source venv/bin/activate
python -c "from src.main import app; print('Backend imports OK')"

# Test frontend
cd frontend
npm run type-check

# Test API connectivity
curl http://localhost:8000/api/config
```

## 📞 Support

For issues and feature requests, please use the GitHub Issues page.

---

**SceneScriber AI** - Making video editing smarter, one scene at a time. 🎬

### Quick Test

After installation, try uploading a short video (under 100MB) to test the workflow. The application will:
1. Upload your video
2. Detect scenes using FFmpeg
3. Generate AI descriptions (if API keys are configured)
4. Export SRT file for video editing

**Note**: AI features require API keys. Without them, you'll get placeholder descriptions that you can edit manually.