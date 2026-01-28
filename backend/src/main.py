"""FastAPI application for Video Scene AI Analyzer."""

import logging
import uuid
import os
import time
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
# Try to import cv2 for video metadata, but make it optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Video duration metadata will be limited.")

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import AnalysisRequest, AnalysisResponse, Scene, ProjectConfig
from .scene_detector_simple import SimpleSceneDetector
from .ai_describer import AIDescriber
from .srt_exporter import SRTExporter
from .filename_formatter import FilenameFormatter
from .cleanup_utils import cleanup_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Check for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_COMPATIBLE_API_BASE = os.getenv("OPENAI_COMPATIBLE_API_BASE")
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if LLaVA dependencies are available
try:
    import torch
    import transformers
    import bitsandbytes
    import accelerate

    LLAVA_AVAILABLE = True
except ImportError as e:
    LLAVA_AVAILABLE = False
    logger.warning(f"LLaVA dependencies check failed: {e}")
    logger.warning("Install with: pip install torch transformers bitsandbytes accelerate")
except Exception as e:
    LLAVA_AVAILABLE = False
    logger.warning(f"Unexpected error checking LLaVA dependencies: {e}")

# Check for Ollama availability
import requests

try:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Quick check if Ollama is running
    requests.get(OLLAMA_HOST, timeout=1)
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False
    OLLAMA_HOST = "http://localhost:11434"  # Default fallback
    # Don't log warning as it's optional


def get_ollama_models():
    """Fetch available models from Ollama."""
    if not OLLAMA_AVAILABLE:
        return []

    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")
    return []


def get_openai_models(provider_type="openai"):
    """Fetch available models from OpenAI or OpenAI-compatible API.

    Args:
        provider_type: Either "openai" or "openai_compatible"

    Returns:
        List of available models
    """
    if provider_type == "openai":
        api_key = OPENAI_API_KEY
        api_base = "https://api.openai.com/v1"
        api_version = None
    elif provider_type == "openai_compatible":
        api_key = OPENAI_COMPATIBLE_API_KEY
        api_base = OPENAI_COMPATIBLE_API_BASE
        api_version = os.getenv("OPENAI_COMPATIBLE_API_VERSION")
    else:
        return []

    if not api_key:
        return []

    try:
        headers = {"Authorization": f"Bearer {api_key}"}

        # Add API version header for Azure OpenAI
        if api_version:
            headers["api-version"] = api_version

        # Try to fetch models list
        response = requests.get(f"{api_base}/models", headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])

            # Filter for vision-capable models and sort by capability
            vision_models = []
            for model in models:
                model_id = model.get("id", "")
                # Include GPT-4 Vision models and other vision-capable models
                # For OpenAI-compatible, we might not know model names, so include all
                if provider_type == "openai_compatible" or any(
                    pattern in model_id.lower()
                    for pattern in ["gpt-4", "gpt-4o", "gpt-4-vision", "gpt-4-turbo", "gpt-4o-mini"]
                ):
                    vision_models.append(
                        {
                            "id": model_id,
                            "name": model_id,
                            "description": f"{provider_type.replace('_', ' ').title()} {model_id}",
                            "capabilities": model.get("capabilities", []),
                            "provider": provider_type,
                        }
                    )

            return vision_models
        else:
            logger.warning(
                f"{provider_type} models API returned status {response.status_code}: {response.text}"
            )
            return []

    except Exception as e:
        logger.warning(f"Failed to fetch {provider_type} models: {e}")
        return []


AI_PROVIDERS_AVAILABLE = {
    "openai": bool(OPENAI_API_KEY),
    "openai_compatible": bool(OPENAI_COMPATIBLE_API_BASE and OPENAI_COMPATIBLE_API_KEY),
    "claude": bool(ANTHROPIC_API_KEY),
    "gemini": bool(GOOGLE_API_KEY),
    "llava": LLAVA_AVAILABLE,  # Available if torch and transformers installed
    "ollama": OLLAMA_AVAILABLE,
}

logger.info(f"AI Providers available: {AI_PROVIDERS_AVAILABLE}")

# Initialize FastAPI app
app = FastAPI(
    title="Video Scene AI Analyzer API",
    description="API for detecting scenes in videos and generating AI descriptions",
    version="0.1.0",
)

# Create keyframes directory if not exists
keyframes_dir = Path("keyframes")
keyframes_dir.mkdir(exist_ok=True)

# Mount keyframes directory
app.mount("/api/keyframes", StaticFiles(directory="keyframes"), name="keyframes")

# Get CORS origins from environment
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for processing jobs (in production, use database)
processing_jobs = {}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Video Scene AI Analyzer API",
        "version": "0.1.0",
        "status": "operational",
        "ai_providers_available": AI_PROVIDERS_AVAILABLE,
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze",
            "status": "/api/status/{job_id}",
            "scenes": "/api/scenes/{job_id}",
            "export": "/api/export/srt/{job_id}",
            "config": "/api/config",
            "filename_placeholders": "/api/filename/placeholders",
        },
    }


@app.get("/api/filename/placeholders")
async def get_filename_placeholders():
    """Get available placeholders for filename formatting.

    Returns:
        Dictionary of available placeholders and their descriptions, plus default/full formats
    """
    formatter = FilenameFormatter()
    
    return {
        "placeholders": formatter.get_available_placeholders(),
        "default_format": formatter.get_default_format(),
        "full_format": formatter.get_full_format(),
        "description": {
            "[videoname]": "Original video filename without extension",
            "[sensitivity]": "Scene detection sensitivity (s1=low, s2=medium, s3=high)",
            "[detail]": "Description detail level (d1=short, d2=medium, d3=detailed)",
            "[provider]": "AI model provider (ollama, openai, etc.)",
            "[model]": "AI model name (e.g. gpt-4o, llama2)",
            "[speed]": "Processing speed (video_duration / processing_time)",
            "[timestamp]": "Export timestamp (YYYYMMDD_HHMMSS)",
        },
    }


@app.get("/api/config")
async def get_config():
    """Get application configuration and available AI providers."""

    providers = {
        "claude": {
            "available": bool(ANTHROPIC_API_KEY),
            "name": "Anthropic Claude 3",
            "description": "Excellent quality, good alternative",
            "needs_api_key": True,
            "key_configured": bool(ANTHROPIC_API_KEY),
        },
        "gemini": {
            "available": bool(GOOGLE_API_KEY),
            "name": "Google Gemini",
            "description": "Cost-effective, good performance",
            "needs_api_key": True,
            "key_configured": bool(GOOGLE_API_KEY),
        },
        "llava": {
            "available": LLAVA_AVAILABLE,
            "name": "Local LLaVA (Transformers)",
            "description": "Privacy-focused, no API costs, requires torch and transformers",
            "needs_api_key": False,
            "key_configured": False,
        },
    }

    # Add OpenAI models dynamically (only if API key is configured)
    if OPENAI_API_KEY:
        openai_models = get_openai_models("openai")
        if openai_models:
            for model in openai_models:
                model_id = model.get("id", "")
                # Create a unique key for each model
                key = f"openai/{model_id}"
                providers[key] = {
                    "available": True,
                    "name": f"{model_id}",
                    "description": f"OpenAI {model_id}",
                    "needs_api_key": True,
                    "key_configured": True,
                }
        else:
            # Fallback to default OpenAI provider if models fetch fails
            providers["openai/gpt-4o"] = {
                "available": True,
                "name": "GPT-4o",
                "description": "OpenAI GPT-4o - Best quality, most expensive",
                "needs_api_key": True,
                "key_configured": True,
            }

    # Add OpenAI-compatible models dynamically (only if endpoint and key are configured)
    if OPENAI_COMPATIBLE_API_BASE and OPENAI_COMPATIBLE_API_KEY:
        compatible_models = get_openai_models("openai_compatible")
        if compatible_models:
            for model in compatible_models:
                model_id = model.get("id", "")
                # Create a unique key for each model
                key = f"openai_compatible/{model_id}"
                providers[key] = {
                    "available": True,
                    "name": f"{model_id}",
                    "description": f"OpenAI-compatible {model_id}",
                    "needs_api_key": True,
                    "key_configured": True,
                }
        else:
            # Fallback to generic OpenAI-compatible provider if models fetch fails
            providers["openai_compatible/gpt-4"] = {
                "available": True,
                "name": "GPT-4 (OpenAI-compatible)",
                "description": "Custom OpenAI-compatible endpoint",
                "needs_api_key": True,
                "key_configured": True,
            }

    # Add Ollama models dynamically
    if OLLAMA_AVAILABLE:
        ollama_models = get_ollama_models()
        for model in ollama_models:
            model_name = model.get("name", "unknown")
            # Create a unique key for each model
            key = f"ollama/{model_name}"
            providers[key] = {
                "available": True,
                "name": f"{model_name} (ollama)",
                "description": f"Local Ollama model: {model_name}",
                "needs_api_key": False,
                "key_configured": False,
            }
    else:
        # Optional: Add a placeholder if Ollama is not available but user wants to know
        providers["ollama"] = {
            "available": False,
            "name": "Ollama (Not Detected)",
            "description": "Ensure Ollama is running at localhost:11434",
            "needs_api_key": False,
            "key_configured": False,
        }

    return {
        "ai_providers": providers,
        "default_settings": {
            "detection_sensitivity": os.getenv("DEFAULT_SENSITIVITY", "medium"),
            "min_scene_duration": float(os.getenv("DEFAULT_MIN_SCENE_DURATION", "2.0")),
            "ai_model": os.getenv("DEFAULT_AI_MODEL", "openai"),
            "description_length": os.getenv("DEFAULT_DESCRIPTION_LENGTH", "medium"),
        },
        "features": {
            "scene_detection": True,
            "ai_description": any(AI_PROVIDERS_AVAILABLE.values()),
            "srt_export": True,
            "theme_support": True,
        },
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing.

    Args:
        file: Video file (MP4, MOV, MKV, AVI supported)

    Returns:
        Upload information including file path
    """
    logger.info(f"Uploading video: {file.filename}")

    # Perform cleanup before upload if auto cleanup is enabled
    if cleanup_manager.should_cleanup():
        logger.info("Performing automatic cleanup before upload...")
        cleanup_results = cleanup_manager.perform_cleanup()
        logger.info(f"Cleanup completed: {cleanup_results.get('total_files_removed', 0)} files removed")

    # Validate file type
    allowed_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    filename = file.filename or "video.mp4"
    file_extension = Path(filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create uploads directory if not exists
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}{file_extension}"

    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    logger.info(f"Video saved to {file_path}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "file_size": len(content),
        "message": "Video uploaded successfully",
    }


@app.get("/api/video/{file_id}")
async def get_video(file_id: str):
    """Serve uploaded video file for playback.

    Args:
        file_id: File ID returned from upload endpoint

    Returns:
        Video file stream
    """
    # Find the video file by file_id
    upload_dir = Path("uploads")
    video_files = list(upload_dir.glob(f"{file_id}.*"))

    if not video_files:
        raise HTTPException(status_code=404, detail="Video file not found")

    video_path = video_files[0]

    # Check if file exists
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    # Return file with appropriate media type
    return FileResponse(path=video_path, media_type="video/mp4", filename=video_path.name)


@app.post("/api/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_path: str,
    original_filename: Optional[str] = None,
    theme: Optional[str] = None,
    detection_sensitivity: str = "medium",
    min_scene_duration: float = 2.0,
    ai_model: str = "openai",
    description_length: str = "medium",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """Start video analysis with scene detection and AI description generation.

    Args:
        video_path: Path to uploaded video file
        theme: Optional theme for description generation
        detection_sensitivity: Scene detection sensitivity ('low', 'medium', 'high')
        min_scene_duration: Minimum scene duration in seconds
        ai_model: AI model to use ('openai', 'claude', 'gemini', 'llava', 'ollama')
        description_length: Description length ('short', 'medium', 'detailed')
        start_time: Start time in seconds (0 = beginning, None = 0)
        end_time: End time in seconds (None = end of video)

    Returns:
        Job information with job_id for status tracking
    """
    logger.info(f"Starting analysis for video: {video_path}")

    # Validate video file exists
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    # Create job ID
    job_id = str(uuid.uuid4())

    # Initialize job status
    processing_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Job created",
        "scenes": [],
        "error": None,
    }

    # Start background processing
    background_tasks.add_task(
        process_video_analysis,
        job_id=job_id,
        video_path=video_path,
        original_filename=original_filename,
        theme=theme,
        detection_sensitivity=detection_sensitivity,
        min_scene_duration=min_scene_duration,
        ai_model=ai_model,
        description_length=description_length,
        start_time=start_time,
        end_time=end_time,
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video analysis started in background",
        "check_status_url": f"/api/status/{job_id}",
    }


def process_video_analysis(
    job_id: str,
    video_path: str,
    original_filename: Optional[str],
    theme: Optional[str],
    detection_sensitivity: str,
    min_scene_duration: float,
    ai_model: str,
    description_length: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """Background task for video analysis processing."""
    processing_start = time.time()
    
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 10
        processing_jobs[job_id]["message"] = "Detecting scenes..."
        
        # Get video metadata (if cv2 is available)
        if CV2_AVAILABLE:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_full_duration = total_frames / fps if fps > 0 else 0
            cap.release()
        else:
            # Fallback: estimate duration from scenes or use default
            video_full_duration = 0
            logger.warning("OpenCV not available, using estimated video duration")

        # Step 1: Detect scenes
        detector = SimpleSceneDetector(
            sensitivity=detection_sensitivity, min_scene_duration=min_scene_duration
        )
        scenes = detector.detect_scenes(video_path)

        # Filter scenes by time range if specified
        if start_time is not None or end_time is not None:
            start = start_time or 0.0
            filtered_scenes = []
            for scene in scenes:
                # Keep scenes that overlap with the specified range
                if end_time is None:
                    # If no end_time, include all scenes from start_time onwards
                    if scene["end_time"] > start:
                        filtered_scenes.append(scene)
                else:
                    # Include scenes that overlap with [start_time, end_time]
                    if scene["end_time"] > start and scene["start_time"] < end_time:
                        filtered_scenes.append(scene)

            scenes = filtered_scenes
            logger.info(f"Filtered to {len(scenes)} scenes within time range [{start}, {end_time}]")

        # Update progress
        processing_jobs[job_id]["progress"] = 40
        processing_jobs[job_id][
            "message"
        ] = f"Detected {len(scenes)} scenes. Extracting keyframes..."

        # Step 2: Extract keyframes
        # Use absolute path for backend extraction
        keyframes_output_dir = os.path.abspath("keyframes")
        scenes = detector.extract_keyframes(
            video_path, scenes, frames_per_scene=3, output_dir=keyframes_output_dir
        )

        # Update progress
        processing_jobs[job_id]["progress"] = 60
        processing_jobs[job_id]["message"] = "Generating AI descriptions..."

        # Step 3: Generate AI descriptions
        # AIDescriber uses the local file paths
        describer = AIDescriber(model=ai_model)
        scenes = describer.generate_descriptions(
            scenes=scenes, theme=theme, description_length=description_length
        )

        # Step 4: Convert file paths to URLs for frontend
        for scene in scenes:
            if "keyframes" in scene:
                new_keyframes = []
                for kf in scene["keyframes"]:
                    filename = os.path.basename(kf)
                    # Create URL path
                    new_keyframes.append(f"/api/keyframes/{filename}")
                scene["keyframes"] = new_keyframes

        # Calculate processing time and segment duration
        processing_end = time.time()
        processing_time = processing_end - processing_start
        
        # Calculate segment duration
        if scenes:
            segment_duration = scenes[-1]["end_time"] - scenes[0]["start_time"]
        else:
            # Fallback to time range if no scenes
            segment_duration = (end_time or video_full_duration) - (start_time or 0)
        
        # Parse model provider and name from ai_model
        if "/" in ai_model:
            model_provider, model_name = ai_model.split("/", 1)
        else:
            model_provider = ai_model
            model_name = ai_model
        
        # Update job with results and metadata
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["message"] = "Analysis completed successfully"
        processing_jobs[job_id]["scenes"] = scenes
        processing_jobs[job_id]["metadata"] = {
            "video_path": video_path,
            "video_name": original_filename or os.path.basename(video_path),
            "sensitivity": detection_sensitivity,
            "detail_level": description_length,
            "model_provider": model_provider,
            "model_name": model_name,
            "segment_duration": segment_duration,
            "processing_time": processing_time,
        }

        logger.info(
            f"Analysis completed for job {job_id}: {len(scenes)} scenes, "
            f"processing time: {processing_time:.2f}s"
        )

    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {e}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        processing_jobs[job_id]["message"] = f"Analysis failed: {e}"


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job.

    Args:
        job_id: Job ID returned from analyze endpoint

    Returns:
        Current job status and progress
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "has_scenes": len(job.get("scenes", [])) > 0,
    }


@app.get("/api/scenes/{job_id}")
async def get_scenes(job_id: str):
    """Get scene data for a completed job.

    Args:
        job_id: Job ID returned from analyze endpoint

    Returns:
        List of scenes with descriptions and timing, plus processing metadata
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job not completed. Current status: {job['status']}"
        )

    # Get metadata from job
    job_metadata = job.get("metadata", {})
    
    # Calculate speed if we have segment_duration and processing_time
    speed = 0.0
    if job_metadata.get("segment_duration") and job_metadata.get("processing_time"):
        if job_metadata["processing_time"] > 0:
            speed = job_metadata["segment_duration"] / job_metadata["processing_time"]
    
    # Add speed to metadata
    metadata_with_speed = {
        **job_metadata,
        "speed": round(speed, 2) if speed > 0 else 0.0,
    }
    
    return {
        "job_id": job_id, 
        "scenes": job["scenes"], 
        "total_scenes": len(job["scenes"]),
        "metadata": metadata_with_speed,
    }


@app.get("/api/export/srt/{job_id}")
async def export_srt(
    job_id: str,
    filename_format: Optional[str] = Query(
        default=None,
        description="Filename format template (e.g., '[videoname]_[timestamp]')",
    ),
):
    """Export scene descriptions as SRT file.

    Args:
        job_id: Job ID returned from analyze endpoint
        filename_format: Optional filename template with placeholders

    Returns:
        SRT file download
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job not completed. Current status: {job['status']}"
        )

    # Generate SRT content
    exporter = SRTExporter()
    srt_content = exporter.export_to_srt(job["scenes"])

    # Validate SRT format
    if not exporter.validate_srt(srt_content):
        raise HTTPException(status_code=500, detail="Generated SRT content is invalid")

    # Generate filename
    formatter = FilenameFormatter()
    
    if filename_format:
        # Use provided format
        template = filename_format
    else:
        # Use default format
        template = FilenameFormatter.get_default_format()
    
    # Get metadata from job
    job_metadata = job.get("metadata", {})
    
    if job_metadata:
        # Create export metadata
        export_metadata = FilenameFormatter.create_metadata(
            video_name=job_metadata.get("video_name", "video"),
            sensitivity=job_metadata.get("sensitivity", "medium"),
            detail_level=job_metadata.get("detail_level", "medium"),
            model_provider=job_metadata.get("model_provider", "unknown"),
            model_name=job_metadata.get("model_name", "unknown"),
            segment_duration=job_metadata.get("segment_duration", 0),
            processing_time=job_metadata.get("processing_time", 0),
        )
        
        # Format filename
        try:
            base_filename = formatter.format_filename(export_metadata, template)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # Fallback if metadata not available
        base_filename = f"scenes_{job_id}"
    
    filename = f"{base_filename}.srt"

    # Save to temporary file
    srt_path = Path(f"exports/{base_filename}.srt")
    srt_path.parent.mkdir(exist_ok=True)

    try:
        exporter.save_to_file(srt_content, str(srt_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save SRT file: {e}")

    # Return file for download
    return FileResponse(path=srt_path, filename=filename, media_type="application/x-subrip")


@app.put("/api/scenes/{scene_id}")
async def update_scene_description(
    scene_id: int,
    job_id: str = Query(..., description="Job ID"),
    description: str = Query(..., description="New description text"),
):
    """Update description for a specific scene.

    Args:
        job_id: Job ID
        scene_id: Scene ID to update
        description: New description text

    Returns:
        Updated scene information
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job not completed. Current status: {job['status']}"
        )

    # Find and update scene
    for scene in job["scenes"]:
        if scene["scene_id"] == scene_id:
            scene["description"] = description
            return {
                "job_id": job_id,
                "scene_id": scene_id,
                "description": description,
                "message": "Scene description updated successfully",
            }

    raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")


@app.get("/api/cleanup/stats")
async def get_cleanup_stats():
    """Get cleanup statistics and directory information.
    
    Returns:
        Dictionary with cleanup configuration and directory statistics
    """
    stats = cleanup_manager.get_directory_stats()
    
    return {
        "cleanup_config": {
            "retention_days": cleanup_manager.retention_days,
            "enable_auto_cleanup": cleanup_manager.enable_auto_cleanup,
            "cleanup_directories": cleanup_manager.cleanup_directories,
            "cleanup_log_files": cleanup_manager.cleanup_log_files,
        },
        "directory_stats": stats,
        "should_cleanup": cleanup_manager.should_cleanup(),
    }


@app.post("/api/cleanup/perform")
async def perform_cleanup():
    """Manually trigger cleanup operation.
    
    Returns:
        Cleanup results
    """
    logger.info("Manual cleanup triggered via API")
    results = cleanup_manager.perform_cleanup()
    return results


@app.post("/api/cleanup/prepare-upload")
async def prepare_for_upload():
    """Prepare for new upload by performing cleanup.
    
    This endpoint should be called before uploading a new video.
    It performs the same cleanup as the automatic cleanup but returns
    results immediately.
    
    Returns:
        Cleanup results
    """
    logger.info("Preparing for new upload - performing cleanup")
    results = cleanup_manager.perform_cleanup()
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
