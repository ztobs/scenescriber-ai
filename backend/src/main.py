"""FastAPI application for Video Scene AI Analyzer."""

import logging
import uuid
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .models import AnalysisRequest, AnalysisResponse, Scene, ProjectConfig
from .scene_detector_simple import SimpleSceneDetector
from .ai_describer import AIDescriber
from .srt_exporter import SRTExporter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

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

AI_PROVIDERS_AVAILABLE = {
    'openai': bool(OPENAI_API_KEY),
    'claude': bool(ANTHROPIC_API_KEY),
    'gemini': bool(GOOGLE_API_KEY),
    'llava': LLAVA_AVAILABLE  # Available if torch and transformers installed
}

logger.info(f"AI Providers available: {AI_PROVIDERS_AVAILABLE}")

# Initialize FastAPI app
app = FastAPI(
    title="Video Scene AI Analyzer API",
    description="API for detecting scenes in videos and generating AI descriptions",
    version="0.1.0"
)

# Get CORS origins from environment
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')

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
        }
    }


@app.get("/api/config")
async def get_config():
    """Get application configuration and available AI providers."""
    return {
        "ai_providers": {
            "openai": {
                "available": bool(OPENAI_API_KEY),
                "name": "OpenAI GPT-4 Vision",
                "description": "Best quality, most expensive",
                "needs_api_key": True,
                "key_configured": bool(OPENAI_API_KEY),
            },
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
                "name": "Local LLaVA",
                "description": "Privacy-focused, no API costs, requires torch and transformers",
                "needs_api_key": False,
                "key_configured": False,
            }
        },
        "default_settings": {
            "detection_sensitivity": os.getenv('DEFAULT_SENSITIVITY', 'medium'),
            "min_scene_duration": float(os.getenv('DEFAULT_MIN_SCENE_DURATION', '2.0')),
            "ai_model": os.getenv('DEFAULT_AI_MODEL', 'openai'),
            "description_length": os.getenv('DEFAULT_DESCRIPTION_LENGTH', 'medium'),
        },
        "features": {
            "scene_detection": True,
            "ai_description": any(AI_PROVIDERS_AVAILABLE.values()),
            "srt_export": True,
            "theme_support": True,
        }
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
    
    # Validate file type
    allowed_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    filename = file.filename or "video.mp4"
    file_extension = Path(filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
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
        "message": "Video uploaded successfully"
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
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_path.name
    )


@app.post("/api/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_path: str,
    theme: Optional[str] = None,
    detection_sensitivity: str = "medium",
    min_scene_duration: float = 2.0,
    ai_model: str = "openai",
    description_length: str = "medium",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """Start video analysis with scene detection and AI description generation.
    
    Args:
        video_path: Path to uploaded video file
        theme: Optional theme for description generation
        detection_sensitivity: Scene detection sensitivity ('low', 'medium', 'high')
        min_scene_duration: Minimum scene duration in seconds
        ai_model: AI model to use ('openai', 'claude', 'gemini', 'llava')
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
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_video_analysis,
        job_id=job_id,
        video_path=video_path,
        theme=theme,
        detection_sensitivity=detection_sensitivity,
        min_scene_duration=min_scene_duration,
        ai_model=ai_model,
        description_length=description_length,
        start_time=start_time,
        end_time=end_time
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video analysis started in background",
        "check_status_url": f"/api/status/{job_id}"
    }


def process_video_analysis(
    job_id: str,
    video_path: str,
    theme: Optional[str],
    detection_sensitivity: str,
    min_scene_duration: float,
    ai_model: str,
    description_length: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """Background task for video analysis processing."""
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 10
        processing_jobs[job_id]["message"] = "Detecting scenes..."
        
        # Step 1: Detect scenes
        detector = SimpleSceneDetector(
            sensitivity=detection_sensitivity,
            min_scene_duration=min_scene_duration
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
        processing_jobs[job_id]["message"] = f"Detected {len(scenes)} scenes. Extracting keyframes..."
        
        # Step 2: Extract keyframes
        scenes = detector.extract_keyframes(video_path, scenes, frames_per_scene=3)
        
        # Update progress
        processing_jobs[job_id]["progress"] = 60
        processing_jobs[job_id]["message"] = "Generating AI descriptions..."
        
        # Step 3: Generate AI descriptions
        describer = AIDescriber(model=ai_model)
        scenes = describer.generate_descriptions(
            scenes=scenes,
            theme=theme,
            description_length=description_length
        )
        
        # Update job with results
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["message"] = "Analysis completed successfully"
        processing_jobs[job_id]["scenes"] = scenes
        
        logger.info(f"Analysis completed for job {job_id}: {len(scenes)} scenes")
        
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
        "has_scenes": len(job.get("scenes", [])) > 0
    }


@app.get("/api/scenes/{job_id}")
async def get_scenes(job_id: str):
    """Get scene data for a completed job.
    
    Args:
        job_id: Job ID returned from analyze endpoint
        
    Returns:
        List of scenes with descriptions and timing
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    return {
        "job_id": job_id,
        "scenes": job["scenes"],
        "total_scenes": len(job["scenes"])
    }


@app.get("/api/export/srt/{job_id}")
async def export_srt(job_id: str):
    """Export scene descriptions as SRT file.
    
    Args:
        job_id: Job ID returned from analyze endpoint
        
    Returns:
        SRT file download
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    # Generate SRT content
    exporter = SRTExporter()
    srt_content = exporter.export_to_srt(job["scenes"])
    
    # Validate SRT format
    if not exporter.validate_srt(srt_content):
        raise HTTPException(status_code=500, detail="Generated SRT content is invalid")
    
    # Save to temporary file
    srt_path = Path(f"exports/{job_id}.srt")
    srt_path.parent.mkdir(exist_ok=True)
    
    try:
        exporter.save_to_file(srt_content, str(srt_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save SRT file: {e}")
    
    # Return file for download
    return FileResponse(
        path=srt_path,
        filename=f"scenes_{job_id}.srt",
        media_type="application/x-subrip"
    )


@app.put("/api/scenes/{scene_id}")
async def update_scene_description(
    job_id: str,
    scene_id: int,
    description: str
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
            status_code=400, 
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    # Find and update scene
    for scene in job["scenes"]:
        if scene["scene_id"] == scene_id:
            scene["description"] = description
            return {
                "job_id": job_id,
                "scene_id": scene_id,
                "description": description,
                "message": "Scene description updated successfully"
            }
    
    raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)