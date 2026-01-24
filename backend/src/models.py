"""Data models for the Video Scene AI Analyzer."""

from typing import List, Optional, TypedDict
from datetime import datetime


class Scene(TypedDict):
    """Represents a detected scene in a video."""
    scene_id: int
    start_time: float
    end_time: float
    duration: float
    description: str
    keyframes: List[str]
    confidence_score: float
    theme_applied: Optional[str]


class ProjectConfig(TypedDict):
    """Project configuration for video analysis."""
    project_id: str
    video_filename: str
    theme: Optional[str]
    detection_sensitivity: str  # 'low', 'medium', 'high'
    min_scene_duration: float
    ai_model: str
    created_at: datetime
    total_scenes: int


class AnalysisRequest(TypedDict):
    """Request for video analysis."""
    video_path: str
    theme: Optional[str]
    detection_sensitivity: str
    min_scene_duration: float
    ai_model: str
    start_time: Optional[float]  # Start time in seconds (0 = beginning)
    end_time: Optional[float]    # End time in seconds (None = end of video)


class AnalysisResponse(TypedDict):
    """Response from video analysis."""
    project_id: str
    scenes: List[Scene]
    processing_time: float
    success: bool
    error_message: Optional[str]