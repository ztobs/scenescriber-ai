"""Tests for data models."""

import pytest
from datetime import datetime
from src.models import Scene, ProjectConfig, AnalysisRequest, AnalysisResponse


def test_scene_model():
    """Test Scene model creation and validation."""
    scene = Scene(
        scene_id=1,
        start_time=0.0,
        end_time=5.5,
        duration=5.5,
        description="Test scene description",
        keyframes=["frame1.jpg", "frame2.jpg"],
        confidence_score=0.95,
        theme_applied="DIY furniture"
    )
    
    assert scene["scene_id"] == 1
    assert scene["start_time"] == 0.0
    assert scene["end_time"] == 5.5
    assert scene["duration"] == 5.5
    assert scene["description"] == "Test scene description"
    assert len(scene["keyframes"]) == 2
    assert scene["confidence_score"] == 0.95
    assert scene["theme_applied"] == "DIY furniture"


def test_project_config_model():
    """Test ProjectConfig model creation."""
    now = datetime.now()
    config = ProjectConfig(
        project_id="test-uuid",
        video_filename="test_video.mp4",
        theme="DIY furniture build",
        detection_sensitivity="medium",
        min_scene_duration=2.0,
        ai_model="openai",
        created_at=now,
        total_scenes=10
    )
    
    assert config["project_id"] == "test-uuid"
    assert config["video_filename"] == "test_video.mp4"
    assert config["theme"] == "DIY furniture build"
    assert config["detection_sensitivity"] == "medium"
    assert config["min_scene_duration"] == 2.0
    assert config["ai_model"] == "openai"
    assert config["created_at"] == now
    assert config["total_scenes"] == 10


def test_analysis_request_model():
    """Test AnalysisRequest model creation."""
    request = AnalysisRequest(
        video_path="/path/to/video.mp4",
        theme="Cooking tutorial",
        detection_sensitivity="high",
        min_scene_duration=1.5,
        ai_model="claude"
    )
    
    assert request["video_path"] == "/path/to/video.mp4"
    assert request["theme"] == "Cooking tutorial"
    assert request["detection_sensitivity"] == "high"
    assert request["min_scene_duration"] == 1.5
    assert request["ai_model"] == "claude"


def test_analysis_response_model():
    """Test AnalysisResponse model creation."""
    scenes = [
        Scene(
            scene_id=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            description="Scene 1",
            keyframes=[],
            confidence_score=0.9,
            theme_applied=None
        )
    ]
    
    response = AnalysisResponse(
        project_id="test-uuid",
        scenes=scenes,
        processing_time=12.5,
        success=True,
        error_message=None
    )
    
    assert response["project_id"] == "test-uuid"
    assert len(response["scenes"]) == 1
    assert response["scenes"][0]["scene_id"] == 1
    assert response["processing_time"] == 12.5
    assert response["success"] is True
    assert response["error_message"] is None


def test_analysis_response_with_error():
    """Test AnalysisResponse model with error."""
    response = AnalysisResponse(
        project_id="test-uuid",
        scenes=[],
        processing_time=0.0,
        success=False,
        error_message="Video file not found"
    )
    
    assert response["success"] is False
    assert response["error_message"] == "Video file not found"
    assert len(response["scenes"]) == 0


def test_optional_fields():
    """Test that optional fields can be None."""
    scene = Scene(
        scene_id=1,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        description="Test",
        keyframes=[],
        confidence_score=1.0,
        theme_applied=None  # This should be allowed
    )
    
    assert scene["theme_applied"] is None
    
    request = AnalysisRequest(
        video_path="/path/to/video.mp4",
        theme=None,  # This should be allowed
        detection_sensitivity="medium",
        min_scene_duration=2.0,
        ai_model="openai"
    )
    
    assert request["theme"] is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])