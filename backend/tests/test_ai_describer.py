"""Tests for AI describer module."""

import pytest
from unittest.mock import Mock, patch
from src.ai_describer import AIDescriber


def test_ai_describer_initialization():
    """Test AI describer initialization with different models."""
    # Test with OpenAI
    describer = AIDescriber(model="openai", api_key="test-key")
    assert describer.model == "openai"
    assert describer.api_key == "test-key"
    
    # Test with Claude
    describer = AIDescriber(model="claude", api_key="test-key")
    assert describer.model == "claude"
    
    # Test with no API key (should use environment variable)
    describer = AIDescriber(model="openai", api_key=None)
    assert describer.api_key is None


def test_build_prompt():
    """Test prompt building with different themes and lengths."""
    describer = AIDescriber(model="openai")
    
    # Test with no theme
    prompt = describer._build_prompt(theme=None, description_length="medium")
    assert "Analyze these video frames" in prompt
    assert "Provide a concise description" in prompt
    
    # Test with theme
    prompt = describer._build_prompt(theme="DIY furniture build", description_length="medium")
    assert "DIY furniture build" in prompt
    assert "tailor your description" in prompt
    
    # Test with different lengths
    short_prompt = describer._build_prompt(theme=None, description_length="short")
    medium_prompt = describer._build_prompt(theme=None, description_length="medium")
    detailed_prompt = describer._build_prompt(theme=None, description_length="detailed")
    
    assert "brief description" in short_prompt.lower()
    assert "concise description" in medium_prompt.lower()
    assert "detailed description" in detailed_prompt.lower()


def test_get_max_tokens():
    """Test token calculation based on description length."""
    describer = AIDescriber(model="openai")
    
    assert describer._get_max_tokens("short") == 100
    assert describer._get_max_tokens("medium") == 200
    assert describer._get_max_tokens("detailed") == 400
    assert describer._get_max_tokens("unknown") == 200  # Default


@patch('src.ai_describer.requests.post')
def test_describe_with_openai(mock_post):
    """Test OpenAI description generation (mocked)."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test description"}}]
    }
    mock_post.return_value = mock_response
    
    describer = AIDescriber(model="openai", api_key="test-key")
    
    # Mock keyframes (empty files for testing)
    keyframes = ["test_frame1.jpg", "test_frame2.jpg"]
    
    # Mock file reading
    with patch('builtins.open', Mock()):
        with patch('src.ai_describer.base64.b64encode', return_value=b"test"):
            description = describer._describe_with_openai(
                keyframes=keyframes,
                theme="DIY furniture",
                description_length="medium"
            )
    
    assert description == "Test description"
    mock_post.assert_called_once()


@patch('src.ai_describer.requests.post')
def test_describe_with_openai_error(mock_post):
    """Test OpenAI error handling."""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "API error"
    mock_post.return_value = mock_response
    
    describer = AIDescriber(model="openai", api_key="test-key")
    
    with patch('builtins.open', Mock()):
        with patch('src.ai_describer.base64.b64encode', return_value=b"test"):
            with pytest.raises(ValueError, match="OpenAI API error"):
                describer._describe_with_openai(
                    keyframes=["test.jpg"],
                    theme=None,
                    description_length="medium"
                )


def test_describe_with_unsupported_model():
    """Test error for unsupported model."""
    describer = AIDescriber(model="unknown", api_key="test-key")
    
    with pytest.raises(ValueError, match="Unsupported model"):
        describer._describe_scene(
            keyframes=["test.jpg"],
            theme=None,
            description_length="medium"
        )


def test_generate_descriptions():
    """Test batch description generation."""
    describer = AIDescriber(model="openai", api_key="test-key")
    
    scenes = [
        {
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": 5.0,
            "duration": 5.0,
            "description": "",
            "keyframes": ["frame1.jpg", "frame2.jpg"],
            "confidence_score": 0.9,
            "theme_applied": None
        },
        {
            "scene_id": 2,
            "start_time": 5.0,
            "end_time": 10.0,
            "duration": 5.0,
            "description": "",
            "keyframes": [],  # No keyframes
            "confidence_score": 0.9,
            "theme_applied": None
        }
    ]
    
    # Mock the _describe_scene method
    with patch.object(describer, '_describe_scene', return_value="AI description"):
        updated_scenes = describer.generate_descriptions(
            scenes=scenes,
            theme="DIY furniture",
            description_length="medium"
        )
    
    assert len(updated_scenes) == 2
    assert updated_scenes[0]["description"] == "AI description"
    assert updated_scenes[0]["theme_applied"] == "DIY furniture"
    assert "No description available" in updated_scenes[1]["description"]  # No keyframes


def test_generate_descriptions_with_error():
    """Test description generation with AI error."""
    describer = AIDescriber(model="openai", api_key="test-key")
    
    scenes = [
        {
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": 5.0,
            "duration": 5.0,
            "description": "",
            "keyframes": ["frame1.jpg"],
            "confidence_score": 0.9,
            "theme_applied": None
        }
    ]
    
    # Mock AI to raise an error
    with patch.object(describer, '_describe_scene', side_effect=Exception("AI error")):
        updated_scenes = describer.generate_descriptions(
            scenes=scenes,
            theme=None,
            description_length="medium"
        )
    
    assert "Failed to generate description" in updated_scenes[0]["description"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])