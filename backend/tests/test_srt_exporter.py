"""Tests for SRT exporter module."""

import pytest
from src.srt_exporter import SRTExporter


def test_format_timestamp():
    """Test timestamp formatting."""
    exporter = SRTExporter()
    
    # Test various timestamps
    assert exporter._format_timestamp(0) == "00:00:00,000"
    assert exporter._format_timestamp(1.5) == "00:00:01,500"
    assert exporter._format_timestamp(65.123) == "00:01:05,123"
    assert exporter._format_timestamp(3661.999) == "01:01:01,999"


def test_wrap_text():
    """Test text wrapping."""
    exporter = SRTExporter(max_line_length=20)
    
    # Short text shouldn't wrap
    short_text = "Hello world"
    assert exporter._wrap_text(short_text) == ["Hello world"]
    
    # Long text should wrap
    long_text = "This is a very long description that needs to be wrapped properly"
    wrapped = exporter._wrap_text(long_text)
    assert len(wrapped) > 1
    for line in wrapped:
        assert len(line) <= 20
    
    # Empty text
    assert exporter._wrap_text("") == [""]


def test_validate_timestamp_format():
    """Test timestamp validation."""
    exporter = SRTExporter()
    
    # Valid timestamps
    assert exporter._validate_timestamp_format("00:00:00,000 --> 00:00:01,000")
    assert exporter._validate_timestamp_format("01:23:45,678 --> 01:23:46,789")
    
    # Invalid timestamps
    assert not exporter._validate_timestamp_format("00:00:00 --> 00:00:01")  # Missing milliseconds
    assert not exporter._validate_timestamp_format("00:00:00,000 -> 00:00:01,000")  # Wrong arrow
    assert not exporter._validate_timestamp_format("00:00:00,0000 --> 00:00:01,000")  # 4-digit milliseconds
    assert not exporter._validate_timestamp_format("25:00:00,000 --> 00:00:01,000")  # Invalid hour


def test_validate_srt():
    """Test SRT validation."""
    exporter = SRTExporter()
    
    # Valid SRT
    valid_srt = """1
00:00:00,000 --> 00:00:01,500
Hello world

2
00:00:01,500 --> 00:00:03,000
This is a test
"""
    assert exporter.validate_srt(valid_srt)
    
    # Invalid SRT (missing timestamp)
    invalid_srt = """1
Hello world
"""
    assert not exporter.validate_srt(invalid_srt)


def test_export_to_srt():
    """Test SRT export with sample scenes."""
    exporter = SRTExporter()
    
    scenes = [
        {
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": 5.5,
            "duration": 5.5,
            "description": "A person drills holes into table legs",
            "keyframes": [],
            "confidence_score": 0.95,
            "theme_applied": "DIY furniture build"
        },
        {
            "scene_id": 2,
            "start_time": 5.5,
            "end_time": 12.3,
            "duration": 6.8,
            "description": "Attaching motor brackets to the table",
            "keyframes": [],
            "confidence_score": 0.92,
            "theme_applied": "DIY furniture build"
        }
    ]
    
    srt_content = exporter.export_to_srt(scenes, include_metadata=True)
    
    # Basic validation
    assert "SceneScriber" not in srt_content  # Should not include branding in SRT
    assert "DIY furniture build" in srt_content  # Should include theme in metadata
    assert "00:00:00,000 --> 00:00:05,500" in srt_content
    assert "00:00:05,500 --> 00:00:12,300" in srt_content
    assert "A person drills holes into table legs" in srt_content
    
    # Should be valid SRT
    assert exporter.validate_srt(srt_content)


def test_export_to_srt_no_metadata():
    """Test SRT export without metadata."""
    exporter = SRTExporter()
    
    scenes = [
        {
            "scene_id": 1,
            "start_time": 0.0,
            "end_time": 1.0,
            "duration": 1.0,
            "description": "Test",
            "keyframes": [],
            "confidence_score": 1.0,
            "theme_applied": None
        }
    ]
    
    srt_content = exporter.export_to_srt(scenes, include_metadata=False)
    
    # Should not include metadata comments
    assert not srt_content.startswith("#")
    assert "00:00:00,000 --> 00:00:01,000" in srt_content
    assert exporter.validate_srt(srt_content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])