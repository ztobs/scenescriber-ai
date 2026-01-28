#!/usr/bin/env python3
"""Test keyframe selection logic with offset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/src'))

# Test the logic without actually extracting frames
def test_scene_detector_logic():
    """Test the frame position calculation logic."""
    
    # Simulate a scene with 30 FPS, 10 seconds duration
    fps = 30.0
    start_time = 10.0  # Scene starts at 10 seconds
    end_time = 20.0    # Scene ends at 20 seconds
    duration = end_time - start_time
    
    print(f"Scene: {start_time}s to {end_time}s (duration: {duration}s)")
    print(f"FPS: {fps}")
    
    # Calculate frame positions as in scene_detector.py
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"\nOriginal frame positions:")
    print(f"  Start frame: {start_frame} (time: {start_frame/fps:.3f}s)")
    print(f"  End frame: {end_frame} (time: {end_frame/fps:.3f}s)")
    
    # Apply 500ms offset
    offset_frames = int(fps * 0.5)  # 500ms offset
    adjusted_start = start_frame + offset_frames
    adjusted_end = end_frame - offset_frames
    
    print(f"\nWith 500ms offset ({offset_frames} frames):")
    print(f"  Adjusted start: {adjusted_start} (time: {adjusted_start/fps:.3f}s)")
    print(f"  Adjusted end: {adjusted_end} (time: {adjusted_end/fps:.3f}s)")
    
    # Ensure adjusted positions are valid
    if adjusted_end <= adjusted_start:
        print("  Warning: Scene too short for offset, using original positions")
        adjusted_start = start_frame
        adjusted_end = end_frame
    
    # Calculate middle frame
    middle_frame = start_frame + (end_frame - start_frame) // 2
    
    print(f"\nKeyframe positions:")
    print(f"  1. Start (with offset): {adjusted_start} (time: {adjusted_start/fps:.3f}s)")
    print(f"  2. Middle: {middle_frame} (time: {middle_frame/fps:.3f}s)")
    print(f"  3. End (with offset): {max(adjusted_start, adjusted_end - 1)} (time: {max(adjusted_start, adjusted_end - 1)/fps:.3f}s)")
    
    # Test with a very short scene (1 second)
    print(f"\n--- Testing short scene (1 second) ---")
    start_time_short = 10.0
    end_time_short = 11.0
    start_frame_short = int(start_time_short * fps)
    end_frame_short = int(end_time_short * fps)
    
    print(f"Scene: {start_time_short}s to {end_time_short}s (duration: 1s)")
    
    adjusted_start_short = start_frame_short + offset_frames
    adjusted_end_short = end_frame_short - offset_frames
    
    if adjusted_end_short <= adjusted_start_short:
        print("  Scene too short for offset, using original positions")
        adjusted_start_short = start_frame_short
        adjusted_end_short = end_frame_short
    
    print(f"  Keyframes will be at: {adjusted_start_short}, middle, {max(adjusted_start_short, adjusted_end_short - 1)}")

def test_simple_detector_logic():
    """Test the time-based logic as in scene_detector_simple.py."""
    
    print(f"\n\n=== Testing Simple Scene Detector Logic ===")
    
    # Simulate a scene
    start_time = 10.0
    end_time = 20.0
    duration = end_time - start_time
    
    print(f"Scene: {start_time}s to {end_time}s (duration: {duration}s)")
    
    offset_seconds = 0.5  # 500ms offset
    
    adjusted_start = start_time + offset_seconds
    adjusted_end = end_time - offset_seconds
    
    print(f"\nWith 500ms offset:")
    print(f"  Adjusted start: {adjusted_start:.3f}s")
    print(f"  Adjusted end: {adjusted_end:.3f}s")
    
    if adjusted_end <= adjusted_start:
        print("  Scene too short for offset, using original times")
        adjusted_start = start_time
        adjusted_end = end_time
    
    middle_time = start_time + duration / 2
    
    print(f"\nKeyframe times:")
    print(f"  1. Start (with offset): {adjusted_start:.3f}s")
    print(f"  2. Middle: {middle_time:.3f}s")
    print(f"  3. End (with offset): {adjusted_end:.3f}s")

if __name__ == "__main__":
    test_scene_detector_logic()
    test_simple_detector_logic()