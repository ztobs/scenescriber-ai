"""Simple scene detection module without OpenCV dependency."""

import logging
import subprocess
import json
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SimpleSceneDetector:
    """Simple scene detector using ffmpeg and basic image comparison."""

    def __init__(self, sensitivity: str = "medium", min_scene_duration: float = 2.0):
        """Initialize scene detector with configuration.

        Args:
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            min_scene_duration: Minimum scene duration in seconds
        """
        self.sensitivity = sensitivity
        self.min_scene_duration = min_scene_duration
        self._setup_threshold()

    def _setup_threshold(self) -> None:
        """Set threshold based on sensitivity level."""
        thresholds = {
            "low": 0.3,  # Higher threshold = fewer scene cuts
            "medium": 0.2,  # Medium threshold = balanced detection
            "high": 0.1,  # Lower threshold = more scene cuts
        }
        self.difference_threshold = thresholds.get(self.sensitivity, 0.2)

    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect scene boundaries in video using ffmpeg scene detection.

        Args:
            video_path: Path to video file

        Returns:
            List of scene dictionaries with start_time, end_time, duration
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Detecting scenes in {video_path} with sensitivity={self.sensitivity}")

        try:
            # Use ffmpeg's scene detection
            scenes = self._detect_with_ffmpeg(video_path)

            # Filter scenes by minimum duration
            filtered_scenes = []
            for scene in scenes:
                if scene["duration"] >= self.min_scene_duration:
                    filtered_scenes.append(scene)

            logger.info(f"Detected {len(filtered_scenes)} scenes (after filtering)")
            return filtered_scenes

        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            # Fallback: split video into equal segments
            return self._fallback_scene_detection(video_path)

    def _detect_with_ffmpeg(self, video_path: str) -> List[Dict[str, Any]]:
        """Use ffmpeg's scene detection filter."""
        try:
            # Get video duration
            duration = self._get_video_duration(video_path)
            if duration <= 0:
                raise ValueError(f"Invalid video duration: {duration}")

            # Create temp file for scene detection output
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
                tmp_file = tmp.name

            # Run ffmpeg scene detection
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"select='gt(scene,{self.difference_threshold})',metadata=print:file={tmp_file}",
                "-vsync",
                "vfr",
                "-f",
                "null",
                "-",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"ffmpeg scene detection failed: {result.stderr}")
                raise RuntimeError("ffmpeg scene detection failed")

            # Parse scene detection output
            scenes = self._parse_ffmpeg_output(tmp_file, duration)

            # Clean up temp file
            os.unlink(tmp_file)

            return scenes

        except Exception as e:
            logger.warning(f"ffmpeg scene detection error: {e}")
            # Clean up temp file if it exists
            if "tmp_file" in locals() and os.path.exists(tmp_file):
                os.unlink(tmp_file)
            raise

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")

        return 0.0

    def _parse_ffmpeg_output(self, output_file: str, total_duration: float) -> List[Dict[str, Any]]:
        """Parse ffmpeg scene detection output."""
        scenes = []

        try:
            with open(output_file, "r") as f:
                lines = f.readlines()

            # Extract scene change timestamps
            scene_times = []
            for line in lines:
                if "pts_time:" in line:
                    # Extract timestamp
                    parts = line.split("pts_time:")
                    if len(parts) > 1:
                        time_str = parts[1].strip()
                        try:
                            timestamp = float(time_str)
                            scene_times.append(timestamp)
                        except ValueError:
                            continue

            # Sort and deduplicate timestamps
            scene_times = sorted(set(scene_times))

            # Create scenes from timestamps
            prev_time = 0.0
            for i, scene_time in enumerate(scene_times):
                scene_duration = scene_time - prev_time

                scenes.append(
                    {
                        "scene_id": i + 1,
                        "start_time": prev_time,
                        "end_time": scene_time,
                        "duration": scene_duration,
                        "description": "",
                        "keyframes": [],
                        "confidence_score": 0.9,
                        "theme_applied": None,
                    }
                )

                prev_time = scene_time

            # Add final scene
            if prev_time < total_duration:
                final_duration = total_duration - prev_time
                scenes.append(
                    {
                        "scene_id": len(scenes) + 1,
                        "start_time": prev_time,
                        "end_time": total_duration,
                        "duration": final_duration,
                        "description": "",
                        "keyframes": [],
                        "confidence_score": 0.9,
                        "theme_applied": None,
                    }
                )

        except Exception as e:
            logger.warning(f"Error parsing ffmpeg output: {e}")

        return scenes

    def _fallback_scene_detection(self, video_path: str) -> List[Dict[str, Any]]:
        """Fallback scene detection: split video into equal segments."""
        try:
            duration = self._get_video_duration(video_path)
            if duration <= 0:
                duration = 300  # Default 5 minutes

            # Split into 10-second segments (or use min_scene_duration)
            segment_duration = max(self.min_scene_duration, 10.0)
            num_segments = int(duration / segment_duration)

            if num_segments < 1:
                num_segments = 1

            actual_segment_duration = duration / num_segments

            scenes = []
            for i in range(num_segments):
                start_time = i * actual_segment_duration
                end_time = (i + 1) * actual_segment_duration if i < num_segments - 1 else duration

                scenes.append(
                    {
                        "scene_id": i + 1,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "description": "",
                        "keyframes": [],
                        "confidence_score": 0.5,  # Low confidence for fallback
                        "theme_applied": None,
                    }
                )

            logger.info(f"Fallback: Created {len(scenes)} equal segments")
            return scenes

        except Exception as e:
            logger.error(f"Fallback scene detection failed: {e}")
            # Return at least one scene
            return [
                {
                    "scene_id": 1,
                    "start_time": 0.0,
                    "end_time": 300.0,  # 5 minutes default
                    "duration": 300.0,
                    "description": "",
                    "keyframes": [],
                    "confidence_score": 0.1,  # Very low confidence
                    "theme_applied": None,
                }
            ]

    def extract_keyframes(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
        frames_per_scene: int = 3,
        output_dir: str = None,
    ) -> List[Dict[str, Any]]:
        """Extract keyframes using ffmpeg.

        Args:
            video_path: Path to video file
            scenes: List of scene dictionaries
            frames_per_scene: Number of frames to extract per scene
            output_dir: Directory to save keyframes (optional, defaults to temp)

        Returns:
            Updated scenes with keyframe paths
        """
        logger.info(f"Extracting {frames_per_scene} keyframes per scene")

        for scene in scenes:
            keyframes = []

            # Calculate frame extraction times
            start_time = scene["start_time"]
            end_time = scene["end_time"]
            duration = scene["duration"]

            if duration <= 0:
                continue

            # Extract frames at start, middle, and end
            frame_times = [
                start_time,
                start_time + duration / 2,
                end_time - 0.1,  # Slightly before end
            ]

            for i, frame_time in enumerate(frame_times[:frames_per_scene]):
                try:
                    if output_dir:
                        # Use provided directory
                        filename = f"scene_{scene['scene_id']}_frame_{i}.jpg"
                        # Make filename unique to avoid collisions if multiple jobs run
                        import uuid

                        filename = f"{uuid.uuid4()}_{filename}"
                        frame_file = os.path.join(output_dir, filename)
                    else:
                        # Create temp file for frame
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            frame_file = tmp.name

                    # Extract frame using ffmpeg
                    cmd = [
                        "ffmpeg",
                        "-y",  # Overwrite output file
                        "-ss",
                        str(frame_time),  # Seek to time
                        "-i",
                        video_path,
                        "-vframes",
                        "1",  # Extract 1 frame
                        "-q:v",
                        "2",  # Quality
                        frame_file,
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0 and os.path.exists(frame_file):
                        # Check if file has content
                        if os.path.getsize(frame_file) > 0:
                            keyframes.append(frame_file)
                        else:
                            os.unlink(frame_file)
                    else:
                        if os.path.exists(frame_file):
                            os.unlink(frame_file)

                except Exception as e:
                    logger.warning(f"Error extracting frame at {frame_time}s: {e}")

            scene["keyframes"] = keyframes

        return scenes

    def calculate_frame_difference(self, image1_path: str, image2_path: str) -> float:
        """Calculate difference between two images using PIL."""
        try:
            img1 = Image.open(image1_path).convert("L")  # Convert to grayscale
            img2 = Image.open(image2_path).convert("L")

            # Resize to same dimensions if needed
            if img1.size != img2.size:
                # Use smaller dimensions
                width = min(img1.width, img2.width)
                height = min(img1.height, img2.height)
                img1 = img1.resize((width, height))
                img2 = img2.resize((width, height))

            # Convert to numpy arrays
            arr1 = np.array(img1).astype(float)
            arr2 = np.array(img2).astype(float)

            # Calculate mean absolute difference
            diff = np.mean(np.abs(arr1 - arr2)) / 255.0  # Normalize to 0-1

            return diff

        except Exception as e:
            logger.warning(f"Error calculating frame difference: {e}")
            return 0.5  # Default difference
