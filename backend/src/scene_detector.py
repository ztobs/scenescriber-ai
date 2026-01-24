"""Scene detection module using PySceneDetect library."""

import logging
from typing import List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detects scene boundaries in video files."""

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
            "low": 40.0,
            "medium": 30.0,
            "high": 20.0,
        }
        self.threshold = thresholds.get(self.sensitivity, 30.0)

    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect scene boundaries in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of scene dictionaries with start_time, end_time, duration
            
        Raises:
            FileNotFoundError: If video file not found
            ValueError: If video cannot be processed
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Detecting scenes in {video_path} with sensitivity={self.sensitivity}")
        
        scenes = []
        try:
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=self.threshold, min_scene_len=int(self.min_scene_duration * 1000))
            )

            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()

            for i, (start_time, end_time) in enumerate(scene_list):
                duration = (end_time - start_time).get_seconds()
                scenes.append({
                    "scene_id": i + 1,
                    "start_time": start_time.get_seconds(),
                    "end_time": end_time.get_seconds(),
                    "duration": duration,
                    "description": "",
                    "keyframes": [],
                    "confidence_score": 1.0,
                    "theme_applied": None
                })

            video_manager.release()
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            raise ValueError(f"Failed to detect scenes: {e}")

        logger.info(f"Detected {len(scenes)} scenes")
        return scenes

    def extract_keyframes(self, video_path: str, scenes: List[Dict[str, Any]], frames_per_scene: int = 3) -> List[Dict[str, Any]]:
        """Extract keyframes for each detected scene.
        
        Args:
            video_path: Path to video file
            scenes: List of scene dictionaries
            frames_per_scene: Number of frames to extract per scene
            
        Returns:
            Updated scenes with keyframe paths
        """
        logger.info(f"Extracting {frames_per_scene} keyframes per scene")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            logger.warning(f"Invalid FPS detected: {fps}, using default 30")
            fps = 30.0

        for scene in scenes:
            keyframes = []
            start_frame = int(scene["start_time"] * fps)
            end_frame = int(scene["end_time"] * fps)
            
            # Extract frames at start, middle, and end of scene
            frame_positions = [
                start_frame,
                start_frame + (end_frame - start_frame) // 2,
                end_frame - 1
            ]
            
            for frame_pos in frame_positions[:frames_per_scene]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if ret:
                    # Save frame as image
                    frame_path = f"keyframe_scene{scene['scene_id']}_frame{frame_pos}.jpg"
                    cv2.imwrite(frame_path, frame)
                    keyframes.append(frame_path)
            
            scene["keyframes"] = keyframes
        
        cap.release()
        return scenes

    def calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between two frames.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            Histogram difference score (0-100)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Convert to difference score (0-100)
        difference = (1 - correlation) * 100
        return difference