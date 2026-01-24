"""Scene detection module using histogram comparison."""

import logging
import subprocess
import json
from typing import List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detects scene boundaries in video files using histogram comparison."""

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
            "low": 0.85,      # Higher threshold = fewer scene cuts
            "medium": 0.75,   # Medium threshold = balanced detection
            "high": 0.65,     # Lower threshold = more scene cuts
        }
        self.similarity_threshold = thresholds.get(self.sensitivity, 0.75)

    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect scene boundaries in video using histogram comparison.
        
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
            # Get video information using ffprobe
            video_info = self._get_video_info(video_path)
            duration = float(video_info.get('duration', 0))
            fps = float(video_info.get('fps', 30))
            
            if duration <= 0 or fps <= 0:
                raise ValueError(f"Invalid video duration ({duration}s) or FPS ({fps})")
            
            # Calculate frame interval for sampling (sample every 0.5 seconds)
            frame_interval = int(fps * 0.5)
            if frame_interval < 1:
                frame_interval = 1
            
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # Estimate total frames if not available
                total_frames = int(duration * fps)
            
            # Sample frames and detect scene changes
            scene_start = 0.0
            prev_frame = None
            prev_time = 0.0
            
            frame_positions = range(0, total_frames, frame_interval)
            
            for frame_num in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                current_time = frame_num / fps
                
                if prev_frame is not None:
                    # Calculate similarity between frames
                    similarity = self.calculate_frame_similarity(prev_frame, frame)
                    
                    # If similarity is below threshold, it's a scene change
                    if similarity < self.similarity_threshold:
                        scene_duration = current_time - scene_start
                        
                        # Only add scene if it meets minimum duration
                        if scene_duration >= self.min_scene_duration:
                            scenes.append({
                                "scene_id": len(scenes) + 1,
                                "start_time": scene_start,
                                "end_time": current_time,
                                "duration": scene_duration,
                                "description": "",
                                "keyframes": [],
                                "confidence_score": 1.0 - similarity,  # Higher difference = higher confidence
                                "theme_applied": None
                            })
                        
                        scene_start = current_time
                
                prev_frame = frame
                prev_time = current_time
            
            # Add final scene
            if scene_start < duration:
                scene_duration = duration - scene_start
                if scene_duration >= self.min_scene_duration:
                    scenes.append({
                        "scene_id": len(scenes) + 1,
                        "start_time": scene_start,
                        "end_time": duration,
                        "duration": scene_duration,
                        "description": "",
                        "keyframes": [],
                        "confidence_score": 0.9,  # Default confidence for final scene
                        "theme_applied": None
                    })
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            raise ValueError(f"Failed to detect scenes: {e}")

        logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"ffprobe failed: {result.stderr}")
            
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_info = {
                'duration': 0,
                'fps': 30,
                'width': 0,
                'height': 0
            }
            
            # Find video stream
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Get duration
                    duration = stream.get('duration')
                    if not duration:
                        duration = info.get('format', {}).get('duration', 0)
                    video_info['duration'] = float(duration) if duration else 0
                    
                    # Get FPS
                    fps_str = stream.get('avg_frame_rate', '30/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        if float(den) > 0:
                            video_info['fps'] = float(num) / float(den)
                    
                    # Get dimensions
                    width = stream.get('width', '0')
                    height = stream.get('height', '0')
                    video_info['width'] = int(float(width)) if width and width != '0' else 0
                    video_info['height'] = int(float(height)) if height and height != '0' else 0
                    break
            
            return video_info
            
        except Exception as e:
            logger.warning(f"Could not get video info with ffprobe: {e}")
            # Return default values
            return {
                'duration': 0,
                'fps': 30,
                'width': 0,
                'height': 0
            }

    def calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames using multiple methods.
        
        Args:
            frame1: First frame as numpy array
            frame2: Second frame as numpy array
            
        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        try:
            # Convert to grayscale for comparison
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frame1
                gray2 = frame2
            
            # Method 1: Structural Similarity Index (SSIM)
            ssim_result = ssim(gray1, gray2, full=True)
            if isinstance(ssim_result, tuple):
                ssim_score = ssim_result[0]  # First element is the score
            else:
                ssim_score = ssim_result
            
            # Method 2: Histogram correlation
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Combine scores (weighted average)
            similarity = (ssim_score * 0.7) + (hist_corr * 0.3)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating frame similarity: {e}")
            return 0.5  # Default similarity if calculation fails

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
                max(start_frame, end_frame - 1)  # Ensure not negative
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
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
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