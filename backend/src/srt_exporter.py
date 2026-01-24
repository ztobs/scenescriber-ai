"""SRT (SubRip Subtitle) file export module."""

import logging
from typing import List, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)


class SRTExporter:
    """Exports scene descriptions to SRT subtitle format."""

    def __init__(self, max_line_length: int = 32):
        """Initialize SRT exporter.
        
        Args:
            max_line_length: Maximum characters per line before wrapping
        """
        self.max_line_length = max_line_length

    def export_to_srt(self, scenes: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """Convert scenes to SRT format.
        
        Args:
            scenes: List of scene dictionaries with timing and descriptions
            include_metadata: Whether to include metadata comments
            
        Returns:
            SRT formatted string
        """
        logger.info(f"Exporting {len(scenes)} scenes to SRT format")
        
        srt_lines = []
        
        if include_metadata:
            srt_lines.extend(self._generate_metadata_header(scenes))
        
        for i, scene in enumerate(scenes):
            srt_lines.extend(self._format_scene_as_srt(scene, i + 1))
        
        return "\n".join(srt_lines)

    def _format_scene_as_srt(self, scene: Dict[str, Any], sequence_number: int) -> List[str]:
        """Format a single scene as SRT entry.
        
        Args:
            scene: Scene dictionary
            sequence_number: SRT sequence number
            
        Returns:
            List of lines for this SRT entry
        """
        lines = []
        
        # Sequence number
        lines.append(str(sequence_number))
        
        # Timecodes
        start_time = self._format_timestamp(scene["start_time"])
        end_time = self._format_timestamp(scene["end_time"])
        lines.append(f"{start_time} --> {end_time}")
        
        # Description text with wrapping
        description = scene.get("description", "")
        wrapped_text = self._wrap_text(description)
        lines.extend(wrapped_text)
        
        # Empty line between entries
        lines.append("")
        
        return lines

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        
        # Extract components
        total_seconds = int(td.total_seconds())
        milliseconds = int((seconds - total_seconds) * 1000)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds_remainder = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:02d},{milliseconds:03d}"

    def _wrap_text(self, text: str) -> List[str]:
        """Wrap text to max line length.
        
        Args:
            text: Text to wrap
            
        Returns:
            List of wrapped lines
        """
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # Check if adding this word would exceed line length
            # Account for space between words
            if current_line:
                potential_length = current_length + 1 + word_length
            else:
                potential_length = word_length
            
            if potential_length <= self.max_line_length:
                current_line.append(word)
                current_length = potential_length
            else:
                # Start new line
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
        
        # Add last line
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines

    def _generate_metadata_header(self, scenes: List[Dict[str, Any]]) -> List[str]:
        """Generate metadata comments for SRT file.
        
        Args:
            scenes: List of scenes for metadata extraction
            
        Returns:
            List of metadata comment lines
        """
        metadata = []
        
        if scenes:
            first_scene = scenes[0]
            theme = first_scene.get("theme_applied")
            
            if theme:
                metadata.append(f"# Theme: {theme}")
            
            metadata.append(f"# Total scenes: {len(scenes)}")
            metadata.append(f"# Scene detection sensitivity: {first_scene.get('detection_sensitivity', 'medium')}")
            metadata.append("")
        
        return metadata

    def validate_srt(self, srt_content: str) -> bool:
        """Validate SRT format compliance.
        
        Args:
            srt_content: SRT content to validate
            
        Returns:
            True if valid, False otherwise
        """
        lines = srt_content.strip().split("\n")
        i = 0
        
        while i < len(lines):
            # Check sequence number
            if not lines[i].strip().isdigit():
                logger.error(f"Invalid sequence number at line {i+1}: {lines[i]}")
                return False
            
            i += 1
            
            # Check timestamp line
            if i >= len(lines):
                logger.error(f"Missing timestamp line after sequence {lines[i-1]}")
                return False
            
            timestamp_line = lines[i].strip()
            if not self._validate_timestamp_format(timestamp_line):
                logger.error(f"Invalid timestamp format at line {i+1}: {timestamp_line}")
                return False
            
            i += 1
            
            # Skip text lines until empty line or next sequence
            while i < len(lines) and lines[i].strip():
                i += 1
            
            # Skip empty line
            if i < len(lines) and not lines[i].strip():
                i += 1
        
        return True

    def _validate_timestamp_format(self, timestamp_line: str) -> bool:
        """Validate SRT timestamp format.
        
        Args:
            timestamp_line: Timestamp line to validate
            
        Returns:
            True if valid format, False otherwise
        """
        if " --> " not in timestamp_line:
            return False
        
        start_time, end_time = timestamp_line.split(" --> ")
        
        for time_str in [start_time, end_time]:
            parts = time_str.split(",")
            if len(parts) != 2:
                return False
            
            time_part, ms_part = parts
            
            # Check milliseconds
            if len(ms_part) != 3 or not ms_part.isdigit():
                return False
            
            # Check HH:MM:SS
            time_parts = time_part.split(":")
            if len(time_parts) != 3:
                return False
            
            for part in time_parts:
                if len(part) != 2 or not part.isdigit():
                    return False
            
            # Validate time ranges
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = int(time_parts[2])
            
            if hours < 0 or minutes < 0 or minutes > 59 or seconds < 0 or seconds > 59:
                return False
        
        return True

    def save_to_file(self, srt_content: str, filepath: str) -> None:
        """Save SRT content to file.
        
        Args:
            srt_content: SRT formatted string
            filepath: Path to save file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(srt_content)
            logger.info(f"SRT file saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save SRT file: {e}")
            raise IOError(f"Failed to save SRT file: {e}")