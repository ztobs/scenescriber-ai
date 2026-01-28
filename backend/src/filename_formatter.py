"""Filename formatting utility for export files."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .models import ExportMetadata

logger = logging.getLogger(__name__)


class FilenameFormatter:
    """Formats export filenames based on templates with placeholders."""

    # Available placeholders
    PLACEHOLDERS = {
        "[videoname]": "video_name",
        "[sensitivity]": "sensitivity",
        "[detail]": "detail_level",
        "[provider]": "model_provider",
        "[model]": "model_name",
        "[speed]": "speed",
        "[timestamp]": "timestamp",
    }

    # Default filename format
    DEFAULT_FORMAT = "[videoname]_[timestamp]"

    # Full format with all fields (for profiling)
    FULL_FORMAT = "[videoname]_[sensitivity]_[detail]_[provider]_[model]_[speed]x_[timestamp]"

    def __init__(self):
        """Initialize the FilenameFormatter."""
        pass

    def format_filename(
        self, metadata: ExportMetadata, template: str = DEFAULT_FORMAT
    ) -> str:
        """Format filename using template and metadata.

        Args:
            metadata: Export metadata containing all placeholder values
            template: Filename template with placeholders (e.g., "[videoname]_[timestamp]")

        Returns:
            Formatted filename string

        Raises:
            ValueError: If template contains invalid placeholders
        """
        logger.info(f"Formatting filename with template: {template}")

        # Validate template
        self._validate_template(template)

        # Create a dictionary for replacement
        replacements = self._create_replacements(metadata)

        # Replace all placeholders
        filename = template
        for placeholder, value in replacements.items():
            filename = filename.replace(placeholder, str(value))

        # Sanitize filename (remove invalid characters)
        filename = self._sanitize_filename(filename)

        logger.debug(f"Formatted filename: {filename}")
        return filename

    def _validate_template(self, template: str) -> None:
        """Validate that template only contains valid placeholders.

        Args:
            template: The template to validate

        Raises:
            ValueError: If template contains invalid placeholders
        """
        # Find all placeholders in template
        placeholders_in_template = re.findall(r"\[[\w]+\]", template)

        # Check if all placeholders are valid
        for placeholder in placeholders_in_template:
            if placeholder not in self.PLACEHOLDERS:
                raise ValueError(
                    f"Invalid placeholder '{placeholder}' in template. "
                    f"Valid placeholders: {', '.join(self.PLACEHOLDERS.keys())}"
                )

    def _create_replacements(self, metadata: ExportMetadata) -> Dict[str, str]:
        """Create dictionary of placeholder replacements from metadata.

        Args:
            metadata: Metadata containing values for all placeholders

        Returns:
            Dictionary mapping placeholders to their values
        """
        # Convert sensitivity to short form (low -> s1, medium -> s2, high -> s3)
        sensitivity_map = {"low": "s1", "medium": "s2", "high": "s3"}
        sensitivity_short = sensitivity_map.get(
            metadata["sensitivity"].lower(), metadata["sensitivity"]
        )

        # Convert detail level to short form (short -> d1, medium -> d2, detailed -> d3)
        detail_map = {"short": "d1", "medium": "d2", "detailed": "d3"}
        detail_short = detail_map.get(
            metadata["detail_level"].lower(), metadata["detail_level"]
        )

        return {
            "[videoname]": self._sanitize_name(metadata["video_name"]),
            "[sensitivity]": sensitivity_short,
            "[detail]": detail_short,
            "[provider]": metadata["model_provider"],
            "[model]": self._sanitize_name(metadata["model_name"]),
            "[speed]": f"{metadata['speed']:.2f}",
            "[timestamp]": metadata["timestamp"],
        }

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name by removing file extension and invalid characters.

        Args:
            name: The name to sanitize

        Returns:
            Sanitized name string
        """
        # Remove file extension if present
        name = Path(name).stem

        # Replace spaces with underscores
        name = name.replace(" ", "_")

        # Remove or replace invalid filename characters
        # Keep only alphanumeric, dash, underscore
        name = re.sub(r"[^\w\-]", "", name)

        return name

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize the complete filename.

        Args:
            filename: The filename to sanitize

        Returns:
            Sanitized filename
        """
        # Replace multiple underscores with single underscore
        filename = re.sub(r"_+", "_", filename)

        # Remove leading/trailing underscores or dashes
        filename = filename.strip("_-")

        # Ensure it's not empty
        if not filename:
            filename = "export"

        return filename

    @staticmethod
    def create_metadata(
        video_name: str,
        sensitivity: str,
        detail_level: str,
        model_provider: str,
        model_name: str,
        segment_duration: float,
        processing_time: float,
    ) -> ExportMetadata:
        """Create export metadata with calculated values.

        Args:
            video_name: Name of the video file
            sensitivity: Detection sensitivity level
            detail_level: Description detail level
            model_provider: AI model provider (e.g., "openai", "ollama")
            model_name: Full model name (e.g., "gpt-4o")
            segment_duration: Duration of processed segment in seconds
            processing_time: Time taken to process in seconds

        Returns:
            ExportMetadata dictionary
        """
        # Calculate speed (segment_duration / processing_time)
        speed = segment_duration / processing_time if processing_time > 0 else 0.0

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return ExportMetadata(
            video_name=video_name,
            sensitivity=sensitivity,
            detail_level=detail_level,
            model_provider=model_provider,
            model_name=model_name,
            speed=speed,
            timestamp=timestamp,
            segment_duration=segment_duration,
            processing_time=processing_time,
        )

    @classmethod
    def get_default_format(cls) -> str:
        """Get the default filename format.

        Returns:
            Default format string
        """
        return cls.DEFAULT_FORMAT

    @classmethod
    def get_full_format(cls) -> str:
        """Get the full filename format (all fields for profiling).

        Returns:
            Full format string
        """
        return cls.FULL_FORMAT

    @classmethod
    def get_available_placeholders(cls) -> Dict[str, str]:
        """Get dictionary of available placeholders and their descriptions.

        Returns:
            Dictionary mapping placeholders to field names
        """
        return cls.PLACEHOLDERS.copy()
