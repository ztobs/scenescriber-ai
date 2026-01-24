# AGENTS.md - Development Guidelines

This document provides guidance for agentic coding agents (and human developers) working on the Video Scene AI Analyzer project.

## Project Overview

**Video Scene AI Analyzer** (SceneScriber AI): An intelligent video analysis tool that detects scene cuts, generates AI-powered descriptions based on user-defined themes, and exports SRT caption files for video editing workflows.

See `app-description.md` for complete technical specification.

---

## Build, Lint, and Test Commands

### Backend (Python)
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for dev/test tools

# Run linting
python -m pylint src/
python -m black --check src/
python -m flake8 src/

# Format code
python -m black src/
python -m isort src/

# Run tests
python -m pytest
python -m pytest tests/test_scene_detection.py  # Single test file
python -m pytest tests/test_scene_detection.py::test_histogram_comparison  # Single test
python -m pytest -v --cov=src  # Verbose with coverage

# Type checking
mypy src/
```

### Frontend (React/Vue - if applicable)
```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Linting & formatting
npm run lint
npm run format
npm run type-check  # TypeScript

# Run tests
npm test
npm test -- tests/components/VideoUpload.test.tsx  # Single test file
npm test -- --testNamePattern="uploads file correctly"  # Single test by name

# Build
npm run build
```

---

## Code Style Guidelines

### Naming Conventions

**Python:**
- Classes: `PascalCase` (e.g., `SceneDetector`, `VideoProcessor`)
- Functions/methods: `snake_case` (e.g., `detect_scenes()`, `extract_frames()`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SENSITIVITY`, `MAX_FRAME_SIZE`)
- Private members: prefix with `_` (e.g., `_validate_video()`)

**TypeScript/React:**
- Components: `PascalCase` (e.g., `VideoUpload`, `SceneList`)
- Functions: `camelCase` (e.g., `detectScenes()`, `extractFrames()`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SENSITIVITY`)
- React hooks: prefix with `use` (e.g., `useVideoUpload`, `useSceneDetection`)

### Imports

**Python:**
```python
# Standard library first
import os
import sys
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import requests

# Local imports
from src.scene_detector import SceneDetector
from src.utils import validate_video
```

**TypeScript:**
```typescript
// React/external libraries
import React, { useState, useEffect } from 'react';
import axios from 'axios';

// UI components
import Button from '@/components/Button';
import VideoPlayer from '@/components/VideoPlayer';

// Utilities
import { detectScenes } from '@/utils/videoAnalysis';
import { formatTimestamp } from '@/utils/time';
```

### Type Definitions

**Python:**
- Use type hints throughout (PEP 484)
- Example: `def detect_scenes(video_path: str) -> List[Scene]:`
- Use `Optional[T]` for nullable values
- Document complex types: `Scene = TypedDict('Scene', {'id': int, 'start': float})`

**TypeScript:**
- Define interfaces for all data structures
- Example:
```typescript
interface Scene {
  id: number;
  startTime: number;
  endTime: number;
  description: string;
  confidence: number;
}
```
- Avoid `any` type; use `unknown` or generics
- Export types at top of file

### Formatting

**Python:**
- Line length: 100 characters (Black default)
- Indentation: 4 spaces
- Use Black for automatic formatting

**TypeScript/JavaScript:**
- Line length: 100 characters
- Indentation: 2 spaces
- Use Prettier for automatic formatting
- Semi-colons required

### Error Handling

**Python:**
```python
# Define custom exceptions
class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass

class SceneDetectionError(Exception):
    """Raised when scene detection encounters issues."""
    pass

# Usage
try:
    scenes = detector.detect_scenes(video_path)
except FileNotFoundError:
    logger.error(f"Video file not found: {video_path}")
    raise VideoProcessingError(f"Cannot process video: {video_path}") from e
except Exception as e:
    logger.exception("Unexpected error during scene detection")
    raise
```

**TypeScript:**
```typescript
// Define error types
enum ErrorCode {
  VIDEO_NOT_FOUND = 'VIDEO_NOT_FOUND',
  INVALID_FORMAT = 'INVALID_FORMAT',
  PROCESSING_FAILED = 'PROCESSING_FAILED',
}

interface AppError extends Error {
  code: ErrorCode;
  details?: string;
}

// Usage with try-catch
try {
  const scenes = await detectScenes(videoPath);
} catch (error) {
  if (error instanceof AppError) {
    handleError(error.code, error.message);
  } else {
    logger.error('Unexpected error', error);
  }
}
```

### Comments & Documentation

- Write docstrings for all public functions/classes
- **Python**: Use Google-style docstrings
```python
def detect_scenes(video_path: str, sensitivity: str = 'medium') -> List[Scene]:
    """Detect scene boundaries in video using histogram comparison.
    
    Args:
        video_path: Path to video file
        sensitivity: Detection sensitivity ('low', 'medium', 'high')
    
    Returns:
        List of Scene objects with timing and confidence scores
    
    Raises:
        FileNotFoundError: If video file not found
        VideoProcessingError: If video cannot be processed
    """
```
- **TypeScript**: Use JSDoc
```typescript
/**
 * Detect scene boundaries in video using histogram comparison.
 * @param videoPath - Path to video file
 * @param sensitivity - Detection sensitivity ('low', 'medium', 'high')
 * @returns Promise resolving to array of Scene objects
 * @throws VideoProcessingError if video cannot be processed
 */
export async function detectScenes(
  videoPath: string,
  sensitivity: 'low' | 'medium' | 'high' = 'medium'
): Promise<Scene[]> {
```

### Logging

**Python:**
```python
import logging

logger = logging.getLogger(__name__)

# In functions
logger.debug("Processing frame %d", frame_num)
logger.info("Scene detection completed: %d scenes", len(scenes))
logger.warning("High API usage detected: %d tokens", token_count)
logger.error("Failed to process video: %s", error_msg)
```

**TypeScript:**
```typescript
// Use console with appropriate levels or dedicated logger
console.debug('Processing frame:', frameNum);
console.info('Scene detection completed:', scenes.length);
console.warn('High API usage detected:', tokenCount);
console.error('Failed to process video:', error);
```

---

## Testing Standards

- **Coverage**: Aim for >80% code coverage
- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test module interactions (scene detection + description generation)
- **Test Naming**: Use descriptive names: `test_detects_scene_cuts_with_histogram()` or `test_uploads_file_correctly()`

---

## Project Structure (Expected)

```
video-scene-tool/
├── backend/                    # Python backend
│   ├── src/
│   │   ├── scene_detector.py
│   │   ├── ai_describer.py
│   │   ├── srt_exporter.py
│   │   └── utils/
│   ├── tests/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── pyproject.toml
├── frontend/                   # React/Vue frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── utils/
│   │   └── App.tsx
│   ├── tests/
│   ├── package.json
│   └── tsconfig.json
└── app-description.md
```

---

## Key Technical Constraints

1. **Video Processing**: Use FFmpeg for frame extraction, OpenCV for preprocessing
2. **Scene Detection**: PySceneDetect library or custom histogram comparison algorithm
3. **AI Integration**: Support GPT-4 Vision, Claude 3, or local LLaVA models
4. **SRT Format**: Must comply with SubRip specification (HH:MM:SS,mmm format)
5. **Performance**: Process 7-minute 1080p video in <3 minutes
6. **Accuracy**: Scene detection >90%, false positive rate <10%

---

## Before Committing Code

- [ ] Run formatter (Black/Prettier)
- [ ] Run linter (Pylint/ESLint)
- [ ] Run type checker (mypy/TypeScript compiler)
- [ ] Run tests and verify coverage >80%
- [ ] Write/update docstrings for new functions
- [ ] Test integration with dependent modules
- [ ] Verify no hardcoded secrets or API keys

---

## Common Tasks

### Adding a New API Endpoint
1. Define request/response types
2. Implement handler in backend
3. Add tests for endpoint
4. Document in API schema
5. Update frontend API client

### Adding New Test
```bash
# Python
touch tests/test_new_feature.py
python -m pytest tests/test_new_feature.py::test_specific_case -v

# TypeScript
touch tests/new-feature.test.ts
npm test -- new-feature.test.ts
```

### Debugging
- **Python**: `import pdb; pdb.set_trace()` or use debugger breakpoints
- **TypeScript**: Use VS Code debugger or `console.log()` statements

---

## References

- **Python Standards**: PEP 8, PEP 484 (type hints)
- **Git Flow**: Commit often with clear messages
- **API Design**: RESTful conventions
- **Video Formats**: Support MP4, MOV, MKV, AVI
