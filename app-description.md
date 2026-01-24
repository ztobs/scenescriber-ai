# **Video Scene AI Analyzer - Technical Specification**

## 1. Product Overview

**App Name**: SceneScriber AI (or your preferred name)  
**Purpose**: An intelligent video analysis tool that automatically detects scene cuts, generates context-aware descriptions for each scene based on user-defined themes, and exports standardized SRT caption files for integration with professional video editing workflows (DaVinci Resolve, Premiere Pro, etc.).

**Target User**: Video editors and content creators working with silent/action-based footage who need AI-assisted scene logging for creative editing guidance.

---

## 2. Core Features & Functional Requirements

### 2.1 Scene Detection Engine
**Function**: Automatically identify and segment video into discrete scenes based on visual changes.

**Technical Requirements**:
- **Algorithm**: Implement shot boundary detection using frame difference analysis (histogram comparison, pixel difference, or deep learning-based approaches like TransNetV2)
- **Sensitivity Control**: User-adjustable detection threshold (Low/Medium/High) to control how aggressive the cut detection is
- **Minimum Scene Duration**: Configurable minimum length (default 2 seconds) to avoid detecting tiny cuts
- **Output**: Array of scenes with start_time, end_time, duration, and scene_id

**Acceptance Criteria**:
- Successfully detect >90% of obvious scene cuts (camera angle changes, major object movements, location changes)
- False positive rate <10% (lighting changes within same scene shouldn't trigger cuts)
- Process 7-minute video in <2 minutes on standard hardware

### 2.2 AI Scene Description Generator
**Function**: Analyze visual content of each detected scene and generate descriptive text.

**Technical Requirements**:
- **Vision AI Model**: Integration with multimodal LLM API (GPT-4 Vision, Claude 3 Opus, Gemini Pro Vision, or open-source LLaVA-1.5/LLaVA-NeXT)
- **Frame Sampling**: Extract 3-5 representative frames per scene (first, middle, last + keyframes if available)
- **Batch Processing**: Send multiple frames per API call with optimized prompts
- **Description Length**: 15-30 words per scene (concise but informative)
- **Language**: Primary English, with architecture support for multilingual expansion

**Acceptance Criteria**:
- Descriptions accurately reflect main action/objects in scene
- Consistent formatting and tone across all scenes
- Processing cost optimization (efficient frame sampling to minimize API calls)

### 2.3 Themed Detection System (Key Differentiator)
**Function**: Allow users to specify a project theme/context to tailor scene descriptions toward specific editing needs.

**Technical Requirements**:
- **Theme Input**: Text field for user to enter project theme (e.g., "DIY motorized furniture build", "cooking tutorial", "product unboxing")
- **Prompt Engineering**: Dynamic prompt construction that injects theme context into vision AI requests
- **Focus Areas**: Theme-aware emphasis on relevant objects, actions, and details
- **Preset Themes**: Optional dropdown with common themes (DIY/Build, Cooking, Gaming, Tutorial, Review, etc.)

**Example Theme Integration**:
```
Base Prompt: "Describe what is happening in this video scene."
Themed Prompt: "This is a DIY motorized furniture build video. Describe what is happening in this scene, focusing on tools being used, assembly steps, mechanical parts, and progress indicators."
```

**Acceptance Criteria**:
- Descriptions noticeably shift focus based on theme input
- Generic fallback when no theme provided
- Theme persists across multiple video analyses in same session

### 2.4 SRT File Generation & Export
**Function**: Convert scene descriptions into standard SubRip Subtitle (SRT) format for universal compatibility.

**Technical Requirements**:
- **Format Compliance**: Strict adherence to SRT specifications (sequence numbers, timecodes HH:MM:SS,mmm --> HH:MM:SS,mmm, text lines)
- **Timestamp Precision**: Millisecond accuracy (00:00:15,420 --> 00:00:22,150)
- **Text Wrapping**: Automatic line breaks at 32 characters (standard subtitle width) or user-configurable
- **Scene Consolidation**: Option to merge consecutive short scenes into single caption if requested
- **Metadata Header**: Optional comments in SRT indicating theme used and processing date

**SRT Output Example**:
```
1
00:00:00,000 --> 00:00:08,500
Drilling holes into table legs for motor mounting brackets

2
00:00:08,500 --> 00:00:15,200
Attaching linear actuator to underside of tabletop
```

**Acceptance Criteria**:
- Import successfully into DaVinci Resolve, Premiere Pro, Final Cut Pro, VLC
- Timecodes align with actual video timing (±100ms tolerance)
- No encoding issues with special characters

---

## 3. User Interface Requirements

### 3.1 Web Application (Recommended) or Desktop App
**Platform**: Cross-platform web app (React/Vue + Python backend) or Electron desktop app

**Key Screens**:

1. **Upload Screen**
   - Drag-and-drop video upload (MP4, MOV, MKV, AVI support)
   - File size limit: 2GB per video (configurable)
   - Progress indicator for upload
   - Video thumbnail preview

2. **Configuration Panel**
   - Theme input text field (with examples)
   - Scene detection sensitivity slider (Low/Medium/High)
   - Minimum scene duration input (seconds)
   - AI model selection dropdown (if multiple providers supported)
   - Description length toggle (Short/Medium/Detailed)

3. **Processing Screen**
   - Real-time progress: "Detecting scenes...", "Analyzing scene 5 of 12...", "Generating SRT..."
   - Estimated time remaining
   - Cancel button
   - Preview of detected scene boundaries (thumbnail strip with cut markers)

4. **Review & Edit Screen**
   - Side-by-side: Video player + synchronized scene list
   - Editable scene descriptions (text fields)
   - Merge/split scene buttons
   - Adjust timestamp handles
   - Theme preview: Show how descriptions change if theme modified

5. **Export Screen**
   - Download SRT button
   - Copy-to-clipboard option
   - Export as CSV/JSON alternative formats
   - Save project file for future editing

### 3.2 DaVinci Resolve Integration (Future/Optional)
- **Markers Import**: Export as DaVinci Resolve marker XML (EDL format)
- **Direct API**: Resolve API integration to auto-import SRT as subtitles

---

## 4. Technical Architecture

### 4.1 Backend Stack
- **Language**: Python 3.9+ (industry standard for video processing)
- **Video Processing**: FFmpeg (scene detection, frame extraction), OpenCV (computer vision preprocessing)
- **Scene Detection**: 
  - Option A: PySceneDetect library (optimized, proven)
  - Option B: Custom frame difference algorithm
  - Option C: TransNetV2 (deep learning, most accurate)
- **AI Integration**: 
  - OpenAI GPT-4 Vision API (best quality)
  - Anthropic Claude 3 API (alternative)
  - Google Gemini API (cost-effective)
  - Local model option: LLaVA (privacy-focused, no API costs)
- **Database**: SQLite for project storage (local) or PostgreSQL (cloud)
- **Task Queue**: Celery + Redis for async processing (if web app)

### 4.2 Frontend Stack
- **Framework**: React.js or Vue.js
- **Video Player**: Video.js or Plyr (frame-accurate seeking)
- **UI Components**: Material-UI or Ant Design
- **State Management**: Redux or Zustand

### 4.3 API Design
```
POST /api/upload - Upload video file
GET /api/status/{job_id} - Check processing status
POST /api/analyze - Start scene analysis with theme/config
GET /api/scenes/{job_id} - Retrieve scene data with descriptions
POST /api/export/srt - Generate and download SRT file
PUT /api/scenes/{scene_id} - Edit scene description/timing
```

---

## 5. Data Models

### Scene Object
```json
{
  "scene_id": 1,
  "start_time": 0.0,
  "end_time": 8.5,
  "duration": 8.5,
  "description": "Drilling holes into table legs for motor mounting",
  "keyframes": ["frame_001.jpg", "frame_042.jpg", "frame_085.jpg"],
  "confidence_score": 0.94,
  "theme_applied": "DIY motorized furniture build"
}
```

### Project Configuration
```json
{
  "project_id": "uuid",
  "video_filename": "table_build.mp4",
  "theme": "DIY motorized furniture build",
  "detection_sensitivity": "medium",
  "min_scene_duration": 2.0,
  "ai_model": "gpt-4-vision",
  "created_at": "timestamp",
  "total_scenes": 12
}
```

---

## 6. Performance Requirements

- **Upload Speed**: Support resumable uploads for large files
- **Processing Time**: 7-minute video processed in <3 minutes (including API calls)
- **Concurrent Users**: Support 5 simultaneous processing jobs (if multi-user)
- **Memory Usage**: <4GB RAM for 1080p video processing
- **Storage**: Temporary file cleanup after 24 hours (if cloud) or immediate (if local)

---

## 7. Error Handling & Edge Cases

- **Corrupted Video**: Graceful error message if FFmpeg cannot parse file
- **API Rate Limits**: Queue system with exponential backoff for AI provider limits
- **No Scenes Detected**: Fallback to time-based segmentation (every 5 seconds) if algorithm fails
- **Long Scenes**: Auto-split scenes longer than 30 seconds into sub-scenes for better granularity
- **Network Failures**: Retry mechanism for API calls (3 attempts)
- **Cost Control**: Token usage estimation before processing, user confirmation for expensive operations

---

## 8. Security & Privacy

- **Local Processing Option**: Allow entirely offline processing using local vision models (LLaVA) for sensitive content
- **API Key Management**: Secure storage of user's own API keys (encrypted at rest)
- **Data Retention**: Auto-delete uploaded videos after processing completion (configurable)
- **HTTPS**: All API communications encrypted

---

## 9. Deliverables for Developer

1. **Functional web application** with all screens described above
2. **Python processing engine** with scene detection and AI integration
3. **SRT export module** with format validation
4. **Documentation**: API docs, deployment guide, user manual
5. **Test suite**: Unit tests for scene detection accuracy, SRT format compliance

---

## 10. Success Metrics

- Scene detection accuracy >90% on test dataset of DIY/build videos
- SRT files import without errors into DaVinci Resolve, Premiere Pro, VLC
- Themed descriptions rated "useful" by test users in 80%+ of scenes
- Processing completes in <3 minutes for 7-minute 1080p video

---

**Next Steps**: 
1. Developer reviews spec and provides technical approach/tech stack confirmation
2. Create wireframes for UI screens
3. Set up test video dataset (your 7-minute table video + 2-3 others)
4. Obtain API keys for chosen AI provider(s)
5. Development sprint planning (recommend 4-6 weeks for MVP)

---
