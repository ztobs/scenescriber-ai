/** Type definitions for Video Scene AI Analyzer */

export interface Scene {
  scene_id: number;
  start_time: number;
  end_time: number;
  duration: number;
  description: string;
  keyframes: string[];
  confidence_score: number;
  theme_applied?: string;
}

export interface UploadResponse {
  file_id: string;
  filename: string;
  file_path: string;
  file_size: number;
  message: string;
}

export interface AnalysisRequest {
  video_path: string;
  theme?: string;
  detection_sensitivity: 'low' | 'medium' | 'high';
  min_scene_duration: number;
  ai_model: 'openai' | 'claude' | 'gemini' | 'llava';
  description_length: 'short' | 'medium' | 'detailed';
  start_time?: number;  // Start time in seconds
  end_time?: number;    // End time in seconds
}

export interface AnalysisResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
  check_status_url: string;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  error?: string;
  has_scenes: boolean;
}

export interface ScenesResponse {
  job_id: string;
  scenes: Scene[];
  total_scenes: number;
}

export interface AIProvider {
  available: boolean;
  name: string;
  description: string;
  needs_api_key: boolean;
  key_configured: boolean;
}

export interface AppConfig {
  ai_providers: {
    openai: AIProvider;
    claude: AIProvider;
    gemini: AIProvider;
    llava: AIProvider;
  };
  default_settings: {
    detection_sensitivity: 'low' | 'medium' | 'high';
    min_scene_duration: number;
    ai_model: 'openai' | 'claude' | 'gemini' | 'llava';
    description_length: 'short' | 'medium' | 'detailed';
  };
  features: {
    scene_detection: boolean;
    ai_description: boolean;
    srt_export: boolean;
    theme_support: boolean;
  };
}

export interface AppState {
  currentStep: 'upload' | 'configure' | 'processing' | 'review' | 'export';
  uploadedFile: UploadResponse | null;
  jobId: string | null;
  jobStatus: JobStatus | null;
  scenes: Scene[];
  theme: string;
  detectionSensitivity: 'low' | 'medium' | 'high';
  minSceneDuration: number;
  aiModel: 'openai' | 'claude' | 'gemini' | 'llava';
  descriptionLength: 'short' | 'medium' | 'detailed';
  videoStartTime: number;      // Start time in seconds (0 = beginning)
  videoEndTime: number | null;  // End time in seconds (null = end of video)
  videoDuration: number;        // Total video duration in seconds
  loading: boolean;
  error: string | null;
  config: AppConfig | null;
}