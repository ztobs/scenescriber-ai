/** Zustand store for application state */

import { create } from 'zustand';
import type { AppState } from '../types';
import { videoApi } from '../utils/api';

const initialState: Omit<AppState, 'setState' | 'reset'> = {
  currentStep: 'upload',
  uploadedFile: null,
  jobId: null,
  jobStatus: null,
  scenes: [],
  theme: '',
  detectionSensitivity: 'medium',
  minSceneDuration: 2.0,
  aiModel: 'openai',
  descriptionLength: 'medium',
  videoStartTime: 0,
  videoEndTime: null,
  videoDuration: 0,
  loading: false,
  error: null,
  config: null,
};

interface AppStore extends AppState {
  setState: (state: Partial<AppState>) => void;
  reset: () => void;
  
  // Actions
  loadConfig: () => Promise<void>;
  uploadVideo: (file: File) => Promise<void>;
  startAnalysis: () => Promise<void>;
  updateSceneDescription: (sceneId: number, description: string) => Promise<void>;
  exportSrt: () => Promise<void>;
  pollJobStatus: () => Promise<void>;
}

export const useAppStore = create<AppStore>((set, get) => ({
  ...initialState,
  
  setState: (state) => set(state),
  
  reset: () => set(initialState),
  
  loadConfig: async () => {
    set({ loading: true, error: null });
    
    try {
      // Add timeout to prevent hanging
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Configuration load timeout')), 10000);
      });
      
      const config = await Promise.race([videoApi.getConfig(), timeoutPromise]) as any;
      set({
        config,
        detectionSensitivity: config.default_settings.detection_sensitivity,
        minSceneDuration: config.default_settings.min_scene_duration,
        aiModel: config.default_settings.ai_model,
        descriptionLength: config.default_settings.description_length,
        loading: false,
      });
    } catch (error) {
      console.error('Failed to load config:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to load configuration. Please check if backend is running.',
        loading: false,
        config: null,
      });
    }
  },
  
  uploadVideo: async (file: File) => {
    set({ loading: true, error: null });
    
    try {
      const response = await videoApi.uploadVideo(file);
      
      // Get video duration using HTML5 video element
      const videoDuration = await new Promise<number>((resolve) => {
        const video = document.createElement('video');
        video.onloadedmetadata = () => {
          resolve(video.duration);
        };
        video.src = URL.createObjectURL(file);
      });
      
      set({
        uploadedFile: response,
        videoDuration: videoDuration,
        videoStartTime: 0,
        videoEndTime: null,
        currentStep: 'configure',
        loading: false,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Upload failed',
        loading: false,
      });
    }
  },
  
  startAnalysis: async () => {
    const state = get();
    if (!state.uploadedFile) {
      set({ error: 'No video uploaded' });
      return;
    }
    
    set({ loading: true, error: null, currentStep: 'processing' });
    
    try {
      const request = {
        video_path: state.uploadedFile.file_path,
        theme: state.theme || undefined,
        detection_sensitivity: state.detectionSensitivity,
        min_scene_duration: state.minSceneDuration,
        ai_model: state.aiModel,
        description_length: state.descriptionLength,
        start_time: state.videoStartTime > 0 ? state.videoStartTime : undefined,
        end_time: state.videoEndTime !== null ? state.videoEndTime : undefined,
      };
      
      const response = await videoApi.analyzeVideo(request);
      set({
        jobId: response.job_id,
        loading: false,
      });
      
      // Start polling for job status
      get().pollJobStatus();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Analysis failed',
        loading: false,
      });
    }
  },
  
  updateSceneDescription: async (sceneId: number, description: string) => {
    const state = get();
    if (!state.jobId) return;
    
    try {
      await videoApi.updateSceneDescription(state.jobId, sceneId, description);
      
      // Update local state
      set((state) => ({
        scenes: state.scenes.map((scene) =>
          scene.scene_id === sceneId ? { ...scene, description } : scene
        ),
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to update description',
      });
    }
  },
  
  exportSrt: async () => {
    const state = get();
    if (!state.jobId) return;
    
    set({ loading: true, error: null });
    
    try {
      const blob = await videoApi.exportSrt(state.jobId);
      videoApi.downloadSrt(state.jobId, blob);
      set({ loading: false, currentStep: 'export' });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Export failed',
        loading: false,
      });
    }
  },
  
  pollJobStatus: async () => {
    const state = get();
    if (!state.jobId) return;
    
    const poll = async () => {
      try {
        const status = await videoApi.getJobStatus(state.jobId!);
        set({ jobStatus: status });
        
        if (status.status === 'completed') {
          // Fetch scenes
          const scenesResponse = await videoApi.getScenes(state.jobId!);
          set({
            scenes: scenesResponse.scenes,
            currentStep: 'review',
            loading: false,
          });
        } else if (status.status === 'failed') {
          set({
            error: status.error || 'Processing failed',
            loading: false,
          });
        } else if (status.status === 'processing') {
          // Continue polling
          setTimeout(poll, 2000);
        }
      } catch (error) {
        set({
          error: error instanceof Error ? error.message : 'Status check failed',
          loading: false,
        });
      }
    };
    
    poll();
  },
}));