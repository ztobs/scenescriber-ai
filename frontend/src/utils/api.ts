/** API client for Video Scene AI Analyzer */

import axios from 'axios';
import type {
  UploadResponse,
  AnalysisRequest,
  AnalysisResponse,
  JobStatus,
  ScenesResponse,
  AppConfig,
} from '../types';

export const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const videoApi = {
  /** Get application configuration */
  getConfig: async (): Promise<AppConfig> => {
    const response = await api.get('/config');
    return response.data;
  },

  /** Upload video file */
  uploadVideo: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  /** Start video analysis */
  analyzeVideo: async (request: AnalysisRequest): Promise<AnalysisResponse> => {
    // Send as query parameters for POST request
    const response = await api.post('/analyze', null, {
      params: request
    });
    return response.data;
  },

  /** Get job status */
  getJobStatus: async (jobId: string): Promise<JobStatus> => {
    const response = await api.get(`/status/${jobId}`);
    return response.data;
  },

  /** Get scenes for completed job */
  getScenes: async (jobId: string): Promise<ScenesResponse> => {
    const response = await api.get(`/scenes/${jobId}`);
    return response.data;
  },

  /** Update scene description */
  updateSceneDescription: async (
    jobId: string,
    sceneId: number,
    description: string
  ): Promise<void> => {
    await api.put(`/scenes/${sceneId}`, null, {
      params: {
        job_id: jobId,
        description,
      },
    });
  },

  /** Export SRT file */
  exportSrt: async (jobId: string): Promise<Blob> => {
    const response = await api.get(`/export/srt/${jobId}`, {
      responseType: 'blob',
    });
    return response.data;
  },

  /** Download SRT file */
  downloadSrt: (jobId: string, blob: Blob): void => {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scenes_${jobId}.srt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  },

  /** Get video URL */
  getVideoUrl: (fileId: string): string => {
    return `${API_BASE_URL}/video/${fileId}`;
  },
};

export default api;