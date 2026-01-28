/** API client for Video Scene AI Analyzer */

import axios from 'axios';
import type {
  UploadResponse,
  AnalysisRequest,
  AnalysisResponse,
  JobStatus,
  ScenesResponse,
  AppConfig,
  FilenameFormatInfo,
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

  /** Get filename placeholders */
  getFilenamePlaceholders: async (): Promise<FilenameFormatInfo> => {
    const response = await api.get('/filename/placeholders');
    return response.data;
  },

  /** Export SRT file */
  exportSrt: async (jobId: string, filenameFormat?: string): Promise<{ blob: Blob; filename: string }> => {
    const params = filenameFormat ? { filename_format: filenameFormat } : {};
    const response = await api.get(`/export/srt/${jobId}`, {
      params,
      responseType: 'blob',
    });
    
    // Extract filename from Content-Disposition header if available
    let filename = 'scenes.srt';
    const contentDisposition = response.headers['content-disposition'];
    if (contentDisposition) {
      const match = contentDisposition.match(/filename="(.+)"/);
      if (match && match[1]) {
        filename = match[1];
      }
    }
    
    return { blob: response.data, filename };
  },

  /** Download SRT file */
  downloadSrt: (blob: Blob, filename: string): void => {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    
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