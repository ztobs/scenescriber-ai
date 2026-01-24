/** Upload screen component */

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  CheckCircle as CheckIcon,
  VideoFile as VideoIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

export const UploadScreen: React.FC = () => {
  const { uploadedFile, uploadVideo, loading } = useAppStore();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        uploadVideo(acceptedFiles[0]);
      }
    },
    [uploadVideo]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.mkv', '.avi', '.webm'],
    },
    maxFiles: 1,
    disabled: loading,
  });

  return (
    <Paper sx={{ p: 4, borderRadius: 2 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          Upload Your Video
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Upload a video file to analyze scenes and generate AI descriptions.
          Supported formats: MP4, MOV, MKV, AVI, WebM.
        </Typography>
      </Box>

      {uploadedFile ? (
        <Alert
          severity="success"
          icon={<CheckIcon />}
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => useAppStore.getState().reset()}>
              Upload Another
            </Button>
          }
        >
          <Typography variant="subtitle1" fontWeight="bold">
            Video uploaded successfully!
          </Typography>
          <Typography variant="body2">
            {uploadedFile.filename} ({Math.round(uploadedFile.file_size / 1024 / 1024)} MB)
          </Typography>
        </Alert>
      ) : (
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            p: 6,
            textAlign: 'center',
            cursor: loading ? 'default' : 'pointer',
            backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
            transition: 'all 0.2s',
            '&:hover': {
              backgroundColor: loading ? 'background.paper' : 'action.hover',
              borderColor: loading ? 'grey.300' : 'primary.main',
            },
          }}
        >
          <input {...getInputProps()} />
          <UploadIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? 'Drop the video here' : 'Drag & drop your video here'}
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            or click to browse files
          </Typography>
          <Button variant="contained" size="large" disabled={loading}>
            Select Video File
          </Button>
        </Box>
      )}

      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          How it works:
        </Typography>
        <List>
          <ListItem>
            <ListItemIcon>
              <VideoIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Upload your video"
              secondary="Supports common video formats up to 2GB"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Configure analysis settings"
              secondary="Set theme, sensitivity, and AI model preferences"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Process and review"
              secondary="AI detects scenes and generates descriptions"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText
              primary="Export SRT file"
              secondary="Download subtitles for video editing software"
            />
          </ListItem>
        </List>
      </Box>

      {uploadedFile && (
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            size="large"
            onClick={() => useAppStore.getState().setState({ currentStep: 'configure' })}
          >
            Continue to Configuration
          </Button>
        </Box>
      )}
    </Paper>
  );
};