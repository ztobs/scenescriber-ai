/** Processing screen component */

import React, { useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Grid,
  CircularProgress,
} from '@mui/material';
import {
  HourglassEmpty as HourglassIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  VideoCameraBack as SceneIcon,
  Image as FrameIcon,
  SmartToy as AiIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

export const ProcessingScreen: React.FC = () => {
  const { jobStatus, pollJobStatus, error } = useAppStore();

  // Poll for status updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (jobStatus?.status === 'processing') {
        pollJobStatus();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [jobStatus?.status, pollJobStatus]);

  const getStatusIcon = () => {
    switch (jobStatus?.status) {
      case 'completed':
        return <CheckIcon color="success" sx={{ fontSize: 64 }} />;
      case 'failed':
        return <ErrorIcon color="error" sx={{ fontSize: 64 }} />;
      default:
        return <HourglassIcon color="primary" sx={{ fontSize: 64 }} />;
    }
  };

  const getStatusMessage = () => {
    if (error) return error;
    if (!jobStatus) return 'Starting analysis...';
    return jobStatus.message || 'Processing video...';
  };

  const getProgressValue = () => {
    if (!jobStatus) return 0;
    if (jobStatus.status === 'completed') return 100;
    if (jobStatus.status === 'failed') return 0;
    return jobStatus.progress || 0;
  };

  const processingSteps = [
    {
      id: 1,
      title: 'Scene Detection',
      description: 'Analyzing video for scene boundaries',
      icon: <SceneIcon />,
      active: getProgressValue() >= 10,
      completed: getProgressValue() >= 40,
    },
    {
      id: 2,
      title: 'Frame Extraction',
      description: 'Extracting keyframes from each scene',
      icon: <FrameIcon />,
      active: getProgressValue() >= 40 && getProgressValue() < 60,
      completed: getProgressValue() >= 60,
    },
    {
      id: 3,
      title: 'AI Description',
      description: 'Generating descriptions using AI',
      icon: <AiIcon />,
      active: getProgressValue() >= 60 && getProgressValue() < 100,
      completed: getProgressValue() >= 100,
    },
  ];

  return (
    <Paper sx={{ p: 4, borderRadius: 2 }}>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        {getStatusIcon()}
        <Typography variant="h5" gutterBottom sx={{ mt: 2 }}>
          {jobStatus?.status === 'completed'
            ? 'Analysis Complete!'
            : jobStatus?.status === 'failed'
            ? 'Analysis Failed'
            : 'Processing Your Video'}
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          {getStatusMessage()}
        </Typography>
      </Box>

      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2">Progress</Typography>
          <Typography variant="body2">{getProgressValue()}%</Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={getProgressValue()}
          sx={{ height: 10, borderRadius: 5 }}
        />
      </Box>

      <Grid container spacing={2} sx={{ mb: 4 }}>
        {processingSteps.map((step) => (
          <Grid item xs={12} md={4} key={step.id}>
            <Card
              variant="outlined"
              sx={{
                borderColor: step.active ? 'primary.main' : 'divider',
                bgcolor: step.active ? 'primary.50' : 'transparent',
                transition: 'all 0.3s',
              }}
            >
              <CardContent sx={{ textAlign: 'center' }}>
                <Box
                  sx={{
                    color: step.completed ? 'success.main' : step.active ? 'primary.main' : 'text.secondary',
                    mb: 2,
                  }}
                >
                  {step.completed ? (
                    <CheckIcon sx={{ fontSize: 40 }} />
                  ) : (
                    React.cloneElement(step.icon, { sx: { fontSize: 40 } })
                  )}
                </Box>
                <Typography variant="h6" gutterBottom>
                  {step.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {step.description}
                </Typography>
                {step.active && !step.completed && (
                  <CircularProgress size={20} sx={{ mt: 2 }} />
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {jobStatus?.status === 'processing' && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            Processing may take several minutes depending on video length and AI model.
            Please don't close this window.
          </Typography>
        </Alert>
      )}

      {jobStatus?.status === 'failed' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="body2">
            Analysis failed: {jobStatus.error || 'Unknown error'}
          </Typography>
        </Alert>
      )}

      {jobStatus?.status === 'completed' && (
        <Box sx={{ textAlign: 'center' }}>
          <Alert severity="success" sx={{ mb: 3 }}>
            <Typography variant="body2">
              Successfully analyzed video and generated descriptions!
            </Typography>
          </Alert>
          <Typography variant="body1" paragraph>
            Ready to review and edit the detected scenes.
          </Typography>
        </Box>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 4 }}>
        {jobStatus?.status === 'failed' && (
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Try adjusting your settings or uploading a different video.
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};