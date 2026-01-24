/** Export screen component */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Download as DownloadIcon,
  CheckCircle as CheckIcon,
  RestartAlt as RestartIcon,
  VideoFile as VideoIcon,
  Subtitles as SrtIcon,
  Edit as EditIcon,
  Share as ShareIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

export const ExportScreen: React.FC = () => {
  const { scenes, uploadedFile, exportSrt, reset, loading } = useAppStore();

  const handleDownloadAgain = () => {
    exportSrt();
  };

  const handleStartNew = () => {
    reset();
  };

  const videoEditingSoftware = [
    {
      name: 'DaVinci Resolve',
      steps: [
        'Open your video project',
        'Go to Edit page',
        'Right-click in Media Pool → Import → Subtitles',
        'Select the downloaded SRT file',
        'Adjust timing if needed',
      ],
    },
    {
      name: 'Adobe Premiere Pro',
      steps: [
        'Open your sequence',
        'File → Import',
        'Select the SRT file',
        'Drag to timeline',
        'Adjust in Essential Graphics panel',
      ],
    },
    {
      name: 'Final Cut Pro',
      steps: [
        'Import SRT file',
        'Drag to timeline above video',
        'Select subtitle clip',
        'Adjust in Inspector',
        'Modify text styling',
      ],
    },
    {
      name: 'VLC Media Player',
      steps: [
        'Play your video',
        'Subtitle → Add Subtitle File',
        'Select the SRT file',
        'Adjust delay if needed',
        'Enjoy with subtitles',
      ],
    },
  ];

  return (
    <Paper sx={{ p: 4, borderRadius: 2 }}>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <CheckIcon color="success" sx={{ fontSize: 64 }} />
        <Typography variant="h5" gutterBottom sx={{ mt: 2 }}>
          Export Complete!
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Your SRT file has been generated with {scenes.length} scenes.
        </Typography>
      </Box>

      <Alert severity="success" sx={{ mb: 4 }}>
        <Typography variant="body1">
          Successfully exported SRT subtitle file with AI-generated scene descriptions.
          The file is ready for use in your video editing workflow.
        </Typography>
      </Alert>

      <Grid container spacing={4} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Analysis Summary
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <VideoIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Video File"
                    secondary={uploadedFile?.filename || 'Unknown'}
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <SrtIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Scenes Detected"
                    secondary={`${scenes.length} scenes with AI descriptions`}
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <EditIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="File Format"
                    secondary="SRT (SubRip Subtitle) - Universal subtitle format"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📥 Download Options
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<DownloadIcon />}
                  onClick={handleDownloadAgain}
                  disabled={loading}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Download SRT File Again
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  startIcon={<ShareIcon />}
                  onClick={() => {
                    // In a real app, this would share the file
                    alert('Share functionality would be implemented here');
                  }}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Share with Team
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  startIcon={<RestartIcon />}
                  onClick={handleStartNew}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Start New Analysis
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          🎬 How to Use in Video Editors
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Import the SRT file into your preferred video editing software:
        </Typography>

        <Grid container spacing={3}>
          {videoEditingSoftware.map((software) => (
            <Grid item xs={12} md={6} key={software.name}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                    {software.name}
                  </Typography>
                  <List dense>
                    {software.steps.map((step, index) => (
                      <ListItem key={index}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <Typography variant="body2" color="primary">
                            {index + 1}.
                          </Typography>
                        </ListItemIcon>
                        <ListItemText primary={step} />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      <Box sx={{ p: 3, bgcolor: 'grey.50', borderRadius: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          💡 Tips for Best Results
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary">
              • <strong>Review descriptions</strong> before final export
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • <strong>Adjust timing</strong> in your video editor if needed
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • <strong>Use themes</strong> for more relevant AI descriptions
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" color="text.secondary">
              • <strong>Export multiple versions</strong> with different themes
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • <strong>Save project files</strong> for future editing
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • <strong>Test with short clips</strong> first for optimal settings
            </Typography>
          </Grid>
        </Grid>
      </Box>

      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="outlined"
          onClick={() => useAppStore.getState().setState({ currentStep: 'review' })}
        >
          Back to Review
        </Button>
        <Button variant="contained" onClick={handleStartNew}>
          Analyze Another Video
        </Button>
      </Box>
    </Paper>
  );
};