/** Main App component for Video Scene AI Analyzer */

import React, { useEffect } from 'react';
import { Container, Box, Typography, Alert, LinearProgress } from '@mui/material';
import { useAppStore } from './store/useAppStore';
import { UploadScreen } from './components/UploadScreen';
import { ConfigurationScreen } from './components/ConfigurationScreen';
import { ProcessingScreen } from './components/ProcessingScreen';
import { ReviewScreen } from './components/ReviewScreen';
import { ExportScreen } from './components/ExportScreen';

const App: React.FC = () => {
  const { currentStep, loading, error, loadConfig, config } = useAppStore();

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const renderStep = () => {
    switch (currentStep) {
      case 'upload':
        return <UploadScreen />;
      case 'configure':
        return <ConfigurationScreen />;
      case 'processing':
        return <ProcessingScreen />;
      case 'review':
        return <ReviewScreen />;
      case 'export':
        return <ExportScreen />;
      default:
        return <UploadScreen />;
    }
  };

  const getStepTitle = () => {
    switch (currentStep) {
      case 'upload':
        return 'Upload Video';
      case 'configure':
        return 'Configure Analysis';
      case 'processing':
        return 'Processing Video';
      case 'review':
        return 'Review & Edit Scenes';
      case 'export':
        return 'Export SRT File';
      default:
        return 'Video Scene AI Analyzer';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          🎬 SceneScriber AI
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom align="center" color="text.secondary">
          {getStepTitle()}
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 3 }}>
          Intelligent video scene detection with AI-powered descriptions
        </Typography>
      </Box>

      {loading && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" align="center" sx={{ mt: 1 }}>
            Processing...
          </Typography>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {renderStep()}

      <Box sx={{ mt: 6, pt: 3, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          Video Scene AI Analyzer • SceneScriber AI • v0.1.0
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          Automatically detect scenes and generate AI descriptions for video editing workflows
        </Typography>
      </Box>
    </Container>
  );
};

export default App;