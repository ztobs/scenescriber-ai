/** Configuration screen component */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Button,
  Grid,
  Alert,
  Card,
  CardContent,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Lightbulb as ThemeIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  SmartToy as AiIcon,
  Description as DescriptionIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';

const themeExamples = [
  'DIY motorized furniture build',
  'Cooking tutorial - Italian pasta',
  'Product unboxing and review',
  'Gaming gameplay highlights',
  'Travel vlog - mountain hiking',
  'Fitness workout routine',
  'Programming tutorial - React hooks',
  'Art painting process',
];

export const ConfigurationScreen: React.FC = () => {
  const {
    theme,
    detectionSensitivity,
    minSceneDuration,
    aiModel,
    descriptionLength,
    setState,
    startAnalysis,
    loading,
    uploadedFile,
  } = useAppStore();

  const handleThemeExampleClick = (example: string) => {
    setState({ theme: example });
  };

  const handleStartAnalysis = () => {
    startAnalysis();
  };

  return (
    <Paper sx={{ p: 4, borderRadius: 2 }}>
      <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <SettingsIcon color="primary" sx={{ fontSize: 32 }} />
        <Box>
          <Typography variant="h5" gutterBottom>
            Configure Analysis Settings
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Customize how your video will be analyzed and described
          </Typography>
        </Box>
      </Box>

      {uploadedFile && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Analyzing: <strong>{uploadedFile.filename}</strong>
        </Alert>
      )}

      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <ThemeIcon color="primary" />
                <Typography variant="h6">Theme & Context</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Provide a theme to help AI generate more relevant descriptions.
                This guides the AI to focus on specific elements.
              </Typography>
              <TextField
                fullWidth
                label="Project Theme (Optional)"
                value={theme}
                onChange={(e) => setState({ theme: e.target.value })}
                placeholder="e.g., DIY motorized furniture build"
                multiline
                rows={2}
                helperText="Describe what your video is about for better AI descriptions"
              />
              
              <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>
                Theme Examples:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {themeExamples.map((example) => (
                  <Button
                    key={example}
                    size="small"
                    variant="outlined"
                    onClick={() => handleThemeExampleClick(example)}
                    sx={{ textTransform: 'none' }}
                  >
                    {example}
                  </Button>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <SpeedIcon color="primary" />
                <Typography variant="h6">Scene Detection</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Configure how aggressively the system detects scene changes.
              </Typography>

              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>Sensitivity</InputLabel>
                <Select
                  value={detectionSensitivity}
                  label="Sensitivity"
                  onChange={(e) =>
                    setState({ detectionSensitivity: e.target.value as any })
                  }
                >
                  <MenuItem value="low">Low (fewer scenes, more conservative)</MenuItem>
                  <MenuItem value="medium">Medium (balanced detection)</MenuItem>
                  <MenuItem value="high">High (more scenes, sensitive to changes)</MenuItem>
                </Select>
              </FormControl>

              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>
                  Minimum Scene Duration: {minSceneDuration.toFixed(1)} seconds
                </Typography>
                <Slider
                  value={minSceneDuration}
                  onChange={(_, value) => setState({ minSceneDuration: value as number })}
                  min={0.5}
                  max={10}
                  step={0.5}
                  marks={[
                    { value: 0.5, label: '0.5s' },
                    { value: 2, label: '2s' },
                    { value: 5, label: '5s' },
                    { value: 10, label: '10s' },
                  ]}
                  valueLabelDisplay="auto"
                />
                <Typography variant="caption" color="text.secondary">
                  Scenes shorter than this will be merged with adjacent scenes
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <AiIcon color="primary" />
                <Typography variant="h6">AI Configuration</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Choose which AI model to use for generating scene descriptions.
              </Typography>

              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>AI Model</InputLabel>
                <Select
                  value={aiModel}
                  label="AI Model"
                  onChange={(e) => setState({ aiModel: e.target.value as any })}
                >
                  <MenuItem value="openai">OpenAI GPT-4 Vision (Best quality)</MenuItem>
                  <MenuItem value="claude">Anthropic Claude 3 (Alternative)</MenuItem>
                  <MenuItem value="gemini">Google Gemini (Cost-effective)</MenuItem>
                  <MenuItem value="llava">Local LLaVA (Privacy-focused)</MenuItem>
                </Select>
              </FormControl>

              <Alert severity="info" sx={{ mb: 2 }}>
                Note: AI models require API keys. Configure in backend settings.
              </Alert>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <DescriptionIcon color="primary" />
                <Typography variant="h6">Description Settings</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                Control the length and detail level of AI-generated descriptions.
              </Typography>

              <FormControl fullWidth>
                <InputLabel>Description Length</InputLabel>
                <Select
                  value={descriptionLength}
                  label="Description Length"
                  onChange={(e) =>
                    setState({ descriptionLength: e.target.value as any })
                  }
                >
                  <MenuItem value="short">Short (5-10 words, concise)</MenuItem>
                  <MenuItem value="medium">Medium (15-30 words, balanced)</MenuItem>
                  <MenuItem value="detailed">Detailed (40-60 words, comprehensive)</MenuItem>
                </Select>
              </FormControl>

              <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Example Output:
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {descriptionLength === 'short' && (
                    <>A person drills holes into wooden table legs.</>
                  )}
                  {descriptionLength === 'medium' && (
                    <>A person uses a power drill to create mounting holes in wooden table legs, preparing for motor installation.</>
                  )}
                  {descriptionLength === 'detailed' && (
                    <>In this DIY scene, a person wearing safety glasses uses a cordless power drill to create precise holes in the legs of a wooden table. Wood shavings are visible as they work, indicating progress in preparing the furniture for motorized components.</>
                  )}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'space-between' }}>
        <Button
          variant="outlined"
          onClick={() => setState({ currentStep: 'upload' })}
          disabled={loading}
        >
          Back to Upload
        </Button>
        <Button
          variant="contained"
          size="large"
          onClick={handleStartAnalysis}
          disabled={loading}
          sx={{ minWidth: 200 }}
        >
          {loading ? 'Starting Analysis...' : 'Start Analysis'}
        </Button>
      </Box>
    </Paper>
  );
};