/** Configuration screen component */

import React, { useMemo } from 'react';
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
  Autocomplete,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Lightbulb as ThemeIcon,
  Speed as SpeedIcon,
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
    videoStartTime,
    videoEndTime,
    videoDuration,
    config,
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

       {/* Video Range Selector */}
       <Card variant="outlined" sx={{ mb: 4, p: 2 }}>
         <CardContent>
           <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
             <SpeedIcon color="primary" />
             <Typography variant="h6">Video Range (Optional)</Typography>
           </Box>
           <Typography variant="body2" color="text.secondary" paragraph>
             Select which part of the video to analyze. Leave at full range to analyze the entire video.
           </Typography>
           
           <Box>
             <Typography variant="subtitle2" gutterBottom>
               Start: {videoStartTime.toFixed(1)}s | End: {videoEndTime ? videoEndTime.toFixed(1) : `${(videoDuration || 0).toFixed(1)}`}s
             </Typography>
             <Slider
               value={[videoStartTime, videoEndTime || videoDuration || 100]}
               min={0}
               max={videoDuration || 100}
               step={0.5}
               onChange={(_, value) => {
                 if (Array.isArray(value)) {
                   const [start, end] = value;
                   setState({
                     videoStartTime: start,
                     videoEndTime: end === (videoDuration || 100) ? null : end,
                   });
                 }
               }}
               valueLabelDisplay="auto"
               valueLabelFormat={(value) => `${value.toFixed(1)}s`}
               marks={[
                 { value: 0, label: '0s' },
                 { value: videoDuration || 100, label: `${(videoDuration || 100).toFixed(1)}s` },
               ]}
               disableSwap
             />
             <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
               Drag the handles to select the time range you want to analyze
             </Typography>
           </Box>
         </CardContent>
       </Card>

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

              {useMemo(() => {
                // Prepare model options
                const modelOptions = config?.ai_providers
                  ? Object.entries(config.ai_providers)
                      .filter(([, provider]) => provider.available)
                      .map(([key, provider]) => ({
                        key,
                        label: provider.name,
                        description: provider.description,
                        providerGroup: key.includes('/')
                          ? key.split('/')[0].replace('_', ' ').toUpperCase()
                          : provider.name.split(' ')[0].toUpperCase(),
                      }))
                      .sort((a, b) => {
                        // Sort by provider group, then by label
                        if (a.providerGroup !== b.providerGroup) {
                          return a.providerGroup.localeCompare(b.providerGroup);
                        }
                        return a.label.localeCompare(b.label);
                      })
                  : [];

                const selectedModel = modelOptions.find((m) => m.key === aiModel);

                return (
                  <Autocomplete
                    fullWidth
                    disabled={!config?.features.ai_description}
                    options={modelOptions}
                    groupBy={(option) => option.providerGroup}
                    getOptionLabel={(option) => option.label}
                    value={selectedModel || null}
                    onChange={(_, value) => {
                      if (value) {
                        setState({ aiModel: value.key });
                      }
                    }}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="AI Model"
                        placeholder="Search or select a model..."
                      />
                    )}
                    renderOption={(props, option) => (
                      <Box
                        component="li"
                        sx={{ py: 1 }}
                        {...props}
                      >
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {option.label}
                          </Typography>
                          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                            {option.description}
                          </Typography>
                        </Box>
                      </Box>
                    )}
                    noOptionsText="No models available"
                    sx={{ mb: 3 }}
                  />
                );
              }, [config, aiModel, setState])}

              {config && !config.features.ai_description && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    ⚠️ No AI API keys configured
                  </Typography>
                  <Typography variant="body2">
                    You'll get mock descriptions. To enable AI:
                    <ol>
                      <li>Copy <code>.env.example</code> to <code>.env</code> in backend/</li>
                      <li>Add your API keys (OpenAI, Claude, or Gemini)</li>
                      <li>Restart the backend server</li>
                    </ol>
                    Scene detection and SRT export will still work without AI.
                  </Typography>
                </Alert>
              )}

              {config?.ai_providers[aiModel]?.needs_api_key && 
               !config?.ai_providers[aiModel]?.key_configured && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    Selected model needs API key. You'll get mock descriptions.
                    Configure in backend/.env file.
                  </Typography>
                </Alert>
              )}
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