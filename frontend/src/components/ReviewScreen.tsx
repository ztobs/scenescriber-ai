/** Review and edit screen component */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  IconButton,
  Alert,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  PlayArrow as PlayIcon,
  Download as DownloadIcon,
  NavigateBefore as BackIcon,
  NavigateNext as NextIcon,
} from '@mui/icons-material';
import { useAppStore } from '../store/useAppStore';
import { formatTimestamp } from '../utils/time';

export const ReviewScreen: React.FC = () => {
  const { scenes, updateSceneDescription, exportSrt, loading, setState } = useAppStore();
  const [editingSceneId, setEditingSceneId] = useState<number | null>(null);
  const [editDescription, setEditDescription] = useState('');
  const [selectedSceneIndex, setSelectedSceneIndex] = useState(0);

  const handleEditClick = (sceneId: number, currentDescription: string) => {
    setEditingSceneId(sceneId);
    setEditDescription(currentDescription);
  };

  const handleSaveEdit = async () => {
    if (editingSceneId) {
      await updateSceneDescription(editingSceneId, editDescription);
      setEditingSceneId(null);
      setEditDescription('');
    }
  };

  const handleCancelEdit = () => {
    setEditingSceneId(null);
    setEditDescription('');
  };

  const handleExportSrt = () => {
    exportSrt();
  };

  const selectedScene = scenes[selectedSceneIndex];

  return (
    <Paper sx={{ p: 4, borderRadius: 2 }}>
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h5" gutterBottom>
            Review & Edit Scenes
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Review AI-generated descriptions and make edits as needed
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={handleExportSrt}
          disabled={loading}
        >
          Export SRT
        </Button>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          Found {scenes.length} scenes. You can edit descriptions before exporting.
          SRT files can be imported into DaVinci Resolve, Premiere Pro, and other video editors.
        </Typography>
      </Alert>

      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Scene List ({scenes.length})
              </Typography>
              <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                {scenes.map((scene, index) => (
                  <React.Fragment key={scene.scene_id}>
                    <ListItem
                      button
                      selected={index === selectedSceneIndex}
                      onClick={() => setSelectedSceneIndex(index)}
                      sx={{
                        borderRadius: 1,
                        mb: 1,
                        '&.Mui-selected': {
                          bgcolor: 'primary.50',
                          borderLeft: '4px solid',
                          borderColor: 'primary.main',
                        },
                      }}
                    >
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="subtitle2">
                              Scene {scene.scene_id}
                            </Typography>
                            <Chip
                              label={formatTimestamp(scene.start_time)}
                              size="small"
                              variant="outlined"
                            />
                          </Box>
                        }
                        secondary={
                          <Typography
                            variant="body2"
                            sx={{
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                              mt: 0.5,
                            }}
                          >
                            {scene.description}
                          </Typography>
                        }
                      />
                    </ListItem>
                    {index < scenes.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          {selectedScene && (
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Box>
                    <Typography variant="h5">
                      Scene {selectedScene.scene_id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {formatTimestamp(selectedScene.start_time)} → {formatTimestamp(selectedScene.end_time)}
                      {' • '}
                      Duration: {selectedScene.duration.toFixed(1)}s
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <IconButton
                      size="small"
                      disabled={selectedSceneIndex === 0}
                      onClick={() => setSelectedSceneIndex(selectedSceneIndex - 1)}
                    >
                      <BackIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      disabled={selectedSceneIndex === scenes.length - 1}
                      onClick={() => setSelectedSceneIndex(selectedSceneIndex + 1)}
                    >
                      <NextIcon />
                    </IconButton>
                  </Box>
                </Box>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom color="text.secondary">
                    AI-Generated Description
                  </Typography>
                  {editingSceneId === selectedScene.scene_id ? (
                    <Box>
                      <TextField
                        fullWidth
                        multiline
                        rows={4}
                        value={editDescription}
                        onChange={(e) => setEditDescription(e.target.value)}
                        sx={{ mb: 2 }}
                      />
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button
                          variant="contained"
                          startIcon={<SaveIcon />}
                          onClick={handleSaveEdit}
                          disabled={loading}
                        >
                          Save
                        </Button>
                        <Button
                          variant="outlined"
                          startIcon={<CancelIcon />}
                          onClick={handleCancelEdit}
                        >
                          Cancel
                        </Button>
                      </Box>
                    </Box>
                  ) : (
                    <Box>
                      <Paper variant="outlined" sx={{ p: 3, bgcolor: 'grey.50' }}>
                        <Typography variant="body1" paragraph>
                          {selectedScene.description}
                        </Typography>
                        {selectedScene.theme_applied && (
                          <Chip
                            label={`Theme: ${selectedScene.theme_applied}`}
                            size="small"
                            color="primary"
                            variant="outlined"
                          />
                        )}
                      </Paper>
                      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                        <Button
                          variant="outlined"
                          startIcon={<EditIcon />}
                          onClick={() =>
                            handleEditClick(selectedScene.scene_id, selectedScene.description)
                          }
                        >
                          Edit Description
                        </Button>
                      </Box>
                    </Box>
                  )}
                </Box>

                {selectedScene.keyframes && selectedScene.keyframes.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom color="text.secondary">
                      Keyframes ({selectedScene.keyframes.length})
                    </Typography>
                    <Grid container spacing={2}>
                      {selectedScene.keyframes.map((_, index) => (
                        <Grid item xs={4} key={index}>
                          <Paper
                            variant="outlined"
                            sx={{
                              p: 1,
                              textAlign: 'center',
                              height: 120,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Typography variant="caption" color="text.secondary">
                              Frame {index + 1}
                            </Typography>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                  <Button
                    variant="outlined"
                    onClick={() => setState({ currentStep: 'configure' })}
                  >
                    Back to Configuration
                  </Button>
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button
                      variant="outlined"
                      startIcon={<PlayIcon />}
                      onClick={() => {
                        // In a real app, this would seek to the scene in a video player
                        alert(`Would seek to ${formatTimestamp(selectedScene.start_time)}`);
                      }}
                    >
                      Play Scene
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<DownloadIcon />}
                      onClick={handleExportSrt}
                      disabled={loading}
                    >
                      Export SRT
                    </Button>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      <Box sx={{ mt: 4, p: 3, bgcolor: 'grey.50', borderRadius: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          SRT Export Preview
        </Typography>
        <Paper variant="outlined" sx={{ p: 2, fontFamily: 'monospace', fontSize: 14 }}>
          {selectedScene && (
            <>
              {selectedScene.scene_id}
              {'\n'}
              {formatTimestamp(selectedScene.start_time).replace('.', ',')} {'-->'} {' '}
              {formatTimestamp(selectedScene.end_time).replace('.', ',')}
              {'\n'}
              {selectedScene.description}
              {'\n\n'}
              <Typography variant="caption" color="text.secondary">
                ... {scenes.length - 1} more scenes
              </Typography>
            </>
          )}
        </Paper>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          This SRT format is compatible with DaVinci Resolve, Premiere Pro, Final Cut Pro, and VLC.
        </Typography>
      </Box>
    </Paper>
  );
};