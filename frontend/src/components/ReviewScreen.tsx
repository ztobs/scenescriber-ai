/** Review and edit screen component */

import React, { useState, useRef } from 'react';
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
  Slider,
  Dialog,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Download as DownloadIcon,
  SkipPrevious as SkipPreviousIcon,
  SkipNext as SkipNextIcon,
} from '@mui/icons-material';
import ReactPlayer from 'react-player';
import { useAppStore } from '../store/useAppStore';
import { formatTimestamp } from '../utils/time';
import { API_BASE_URL } from '../utils/api';

export const ReviewScreen: React.FC = () => {
  const { scenes, updateSceneDescription, exportSrt, loading, setState, uploadedFile } = useAppStore();
  const [editingSceneId, setEditingSceneId] = useState<number | null>(null);
  const [editDescription, setEditDescription] = useState('');
  const [selectedSceneIndex, setSelectedSceneIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [selectedKeyframe, setSelectedKeyframe] = useState<string | null>(null);
  const playerRef = useRef<any>(null);

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

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSeekToScene = () => {
    if (playerRef.current && selectedScene) {
      playerRef.current.seekTo(selectedScene.start_time, 'seconds');
      setIsPlaying(true); // This is for the "Play Scene" button, so it should play
    }
  };

  const handleProgress = (state: { playedSeconds: number }) => {
    setCurrentTime(state.playedSeconds);
  };

  const handleDuration = (dur: number) => {
    setDuration(dur);
  };

  const handleSliderChange = (_event: Event, newValue: number | number[]) => {
    const time = Array.isArray(newValue) ? newValue[0] : newValue;
    setCurrentTime(time);
    if (playerRef.current) {
      playerRef.current.seekTo(time, 'seconds');
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
           <Card variant="outlined" sx={{ maxHeight: '70vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
             <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 2, overflow: 'hidden' }}>
               <Typography variant="h6" gutterBottom sx={{ flexShrink: 0 }}>
                 Scene List ({scenes.length})
               </Typography>
                <List sx={{ flex: 1, overflow: 'auto' }}>
                {scenes.map((scene, index) => (
                  <React.Fragment key={scene.scene_id}>
                     <ListItem
                       button
                       selected={index === selectedSceneIndex}
                       onClick={() => {
                         setSelectedSceneIndex(index);
                         if (playerRef.current) {
                           playerRef.current.seekTo(scene.start_time, 'seconds');
                           // Don't auto-play, just seek
                         }
                       }}
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
                       title="Previous scene"
                     >
                       <SkipPreviousIcon />
                     </IconButton>
                     <IconButton
                       size="small"
                       disabled={selectedSceneIndex === scenes.length - 1}
                       onClick={() => setSelectedSceneIndex(selectedSceneIndex + 1)}
                       title="Next scene"
                     >
                       <SkipNextIcon />
                     </IconButton>
                   </Box>
                 </Box>

                 {/* Video Player Section */}
                 {uploadedFile && (
                   <Box sx={{ mb: 4 }}>
                     <Typography variant="subtitle2" gutterBottom color="text.secondary">
                       Video Preview
                     </Typography>
                     <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                       <Box sx={{ position: 'relative', paddingTop: '56.25%', bgcolor: 'black', borderRadius: 1 }}>
                         <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                           <ReactPlayer
                             ref={playerRef}
                             url={`${API_BASE_URL}/video/${uploadedFile.file_id}`}
                             playing={isPlaying}
                             controls={true}
                             width="100%"
                             height="100%"
                             onProgress={handleProgress}
                             onDuration={handleDuration}
                             config={{
                               file: {
                                 attributes: {
                                   controlsList: 'nodownload',
                                 },
                               },
                             }}
                           />
                         </Box>
                       </Box>
                       
                       {/* Custom Controls */}
                       <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                         <IconButton onClick={handlePlayPause} size="small">
                           {isPlaying ? <PauseIcon /> : <PlayIcon />}
                         </IconButton>
                         <Typography variant="body2" color="text.secondary" sx={{ minWidth: 40 }}>
                           {formatTime(currentTime)}
                         </Typography>
                         <Slider
                           size="small"
                           value={currentTime}
                           min={0}
                           max={duration || 100}
                           onChange={handleSliderChange}
                           sx={{ flex: 1 }}
                           aria-label="Video timeline"
                         />
                         <Typography variant="body2" color="text.secondary" sx={{ minWidth: 40 }}>
                           {formatTime(duration)}
                         </Typography>
                         <Button
                           variant="outlined"
                           size="small"
                           startIcon={<PlayIcon />}
                           onClick={handleSeekToScene}
                           disabled={!selectedScene}
                         >
                           Play Scene
                         </Button>
                       </Box>
                       
                       {/* Scene Timeline */}
                       {scenes.length > 0 && (
                         <Box sx={{ mt: 2 }}>
                           <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                             Scene Markers
                           </Typography>
                           <Box sx={{ position: 'relative', height: 24, bgcolor: 'grey.100', borderRadius: 1 }}>
                             {scenes.map((scene, index) => {
                               const position = (scene.start_time / (duration || 1)) * 100;
                               const isSelected = index === selectedSceneIndex;
                               return (
                                 <Box
                                   key={scene.scene_id}
                                   sx={{
                                     position: 'absolute',
                                     left: `${position}%`,
                                     top: 0,
                                     bottom: 0,
                                     width: 4,
                                     bgcolor: isSelected ? 'primary.main' : 'grey.400',
                                     cursor: 'pointer',
                                     '&:hover': {
                                       bgcolor: isSelected ? 'primary.dark' : 'grey.600',
                                     },
                                   }}
                                    onClick={() => {
                                      setSelectedSceneIndex(index);
                                      if (playerRef.current) {
                                        playerRef.current.seekTo(scene.start_time, 'seconds');
                                        // Don't auto-play, just seek
                                      }
                                    }}
                                   title={`Scene ${scene.scene_id}: ${formatTimestamp(scene.start_time)}`}
                                 />
                               );
                             })}
                           </Box>
                         </Box>
                       )}
                     </Paper>
                   </Box>
                 )}

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
                      {selectedScene.keyframes.map((keyframeUrl, index) => (
                        <Grid item xs={4} key={index}>
                          <Paper
                            variant="outlined"
                            sx={{
                              p: 0,
                              textAlign: 'center',
                              height: 120,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              overflow: 'hidden',
                              bgcolor: 'black',
                              cursor: 'pointer',
                              '&:hover': {
                                opacity: 0.8,
                                borderColor: 'primary.main',
                              }
                            }}
                            onClick={() => setSelectedKeyframe(keyframeUrl)}
                          >
                             <img 
                               src={keyframeUrl} 
                               alt={`Frame ${index + 1}`}
                               style={{ 
                                 width: '100%', 
                                 height: '100%', 
                                 objectFit: 'contain' 
                               }}
                               onError={(e) => {
                                 // Fallback if image fails to load
                                 (e.target as HTMLImageElement).style.display = 'none';
                                 (e.target as HTMLImageElement).parentElement!.innerText = `Frame ${index + 1}`;
                               }}
                             />
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
                       onClick={handleSeekToScene}
                       disabled={!selectedScene || !uploadedFile}
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

      {/* Keyframe Preview Dialog */}
      <Dialog
        open={!!selectedKeyframe}
        onClose={() => setSelectedKeyframe(null)}
        maxWidth="lg"
        fullWidth
      >
        <DialogContent sx={{ p: 0, bgcolor: 'black', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          {selectedKeyframe && (
            <img
              src={selectedKeyframe}
              alt="Full size keyframe"
              style={{ maxWidth: '100%', maxHeight: '80vh', objectFit: 'contain' }}
            />
          )}
        </DialogContent>
        <DialogActions sx={{ bgcolor: 'black', p: 2 }}>
          <Button onClick={() => setSelectedKeyframe(null)} variant="contained" color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};