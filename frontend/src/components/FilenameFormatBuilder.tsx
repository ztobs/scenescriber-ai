/** Filename format builder component */

import React, {useState, useEffect} from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Chip,
  Stack,
  Tooltip,
  Alert,
} from '@mui/material';
import {
  RestartAlt as ResetIcon,
  ListAlt as FullIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { videoApi } from '../utils/api';
import type { FilenameFormatInfo } from '../types';

interface FilenameFormatBuilderProps {
  value: string;
  onChange: (format: string) => void;
  disabled?: boolean;
}

export const FilenameFormatBuilder: React.FC<FilenameFormatBuilderProps> = ({
  value,
  onChange,
  disabled = false,
}) => {
  const [formatInfo, setFormatInfo] = useState<FilenameFormatInfo | null>(null);
  const [cursorPosition, setCursorPosition] = useState<number>(0);

  useEffect(() => {
    // Load placeholder information
    const loadPlaceholders = async () => {
      try {
        const info = await videoApi.getFilenamePlaceholders();
        setFormatInfo(info);
      } catch (error) {
        console.error('Failed to load filename placeholders:', error);
      }
    };

    loadPlaceholders();
  }, []);

  const handleInsertPlaceholder = (placeholder: string) => {
    // Insert placeholder at cursor position
    const before = value.substring(0, cursorPosition);
    const after = value.substring(cursorPosition);
    const newValue = before + placeholder + after;
    onChange(newValue);
    setCursorPosition(cursorPosition + placeholder.length);
  };

  const handleTextFieldChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  };

  const handleInputRef = (input: HTMLInputElement | null) => {
    if (input) {
      input.selectionStart = cursorPosition;
      input.selectionEnd = cursorPosition;
    }
  };

  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    setCursorPosition(e.target.selectionStart || 0);
  };

  const handleDefaultFormat = () => {
    if (formatInfo) {
      onChange(formatInfo.default_format);
    }
  };

  const handleFullFormat = () => {
    if (formatInfo) {
      onChange(formatInfo.full_format);
    }
  };

  if (!formatInfo) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Loading filename format options...
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        📝 Customize Export Filename
      </Typography>

      <Alert severity="info" sx={{ mb: 2 }}>
        <Typography variant="body2">
          Click placeholders below to build your filename. Default format uses only video name and
          timestamp. Use "All Fields" for detailed profiling of test cases.
        </Typography>
      </Alert>

      <Box sx={{ mb: 2 }}>
        <TextField
          fullWidth
          label="Filename Format"
          value={value}
          onChange={handleTextFieldChange}
          onBlur={handleBlur}
          inputRef={handleInputRef}
          disabled={disabled}
          helperText="Click placeholders to insert or type manually"
          variant="outlined"
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Available Placeholders:
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ gap: 1 }}>
          {Object.entries(formatInfo.description).map(([placeholder, description]) => (
            <Tooltip key={placeholder} title={description} arrow>
              <Chip
                label={placeholder}
                onClick={() => handleInsertPlaceholder(placeholder)}
                clickable
                variant="outlined"
                icon={<AddIcon />}
                disabled={disabled}
                sx={{ cursor: 'pointer' }}
              />
            </Tooltip>
          ))}
        </Stack>
      </Box>

      <Box sx={{ display: 'flex', gap: 2 }}>
        <Button
          variant="outlined"
          startIcon={<ResetIcon />}
          onClick={handleDefaultFormat}
          disabled={disabled}
          size="small"
        >
          Use Default
        </Button>
        <Button
          variant="outlined"
          startIcon={<FullIcon />}
          onClick={handleFullFormat}
          disabled={disabled}
          size="small"
        >
          Use All Fields (Profiling)
        </Button>
      </Box>

      <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          Examples:
        </Typography>
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
          <strong>Default:</strong> my_video_20260128_120000.srt
        </Typography>
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
          <strong>Full:</strong> my_video_s2_d2_ollama_llama2_1.45x_20260128_120000.srt
        </Typography>
      </Box>
    </Paper>
  );
};
