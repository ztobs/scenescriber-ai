/** Time formatting utilities */

/**
 * Format seconds to HH:MM:SS.mmm format
 */
export const formatTimestamp = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const milliseconds = Math.floor((seconds - Math.floor(seconds)) * 1000);
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
};

/**
 * Format seconds to SRT timestamp format (HH:MM:SS,mmm)
 */
export const formatSrtTimestamp = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const milliseconds = Math.floor((seconds - Math.floor(seconds)) * 1000);
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${milliseconds.toString().padStart(3, '0')}`;
};

/**
 * Format duration in seconds to human readable format
 */
export const formatDuration = (seconds: number): string => {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
};

/**
 * Calculate time remaining based on progress
 */
export const calculateTimeRemaining = (
  progress: number,
  elapsedTime: number
): number => {
  if (progress <= 0) return 0;
  const estimatedTotalTime = elapsedTime / (progress / 100);
  return Math.max(0, estimatedTotalTime - elapsedTime);
};