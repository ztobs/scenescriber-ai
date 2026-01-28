"""Cleanup utilities for removing old files and logs."""

import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CleanupManager:
    """Manages cleanup of old files and logs based on retention settings."""

    def __init__(self):
        """Initialize cleanup manager with environment settings."""
        self.retention_days = int(os.getenv("CLEANUP_RETENTION_DAYS", "7"))
        self.enable_auto_cleanup = os.getenv("ENABLE_AUTO_CLEANUP", "true").lower() == "true"
        
        # Parse directories to clean up
        cleanup_dirs_str = os.getenv("CLEANUP_DIRECTORIES", "uploads,exports,keyframes")
        self.cleanup_directories = [d.strip() for d in cleanup_dirs_str.split(",") if d.strip()]
        
        # Parse log files to clean up
        log_files_str = os.getenv("CLEANUP_LOG_FILES", "scenescriber.log,backend.log,server.log")
        self.log_files_to_clean = [f.strip() for f in log_files_str.split(",") if f.strip()]
        
        logger.info(
            f"CleanupManager initialized: retention={self.retention_days} days, "
            f"auto_cleanup={self.enable_auto_cleanup}, "
            f"directories={self.cleanup_directories}, "
            f"log_files={self.log_files_to_clean}"
        )

    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed.
        
        Returns:
            True if auto cleanup is enabled, False otherwise
        """
        return self.enable_auto_cleanup

    def cleanup_old_files(self) -> Dict[str, int]:
        """Remove files older than retention days from configured directories.
        
        Returns:
            Dictionary with counts of files removed from each directory
        """
        if not self.should_cleanup():
            logger.info("Auto cleanup is disabled, skipping cleanup")
            return {"skipped": 0}
        
        cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
        results = {}
        
        for dir_name in self.cleanup_directories:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                logger.debug(f"Directory {dir_name} does not exist, skipping")
                continue
            
            files_removed = 0
            try:
                for item in dir_path.iterdir():
                    if item.is_file():
                        # Check if file is older than retention days
                        if item.stat().st_mtime < cutoff_time:
                            try:
                                item.unlink()
                                files_removed += 1
                                logger.debug(f"Removed old file: {item}")
                            except Exception as e:
                                logger.warning(f"Failed to remove file {item}: {e}")
                    elif item.is_dir():
                        # For directories, check if empty or old
                        try:
                            # Remove empty directories
                            if not any(item.iterdir()):
                                item.rmdir()
                                logger.debug(f"Removed empty directory: {item}")
                        except Exception as e:
                            logger.debug(f"Could not remove directory {item}: {e}")
                
                results[dir_name] = files_removed
                if files_removed > 0:
                    logger.info(f"Removed {files_removed} old files from {dir_name}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up directory {dir_name}: {e}")
                results[dir_name] = 0
        
        return results

    def cleanup_log_files(self) -> Dict[str, int]:
        """Clean up old log files.
        
        Returns:
            Dictionary with counts of log files cleaned up
        """
        if not self.should_cleanup():
            return {"skipped": 0}
        
        cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
        results = {}
        
        for log_file in self.log_files_to_clean:
            log_path = Path(log_file)
            if not log_path.exists():
                logger.debug(f"Log file {log_file} does not exist, skipping")
                continue
            
            try:
                # Check if log file is older than retention days
                if log_path.stat().st_mtime < cutoff_time:
                    # Instead of deleting, we can truncate or rotate
                    # For now, just delete old log files
                    log_path.unlink()
                    results[log_file] = 1
                    logger.info(f"Removed old log file: {log_file}")
                else:
                    results[log_file] = 0
                    
            except Exception as e:
                logger.error(f"Error cleaning up log file {log_file}: {e}")
                results[log_file] = 0
        
        return results

    def remove_last_uploaded_file(self) -> Optional[str]:
        """Remove the most recently uploaded video file.
        
        Returns:
            Name of removed file if successful, None otherwise
        """
        upload_dir = Path("uploads")
        if not upload_dir.exists():
            logger.debug("Uploads directory does not exist")
            return None
        
        try:
            # Get all files in uploads directory
            files = list(upload_dir.iterdir())
            if not files:
                logger.debug("No files in uploads directory")
                return None
            
            # Find the most recently modified file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            # Remove it
            latest_file.unlink()
            logger.info(f"Removed last uploaded file: {latest_file.name}")
            return latest_file.name
            
        except Exception as e:
            logger.error(f"Failed to remove last uploaded file: {e}")
            return None

    def perform_cleanup(self) -> Dict[str, any]:
        """Perform complete cleanup operation.
        
        This includes:
        1. Removing old files from configured directories
        2. Cleaning up old log files
        3. Removing the last uploaded file
        
        Returns:
            Dictionary with cleanup results
        """
        if not self.should_cleanup():
            return {"status": "skipped", "reason": "Auto cleanup disabled"}
        
        logger.info("Starting automatic cleanup...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "retention_days": self.retention_days,
            "old_files_removed": self.cleanup_old_files(),
            "log_files_cleaned": self.cleanup_log_files(),
            "last_upload_removed": self.remove_last_uploaded_file(),
        }
        
        # Count total files removed
        total_removed = 0
        for dir_name, count in results["old_files_removed"].items():
            if isinstance(count, int):
                total_removed += count
        
        for log_file, count in results["log_files_cleaned"].items():
            if isinstance(count, int):
                total_removed += count
        
        if results["last_upload_removed"]:
            total_removed += 1
        
        results["total_files_removed"] = total_removed
        results["status"] = "completed"
        
        logger.info(f"Cleanup completed: {total_removed} files removed")
        return results

    def get_directory_stats(self) -> Dict[str, Dict[str, any]]:
        """Get statistics about directories being cleaned up.
        
        Returns:
            Dictionary with directory statistics
        """
        stats = {}
        
        for dir_name in self.cleanup_directories:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                stats[dir_name] = {"exists": False, "file_count": 0, "size_bytes": 0}
                continue
            
            try:
                file_count = 0
                total_size = 0
                oldest_file = None
                newest_file = None
                
                for item in dir_path.iterdir():
                    if item.is_file():
                        file_count += 1
                        stat = item.stat()
                        total_size += stat.st_size
                        
                        # Track oldest and newest files
                        if oldest_file is None or stat.st_mtime < oldest_file[1]:
                            oldest_file = (item.name, stat.st_mtime)
                        if newest_file is None or stat.st_mtime > newest_file[1]:
                            newest_file = (item.name, stat.st_mtime)
                
                stats[dir_name] = {
                    "exists": True,
                    "file_count": file_count,
                    "size_bytes": total_size,
                    "size_mb": total_size / (1024 * 1024),
                    "oldest_file": oldest_file[0] if oldest_file else None,
                    "newest_file": newest_file[0] if newest_file else None,
                }
                
            except Exception as e:
                logger.error(f"Error getting stats for directory {dir_name}: {e}")
                stats[dir_name] = {"exists": True, "error": str(e)}
        
        return stats


# Global instance for easy access
cleanup_manager = CleanupManager()