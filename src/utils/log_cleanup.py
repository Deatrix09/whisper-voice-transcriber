import os
import time
from datetime import datetime, timedelta
from typing import List
import logging
from pathlib import Path

class LogCleanup:
    def __init__(self, logs_dir: str, retention_days: int):
        """
        Initialize LogCleanup with logs directory and retention period.
        
        Args:
            logs_dir (str): Path to logs directory
            retention_days (int): Number of days to keep logs
        """
        self.logs_dir = Path(logs_dir)
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)

    def cleanup_old_logs(self) -> None:
        """Remove log files older than retention_days."""
        if not self.logs_dir.exists():
            self.logger.warning(f"Logs directory does not exist: {self.logs_dir}")
            return

        current_time = time.time()
        retention_seconds = self.retention_days * 24 * 60 * 60

        try:
            for log_file in self.logs_dir.glob("*.log"):
                file_age = current_time - log_file.stat().st_mtime
                
                if file_age > retention_seconds:
                    try:
                        log_file.unlink()
                        self.logger.info(f"Removed old log file: {log_file.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove old log file {log_file.name}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error during log cleanup: {str(e)}")

    def get_recent_logs(self, days: int = 1) -> List[str]:
        """
        Get list of log files from the last n days.
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            List[str]: List of recent log filenames
        """
        if not self.logs_dir.exists():
            self.logger.warning(f"Logs directory does not exist: {self.logs_dir}")
            return []

        current_time = datetime.now()
        recent_logs = []

        try:
            for log_file in self.logs_dir.glob("*.log"):
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if (current_time - file_time) <= timedelta(days=days):
                    recent_logs.append(log_file.name)

            # Sort by modification time (most recent first)
            recent_logs.sort(
                key=lambda x: os.path.getmtime(self.logs_dir / x),
                reverse=True
            )
            
            return recent_logs
            
        except Exception as e:
            self.logger.error(f"Error getting recent logs: {str(e)}")
            return []

    def get_log_stats(self) -> dict:
        """
        Get statistics about log files.
        
        Returns:
            dict: Statistics about log files
        """
        if not self.logs_dir.exists():
            return {
                "total_logs": 0,
                "total_size": 0,
                "oldest_log": None,
                "newest_log": None
            }

        try:
            log_files = list(self.logs_dir.glob("*.log"))
            
            if not log_files:
                return {
                    "total_logs": 0,
                    "total_size": 0,
                    "oldest_log": None,
                    "newest_log": None
                }

            # Get file statistics
            file_stats = [
                (f, f.stat().st_mtime, f.stat().st_size)
                for f in log_files
            ]
            
            # Sort by modification time
            file_stats.sort(key=lambda x: x[1])
            
            return {
                "total_logs": len(file_stats),
                "total_size": sum(size for _, _, size in file_stats),
                "oldest_log": file_stats[0][0].name,
                "newest_log": file_stats[-1][0].name
            }
            
        except Exception as e:
            self.logger.error(f"Error getting log statistics: {str(e)}")
            return {
                "total_logs": 0,
                "total_size": 0,
                "oldest_log": None,
                "newest_log": None
            } 