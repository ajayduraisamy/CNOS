"""Configuration module for the CNOS Memory Profiler.

Centralizes all user-configurable settings including sampling
intervals, output paths, alert thresholds, and process targets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ProfilerConfig:
    """Top-level configuration for the memory profiler.

    Attributes:
        sampling_interval: Seconds between consecutive measurements.
        duration: Total profiling duration in seconds (None = run until Ctrl+C).
        output_dir: Directory for log files and reports.
        log_filename: Name of the CSV log file.
        process_name: Name of the target process to monitor (None = system-wide).
        process_pid: Specific PID to monitor (overrides process_name if set).
        cpu_warn_threshold: CPU % threshold for warning alerts.
        memory_warn_threshold: Memory % threshold for warning alerts.
        verbose: Enable verbose console output during profiling.
    """

    sampling_interval: float = 1.0
    duration: Optional[float] = None
    output_dir: str = "logs"
    log_filename: str = "profile_results.csv"
    process_name: Optional[str] = None
    process_pid: Optional[int] = None
    cpu_warn_threshold: float = 90.0
    memory_warn_threshold: float = 85.0
    verbose: bool = True

    monitored_processes: List[str] = field(default_factory=lambda: ["python"])

    @property
    def output_path(self) -> Path:
        """Full path to the output CSV file."""
        return Path(self.output_dir) / self.log_filename

    def validate(self) -> None:
        """Validate configuration values, raising on invalid input."""
        if self.sampling_interval <= 0:
            raise ValueError("sampling_interval must be positive")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive or None")
        if self.cpu_warn_threshold < 0 or self.cpu_warn_threshold > 100:
            raise ValueError("cpu_warn_threshold must be in [0, 100]")
        if self.memory_warn_threshold < 0 or self.memory_warn_threshold > 100:
            raise ValueError("memory_warn_threshold must be in [0, 100]")
