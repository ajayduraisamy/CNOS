"""Logging module for the CNOS Memory Profiler.

Handles structured logging of profiling data to both CSV files
and the console with configurable verbosity levels.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import ProfilerConfig


class DataLogger:
    """Writes profiling measurements to CSV and manages console logging.

    Each call to ``log_record`` appends a row to the CSV file. The file
    is created with a header on first write. Console output is handled
    through Python's standard ``logging`` module.

    Args:
        config: ProfilerConfig instance controlling output paths and verbosity.
    """

    def __init__(self, config: ProfilerConfig) -> None:
        self.config = config
        self._setup_logging()
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file: Optional[TextIO] = None
        self._header_written = False

    def _setup_logging(self) -> None:
        """Configure the root logger with console handler."""
        self.logger = logging.getLogger("memory_profiler")
        self.logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def open_csv(self) -> None:
        """Open the CSV file for writing."""
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(path, mode="w", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp",
            "cpu_percent",
            "memory_rss_mb",
            "memory_percent",
            "disk_read_mb",
            "disk_write_mb",
            "num_threads",
            "process_count",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        self._header_written = True
        self.logger.info("CSV log created at %s", path.resolve())

    def log_record(self, record: Dict[str, float]) -> None:
        """Append a single measurement record to the CSV and optionally console.

        Args:
            record: Dictionary of metric_name -> value.
        """
        record["timestamp"] = datetime.now().isoformat(timespec="milliseconds")

        if self._csv_writer is not None:
            self._csv_writer.writerow(record)
            self._csv_file.flush()

        if self.config.verbose:
            mem = record.get("memory_rss_mb", 0)
            cpu = record.get("cpu_percent", 0)
            disk_r = record.get("disk_read_mb", 0)
            disk_w = record.get("disk_write_mb", 0)
            self.logger.info(
                "CPU: %5.1f%%  |  RAM: %6.1f MB  |  Disk R: %5.1f MB  W: %5.1f MB",
                cpu,
                mem,
                disk_r,
                disk_w,
            )

    def close(self) -> None:
        """Close the CSV file if open."""
        if self._csv_file is not None:
            self._csv_file.close()
            self.logger.info("CSV log closed.")
