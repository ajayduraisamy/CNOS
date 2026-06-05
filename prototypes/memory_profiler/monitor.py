"""System resource monitor for the CNOS Memory Profiler.

Collects per-process or system-wide metrics including CPU utilisation,
memory (RSS), disk I/O, and thread counts using the ``psutil`` library.
"""

import os
from typing import Dict, Optional

import psutil

from config import ProfilerConfig


class SystemMonitor:
    """Captures snapshot measurements of system and process resources.

    Can target a specific process (by name or PID) or collect
    system-wide aggregates.

    Args:
        config: ProfilerConfig instance with process targeting settings.
    """

    def __init__(self, config: ProfilerConfig) -> None:
        self.config = config
        self._process: Optional[psutil.Process] = None
        self._disk_io_before: Optional[Dict[str, int]] = None
        self._resolve_target()

    def _resolve_target(self) -> None:
        """Locate the target process, if one is specified."""
        if self.config.process_pid is not None:
            try:
                self._process = psutil.Process(self.config.process_pid)
            except psutil.NoSuchProcess:
                print(f"Warning: no process with PID {self.config.process_pid}")
                self._process = None
            return

        if self.config.process_name is not None:
            for proc in psutil.process_iter(["pid", "name"]):
                if proc.info["name"] == self.config.process_name:
                    self._process = psutil.Process(proc.info["pid"])
                    return
            print(f"Warning: no process named '{self.config.process_name}'")

    def _sample_disk_io(self) -> Dict[str, int]:
        """Return cumulative disk read/write bytes since boot."""
        try:
            counter = psutil.disk_io_counters()
            if counter is None:
                return {"read_bytes": 0, "write_bytes": 0}
            return {
                "read_bytes": counter.read_bytes,
                "write_bytes": counter.write_bytes,
            }
        except RuntimeError:
            return {"read_bytes": 0, "write_bytes": 0}

    def start_disk_sampling(self) -> None:
        """Record initial disk counters for delta calculation."""
        self._disk_io_before = self._sample_disk_io()

    def snapshot(self) -> Dict[str, float]:
        """Collect a single measurement snapshot.

        Returns:
            Dictionary containing:
                - cpu_percent: CPU usage percent (system-wide).
                - memory_rss_mb: Resident set size in MB.
                - memory_percent: Fraction of total physical RAM used.
                - disk_read_mb: MB read since last snapshot.
                - disk_write_mb: MB written since last snapshot.
                - num_threads: Thread count of target process (or 0).
                - process_count: Total number of running processes.
        """
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()

        rss_mb = 0.0
        mem_pct = 0.0
        threads = 0

        if self._process is not None:
            try:
                with self._process.oneshot():
                    rss_mb = self._process.memory_info().rss / (1024 * 1024)
                    mem_pct = self._process.memory_percent()
                    threads = self._process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        disk_now = self._sample_disk_io()
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        if self._disk_io_before is not None:
            disk_read_mb = (disk_now["read_bytes"] - self._disk_io_before["read_bytes"]) / (1024 * 1024)
            disk_write_mb = (disk_now["write_bytes"] - self._disk_io_before["write_bytes"]) / (1024 * 1024)
        self._disk_io_before = disk_now

        return {
            "cpu_percent": cpu,
            "memory_rss_mb": rss_mb,
            "memory_percent": mem_pct,
            "disk_read_mb": disk_read_mb,
            "disk_write_mb": disk_write_mb,
            "num_threads": float(threads),
            "process_count": float(len(psutil.pids())),
        }

    def check_alerts(self, record: Dict[str, float]) -> None:
        """Print warning messages if thresholds are exceeded.

        Args:
            record: A snapshot dictionary from ``self.snapshot()``.
        """
        if record["cpu_percent"] > self.config.cpu_warn_threshold:
            print(
                f"⚠  High CPU: {record['cpu_percent']:.1f}% "
                f"(threshold: {self.config.cpu_warn_threshold}%)"
            )
        if record["memory_percent"] > self.config.memory_warn_threshold:
            print(
                f"⚠  High Memory: {record['memory_percent']:.1f}% "
                f"(threshold: {self.config.memory_warn_threshold}%)"
            )
