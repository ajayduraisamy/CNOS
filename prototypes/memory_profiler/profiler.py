#!/usr/bin/env python3
"""CNOS Memory Profiler — main entry point.

Measures system and process-level resource usage (RAM, CPU, disk I/O)
over time and logs results to a CSV file for offline analysis.

Usage:
    python profiler.py                      # Profile with defaults
    python profiler.py --interval 0.5       # Sample every 500 ms
    python profiler.py --duration 60        # Run for 60 seconds
    python profiler.py --name python        # Monitor python processes only
    python profiler.py --pid 1234           # Monitor a specific PID
    python profiler.py --verbose            # Show detailed console output
"""

import argparse
import signal
import sys
import time
from typing import NoReturn

from config import ProfilerConfig
from logger import DataLogger
from monitor import SystemMonitor


def parse_args(argv: list[str] | None = None) -> ProfilerConfig:
    """Parse command-line arguments into a ProfilerConfig."""
    parser = argparse.ArgumentParser(
        description="CNOS Memory Profiler — measure RAM, CPU, and disk I/O.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total profiling duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory for CSV output (default: logs/)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Target process name to monitor (default: system-wide)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target process PID (overrides --name)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose console output (default: true)",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Disable verbose console output",
    )

    parsed = parser.parse_args(argv)

    config = ProfilerConfig(
        sampling_interval=parsed.interval,
        duration=parsed.duration,
        output_dir=parsed.output_dir,
        process_name=parsed.name,
        process_pid=parsed.pid,
        verbose=parsed.verbose,
    )
    config.validate()
    return config


def _handle_signal(signum: int, _frame) -> None:
    """Raise KeyboardInterrupt on SIGINT / SIGTERM for clean shutdown."""
    print(f"\nReceived signal {signum}. Shutting down...")
    raise KeyboardInterrupt


def run(config: ProfilerConfig) -> None:
    """Main profiling loop.

    Args:
        config: Validated ProfilerConfig.
    """
    logger = DataLogger(config)
    monitor = SystemMonitor(config)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.open_csv()
    monitor.start_disk_sampling()
    logger.logger.info("Profiling started  (interval=%ss, duration=%s)", config.sampling_interval, config.duration or "∞")

    start_time = time.monotonic()
    sample_count = 0

    try:
        while True:
            # Check duration limit
            if config.duration is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= config.duration:
                    logger.logger.info("Duration limit reached (%.1f s). Stopping.", config.duration)
                    break

            record = monitor.snapshot()
            logger.log_record(record)
            monitor.check_alerts(record)
            sample_count += 1

            time.sleep(config.sampling_interval)

    except KeyboardInterrupt:
        logger.logger.info("Profiling interrupted by user.")

    finally:
        elapsed = time.monotonic() - start_time
        logger.logger.info(
            "Profiling finished  —  %d samples over %.1f seconds (%.2f samples/s)",
            sample_count,
            elapsed,
            sample_count / elapsed if elapsed > 0 else 0,
        )
        logger.close()


def main() -> None:
    """Parse args and run the profiler."""
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
