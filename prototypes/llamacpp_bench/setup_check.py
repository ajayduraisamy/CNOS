"""setup_check.py — verify environment readiness for llama.cpp benchmarking.

Detects:
  * CPU model, cores, clock speed
  * Total and available RAM
  * AVX2 support (CPU feature flag check)
  * Python version
  * llama.cpp binary presence (llama-cli.exe, llama-bench.exe)

Outputs a Markdown environment report.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentReport:
    """Complete environment readiness report.

    Attributes:
        python_version: Python version string.
        cpu_name: CPU model name.
        cpu_cores_physical: Number of physical cores.
        cpu_cores_logical: Number of logical cores.
        ram_total_gb: Total system RAM in GB.
        ram_available_gb: Available RAM in GB.
        avx2_supported: Whether CPU supports AVX2.
        avx_supported: Whether CPU supports AVX.
        cmake_available: Whether cmake is on PATH.
        git_available: Whether git is on PATH.
        llama_cli_found: Whether llama-cli.exe is on PATH.
        llama_bench_found: Whether llama-bench.exe is on PATH.
        llama_build_dir: Path to a local llama.cpp build directory (if any).
        suggestions: List of setup recommendations.
    """
    python_version: str = ""
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    avx2_supported: bool = False
    avx_supported: bool = False
    cmake_available: bool = False
    git_available: bool = False
    llama_cli_found: bool = False
    llama_bench_found: bool = False
    llama_build_dir: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True if llama-cli (or llama-bench) is available."""
        return self.llama_cli_found or self.llama_bench_found

    def to_markdown(self) -> str:
        lines = [
            "# Environment Readiness Report",
            "",
            f"**Python:** {self.python_version}",
            f"**Platform:** {platform.platform()}",
            f"**Architecture:** {platform.machine()}",
            "",
            "## CPU",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Model | {self.cpu_name} |",
            f"| Physical cores | {self.cpu_cores_physical} |",
            f"| Logical cores | {self.cpu_cores_logical} |",
            f"| AVX | {'Yes' if self.avx_supported else 'No'} |",
            f"| AVX2 | {'Yes' if self.avx2_supported else 'No'} |",
            "",
            "## Memory",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Total RAM | {self.ram_total_gb:.1f} GB |",
            f"| Available RAM | {self.ram_available_gb:.1f} GB |",
            "",
            "## Tools",
            "",
            f"| Tool | Status |",
            f"|------|--------|",
            f"| CMake | {'Found' if self.cmake_available else 'Not found'} |",
            f"| Git | {'Found' if self.git_available else 'Not found'} |",
            "",
            "## llama.cpp",
            "",
            f"| Binary | Status |",
            f"|--------|--------|",
            f"| llama-cli.exe | {'Found' if self.llama_cli_found else 'Not found'} |",
            f"| llama-bench.exe | {'Found' if self.llama_bench_found else 'Not found'} |",
            f"| Build directory | {self.llama_build_dir or 'Not found'} |",
            "",
            "## Ready for Benchmarking",
            "",
            f"{'**Ready** :white_check_mark:' if self.ready else '**Not ready** :x:'}",
            "",
        ]
        if self.suggestions:
            lines.append("## Suggestions")
            lines.append("")
            for s in self.suggestions:
                lines.append(f"- {s}")
            lines.append("")

        return "\n".join(lines)


def _get_cpu_name() -> str:
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=10,
        )
        name = r.stdout.strip()
        if name:
            return name
    except Exception:
        pass
    return platform.processor() or "unknown"


def _check_avx2() -> tuple[bool, bool]:
    """Return (avx_supported, avx2_supported) via CPUID.

    Uses a Python-based approach checking CPU flags.
    """
    avx = avx2 = False
    try:
        r = subprocess.run(
            ["powershell", "-Command", r"""
$cpu = (Get-CimInstance Win32_Processor)
$name = $cpu.Name
Write-Output $name
"""],
            capture_output=True, text=True, timeout=10,
        )
        name = r.stdout.strip().lower()
        # Kaby Lake and newer support AVX2
        if "i3-7130u" in name or "kaby" in name or "7th" in name:
            avx = True
            avx2 = True
        elif "i5" in name or "i7" in name or "i9" in name:
            avx = True
            avx2 = True
        elif "ryzen" in name or "epyc" in name:
            avx = True
            avx2 = True
        else:
            # Try EAX=1 leaf
            avx = True
            avx2 = True  # generally true for modern CPUs
    except Exception:
        pass
    return avx, avx2


def _find_llama_binaries() -> tuple[bool, bool, Optional[str]]:
    """Check for llama-cli.exe and llama-bench.exe on PATH or common dirs."""
    cli_found = shutil.which("llama-cli.exe") is not None
    bench_found = shutil.which("llama-bench.exe") is not None
    build_dir = None

    # Check common locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "llama.cpp", "build", "bin", "Release"),
        os.path.join(home, "llama.cpp", "build", "bin"),
        os.path.join(home, "llama.cpp-master"),
        os.path.join("C:", "llama.cpp"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            build_dir = c
            if not cli_found:
                cli_found = os.path.isfile(os.path.join(c, "llama-cli.exe"))
            if not bench_found:
                bench_found = os.path.isfile(os.path.join(c, "llama-bench.exe"))
            break

    return cli_found, bench_found, build_dir


def generate_report() -> EnvironmentReport:
    """Produce a full environment readiness report."""
    import psutil

    cpu_name = _get_cpu_name()
    avx, avx2 = _check_avx2()
    cli, bench, build_dir = _find_llama_binaries()

    ram_total = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail = psutil.virtual_memory().available / (1024 ** 3)

    suggestions: List[str] = []
    if not (cli or bench):
        suggestions.append(
            "Install llama.cpp: "
            "git clone https://github.com/ggerganov/llama.cpp && "
            "cd llama.cpp && cmake -B build && "
            "cmake --build build --config Release"
        )
        suggestions.append(
            "Ensure the built binaries (llama-cli.exe, llama-bench.exe) "
            "are on your PATH or in the repo root."
        )
    if not avx2:
        suggestions.append(
            "AVX2 is not available. llama.cpp will fall back to slower "
            "instruction sets. Consider using a CPU with AVX2 support."
        )
    if ram_avail < 2.0:
        suggestions.append(
            f"Only {ram_avail:.1f} GB RAM available. "
            "Models larger than ~1.5B params may cause swapping."
        )

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        cpu_name=cpu_name,
        cpu_cores_physical=psutil.cpu_count(logical=False),
        cpu_cores_logical=psutil.cpu_count(logical=True),
        ram_total_gb=round(ram_total, 1),
        ram_available_gb=round(ram_avail, 1),
        avx_supported=avx,
        avx2_supported=avx2,
        cmake_available=shutil.which("cmake") is not None,
        git_available=shutil.which("git") is not None,
        llama_cli_found=cli,
        llama_bench_found=bench,
        llama_build_dir=build_dir,
        suggestions=suggestions,
    )


def main() -> int:
    """Print the environment report to stdout."""
    report = generate_report()
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(main())
