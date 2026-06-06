"""benchmark.py — run llama.cpp inference benchmarks on CNOS test queries.

This module prepares and invokes llama-cli.exe or llama-bench.exe for
CPU-only inference.  No actual benchmarks are run (the user must first
set up llama.cpp and download a GGUF model).

Usage:
    python prototypes/llamacpp_bench/benchmark.py --help
    python prototypes/llamacpp_bench/benchmark.py --list-models
    python prototypes/llamacpp_bench/benchmark.py --model tinyllama --n-gpu-layers 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry — maps short names to recommended GGUF filenames / URLs
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tinyllama": {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "params": "1.1B",
        "ram_estimate_gb": 1.5,
    },
    "llama-3.2-1b": {
        "name": "Llama-3.2-1B-Instruct",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "params": "1B",
        "ram_estimate_gb": 1.2,
    },
    "qwen-2.5-1.5b": {
        "name": "Qwen2.5-1.5B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "params": "1.5B",
        "ram_estimate_gb": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Benchmark queries (matching the CNOS benchmark suite)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: List[Dict[str, str]] = [
    {"query": "What is 2+2?", "type": "simple"},
    {"query": "What is the capital of France?", "type": "simple"},
    {"query": "Explain REST API", "type": "medium"},
    {"query": "Write Python binary search", "type": "medium"},
    {"query": "Explain transformer attention", "type": "complex"},
]


# ---------------------------------------------------------------------------
# llama.cpp invocation helpers
# ---------------------------------------------------------------------------


def find_llama_cli() -> Optional[str]:
    """Locate llama-cli.exe on PATH or in common build directories."""
    exe = shutil.which("llama-cli.exe")
    if exe:
        return exe
    # Check common locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "llama.cpp", "build", "bin", "Release", "llama-cli.exe"),
        os.path.join(home, "llama.cpp", "build", "bin", "llama-cli.exe"),
        os.path.join("C:", "llama.cpp", "build", "bin", "Release", "llama-cli.exe"),
        os.path.join("C:", "llama.cpp", "build", "bin", "llama-cli.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def find_llama_bench() -> Optional[str]:
    """Locate llama-bench.exe on PATH or in common build directories."""
    exe = shutil.which("llama-bench.exe")
    if exe:
        return exe
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "llama.cpp", "build", "bin", "Release", "llama-bench.exe"),
        os.path.join(home, "llama.cpp", "build", "bin", "llama-bench.exe"),
        os.path.join("C:", "llama.cpp", "build", "bin", "Release", "llama-bench.exe"),
        os.path.join("C:", "llama.cpp", "build", "bin", "llama-bench.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@dataclass
class BenchResult:
    """Result of a single llama-cli inference run.

    Attributes:
        query: Input text.
        response: Generated output.
        latency_s: Wall-clock inference time.
        tokens_generated: Number of output tokens.
        tokens_per_second: Throughput.
        model: GGUF model filename.
        n_gpu_layers: GPU layers used (0 = CPU only).
    """
    query: str = ""
    response: str = ""
    latency_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    model: str = ""
    n_gpu_layers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "latency_s": round(self.latency_s, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "model": self.model,
            "n_gpu_layers": self.n_gpu_layers,
        }


def run_llama_cli(
    model_path: str,
    prompt: str,
    max_tokens: int = 128,
    n_gpu_layers: int = 0,
    temp: float = 0.0,
    verbose: bool = False,
) -> BenchResult:
    """Run a single inference with llama-cli.exe.

    Args:
        model_path: Path to the GGUF model file.
        prompt: Input prompt text.
        max_tokens: Maximum tokens to generate.
        n_gpu_layers: GPU offloading (0 = CPU only).
        temp: Temperature (0 = greedy).
        verbose: Print stderr from llama-cli.

    Returns:
        A :class:`BenchResult`.
    """
    cli = find_llama_cli()
    if not cli:
        raise RuntimeError(
            "llama-cli.exe not found. "
            "Build llama.cpp first: see README.md for instructions."
        )

    cmd = [
        cli,
        "-m", model_path,
        "--prompt", prompt,
        "-n", str(max_tokens),
        "--temp", str(temp),
        "--no-mmap",  # works better on Windows without CUDA
        "-ngl", str(n_gpu_layers),
    ]

    logger.debug("Running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.perf_counter() - t0

    if verbose and r.stderr:
        for line in r.stderr.splitlines():
            logger.debug("  llama: %s", line)

    response = r.stdout.strip()
    # Parse timing info from stderr if available
    tokens_gen = 0
    tokens_sec = 0.0
    for line in r.stderr.splitlines():
        if "tokens generated" in line.lower():
            import re
            m = re.search(r"(\d+)\s+tokens\s+generated", line)
            if m:
                tokens_gen = int(m.group(1))
        if "tokens per second" in line.lower():
            import re
            m = re.search(r"([\d.]+)\s+tokens\s+per\s+second", line)
            if m:
                tokens_sec = float(m.group(1))

    return BenchResult(
        query=prompt,
        response=response,
        latency_s=round(elapsed, 4),
        tokens_generated=tokens_gen,
        tokens_per_second=tokens_sec,
        model=os.path.basename(model_path),
        n_gpu_layers=n_gpu_layers,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="llama.cpp benchmark harness for CNOS",
    )
    p.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="tinyllama",
        help="Model key (default: tinyllama)",
    )
    p.add_argument(
        "--model-path", type=str, default=None,
        help="Path to GGUF file (overrides model registry)",
    )
    p.add_argument(
        "--n-gpu-layers", type=int, default=0,
        help="GPU layers (0 = CPU only, default: 0)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max tokens to generate per query (default: 128)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature, 0=greedy (default: 0.0)",
    )
    p.add_argument(
        "--queries", type=int, default=None,
        help="Number of benchmark queries to run (default: all)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: prototypes/llamacpp_bench/output/)",
    )
    p.add_argument(
        "--list-models", action="store_true",
        help="List available model registry entries and exit",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    if args.list_models:
        print("# Available Models\n")
        for key, info in MODEL_REGISTRY.items():
            print(f"- **{key}**: {info['name']} ({info['params']})")
            print(f"  - GGUF: {info['file']}")
            print(f"  - URL: {info['url']}")
            print(f"  - RAM estimate: {info['ram_estimate_gb']} GB")
            print()
        return 0

    # Locate llama-cli
    cli = find_llama_cli()
    if not cli:
        logger.error(
            "llama-cli.exe not found. "
            "Please build llama.cpp first:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cd llama.cpp\n"
            "  cmake -B build\n"
            "  cmake --build build --config Release\n"
        )
        return 1
    logger.info("llama-cli.exe found: %s", cli)

    # Model path
    if args.model_path:
        model_path = args.model_path
    elif args.model in MODEL_REGISTRY:
        info = MODEL_REGISTRY[args.model]
        model_path = info["file"]
        logger.info(
            "Model: %s (%s)  RAM est: %.1f GB",
            args.model, info["name"], info["ram_estimate_gb"],
        )
        logger.info(
            "GGUF file: %s  (download from %s if not present)",
            model_path, info["url"],
        )
        if not os.path.isfile(model_path):
            logger.warning(
                "GGUF file not found: %s. "
                "Download it from %s and place it in the working directory.",
                model_path, info["url"],
            )
            logger.warning("Use --model-path to specify a custom location.")
            return 1
    else:
        logger.error("Unknown model: %s", args.model)
        return 1

    # Queries
    queries = BENCHMARK_QUERIES
    if args.queries is not None and args.queries < len(queries):
        queries = queries[: args.queries]

    logger.info(
        "Running %d queries  max_tokens=%d  temp=%.1f",
        len(queries), args.max_tokens, args.temperature,
    )

    results: List[BenchResult] = []
    for i, qd in enumerate(queries):
        query = qd["query"]
        logger.info("Query %d/%d: %s", i + 1, len(queries), query)
        try:
            br = run_llama_cli(
                model_path=model_path,
                prompt=query,
                max_tokens=args.max_tokens,
                n_gpu_layers=args.n_gpu_layers,
                temp=args.temperature,
                verbose=args.verbose,
            )
            results.append(br)
            logger.info(
                "  latency=%.2fs  tokens=%d  tok/s=%.1f",
                br.latency_s, br.tokens_generated, br.tokens_per_second,
            )
            logger.info("  response: %s", br.response[:80])
        except subprocess.TimeoutExpired:
            logger.error("  Query timed out (>600s)")
        except Exception as exc:
            logger.error("  Query failed: %s", exc)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    out_json = os.path.join(output_dir, "llamacpp_bench_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            [r.to_dict() for r in results],
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Wrote: %s", out_json)
    logger.info("Done. %d/%d queries succeeded.", len(results), len(queries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
