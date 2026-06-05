"""baseline_runner -- runs standard transformer inference (no CNOS optimizations).

Collects:
  * RAM usage (peak, average)
  * CPU usage (peak, average)
  * Latency
  * Tokens generated
  * Tokens per second
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_PROTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "real_inference"))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

logger = logging.getLogger(__name__)

_OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@dataclass
class BaselineQueryResult:
    query: str = ""
    response: str = ""
    latency_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    ram_peak_mb: float = 0.0
    ram_avg_mb: float = 0.0
    cpu_peak_pct: float = 0.0
    cpu_avg_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "latency_s": round(self.latency_s, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "ram_peak_mb": round(self.ram_peak_mb, 1),
            "ram_avg_mb": round(self.ram_avg_mb, 1),
            "cpu_peak_pct": round(self.cpu_peak_pct, 1),
            "cpu_avg_pct": round(self.cpu_avg_pct, 1),
        }


@dataclass
class BaselineResult:
    queries: List[BaselineQueryResult] = field(default_factory=list)
    model_key: str = ""
    num_layers: int = 0
    total_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "num_layers": self.num_layers,
            "total_time_s": round(self.total_time_s, 2),
            "queries": [q.to_dict() for q in self.queries],
        }

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            os.makedirs(_OUT_DIR, exist_ok=True)
            path = os.path.join(_OUT_DIR, "baseline_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Baseline results saved to %s", path)
        return path


class BaselineRunner:
    """Runs standard (unoptimized) model inference.

    Args:
        bundle: The ``ModelBundle`` (from real_inference model_loader).
        max_tokens: Maximum tokens to generate per query.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        bundle: Any,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.bundle = bundle
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_layers = bundle.num_layers

        from routed_inference import RoutedInferenceEngine
        self.engine = RoutedInferenceEngine(
            bundle=bundle,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info("BaselineRunner ready -- %s, %d layers",
                     bundle.model_name, self.num_layers)

    def run_query(self, query: str) -> BaselineQueryResult:
        result = BaselineQueryResult(query=query)
        ram_samples: List[float] = []
        cpu_samples: List[float] = []

        import psutil
        proc = psutil.Process(os.getpid())

        start = time.perf_counter()
        ram_start = proc.memory_info().rss / (1024 * 1024)
        cpu_start = proc.cpu_percent(interval=None)

        response, metrics = self.engine.generate_baseline(query)

        elapsed = time.perf_counter() - start
        ram_end = proc.memory_info().rss / (1024 * 1024)
        cpu_end = proc.cpu_percent(interval=None)

        result.response = response
        result.latency_s = elapsed
        result.tokens_generated = metrics.num_tokens_generated
        result.tokens_per_sec = metrics.num_tokens_generated / max(elapsed, 1e-9)
        result.ram_peak_mb = max(ram_start, ram_end)
        result.ram_avg_mb = (ram_start + ram_end) / 2
        result.cpu_peak_pct = max(cpu_start, cpu_end)
        result.cpu_avg_pct = (cpu_start + cpu_end) / 2

        return result

    def run_queries(self, queries: List[str]) -> BaselineResult:
        all_results: List[BaselineQueryResult] = []
        t0 = time.perf_counter()

        for i, q in enumerate(queries):
            logger.info("Baseline query %d/%d: %s", i + 1, len(queries), q[:50])
            try:
                r = self.run_query(q)
                all_results.append(r)
                logger.info(
                    "  latency=%.2fs tokens=%d RAM=%.0fMB",
                    r.latency_s, r.tokens_generated, r.ram_peak_mb,
                )
            except Exception as exc:
                logger.error("Baseline query %d failed: %s", i, exc)
                all_results.append(BaselineQueryResult(
                    query=q, response=f"ERROR: {exc}",
                ))

        total = time.perf_counter() - t0
        return BaselineResult(
            queries=all_results,
            model_key=self.bundle.model_name,
            num_layers=self.num_layers,
            total_time_s=total,
        )

    def cleanup(self) -> None:
        self.engine.cleanup()
        logger.info("BaselineRunner cleaned up")
