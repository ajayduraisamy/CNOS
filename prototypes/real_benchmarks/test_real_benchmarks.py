"""Tests for the real_benchmarks prototype.

These tests use SimulatedModelAdapter (from integration_engine) to verify
the benchmark suite logic without downloading TinyLlama.
"""

from __future__ import annotations

import json
import logging
import os
import sys

_PROTO = os.path.abspath(os.path.dirname(__file__))
_PROTO_REAL = os.path.abspath(os.path.join(_PROTO, "..", "real_inference"))
_PROTO_INT = os.path.abspath(os.path.join(_PROTO, "..", "integration_engine"))

for _p in (_PROTO, _PROTO_REAL, _PROTO_INT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"  -  {detail}"
        print(msg)


def approx(a: float, b: float, eps: float = 0.01) -> bool:
    return abs(a - b) < eps


# ---------------------------------------------------------------------------
# Fake bundle for testing runners without real model
# ---------------------------------------------------------------------------

class FakeBundle:
    """Minimal bundle stub that satisfies the interfaces expected by runners."""
    def __init__(self, model_key: str = "tinyllama", num_layers: int = 22):
        self.model_name = model_key
        self.model_key = model_key
        self.num_layers = num_layers
        self.model = self._make_model(num_layers)
        self.tokenizer = self._make_tokenizer()
        self.device = "cpu"
        self.dtype = "float32"

    class _FakeConfig:
        num_attention_heads = 32
        hidden_size = 2048

    def _make_model(self, num_layers: int):
        import torch
        import torch.nn as nn
        model = nn.Module()
        model.config = self._FakeConfig()
        model.device_indices = {"cpu": 0}
        model.hf_device_map = {"cpu": "cpu"}
        model.model = nn.Module()
        model.model.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            model.model.layers.append(layer)
        model.lm_head = nn.Linear(2048, 32000)
        def _generate(**kwargs):
            inp = kwargs.get("input_ids", torch.randint(0, 100, (1, 5)))
            batch, seq = inp.shape
            fake_out = torch.cat([inp, torch.randint(0, 100, (batch, 10))], dim=1)
            return fake_out
        model.generate = _generate
        return model

    def _make_tokenizer(self):
        class FakeBatchEncoding(dict):
            def to(self, device):
                return self
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def __call__(self, text, return_tensors="pt"):
                import torch
                return FakeBatchEncoding({"input_ids": torch.randint(0, 100, (1, 5))})
            def decode(self, ids, skip_special_tokens=True):
                return f"[fake: {ids.tolist() if hasattr(ids, 'tolist') else ids}]"
        return FakeTokenizer()


# ---------------------------------------------------------------------------
# Test: benchmark_loader
# ---------------------------------------------------------------------------

def test_benchmark_loader() -> None:
    print("\n--- benchmark_loader ---")
    import benchmark_loader as bl

    check("MODEL_KEYS has tinyllama",
          "tinyllama" in bl.MODEL_KEYS)
    check("MODEL_KEYS has qwen-1.5b",
          "qwen-1.5b" in bl.MODEL_KEYS)
    check("MODEL_LAYERS matches",
          bl.MODEL_LAYERS["tinyllama"] == 22)

    # LoadResult with error
    fail = bl.LoadResult(success=False, error="test error")
    check("failed LoadResult.summary",
          fail.summary()["success"] is False)


# ---------------------------------------------------------------------------
# Test: baseline_runner (simulated)
# ---------------------------------------------------------------------------

def test_baseline_runner() -> None:
    print("\n--- baseline_runner (simulated) ---")
    from baseline_runner import BaselineRunner, BaselineQueryResult, BaselineResult

    bundle = FakeBundle()
    runner = BaselineRunner(bundle, max_tokens=16)
    check("BaselineRunner created", runner is not None)

    qr = runner.run_query("What is 2+2?")
    check("query result has response", bool(qr.response))
    check("latency > 0", qr.latency_s >= 0)
    check("tokens_generated >= 0", qr.tokens_generated >= 0)
    check("tokens_per_sec >= 0", qr.tokens_per_sec >= 0)
    check("RAM peak >= 0", qr.ram_peak_mb >= 0)

    queries = ["Hello", "World"]
    result = runner.run_queries(queries)
    check("BaselineResult has 2 queries",
          len(result.queries) == 2)
    check("BaselineResult total_time > 0",
          result.total_time_s >= 0)

    d = result.to_dict()
    check("to_dict has model_key",
          "model_key" in d)
    check("to_dict has queries",
          len(d["queries"]) == 2)

    runner.cleanup()


# ---------------------------------------------------------------------------
# Test: cnos_runner (simulated)
# ---------------------------------------------------------------------------

def test_cnos_runner() -> None:
    print("\n--- cnos_runner (simulated) ---")
    from cnos_runner import CnosRunner, CnosQueryResult, CnosResult

    bundle = FakeBundle()
    runner = CnosRunner(
        bundle=bundle,
        max_tokens=16,
        routing_policy="adaptive",
        quantisation="int8",
        ram_gb=4.0,
    )
    check("CnosRunner created", runner is not None)

    qr = runner.run_query("Explain gravity.")
    check("CNOS query result has response", bool(qr.response))
    check("latency >= 0", qr.latency_s >= 0)
    check("tokens_generated >= 0", qr.tokens_generated >= 0)
    check("layers_executed > 0", qr.layers_executed > 0)
    check("compute_reduction_pct >= 0", qr.compute_reduction_pct >= 0)
    check("cache_hit_rate_pct >= 0", qr.cache_hit_rate_pct >= 0)

    queries = ["Q1", "Q2"]
    result = runner.run_queries(queries)
    check("CnosResult has 2 queries",
          len(result.queries) == 2)
    check("CnosResult routing_policy set",
          result.routing_policy == "adaptive")

    d = result.to_dict()
    check("to_dict has model_key", "model_key" in d)
    check("to_dict has routing_policy", "routing_policy" in d)

    runner.cleanup()


# ---------------------------------------------------------------------------
# Test: metrics_collector
# ---------------------------------------------------------------------------

def test_metrics_collector() -> None:
    print("\n--- metrics_collector ---")
    from metrics_collector import compute_metrics, save_metrics_json, MetricsReport, ComparisonRow
    from baseline_runner import BaselineResult, BaselineQueryResult
    from cnos_runner import CnosResult, CnosQueryResult

    bq = BaselineQueryResult(
        query="test", response="baseline answer",
        latency_s=1.0, tokens_generated=10, tokens_per_sec=10.0,
        ram_peak_mb=500.0, ram_avg_mb=450.0,
        cpu_peak_pct=50.0, cpu_avg_pct=40.0,
    )
    baseline = BaselineResult(
        queries=[bq],
        model_key="tinyllama", num_layers=22,
        total_time_s=1.0,
    )

    cq = CnosQueryResult(
        query="test", response="cnos answer",
        baseline_response="baseline answer",
        latency_s=0.8, tokens_generated=10, tokens_per_sec=12.5,
        ram_peak_mb=400.0, ram_avg_mb=380.0,
        cpu_peak_pct=45.0, cpu_avg_pct=35.0,
        layers_executed=15, layers_skipped=7,
        compute_reduction_pct=31.8,
        page_faults=10, page_hits=90,
        cache_hit_rate_pct=90.0,
        compression_ratio=2.0,
        query_type="code", complexity_score=0.75,
    )
    cnos = CnosResult(
        queries=[cq],
        model_key="tinyllama", num_layers=22,
        routing_policy="adaptive", quantisation="int8",
        total_time_s=0.8,
    )

    report = compute_metrics(baseline, cnos)
    check("MetricsReport num_queries == 1",
          report.num_queries == 1)
    check("avg_latency_reduction_pct ~ 20%",
          approx(report.avg_latency_reduction_pct, 20.0, 1.0),
          str(report.avg_latency_reduction_pct))
    check("avg_ram_reduction_pct ~ 20%",
          approx(report.avg_ram_reduction_pct, 20.0, 1.0),
          str(report.avg_ram_reduction_pct))
    check("routing_policy matches",
          report.routing_policy == "adaptive")
    check("quantisation matches",
          report.quantisation == "int8")

    check("avg_cache_hit_rate_pct == 90",
          approx(report.avg_cache_hit_rate_pct, 90.0, 0.1))
    check("avg_compression_ratio == 2",
          approx(report.avg_compression_ratio, 2.0, 0.1))

    d = report.to_dict()
    check("to_dict has avg_jaccard_sim",
          "avg_jaccard_sim" in d)

    # Test per-query detail
    check("detail[0].latency_reduction_pct ~ 20%",
          approx(report.details[0].latency_reduction_pct, 20.0, 1.0))


# ---------------------------------------------------------------------------
# Test: report_generator
# ---------------------------------------------------------------------------

def test_report_generator(tmp_dir: str) -> None:
    print("\n--- report_generator ---")
    from report_generator import generate_markdown_report, generate_csv_report, generate_json_report, generate_all_reports
    from metrics_collector import MetricsReport, ComparisonRow

    report = MetricsReport(
        model_key="tinyllama",
        num_layers=22,
        routing_policy="adaptive",
        quantisation="int8",
        num_queries=1,
        avg_baseline_latency_s=1.0,
        avg_cnos_latency_s=0.8,
        avg_latency_reduction_pct=20.0,
        avg_baseline_ram_peak_mb=500.0,
        avg_cnos_ram_peak_mb=400.0,
        avg_ram_reduction_pct=20.0,
        avg_compute_reduction_pct=31.8,
        avg_jaccard_sim=0.85,
        avg_rouge_l=0.78,
        avg_tokens_per_sec=12.5,
        avg_cache_hit_rate_pct=90.0,
        avg_compression_ratio=2.0,
        details=[
            ComparisonRow(
                query="test query",
                baseline_response="baseline",
                cnos_response="cnos",
                baseline_latency_s=1.0,
                cnos_latency_s=0.8,
                latency_reduction_pct=20.0,
                baseline_ram_peak_mb=500.0,
                cnos_ram_peak_mb=400.0,
                ram_reduction_pct=20.0,
                tokens_generated=10,
                layers_skipped=7,
                compute_reduction_pct=31.8,
                jaccard_sim=0.85,
                rouge_l=0.78,
            ),
        ],
    )

    md_path = generate_markdown_report(report, os.path.join(tmp_dir, "test_report.md"))
    csv_path = generate_csv_report(report, os.path.join(tmp_dir, "test_report.csv"))
    json_path = generate_json_report(report, os.path.join(tmp_dir, "test_report.json"))

    check("md file exists", os.path.exists(md_path))
    check("csv file exists", os.path.exists(csv_path))
    check("json file exists", os.path.exists(json_path))

    with open(md_path, encoding="utf-8") as f:
        md = f.read()
    check("md has title", "# CNOS" in md)
    check("md has summary table", "| Metric |" in md)

    with open(csv_path, encoding="utf-8") as f:
        csv_content = f.read()
    check("csv has header", "query" in csv_content)

    with open(json_path, encoding="utf-8") as f:
        js = json.load(f)
    check("json has model_key",
          js.get("model_key") == "tinyllama")

    paths = generate_all_reports(report)
    check("generate_all_reports returns 3 paths",
          len(paths) == 3)


# ---------------------------------------------------------------------------
# Test: benchmark_suite
# ---------------------------------------------------------------------------

def test_benchmark_suite_api() -> None:
    print("\n--- benchmark_suite API (simulated) ---")
    # Test that we can import and construct the module-level things
    import benchmark_suite as bs
    check("DEFAULT_QUERIES count", len(bs.DEFAULT_QUERIES) == 10)
    check("parse_args works", bs.parse_args(["--model", "tinyllama"]) is not None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="cnos_benchmark_test_")

    test_benchmark_loader()
    test_baseline_runner()
    test_cnos_runner()
    test_metrics_collector()
    test_report_generator(tmp_dir)
    test_benchmark_suite_api()

    total = PASS + FAIL
    print(f"\n{'='*40}")
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")

    # Clean up tmp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
