"""Tests for the importance_router module.

Covers:
  * layer_profile — loading, candidates, critical layers
  * importance_router — decisions per mode, custom budget, never skips critical
  * quality_evaluator — comparison metrics
  * benchmark — simulated full run with report generation

Run:
    python -m pytest prototypes/importance_router/test_importance_router.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest

from layer_profile import LayerProfile, LayerScore
from importance_router import ImportanceRouter, RoutingMode, RoutingDecision
from quality_evaluator import (
    evaluate, jaccard_similarity, rouge_l_f1, QualityMetrics,
)
from benchmark import generate_report, BenchmarkResult, BenchmarkSuiteResult


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def profile():
    """Load the real layer importance profile."""
    return LayerProfile()


@pytest.fixture(scope="module")
def router(profile):
    return ImportanceRouter(profile)


# ===================================================================
# LayerProfile
# ===================================================================


class TestLayerProfile:
    def test_loads_all_layers(self, profile):
        assert profile.num_layers == 22
        assert len(profile._scores) == 22

    def test_get_score_exists(self, profile):
        s = profile.get_score(0)
        assert s is not None
        assert s.layer == 0
        assert s.avg_impact_score > 0
        assert s.classification in ("high", "medium", "low")

    def test_get_score_missing(self, profile):
        assert profile.get_score(99) is None

    def test_get_classification(self, profile):
        assert profile.get_classification(2) == "high"
        assert profile.get_classification(8) == "medium"

    def test_critical_layers_excludes_medium(self, profile):
        critical = set(profile.critical_layers)
        assert 0 in critical
        assert 2 in critical
        assert 8 not in critical
        assert 9 not in critical
        assert 19 not in critical

    def test_skip_candidates_are_medium_low(self, profile):
        candidates = profile.skip_candidates
        for layer, score in candidates:
            cls = profile.get_classification(layer)
            assert cls in ("medium", "low"), f"Layer {layer} is {cls}"

    def test_skip_candidates_sorted(self, profile):
        candidates = profile.skip_candidates
        scores = [s for _, s in candidates]
        assert scores == sorted(scores)

    def test_max_skippable(self, profile):
        assert profile.max_skippable() == len(profile.skip_candidates)
        assert profile.max_skippable() >= 2  # at least medium layers 8, 9, 19

    def test_profile_from_missing_file(self):
        with pytest.raises(FileNotFoundError):
            LayerProfile(filepath="/nonexistent/profile.json")


# ===================================================================
# ImportanceRouter
# ===================================================================


class TestImportanceRouter:
    def test_conservative_mode(self, router):
        dec = router.decide(RoutingMode.CONSERVATIVE)
        assert 1 <= dec.num_skipped <= 2
        assert dec.compute_reduction_pct > 0
        # Never skip critical layers
        for l in dec.skip_layers:
            assert l not in router.profile.critical_layers
        # Active + skipped = all layers
        assert len(dec.active_layers) + len(dec.skip_layers) == router.num_layers

    def test_balanced_mode(self, router):
        dec = router.decide(RoutingMode.BALANCED)
        assert 2 <= dec.num_skipped <= 3
        for l in dec.skip_layers:
            assert l not in router.profile.critical_layers

    def test_aggressive_mode(self, router):
        dec = router.decide(RoutingMode.AGGRESSIVE)
        assert 3 <= dec.num_skipped <= 4
        for l in dec.skip_layers:
            assert l not in router.profile.critical_layers

    def test_never_skips_critical(self, router):
        """All modes must never skip a high-impact layer."""
        for mode in RoutingMode:
            dec = router.decide(mode)
            for l in dec.skip_layers:
                assert router.profile.get_classification(l) != "high", (
                    f"Mode {mode} skips critical layer {l}"
                )

    def test_skip_set_is_valid(self, router):
        dec = router.decide(RoutingMode.BALANCED)
        # Every layer is either active or skipped (no overlap, no gaps)
        all_layers = set(range(router.num_layers))
        union = dec.active_layers | dec.skip_layers
        assert union == all_layers
        assert len(dec.active_layers & dec.skip_layers) == 0

    def test_custom_budget(self, router):
        dec = router.decide(RoutingMode.CONSERVATIVE, custom_budget=(1, 1))
        assert dec.num_skipped == 1
        assert dec.budget == (1, 1)

    def test_decide_pct(self, router):
        dec = router.decide_pct(reduction_target_pct=5.0)
        assert dec.num_skipped >= 1  # at least 1 layer for 5% of 22

    def test_routing_decision_to_dict(self, router):
        dec = router.decide(RoutingMode.BALANCED)
        d = dec.to_dict()
        assert "mode" in d
        assert "skip_layers" in d
        assert "compute_reduction_pct" in d

    def test_mode_from_str(self):
        assert RoutingMode.from_str("conservative") == RoutingMode.CONSERVATIVE
        assert RoutingMode.from_str("BALANCED") == RoutingMode.BALANCED
        assert RoutingMode.from_str("Aggressive") == RoutingMode.AGGRESSIVE
        with pytest.raises(ValueError):
            RoutingMode.from_str("invalid")


# ===================================================================
# QualityEvaluator
# ===================================================================


class TestQualityEvaluator:
    def test_identical_responses(self):
        qm = evaluate("hello world", "hello world", mode="test")
        assert qm.quality_score == 1.0
        assert qm.jaccard_similarity == 1.0
        assert qm.rouge_l_f1 == 1.0
        assert qm.quality_loss == 0.0

    def test_completely_different(self):
        qm = evaluate("hello world", "zzzzz bbbbb", mode="test")
        assert qm.quality_score == 0.0
        assert qm.jaccard_similarity == 0.0

    def test_partial_similarity(self):
        qm = evaluate("hello world foo", "hello bar foo", mode="test")
        assert 0.0 < qm.quality_score < 1.0
        assert qm.similarity_score == qm.jaccard_similarity

    def test_custom_fields(self):
        qm = evaluate(
            "a b c", "a b", query="test query", mode="balanced",
            latency_s=2.5, num_layers_skipped=2, compute_reduction_pct=9.09,
        )
        assert qm.query == "test query"
        assert qm.mode == "balanced"
        assert qm.latency_s == 2.5
        assert qm.num_layers_skipped == 2
        assert qm.compute_reduction_pct == 9.09

    def test_both_empty(self):
        qm = evaluate("", "", mode="test")
        assert qm.quality_score == 1.0

    def test_jaccard_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_jaccard_disjoint(self):
        assert jaccard_similarity("abc def", "ghi jkl") == 0.0

    def test_rouge_l_identical(self):
        assert rouge_l_f1("a b c", "a b c") == 1.0

    def test_quality_metrics_to_dict(self):
        qm = evaluate("base", "routed", mode="test")
        d = qm.to_dict()
        assert "quality_score" in d
        assert "jaccard_similarity" in d
        assert "quality_loss" in d


# ===================================================================
# Benchmark report generation
# ===================================================================


class TestReportGeneration:
    def _make_result(self) -> BenchmarkSuiteResult:
        metrics = [
            QualityMetrics(
                query="What is 2+2?", baseline_response="4",
                routed_response="5", mode="conservative",
                jaccard_similarity=0.8, rouge_l_f1=0.7,
                quality_score=0.75, similarity_score=0.8,
                quality_loss=0.25, latency_s=1.0,
                num_layers_skipped=1, compute_reduction_pct=4.55,
            ),
            QualityMetrics(
                query="Explain REST", baseline_response="REST is...",
                routed_response="REST...", mode="conservative",
                jaccard_similarity=0.6, rouge_l_f1=0.5,
                quality_score=0.55, similarity_score=0.6,
                quality_loss=0.45, latency_s=1.2,
                num_layers_skipped=1, compute_reduction_pct=4.55,
            ),
        ]
        modes = {
            "conservative": BenchmarkResult(
                mode="conservative", all_metrics=metrics,
            ),
            "balanced": BenchmarkResult(
                mode="balanced", all_metrics=[
                    QualityMetrics(query="test", baseline_response="a",
                                   routed_response="b", mode="balanced",
                                   quality_score=0.5, latency_s=0.9,
                                   num_layers_skipped=2, compute_reduction_pct=9.09),
                ],
            ),
            "aggressive": BenchmarkResult(
                mode="aggressive", all_metrics=[
                    QualityMetrics(query="test", baseline_response="a",
                                   routed_response="c", mode="aggressive",
                                   quality_score=0.4, latency_s=0.8,
                                   num_layers_skipped=3, compute_reduction_pct=13.64),
                ],
            ),
        }
        from datetime import datetime
        return BenchmarkSuiteResult(
            baseline_latency_avg=1.5,
            modes=modes,
            config={"mode": "simulate", "max_tokens": 16},
            total_time_s=5.0,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def test_generates_markdown(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(result, tmpdir)
            assert os.path.isfile(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "# Importance-Based Layer Router" in content
            assert "conservative" in content
            assert "balanced" in content
            assert "aggressive" in content
            assert "Summary" in content
            assert "Per-Query Details" in content

    def test_generates_json(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(result, tmpdir)
            json_path = os.path.join(tmpdir, "importance_router_results.json")
            assert os.path.isfile(json_path)
            with open(json_path, "r") as f:
                data = json.load(f)
            assert "modes" in data
            assert "conservative" in data["modes"]
            assert "per_query" in data["modes"]["conservative"]

    def test_benchmark_result_properties(self):
        qms = [
            QualityMetrics(
                query="q1", baseline_response="a", routed_response="b",
                mode="test", jaccard_similarity=0.8, rouge_l_f1=0.6,
                quality_score=0.7, similarity_score=0.8, quality_loss=0.3,
                latency_s=1.0, num_layers_skipped=1, compute_reduction_pct=4.55,
            ),
            QualityMetrics(
                query="q2", baseline_response="c", routed_response="d",
                mode="test", jaccard_similarity=0.4, rouge_l_f1=0.4,
                quality_score=0.4, similarity_score=0.4, quality_loss=0.6,
                latency_s=2.0, num_layers_skipped=2, compute_reduction_pct=9.09,
            ),
        ]
        br = BenchmarkResult(mode="test", all_metrics=qms)
        assert br.avg_quality_score == pytest.approx(0.55)
        assert br.avg_jaccard == pytest.approx(0.6)
        assert br.avg_latency_s == 1.5
        assert br.avg_layers_skipped == 1.5
        assert br.avg_reduction_pct == pytest.approx(6.82)

    def test_benchmark_result_empty(self):
        br = BenchmarkResult(mode="empty")
        assert br.avg_quality_score == 0.0
        assert br.avg_latency_s == 0.0


# ===================================================================
# Simulated integration test
# ===================================================================


@pytest.mark.slow
class TestSimulatedBenchmark:
    def test_full_simulated_run(self):
        from benchmark import SimulatedBenchmark

        engine = SimulatedBenchmark(max_tokens=16)
        queries = [
            {"query": "What is 2+2?", "type": "simple"},
            {"query": "Explain REST API", "type": "medium"},
        ]
        result = engine.run(queries)
        assert len(result.modes) == 3
        for mode_name in ("conservative", "balanced", "aggressive"):
            assert mode_name in result.modes
            mr = result.modes[mode_name]
            assert len(mr.all_metrics) == 2
            assert mr.avg_quality_score > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(result, tmpdir)
            assert os.path.isfile(path)

    def test_correct_skip_properties(self):
        from benchmark import SimulatedBenchmark

        engine = SimulatedBenchmark()
        queries = [{"query": "test", "type": "simple"}]
        result = engine.run(queries)
        cons = result.modes["conservative"]
        bal = result.modes["balanced"]
        agg = result.modes["aggressive"]
        assert cons.avg_layers_skipped <= bal.avg_layers_skipped <= agg.avg_layers_skipped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
