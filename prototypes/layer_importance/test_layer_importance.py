"""Tests for the layer_importance module.

Covers:
  * quality_metrics — comparison functions, classification
  * report_generator — JSON, CSV, Markdown output
  * benchmark SimulatedAblationEngine — full study lifecycle

Run with:
    python -m pytest prototypes/layer_importance/test_layer_importance.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure quality_metrics is importable
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest

from quality_metrics import (
    jaccard_similarity,
    rouge_l_f1,
    compare_responses,
    classify_importance,
    ComparisonResult,
)
from report_generator import (
    write_json,
    write_csv,
    generate_markdown_report,
    generate_all_reports,
)
from layer_ablation import (
    AblationStudyResult,
    LayerImportanceResult,
)


# ===================================================================
# quality_metrics
# ===================================================================


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert jaccard_similarity("hello world", "foo bar baz") == 0.0

    def test_partial_overlap(self):
        # tokens: {hello, world} ∩ {hello, there} = {hello}
        # union = {hello, world, there} = 3
        # overlap = 1/3 ≈ 0.333
        sim = jaccard_similarity("hello world", "hello there")
        assert abs(sim - 1.0 / 3.0) < 1e-6

    def test_both_empty(self):
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty(self):
        assert jaccard_similarity("hello", "") == 0.0

    def test_case_insensitive(self):
        sim = jaccard_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_punctuation_treated_as_tokens(self):
        sim = jaccard_similarity("hello, world!", "hello world")
        # "hello," and "hello" are distinct tokens; "world!" and "world" distinct
        assert sim == 0.0


class TestRougeLF1:
    def test_identical(self):
        assert rouge_l_f1("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert rouge_l_f1("hello world", "foo bar") == 0.0

    def test_partial(self):
        f1 = rouge_l_f1("hello world foo", "hello bar foo")
        assert 0.0 < f1 < 1.0

    def test_both_empty(self):
        assert rouge_l_f1("", "") == 1.0


class TestCompareResponses:
    def test_identical_responses(self):
        comp = compare_responses("hello world", "hello world", layer=5)
        assert comp.impact_score == 0.0
        assert comp.jaccard_similarity == 1.0
        assert comp.rouge_l_f1 == 1.0
        assert comp.ablated_layer == 5

    def test_completely_different(self):
        comp = compare_responses("hello world", "zzzzz bbbbb", layer=10)
        assert comp.impact_score > 0.9
        assert comp.jaccard_similarity == 0.0

    def test_custom_fields(self):
        comp = compare_responses("a b c", "a b", layer=3, latency_s=1.5, query="test")
        assert comp.query == "test"
        assert comp.latency_s == 1.5
        assert comp.ablated_layer == 3


class TestClassifyImportance:
    def test_high(self):
        assert classify_importance(0.50) == "high"
        assert classify_importance(0.30) == "high"

    def test_medium(self):
        assert classify_importance(0.20) == "medium"
        assert classify_importance(0.10) == "medium"

    def test_low(self):
        assert classify_importance(0.05) == "low"
        assert classify_importance(0.0) == "low"


# ===================================================================
# Report generator
# ===================================================================


def _make_result() -> AblationStudyResult:
    layers = [
        LayerImportanceResult(
            layer=i,
            impact_scores=[0.1, 0.2],
            avg_impact_score=0.15,
            classification="medium" if i % 2 == 0 else "low",
            per_query=[
                ComparisonResult(
                    query="What is 2+2?", baseline_response="4",
                    ablated_response="5", ablated_layer=i,
                    jaccard_similarity=0.8, rouge_l_f1=0.7,
                    length_ratio=1.0, latency_s=0.5, impact_score=0.15,
                ),
            ],
        )
        for i in range(22)
    ]
    from datetime import datetime
    return AblationStudyResult(
        model_key="test-model",
        num_layers=22,
        per_layer=layers,
        config={"mode": "simulate"},
        total_time_s=42.0,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


class TestWriteJson:
    def test_writes_json(self):
        result = _make_result()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_json(result, path)
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            assert data["model_key"] == "test-model"
            assert data["num_layers"] == 22
            assert len(data["per_layer"]) == 22
            assert "layer" in data["per_layer"][0]
        finally:
            os.unlink(path)


class TestWriteCsv:
    def test_writes_csv(self):
        result = _make_result()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write_csv(result, path)
            with open(path, "r", encoding="utf-8") as fp:
                lines = fp.readlines()
            assert len(lines) == 23  # header + 22 layers
            assert "layer" in lines[0]
            assert lines[1].startswith("0,")
        finally:
            os.unlink(path)


class TestGenerateMarkdown:
    def test_writes_markdown(self):
        result = _make_result()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            generate_markdown_report(result, path)
            with open(path, "r", encoding="utf-8") as fp:
                content = fp.read()
            assert "# Layer Importance Study" in content
            assert "test-model" in content
            assert "High Impact" in content
            assert "Medium Impact" in content
            assert "Low Impact" in content
        finally:
            os.unlink(path)


class TestGenerateAllReports:
    def test_generates_all_formats(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = generate_all_reports(result, output_dir=tmpdir)
            assert "json" in files
            assert "csv" in files
            assert "md" in files
            for fpath in files.values():
                assert os.path.isfile(fpath)


# ===================================================================
# Simulated benchmark integration test
# ===================================================================


@pytest.mark.slow
class TestSimulatedBenchmark:
    """Integration test using the SimulatedAblationEngine.

    Marked ``slow`` because it exercises the full study pipeline.
    """

    def test_simulated_study_produces_valid_result(self):
        from benchmark import SimulatedAblationEngine

        engine = SimulatedAblationEngine(max_tokens=8)
        try:
            # Use only 2 queries for speed
            queries = [
                {"query": "What is 2+2?", "type": "simple"},
                {"query": "Explain REST API", "type": "medium"},
            ]
            result = engine.run_study(queries=queries, config={"mode": "simulate"})
            assert result.model_key is not None
            assert result.num_layers == 22
            assert len(result.per_layer) == 22
            assert result.total_time_s >= 0
            # Scores should be monotonic (approximately)
            scores = [l.avg_impact_score for l in result.per_layer]
            assert all(0.0 <= s <= 1.0 for s in scores)
            assert result.high_impact_layers is not None
            assert result.medium_impact_layers is not None
            assert result.low_impact_layers is not None
            assert (
                len(result.high_impact_layers)
                + len(result.medium_impact_layers)
                + len(result.low_impact_layers)
                == 22
            )
        finally:
            engine.cleanup()

    def test_simulated_full_reports(self):
        from benchmark import SimulatedAblationEngine

        engine = SimulatedAblationEngine(max_tokens=8)
        try:
            queries = [{"query": "Hi", "type": "simple"}]
            result = engine.run_study(queries=queries)

            with tempfile.TemporaryDirectory() as tmpdir:
                files = generate_all_reports(result, output_dir=tmpdir)
                for path in files.values():
                    assert os.path.isfile(path), f"Missing: {path}"
                    assert os.path.getsize(path) > 0, f"Empty: {path}"

                # Validate JSON structure
                with open(files["json"], "r") as f:
                    data = json.load(f)
                assert "per_layer" in data
                assert data["num_layers"] == 22
        finally:
            engine.cleanup()


class TestAblationStudyResultProperties:
    def test_classification_properties(self):
        layers = [
            LayerImportanceResult(
                layer=0, impact_scores=[0.8], avg_impact_score=0.8,
                classification="high",
            ),
            LayerImportanceResult(
                layer=1, impact_scores=[0.2], avg_impact_score=0.2,
                classification="medium",
            ),
            LayerImportanceResult(
                layer=2, impact_scores=[0.05], avg_impact_score=0.05,
                classification="low",
            ),
        ]
        result = AblationStudyResult(
            model_key="test", num_layers=3, per_layer=layers,
        )
        assert result.high_impact_layers == [0]
        assert result.medium_impact_layers == [1]
        assert result.low_impact_layers == [2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
