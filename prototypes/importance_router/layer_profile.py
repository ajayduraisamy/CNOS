"""layer_profile — loads and queries layer importance scores from the ablation study."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "layer_importance", "output",
        "layer_importance.json",
    )
)


@dataclass
class LayerScore:
    """Score and classification for a single layer.

    Attributes:
        layer: Layer index.
        avg_impact_score: Mean impact score across queries (0-1).
        classification: ``"high"``, ``"medium"``, or ``"low"``.
        num_queries: Number of queries that contributed.
    """
    layer: int
    avg_impact_score: float
    classification: str
    num_queries: int


class LayerProfile:
    """Loads and provides access to per-layer importance data.

    The profile is loaded from the JSON produced by the ablation study
    (``layer_importance.json``).  It classifies each layer and provides
    safe skip candidates — only medium/low impact layers are eligible.

    Args:
        filepath: Path to the JSON profile.  Defaults to the output of
            the layer importance study.
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = filepath or _DEFAULT_PROFILE
        self._scores: Dict[int, LayerScore] = {}
        self.num_layers: int = 0
        self.model_key: str = ""
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                f"Layer importance profile not found: {self.filepath}"
            )
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.model_key = data.get("model_key", "unknown")
        self.num_layers = data.get("num_layers", 0)

        for entry in data.get("per_layer", []):
            ls = LayerScore(
                layer=entry["layer"],
                avg_impact_score=entry["avg_impact_score"],
                classification=entry["classification"],
                num_queries=entry.get("num_queries", 0),
            )
            self._scores[ls.layer] = ls

        if len(self._scores) != self.num_layers:
            logger.warning(
                "Profile has %d/%d layers",
                len(self._scores), self.num_layers,
            )
        logger.info(
            "LayerProfile loaded  %s  %d layers",
            self.model_key, self.num_layers,
        )

    def get_score(self, layer: int) -> Optional[LayerScore]:
        """Return the score for a single layer, or *None*."""
        return self._scores.get(layer)

    def get_classification(self, layer: int) -> str:
        """Return ``"high"``, ``"medium"``, or ``"low"`` for a layer."""
        ls = self._scores.get(layer)
        return ls.classification if ls else "unknown"

    @property
    def critical_layers(self) -> List[int]:
        """Layers classified as high impact — never skip."""
        return sorted(
            l for l, s in self._scores.items() if s.classification == "high"
        )

    @property
    def skip_candidates(self) -> List[Tuple[int, float]]:
        """Sorted (layer, impact_score) eligible for skipping, ascending.

        Only medium and low impact layers are eligible.
        """
        candidates = [
            (l, s.avg_impact_score)
            for l, s in self._scores.items()
            if s.classification in ("medium", "low")
        ]
        candidates.sort(key=lambda x: x[1])
        return candidates

    def max_skippable(self) -> int:
        """Maximum number of layers that can be safely skipped."""
        return len(self.skip_candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "num_layers": self.num_layers,
            "skip_candidates": [
                {"layer": l, "impact_score": s}
                for l, s in self.skip_candidates
            ],
            "critical_layers": self.critical_layers,
            "max_skippable": self.max_skippable(),
        }
