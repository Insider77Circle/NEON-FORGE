"""Reconstructed-logic validation for the REA stage.

Two jobs:

* ``validate``  - is the extracted dataset usable (coverage, finiteness,
  class separability)? Gate before anything is ablated.
* ``validate_reconstruction`` - did the reconstruction reproduce the target's
  behavior (per-probe cosine fidelity)?
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .dataset_gen import CATEGORIES, BehaviorDataset


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(_dot(v, v))


def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    total = [0.0] * dim
    for vec in vectors:
        for i, x in enumerate(vec):
            total[i] += x
    return [x / len(vectors) for x in total]


def _spread(vectors: List[List[float]], centroid: List[float]) -> float:
    """Mean Euclidean distance to the centroid."""
    if not vectors:
        return 0.0
    return sum(_norm([x - c for x, c in zip(v, centroid)]) for v in vectors) / len(vectors)


@dataclass
class ValidationReport:
    dataset_size: int
    dim: int
    category_counts: Dict[str, int] = field(default_factory=dict)
    dims_consistent: bool = False
    finite_values: bool = False
    category_coverage: bool = False
    centroid_separation: float = 0.0
    score: float = 0.0
    ok: bool = False
    findings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class LogicValidator:
    """Structural + separability checks on extracted behavior evidence."""

    def __init__(self, min_per_category: int = 8, min_separation: float = 2.0):
        self.min_per_category = min_per_category
        self.min_separation = min_separation

    def validate(self, dataset: BehaviorDataset) -> ValidationReport:
        findings: List[str] = []

        counts = {category: len(dataset.by_category(category)) for category in CATEGORIES}
        dims_consistent = all(len(r.activation) == dataset.dim for r in dataset.records)
        if not dims_consistent:
            findings.append("activation vectors disagree on dimensionality")

        finite_values = all(
            math.isfinite(x) for r in dataset.records for x in r.activation
        )
        if not finite_values:
            findings.append("non-finite value detected in activation tensor")

        category_coverage = all(c >= self.min_per_category for c in counts.values())
        for category, count in counts.items():
            if count < self.min_per_category:
                findings.append(
                    f"category '{category}' under-sampled: {count} < {self.min_per_category}"
                )

        separation = 0.0
        triggers = dataset.by_category("refusal_trigger")
        neutrals = dataset.by_category("neutral")
        if triggers and neutrals:
            trig_centroid = _centroid([r.activation for r in triggers])
            neut_centroid = _centroid([r.activation for r in neutrals])
            between = _norm([a - b for a, b in zip(trig_centroid, neut_centroid)])
            within = (
                _spread([r.activation for r in triggers], trig_centroid)
                + _spread([r.activation for r in neutrals], neut_centroid)
            ) / 2.0
            separation = between / within if within > 1e-9 else 0.0

        checks = [dims_consistent, finite_values, category_coverage, separation >= self.min_separation]
        score = sum(1 for c in checks if c) / len(checks)

        if separation < self.min_separation:
            findings.append(
                f"refusal direction weak: separation {separation:.2f} < {self.min_separation}"
            )

        return ValidationReport(
            dataset_size=len(dataset),
            dim=dataset.dim,
            category_counts=counts,
            dims_consistent=dims_consistent,
            finite_values=finite_values,
            category_coverage=category_coverage,
            centroid_separation=round(separation, 4),
            score=round(score, 4),
            ok=all(checks),
            findings=findings,
        )

    def validate_reconstruction(
        self, original: BehaviorDataset, reconstructed: BehaviorDataset
    ) -> Dict[str, object]:
        """Per-probe cosine fidelity between the target and its reconstruction."""
        recon = {r.probe_id: r.activation for r in reconstructed.records}
        scores: List[float] = []
        for record in original.records:
            other = recon.get(record.probe_id)
            if other is None:
                continue
            denom = _norm(record.activation) * _norm(other)
            if denom < 1e-9:
                continue
            scores.append(_dot(record.activation, other) / denom)

        if not scores:
            return {"fidelity": 0.0, "probes_compared": 0, "ok": False,
                    "finding": "no overlapping probes between original and reconstruction"}

        fidelity = sum(scores) / len(scores)
        return {
            "fidelity": round(fidelity, 4),
            "probes_compared": len(scores),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "ok": fidelity >= 0.95,
        }
