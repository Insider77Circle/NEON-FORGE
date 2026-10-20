"""Telemetry aggregation, drift scoring, capability preservation.

All numbers here are computed from recorded evidence. Nothing is estimated: if a
probe is missing from either side of a comparison, it is dropped from that
metric and the count is reported alongside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rea_engine.dataset_gen import CATEGORIES, BehaviorDataset

from .scheduler import ParallelScheduler


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(_dot(v, v))


def _cosine(a: List[float], b: List[float]) -> float:
    denom = _norm(a) * _norm(b)
    return _dot(a, b) / denom if denom > 1e-12 else 0.0


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class MetricSample:
    name: str
    value: float
    step: int = 0
    tags: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "value": self.value, "step": self.step, "tags": self.tags}


class MetricsAggregator:
    """Aggregates per-category telemetry and before/after drift."""

    def __init__(self, scheduler: Optional[ParallelScheduler] = None, max_workers: int = 3):
        self.scheduler = scheduler or ParallelScheduler(max_workers)
        self.samples: List[MetricSample] = []
        self.last_report: Dict[str, object] = {}

    def record(self, name: str, value: float, step: int = 0, **tags: str) -> MetricSample:
        sample = MetricSample(name=name, value=float(value), step=step, tags=tags)
        self.samples.append(sample)
        return sample

    def latest(self, name: str) -> Optional[MetricSample]:
        for sample in reversed(self.samples):
            if sample.name == name:
                return sample
        return None

    def category_stats(self, dataset: BehaviorDataset) -> Dict[str, Dict[str, float]]:
        """Per-category refusal/compliance counts for one dataset."""
        stats: Dict[str, Dict[str, float]] = {}
        for category in CATEGORIES:
            records = dataset.by_category(category)
            total = len(records)
            refused = sum(1 for r in records if r.refused)
            stats[category] = {
                "probes": total,
                "refused": refused,
                "refusal_rate": refused / total if total else 0.0,
                "compliance_rate": (total - refused) / total if total else 0.0,
            }
        return stats

    def _category_delta(self, category: str) -> Dict[str, float]:
        before = {r.probe_id: r for r in self._before.by_category(category)}
        after = {r.probe_id: r for r in self._after.by_category(category)}
        shared = [pid for pid in before if pid in after]
        if not shared:
            return {"probes": 0, "drift": 0.0, "refusal_rate_before": 0.0,
                    "refusal_rate_after": 0.0, "refusal_drop": 0.0}

        drift_values = [1.0 - _cosine(before[p].activation, after[p].activation) for p in shared]
        rate_before = sum(1 for p in shared if before[p].refused) / len(shared)
        rate_after = sum(1 for p in shared if after[p].refused) / len(shared)
        return {
            "probes": len(shared),
            "drift": sum(drift_values) / len(drift_values),
            "refusal_rate_before": rate_before,
            "refusal_rate_after": rate_after,
            "refusal_drop": rate_before - rate_after,
        }

    def drift(self, before: BehaviorDataset, after: BehaviorDataset) -> float:
        """Mean cosine distance across probes present in both runs."""
        after_map = {r.probe_id: r for r in after.records}
        distances = [
           1.0 - _cosine(record.activation, after_map[record.probe_id].activation)
            for record in before.records
            if record.probe_id in after_map
        ]
        return sum(distances) / len(distances) if distances else 0.0

    def capability_preservation(self, before: BehaviorDataset, after: BehaviorDataset) -> float:
        """Mean retained capability signal on probes that were compliant before.

        Probes that were already refusing express little capability signal, so
        including them would flatter the score. They are excluded, and the
        denominator is reported by the caller via probe counts.
        """
        after_map = {r.probe_id: r for r in after.records}
        ratios: List[float] = []
        for record in before.records:
            if record.refused:
                continue
            other = after_map.get(record.probe_id)
            if other is None or abs(record.capability_signal) < 1e-9:
                continue
            ratios.append(_clip(other.capability_signal / record.capability_signal))
        return sum(ratios) / len(ratios) if ratios else 1.0

    def compute(self, before: BehaviorDataset, after: BehaviorDataset) -> Dict[str, object]:
        """Full before/after report, per-category stats computed in parallel."""
        self._before, self._after = before, after

        self.scheduler.reset()
        for category in CATEGORIES:
            self.scheduler.add(f"category::{category}", self._category_delta, category)
        self.scheduler.run()

        per_category: Dict[str, object] = {}
        for category in CATEGORIES:
            task = self.scheduler.tasks[f"category::{category}"]
            if task.status == "done" and task.result is not None:
                per_category[category] = {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in task.result.items()
                }
            else:
                per_category[category] = {"error": task.error or task.status}

        rate_before = before.refusal_rate()
        rate_after = after.refusal_rate()
        preservation = self.capability_preservation(before, after)
        drift = self.drift(before, after)

        report: Dict[str, object] = {
            "probes": {"before": len(before), "after": len(after)},
            "refusal_rate": {
                "before": round(rate_before, 4),
                "after": round(rate_after, 4),
                "delta": round(rate_before - rate_after, 4),
                "relative_reduction": round((rate_before - rate_after) / rate_before, 4)
                if rate_before > 1e-9
                else 0.0,
            },
            "compliance_rate": {
                "before": round(before.compliance_rate(), 4),
                "after": round(after.compliance_rate(), 4),
            },
            "capability_preservation": round(preservation, 4),
            "drift": round(drift, 4),
            "per_category": per_category,
            "scheduler": self.scheduler.summary(),
            "samples": len(self.samples),
        }

        for name in ("capability_preservation", "drift"):
            self.record(name, report[name], step=1)  # type: ignore[arg-type]
        self.record("refusal_rate", rate_after, step=1, phase="after")
        self.record("refusal_rate", rate_before, step=0, phase="before")

        self.last_report = report
        return report
