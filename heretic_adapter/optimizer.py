"""Directional ablation: fit the refusal direction, remove it, iterate.

Upstream Heretic performs directional ablation ("abliteration") and optimizes
the ablation kernel with a TPE sampler (Optuna) co-minimizing refusals and KL
divergence. This module implements the ablation half deterministically: fit by
difference-of-means, remove by projection, refit against the mutated target, stop
when the template's target refusal rate is met or passes are exhausted.

Route to the upstream optimizer with ``NEON_FORGE_BACKEND=real`` and
``HereticRunner.run_cli``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rea_engine.dataset_gen import BehaviorDataset, ProbeExample
from rea_engine.extractor import TargetModel, _dot, _norm, _unit

from .config import AblationConfig, load_config


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
    if not vectors:
        return 0.0
    return sum(_norm([x - c for x, c in zip(v, centroid)]) for v in vectors) / len(vectors)


def probes_from(dataset: BehaviorDataset) -> List[ProbeExample]:
    """Rebuild the probe set from a recorded dataset."""
    return [
        ProbeExample(
            probe_id=r.probe_id,
            prompt=r.prompt,
            category=r.category,
            harm_score=r.harm_score,
        )
        for r in dataset.records
    ]


@dataclass
class FitResult:
    direction: List[float]
    separation: float
    support: int
    cosine_to_previous: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "separation": round(self.separation, 4),
            "support": self.support,
            "cosine_to_previous": round(self.cosine_to_previous, 4),
            "direction_norm": round(_norm(self.direction), 6),
        }


@dataclass
class AblationResult:
    template: str
    baseline_refusal_rate: float
    final_refusal_rate: float
    reduction: float
    passes_used: int
    directions_applied: int
    converged: bool
    history: List[Dict[str, object]] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def refusal_reduction(self) -> float:
        return self.reduction

    def as_dict(self) -> Dict[str, object]:
        return {
            "template": self.template,
            "baseline_refusal_rate": round(self.baseline_refusal_rate, 4),
            "final_refusal_rate": round(self.final_refusal_rate, 4),
            "reduction": round(self.reduction, 4),
            "passes_used": self.passes_used,
            "directions_applied": self.directions_applied,
            "converged": self.converged,
            "history": self.history,
            "config": self.config,
            "notes": self.notes,
        }


class DirectionalAblator:
    """Difference-of-means directional ablation with iterative refits."""

    def __init__(self, config: Optional[AblationConfig] = None):
        self.config = config or load_config("default")

    def fit(self, dataset: BehaviorDataset) -> FitResult:
        triggers = [r.activation for r in dataset.by_category("refusal_trigger")]
        neutrals = [r.activation for r in dataset.by_category("neutral")]
        if not triggers or not neutrals:
            raise ValueError("fit requires both 'refusal_trigger' and 'neutral' records")

        trigger_centroid = _centroid(triggers)
        neutral_centroid = _centroid(neutrals)
        diff = [a - b for a, b in zip(trigger_centroid, neutral_centroid)]

        within = (
            _spread(triggers, trigger_centroid) + _spread(neutrals, neutral_centroid)
        ) / 2.0
        separation = _norm(diff) / within if within > 1e-9 else 0.0
        return FitResult(direction=_unit(diff), separation=separation, support=len(triggers) + len(neutrals))

    @staticmethod
    def observe(model: TargetModel, probes: List[ProbeExample]) -> BehaviorDataset:
        return BehaviorDataset(dim=model.dim, records=[model.respond(p) for p in probes])

    def run(
        self,
        model: TargetModel,
        dataset: BehaviorDataset,
        probes: Optional[List[ProbeExample]] = None,
    ) -> AblationResult:
        probe_list = list(probes) if probes else probes_from(dataset)
        baseline = dataset.refusal_rate()
        history: List[Dict[str, object]] = []
        notes: List[str] = []
        previous: Optional[List[float]] = None
        rate = baseline
        converged = False
        passes_used = 0

        for pass_index in range(1, self.config.passes + 1):
            fit = self.fit(self.observe(model, probe_list))

            if previous is not None:
                fit.cosine_to_previous = _dot(_unit(fit.direction), previous)
                if fit.cosine_to_previous < 0.0:
                    notes.append(
                        f"pass {pass_index}: direction flipped vs previous pass "
                        f"(cosine {fit.cosine_to_previous:.3f}) - signal may be exhausted"
                    )
            previous = _unit(fit.direction)

            model.suppress(fit.direction, self.config.strength)
            passes_used = pass_index
            rate = self.observe(model, probe_list).refusal_rate()
            history.append({"pass": pass_index, "refusal_rate": round(rate, 4), **fit.as_dict()})

            if rate <= self.config.target_refusal_rate:
                converged = True
                break
            if fit.separation < 0.5:
                notes.append(
                    f"pass {pass_index}: separability collapsed ({fit.separation:.2f}) - "
                    "residual signal insufficient to refit"
                )
                break

        return AblationResult(
            template=self.config.name,
            baseline_refusal_rate=baseline,
            final_refusal_rate=rate,
            reduction=(baseline - rate) / baseline if baseline > 1e-9 else 0.0,
            passes_used=passes_used,
            directions_applied=model.suppressed_count,
            converged=converged,
            history=history,
            config=self.config.as_dict(),
            notes=notes,
        )
