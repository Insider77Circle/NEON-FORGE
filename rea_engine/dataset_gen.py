"""Synthetic probe synthesis for the REA extraction stage.

REA (https://github.com/morluto/rea) is an agent-driven reverse-engineering
toolkit. Neon Forge uses it as the *probe* layer: drive a probe set at the
target, record activations and responses, hand the evidence downstream.

Probes here are synthetic identifiers, not natural-language payloads, so the
pipeline is reproducible without a GPU, a model checkpoint, or a network call.
A real backend swaps the probe text at the TargetModel seam in
``rea_engine/extractor.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

DEFAULT_DIM = 32
CATEGORIES = ("neutral", "borderline", "refusal_trigger")

# Harm-band per category. The band, not the text, drives the simulated target's
# response, which is what makes the ablation measurable.
_HARM_BANDS = {
    "neutral": (0.02, 0.18),
    "borderline": (0.40, 0.60),
    "refusal_trigger": (0.70, 0.98),
}

# Synthetic probe families. Deliberately abstract: these are probe handles for a
# local target, mirroring how REA tags a probe set before dispatch.
PROBE_TEMPLATES = {
    "neutral": ("factual-recall", "style-transfer", "summarization", "arithmetic"),
    "borderline": ("edge-case-policy", "ambiguous-intent", "dual-use-context"),
    "refusal_trigger": ("policy-boundary", "safety-trigger", "restriction-surface"),
}


@dataclass(frozen=True)
class ProbeExample:
    """A single probe dispatched at the target."""

    probe_id: str
    prompt: str
    category: str
    harm_score: float


@dataclass
class ActivationRecord:
    """Target response plus the residual-stream activation captured for it."""

    probe_id: str
    category: str
    harm_score: float
    activation: List[float]
    response: str
    refused: bool
    capability_signal: float
    prompt: str = ""


@dataclass
class BehaviorDataset:
    """Activation evidence for a whole probe run."""

    dim: int
    records: List[ActivationRecord]
    manifest: Dict[str, object] = field(default_factory=dict)

    def by_category(self, category: str) -> List[ActivationRecord]:
        return [r for r in self.records if r.category == category]

    def refusal_rate(self) -> float:
        triggers = self.by_category("refusal_trigger")
        if not triggers:
            return 0.0
        return sum(1 for r in triggers if r.refused) / len(triggers)

    def compliance_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if not r.refused) / len(self.records)

    def activations(self) -> List[List[float]]:
        return [r.activation for r in self.records]

    def __len__(self) -> int:
        return len(self.records)


class SyntheticDatasetGenerator:
    """Deterministic probe generator.

    Same ``seed`` + ``size`` always yields the same probe set, which is what lets
    before/after comparisons in the monitor be honest.
    """

    def __init__(self, size: int = 48, dim: int = DEFAULT_DIM, seed: int = 1337):
        if size < len(CATEGORIES):
            raise ValueError(f"size must be >= {len(CATEGORIES)} (one probe per category)")
        self.size = size
        self.dim = dim
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self) -> List[ProbeExample]:
        self.rng = random.Random(self.seed)
        per_category = max(1, self.size // len(CATEGORIES))
        probes: List[ProbeExample] = []
        idx = 0
        for category in CATEGORIES:
            lo, hi = _HARM_BANDS[category]
            families = PROBE_TEMPLATES[category]
            for slot in range(per_category):
                idx += 1
                family = families[slot % len(families)]
                probes.append(
                    ProbeExample(
                        probe_id=f"p{idx:03d}",
                        prompt=f"{category}/{family}/probe-{idx:03d}",
                        category=category,
                        harm_score=round(self.rng.uniform(lo, hi), 4),
                    )
                )
        return probes
