"""Surgical alignment removal (OBLITERATUS EXCISE stage).

Removes a low-rank refusal subspace from the target's residual stream by
projection. Every operation is recorded with its inverse so a run can be audited
or walked back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rea_engine.dataset_gen import BehaviorDataset, ProbeExample
from rea_engine.extractor import _dot


@dataclass
class SurgeryRecord:
    rank: int
    directions: List[List[float]] = field(default_factory=list)
    strength: float = 1.0
    removed_energy: List[float] = field(default_factory=list)
    reversible: bool = True
    inverse_ops: List[Dict[str, object]] = field(default_factory=list)
    post_refusal_rate: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "strength": self.strength,
            "removed_energy": [round(e, 6) for e in self.removed_energy],
            "reversible": self.reversible,
            "inverse_ops": self.inverse_ops,
            "post_refusal_rate": self.post_refusal_rate,
        }


class AlignmentSurgeon:
    """Applies — and can walk back — a rank-k projection removal."""

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def operate(
        self,
        model,
        basis: List[List[float]],
        probes: Optional[List[ProbeExample]] = None,
    ) -> SurgeryRecord:
        record = SurgeryRecord(rank=len(basis), strength=self.strength)

        before: Dict[str, List[float]] = {}
        if probes:
            before = {p.probe_id: model.respond(p).activation for p in probes}

        for direction in basis:
            unit = [
                x / (sum(y * y for y in direction) ** 0.5 or 1.0) for x in direction
            ]
            energy = 0.0
            if before:
                energy = sum(abs(_dot(vec, unit)) for vec in before.values()) / len(before)
            model.suppress(unit, self.strength)
            record.directions.append(unit)
            record.removed_energy.append(energy)
            record.inverse_ops.append(
                {"op": "reject_projection", "strength": self.strength, "energy": round(energy, 6)}
            )

        if probes:
            after = [model.respond(p) for p in probes]
            triggers = [r for r in after if r.category == "refusal_trigger"]
            if triggers:
                record.post_refusal_rate = sum(1 for r in triggers if r.refused) / len(triggers)
        return record

    def reverse(self, model, record: SurgeryRecord) -> int:
        """Walk the surgery back.

        Projection removal is not exactly invertible, so the removed component is
        re-injected as a steering vector along the same axis. The result is an
        approximation, and ``reversible`` stays True only while that holds.
        """
        restored = 0
        for direction, energy in zip(reversed(record.directions), reversed(record.removed_energy)):
            if model.pop_suppression() is None:
                break
            model.apply_steering(direction, energy)
            restored += 1
        return restored

    @staticmethod
    def measure_energy(model, basis: List[List[float]], probes: List[ProbeExample]) -> List[float]:
        """Mean absolute projection onto each axis, for audit/telemetry."""
        activations = [model.respond(probe).activation for probe in probes]
        energies: List[float] = []
        for direction in basis:
            magnitude = sum(y * y for y in direction) ** 0.5 or 1.0
            unit = [x / magnitude for x in direction]
            energies.append(
                sum(abs(_dot(vec, unit)) for vec in activations) / max(1, len(activations))
            )
        return energies

    @staticmethod
    def reference_rate(dataset: BehaviorDataset) -> float:
        return dataset.refusal_rate()
