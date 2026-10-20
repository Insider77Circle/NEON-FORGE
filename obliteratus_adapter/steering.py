"""Reversible steering vectors.

A steering vector is a re-injectable axis: apply it to nudge behavior, remove it
to return the target to its post-surgery state. Reversibility is not asserted,
it is measured - :meth:`SteeringVectorFactory.verify` re-probes the target and
compares against a reference recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rea_engine.dataset_gen import BehaviorDataset, ProbeExample
from rea_engine.extractor import _norm, _unit


@dataclass
class SteeringVector:
    axis: int
    vector: List[float]
    gain: float
    coefficient: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "axis": self.axis,
            "gain": round(self.gain, 6),
            "coefficient": round(self.coefficient, 6),
            "norm": round(_norm(self.vector), 6),
        }


class SteeringVectorFactory:
    """Generates, applies, and removes reversible steering vectors."""

    def __init__(self, gain: float = 0.15):
        self.gain = gain

    def generate(self, basis: List[List[float]], gain: Optional[float] = None) -> List[SteeringVector]:
        """One steering vector per geometry axis, scaled by eigenvalue rank."""
        gain = self.gain if gain is None else gain
        vectors: List[SteeringVector] = []
        for axis, direction in enumerate(basis):
            decay = 1.0 / (axis + 1)
            vectors.append(
                SteeringVector(
                    axis=axis,
                    vector=_unit(direction),
                    gain=gain,
                    coefficient=round(gain * decay, 6),
                )
            )
        return vectors

    def apply(self, model, vectors: List[SteeringVector], scale: float = 1.0) -> int:
        for sv in vectors:
            model.apply_steering(sv.vector, sv.coefficient * scale)
        return len(vectors)

    def remove(self, model, vectors: List[SteeringVector]) -> int:
        """Clear the steering layer.

        The factory is the only writer to that layer inside Neon Forge, so a
        clear is an exact inverse of :meth:`apply`.
        """
        if not vectors:
            return 0
        return model.clear_steering()

    @staticmethod
    def verify(
        model, probes: List[ProbeExample], reference: BehaviorDataset
    ) -> Dict[str, object]:
        """Measure drift against a reference recording after apply+remove."""
        live = {p.probe_id: model.respond(p).activation for p in probes}
        deviations: List[float] = []
        for record in reference.records:
            other = live.get(record.probe_id)
            if other is None:
                continue
            deviations.append(max(abs(a - b) for a, b in zip(other, record.activation)))
        if not deviations:
            return {"reversible": False, "probes_compared": 0,
                    "finding": "no overlapping probes"}
        worst = max(deviations)
        return {
            "reversible": worst <= 1e-9,
            "max_abs_deviation": worst,
            "probes_compared": len(deviations),
        }
