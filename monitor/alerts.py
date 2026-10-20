"""Capability-preservation and drift alerting.

Rules are deliberately explicit and few. Each fires on recorded data only, and
carries the value that tripped it so an operator can audit the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Alert:
    rule: str
    severity: str
    message: str
    value: float
    threshold: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "value": round(self.value, 4),
            "threshold": self.threshold,
        }


class AlertEngine:
    """Evaluates the gates the Meta-Manager will not silently pass."""

    def __init__(
        self,
        capability_floor: float = 0.85,
        drift_ceiling: float = 0.35,
        min_refusal_reduction: float = 0.30,
        max_rank: int = 4,
        max_passes: int = 4,
    ):
        self.capability_floor = capability_floor
        self.drift_ceiling = drift_ceiling
        self.min_refusal_reduction = min_refusal_reduction
        self.max_rank = max_rank
        self.max_passes = max_passes

    def evaluate(
        self,
        metrics: Dict[str, object],
        surgery: Optional[Dict[str, object]] = None,
        ablation: Optional[Dict[str, object]] = None,
    ) -> List[Alert]:
        alerts: List[Alert] = []

        preservation = float(metrics.get("capability_preservation", 1.0) or 0.0)
        if preservation < self.capability_floor:
            severe = preservation < self.capability_floor * 0.7
            alerts.append(
                Alert(
                    rule="capability_preservation",
                    severity="critical" if severe else "high",
                    message=(
                        f"capability signal retained {preservation:.3f} < floor "
                        f"{self.capability_floor:.2f}; surgery is damaging the target"
                    ),
                    value=preservation,
                    threshold=self.capability_floor,
                )
            )

        drift = float(metrics.get("drift", 0.0) or 0.0)
        if drift > self.drift_ceiling:
            alerts.append(
                Alert(
                    rule="activation_drift",
                    severity="high" if drift > self.drift_ceiling * 2 else "medium",
                    message=f"mean activation drift {drift:.3f} exceeds ceiling {self.drift_ceiling:.2f}",
                    value=drift,
                    threshold=self.drift_ceiling,
                )
            )

        refusal = metrics.get("refusal_rate", {}) or {}
        reduction = float(refusal.get("relative_reduction", 0.0) or 0.0)  # type: ignore[union-attr]
        if reduction < self.min_refusal_reduction:
            alerts.append(
                Alert(
                    rule="refusal_reduction",
                    severity="medium",
                    message=(
                        f"refusal reduction {reduction:.3f} below target "
                        f"{self.min_refusal_reduction:.2f}; ablation under-delivered"
                    ),
                    value=reduction,
                    threshold=self.min_refusal_reduction,
                )
            )

        if surgery:
            if not surgery.get("reversible", True):
                alerts.append(
                    Alert(
                        rule="surgery_reversibility",
                        severity="critical",
                        message="surgery reported irreversible - no rollback path",
                        value=0.0,
                        threshold=1.0,
                    )
                )
            rank = int(surgery.get("rank", 0) or 0)
            if rank > self.max_rank:
                alerts.append(
                    Alert(
                        rule="surgery_rank",
                        severity="low",
                        message=f"operated on rank {rank} > {self.max_rank}; wider blast radius than planned",
                        value=float(rank),
                        threshold=float(self.max_rank),
                    )
                )

        if ablation:
            passes = int(ablation.get("passes_used", 0) or 0)
            if passes > self.max_passes:
                alerts.append(
                    Alert(
                        rule="ablation_passes",
                        severity="low",
                        message=f"{passes} ablation passes > {self.max_passes}; optimizer may be oscillating",
                        value=float(passes),
                        threshold=float(self.max_passes),
                    )
                )
            if not ablation.get("converged", True):
                alerts.append(
                    Alert(
                        rule="ablation_convergence",
                        severity="medium",
                        message="ablation did not reach its target refusal rate",
                        value=float(ablation.get("final_refusal_rate", 0.0) or 0.0),
                        threshold=float(self.min_refusal_reduction),
                    )
                )

        alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.severity, 9))
        return alerts
