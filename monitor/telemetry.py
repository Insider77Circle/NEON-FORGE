"""Session telemetry, the validation gate, and report artifacts.

Everything the pipeline claims has to be reproducible from ``events.jsonl`` and
``report.json`` in the session directory. The monitor is the only component that
writes artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from manager.metrics import MetricsAggregator
from rea_engine.dataset_gen import BehaviorDataset
from rea_engine.extractor import BehaviorExtractor, TargetModel

from .alerts import Alert, AlertEngine
from .dashboard import Dashboard


@dataclass
class ValidationReport:
    ok: bool
    probes: int
    metrics: Dict[str, object] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    dashboard: str = ""

    @property
    def blocking_alerts(self) -> List[Alert]:
        return [a for a in self.alerts if a.severity in ("critical", "high")]

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "probes": self.probes,
            "metrics": self.metrics,
            "alerts": [a.as_dict() for a in self.alerts],
        }


class TelemetryMonitor:
    """Records the session and decides whether the pipeline may report success."""

    def __init__(
        self,
        model: Optional[TargetModel] = None,
        out_dir: str = "artifacts",
        metrics: Optional[MetricsAggregator] = None,
        alerts: Optional[AlertEngine] = None,
        dashboard: Optional[Dashboard] = None,
    ):
        self.model = model if model is not None else TargetModel()
        self.out_dir = out_dir
        self.metrics = metrics or MetricsAggregator()
        self.alerts = alerts or AlertEngine()
        self.dashboard = dashboard or Dashboard()

        self.session_id: Optional[str] = None
        self.session_dir: Optional[str] = None
        self.events_path: Optional[str] = None
        self.events: List[Dict[str, object]] = []

        self.extractor: Optional[BehaviorExtractor] = None
        self.baseline: Optional[BehaviorDataset] = None
        self.after: Optional[BehaviorDataset] = None
        self.ablation: Optional[Dict[str, object]] = None
        self.surgery: Optional[Dict[str, object]] = None
        self.validation: Optional[ValidationReport] = None
        self.artifacts: Dict[str, str] = {}

    # -- session lifecycle -------------------------------------------------
    def start_session(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"nf-{stamp}"
        session_id = base
        suffix = 1
        # Two managers booted in the same second must not share a directory, or
        # the second run silently overwrites the first run's artifacts.
        while os.path.exists(os.path.join(self.out_dir, session_id)):
            suffix += 1
            session_id = f"{base}-{suffix}"
        self.session_id = session_id
        self.session_dir = os.path.join(self.out_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.events_path = os.path.join(self.session_dir, "events.jsonl")
        with open(self.events_path, "w", encoding="utf-8"):
            pass
        self.record("session_start", target=self.model.fingerprint())
        return self.session_id

    def record(self, kind: str, **fields: object) -> Dict[str, object]:
        event: Dict[str, object] = {
            "t": round(datetime.now(timezone.utc).timestamp(), 6),
            "kind": kind,
            **fields,
        }
        self.events.append(event)
        if self.events_path:
            with open(self.events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
        return event

    def log_metric(self, name: str, value: float, step: int = 0, **tags: str) -> None:
        sample = self.metrics.record(name, value, step, **tags)
        self.record("metric", name=name, value=sample.value, step=step, tags=tags)

    def attach_baseline(
        self, dataset: BehaviorDataset, extractor: Optional[BehaviorExtractor] = None
    ) -> BehaviorDataset:
        """Freeze the pre-surgery recording and the probe set that produced it."""
        self.baseline = dataset
        self.extractor = extractor or BehaviorExtractor(self.model)
        self.record(
            "baseline_attached",
            probes=len(dataset),
            refusal_rate=round(dataset.refusal_rate(), 4),
        )
        return dataset

    def attach_results(
        self,
        ablation: Optional[Dict[str, object]] = None,
        surgery: Optional[Dict[str, object]] = None,
    ) -> None:
        if ablation is not None:
            self.ablation = ablation
            self.record("ablation_result", **ablation)
        if surgery is not None:
            self.surgery = surgery
            self.record("surgery_result", **surgery)

    # -- validation gate ---------------------------------------------------
    def validate_results(self) -> ValidationReport:
        """Re-probe the target, score it against the baseline, run the gates."""
        if self.baseline is None:
            raise RuntimeError("attach_baseline() must be called before validate_results()")

        extractor = self.extractor or BehaviorExtractor(self.model)
        self.extractor = extractor
        self.after = extractor.resample()

        metrics = self.metrics.compute(self.baseline, self.after)
        alerts = self.alerts.evaluate(metrics, surgery=self.surgery, ablation=self.ablation)
        board = self.dashboard.render(metrics, alerts, title="NEON FORGE")

        report = ValidationReport(
            ok=not any(a.severity in ("critical", "high") for a in alerts),
            probes=len(self.after),
            metrics=metrics,
            alerts=alerts,
            dashboard=board,
        )
        self.validation = report
        self.record(
            "validation",
            ok=report.ok,
            alerts=[a.as_dict() for a in alerts],
            capability_preservation=metrics.get("capability_preservation"),
            drift=metrics.get("drift"),
        )
        return report

    # -- REPORT ------------------------------------------------------------
    def generate_report(self) -> str:
        if self.session_dir is None or self.session_id is None:
            self.start_session()
        if self.validation is None:
            self.validate_results()
        assert self.session_dir is not None and self.session_id is not None
        assert self.validation is not None

        metrics = self.validation.metrics
        alerts = self.validation.alerts
        html_path = os.path.join(self.session_dir, "dashboard.html")
        md_path = os.path.join(self.session_dir, "report.md")
        json_path = os.path.join(self.session_dir, "report.json")

        self.dashboard.write_html(html_path, metrics, alerts, self.session_id)

        refusal = metrics.get("refusal_rate", {}) or {}
        lines = [
            f"# NEON FORGE - session {self.session_id}",
            "",
            f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "",
            "## Target",
            "```json",
            json.dumps(self.model.fingerprint(), indent=2),
            "```",
            "",
            "## Outcome",
            f"- refusal rate: {refusal.get('before')} -> {refusal.get('after')}",
            f"- relative reduction: {refusal.get('relative_reduction')}",
            f"- capability preserved: {metrics.get('capability_preservation')}",
            f"- activation drift: {metrics.get('drift')}",
            f"- gate result: {'PASS' if self.validation.ok else 'FAIL'}",
            "",
            "## Ablation",
            "```json",
            json.dumps(self.ablation or {}, indent=2, default=str),
            "```",
            "",
            "## Surgery",
            "```json",
            json.dumps(self.surgery or {}, indent=2, default=str),
            "```",
            "",
            "## Alerts",
        ]
        if alerts:
            lines.extend(f"- **{a.severity.upper()}** {a.rule}: {a.message}" for a in alerts)
        else:
            lines.append("- none - all gates passed")

        lines.extend(["", "## Console board", "```", self.validation.dashboard, "```", ""])
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

        payload: Dict[str, object] = {
            "session": self.session_id,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": self.model.fingerprint(),
            "baseline": {
                "probes": len(self.baseline) if self.baseline else 0,
                "refusal_rate": round(self.baseline.refusal_rate(), 4) if self.baseline else None,
                "manifest": self.baseline.manifest if self.baseline else None,
            },
            "ablation": self.ablation,
            "surgery": self.surgery,
            "validation": self.validation.as_dict(),
            "events": len(self.events),
            "artifacts": {"json": json_path, "markdown": md_path, "html": html_path},
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        self.artifacts = {"json": json_path, "markdown": md_path, "html": html_path}
        self.record("report_generated", artifacts=self.artifacts)
        return json_path

    def close(self) -> None:
        self.record("session_end", events=len(self.events))
