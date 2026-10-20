"""AI Meta-Manager: schedules, validates, and monitors the whole pipeline.

Stage order is fixed (REA -> Heretic -> OBLITERATUS -> VERIFY) because each stage
consumes the previous stage's state. The scheduler is used where the work is
genuinely independent: session pre-flight in :meth:`MetaManager.initialize`.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from heretic_adapter.config import AblationConfig
from heretic_adapter.runner import HereticRunner
from monitor.telemetry import TelemetryMonitor
from obliteratus_adapter.analyzer import GeometryAnalyzer
from rea_engine.extractor import BehaviorExtractor, TargetModel, resolve_backend

from .scheduler import DONE, ParallelScheduler

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SYSTEM_PROMPT.md"
)


class MetaManager:
    """Oversees, schedules, validates."""

    def __init__(
        self,
        model: Optional[TargetModel] = None,
        ablation_config: Optional[AblationConfig] = None,
        rank: int = 3,
        out_dir: str = "artifacts",
        workers: int = 4,
        selection: str = "refusal_aligned",
    ):
        self.backend = resolve_backend()
        self.model = model if model is not None else TargetModel()
        self.scheduler = ParallelScheduler(workers)
        self.out_dir = out_dir

        self.rea = BehaviorExtractor(self.model)
        self.heretic = HereticRunner(self.model, ablation_config)
        self.obliteratus = GeometryAnalyzer(self.model, rank=rank, selection=selection)
        self.monitor = TelemetryMonitor(self.model, out_dir=out_dir)

        self.dataset = None
        self.ablation: Optional[Dict[str, object]] = None
        self.surgery: Optional[Dict[str, object]] = None
        self.validation = None
        self.preflight: Dict[str, object] = {}

    # -- prompt ------------------------------------------------------------
    def system_prompt(self) -> str:
        """The Meta-Manager's operating instructions, as shipped."""
        with open(PROMPT_PATH, "r", encoding="utf-8") as handle:
            return handle.read()

    # -- boot --------------------------------------------------------------
    def initialize(self) -> None:
        print("[NEON FORGE] Boot sequence initiated.")
        session_id = self.monitor.start_session()
        assert self.monitor.session_dir is not None

        config_path = os.path.join(
            self.monitor.session_dir, f"heretic.{self.heretic.config.name}.toml"
        )
        self.scheduler.add("export_heretic_config", self.heretic.export_config, config_path)
        self.scheduler.add("build_probe_set", self.rea.probe_set)
        self.scheduler.add("load_system_prompt", self.system_prompt)
        self.scheduler.run()

        self.preflight = self.scheduler.summary()
        for task in ("export_heretic_config", "build_probe_set", "load_system_prompt"):
            record = self.scheduler.tasks[task]
            if record.status != DONE:
                print(f"[NEON FORGE] pre-flight '{task}' failed: {record.error}")

        self.monitor.record(
            "boot",
            session=session_id,
            backend=self.backend,
            rank=self.obliteratus.rank,
            ablation_template=self.heretic.config.name,
            preflight=self.preflight,
        )

    # -- pipeline ----------------------------------------------------------
    def run_pipeline(self) -> None:
        print("[NEON FORGE] Extracting behavior via REA...")
        self.dataset = self.rea.extract_behavior()
        self.monitor.attach_baseline(self.dataset, self.rea)
        self.monitor.record(
            "rea_extraction",
            probes=len(self.dataset),
            refusal_rate=round(self.dataset.refusal_rate(), 4),
            validation=self.dataset.manifest.get("validation"),
        )

        print("[NEON FORGE] Running Heretic ablation...")
        ablation = self.heretic.run_ablation(self.dataset)
        self.ablation = ablation.as_dict()
        self.monitor.attach_results(ablation=self.ablation)

        print("[NEON FORGE] Performing OBLITERATUS geometry surgery...")
        self.obliteratus.attach(self.dataset)
        outcome = self.obliteratus.map_and_surgery()
        self.surgery = outcome.as_dict()
        self.monitor.attach_results(surgery=self.surgery)
        self.monitor.record("rebirth_snapshot", **self.obliteratus.snapshot())

        print("[NEON FORGE] Validating outputs...")
        self.validation = self.monitor.validate_results()

    def report(self) -> str:
        path = self.monitor.generate_report()
        if self.validation is not None:
            print(self.validation.dashboard)
        print(f"[NEON FORGE] Artifacts: {path}")
        self.monitor.close()
        print("[NEON FORGE] Pipeline complete.")
        return path
