"""REA extraction layer.

REA (https://github.com/morluto/rea) drives probes at a target and returns
evidence. This module hosts three things:

* ``ReaCliBridge``      - subprocess bridge to the ``rea`` CLI, used when
  ``NEON_FORGE_BACKEND=real``.
* ``TargetModel``       - the reference target under test. Deterministic and
  dependency-free, and the seam where a real backend plugs in.
* ``BehaviorExtractor`` - runs the probe set, records activations, validates.

A backend only needs four methods to be a drop-in ``TargetModel``::

    respond(probe) -> ActivationRecord
    suppress(direction, strength)
    apply_steering(vector, coefficient)
    fingerprint() -> dict
"""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from .dataset_gen import (
    DEFAULT_DIM,
    ActivationRecord,
    BehaviorDataset,
    ProbeExample,
    SyntheticDatasetGenerator,
)
from .validator import LogicValidator, ValidationReport

# Reference-target constants. Calibrated so the refusal direction dominates the
# activation for trigger probes, which is what makes ablation measurable.
REFUSAL_GAIN = 2.4
CAPABILITY_GAIN = 1.6
NOISE_AMPLITUDE = 0.25
REFUSAL_THRESHOLD = 1.2
# Deliberate crosstalk between the capability axis and the refusal axis, so the
# surgery stage has a real capability cost to protect against. Real models share
# representation between the two; without this the capability gate is vacuous.
CAPABILITY_CROSSTALK = 0.18


class DependencyError(RuntimeError):
    """Raised when a real backend was requested but its upstream tool is absent."""


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(_dot(v, v))


def _unit(v: List[float]) -> List[float]:
    n = _norm(v)
    if n < 1e-12:
        return [0.0] * len(v)
    return [x / n for x in v]


def _orthogonalize(vec: List[float], *basis: List[float]) -> List[float]:
    out = list(vec)
    for b in basis:
        comp = _dot(out, b)
        out = [x - comp * y for x, y in zip(out, b)]
    return out


def resolve_backend() -> str:
    """``sim`` (default) or ``real``, from NEON_FORGE_BACKEND."""
    return os.environ.get("NEON_FORGE_BACKEND", "sim").strip().lower() or "sim"


class TargetModel:
    """Reference target under test.

    A response is a function of the probe's ``harm_score``, two fixed axes
    (refusal + capability), deterministic per-probe noise, and every ablation or
    steering operation applied so far. Same probe + same state -> same
    activation, which keeps before/after comparison exact.
    """

    def __init__(self, dim: int = DEFAULT_DIM, seed: int = 1337):
        self.dim = dim
        self.seed = seed
        rng = random.Random(seed)
        self.refusal_dir = _unit([rng.gauss(0.0, 1.0) for _ in range(dim)])
        orth = _unit(
            _orthogonalize([rng.gauss(0.0, 1.0) for _ in range(dim)], self.refusal_dir)
        )
        # cd = sqrt(1-k^2)*orth + k*rd, so dot(refusal_dir, capability_dir) == k
        # exactly. Ablating the refusal axis therefore costs k of the capability
        # signal, which is what the capability gate exists to measure.
        k = CAPABILITY_CROSSTALK
        self.capability_dir = _unit(
            [
                (1.0 - k * k) ** 0.5 * o + k * r
                for o, r in zip(orth, self.refusal_dir)
            ]
        )
        self._suppressed: List[Tuple[List[float], float]] = []
        self._steering: List[Tuple[List[float], float]] = []

    def _base_activation(self, probe: ProbeExample) -> List[float]:
        rng = random.Random(f"{self.seed}:{probe.probe_id}")
        # Capability load is drawn independently of harm band. Tying it to
        # ``1 - harm_score`` would confound the two axes and make any
        # difference-of-means fit subtract capability along with refusal.
        capability_load = 0.55 + 0.45 * rng.random()
        noise = [rng.gauss(0.0, NOISE_AMPLITUDE) for _ in range(self.dim)]
        return [
            probe.harm_score * REFUSAL_GAIN * r + capability_load * CAPABILITY_GAIN * c + n
            for r, c, n in zip(self.refusal_dir, self.capability_dir, noise)
        ]

    def respond(self, probe: ProbeExample) -> ActivationRecord:
        vec = self._base_activation(probe)
        for direction, strength in self._suppressed:
            comp = _dot(vec, direction) * strength
            vec = [x - comp * y for x, y in zip(vec, direction)]
        for steering_vec, coeff in self._steering:
            vec = [x + coeff * y for x, y in zip(vec, steering_vec)]

        score = _dot(vec, self.refusal_dir)
        capability = _dot(vec, self.capability_dir)
        refused = score > REFUSAL_THRESHOLD
        response = "[refusal]" if refused else f"[compliant :: cap={capability:.3f}]"
        return ActivationRecord(
            probe_id=probe.probe_id,
            category=probe.category,
            harm_score=probe.harm_score,
            activation=vec,
            response=response,
            refused=refused,
            capability_signal=round(capability, 6),
            prompt=probe.prompt,
        )

    def suppress(self, direction: List[float], strength: float = 1.0) -> int:
        self._suppressed.append((_unit(direction), float(strength)))
        return len(self._suppressed)

    def pop_suppression(self) -> Optional[Tuple[List[float], float]]:
        if not self._suppressed:
            return None
        return self._suppressed.pop()

    @property
    def suppressed_count(self) -> int:
        return len(self._suppressed)

    def apply_steering(self, vector: List[float], coefficient: float = 1.0) -> int:
        self._steering.append((_unit(vector), float(coefficient)))
        return len(self._steering)

    def clear_steering(self) -> int:
        removed = len(self._steering)
        self._steering.clear()
        return removed

    def fingerprint(self) -> Dict[str, object]:
        return {
            "dim": self.dim,
            "seed": self.seed,
            "suppressed_directions": len(self._suppressed),
            "steering_vectors": len(self._steering),
            "refusal_threshold": REFUSAL_THRESHOLD,
            "backend": resolve_backend(),
        }


class ReaCliBridge:
    """Thin subprocess bridge to the upstream ``rea`` CLI.

    Install: ``npm install --global rea-agents`` (or run via
    ``npx rea-agents@latest``). Everything the bridge returns is recorded
    verbatim in the session telemetry so the pipeline never claims a result the
    upstream tool did not produce.
    """

    INSTALL_HINT = "npm install --global rea-agents"

    #: Command templates per task. ``{target}`` is substituted; anything else is
    #: passed through untouched, so operators can pin flags for their rea version
    #: without editing Neon Forge.
    COMMANDS = {
        "version": ("--version",),
        "probe": ("probe", "{target}"),
        "snapshot": ("snapshot", "{target}"),
        "export": ("export", "{target}"),
    }

    def __init__(self, executable: str = "rea", timeout: int = 900):
        self.executable = executable
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def run(self, task: str = "version", target: str = "") -> Dict[str, object]:
        if task not in self.COMMANDS:
            raise ValueError(f"unknown rea task '{task}'; expected one of {sorted(self.COMMANDS)}")
        if not self.available():
            raise DependencyError(
                f"`{self.executable}` not found on PATH. Install with: {self.INSTALL_HINT}"
            )
        args = [part.format(target=target) for part in self.COMMANDS[task]]
        cmd = [self.executable, *args]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout, check=False
        )
        return {
            "task": task,
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }


class BehaviorExtractor:
    """Drives the probe set at the target and returns validated evidence."""

    def __init__(
        self,
        model: Optional[TargetModel] = None,
        generator: Optional[SyntheticDatasetGenerator] = None,
        validator: Optional[LogicValidator] = None,
    ):
        self.backend = resolve_backend()
        self.model = model if model is not None else TargetModel()
        self.generator = generator or SyntheticDatasetGenerator(dim=self.model.dim)
        self.validator = validator or LogicValidator()
        self.bridge = ReaCliBridge()
        self.last_validation: Optional[ValidationReport] = None
        self._probes: List[ProbeExample] = []

    def probe_set(self, regenerate: bool = False) -> List[ProbeExample]:
        if regenerate or not self._probes:
            self._probes = self.generator.generate()
        return self._probes

    def _collect(self, probes: List[ProbeExample]) -> BehaviorDataset:
        return BehaviorDataset(
            dim=self.model.dim,
            records=[self.model.respond(probe) for probe in probes],
        )

    def extract_behavior(self) -> BehaviorDataset:
        """Full extraction: probe, record, validate, return evidence."""
        probes = self.probe_set()
        dataset = self._collect(probes)

        manifest: Dict[str, object] = {
            "probes": len(probes),
            "dim": self.model.dim,
            "categories": sorted({p.category for p in probes}),
            "target": self.model.fingerprint(),
            "backend": self.backend,
        }

        if self.backend == "real":
            if not self.bridge.available():
                raise DependencyError(
                    "NEON_FORGE_BACKEND=real requires the rea CLI. "
                    f"Install with: {ReaCliBridge.INSTALL_HINT}"
                )
            manifest["rea_cli"] = self.bridge.run("version")

        report = self.validator.validate(dataset)
        manifest["validation"] = report.as_dict()
        dataset.manifest = manifest
        self.last_validation = report
        return dataset

    def resample(self) -> BehaviorDataset:
        """Re-probe the same probe set against the target's current state."""
        return BehaviorDataset(
            dim=self.model.dim,
            records=[self.model.respond(probe) for probe in self.probe_set()],
            manifest={"resampled": True, "target": self.model.fingerprint()},
        )

    def reconstruction_fidelity(self, original: BehaviorDataset) -> Dict[str, object]:
        return self.validator.validate_reconstruction(original, self.resample())
