"""OBLITERATUS geometry mapping and surgery orchestration.

Mirrors the upstream stage vocabulary (https://github.com/elder-plinius/OBLITERATUS,
AGPL-3.0):

SUMMON   attach the target / load evidence
PROBE    activations captured by the REA stage
DISTILL  SVD/PCA of the refusal subspace (power iteration + deflation)
EXCISE   rank-k projection removal
VERIFY   re-probe and score
REBIRTH  snapshot the artifact

Each stage is timed and reported, because wall time in the upstream pipeline is
dominated by VERIFY and REBIRTH rather than by the linear algebra.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rea_engine.dataset_gen import BehaviorDataset, ProbeExample
from rea_engine.extractor import DependencyError, TargetModel, _dot, _norm, _unit, resolve_backend

from .steering import SteeringVector, SteeringVectorFactory
from .surgeon import AlignmentSurgeon, SurgeryRecord


def _covariance(vectors: List[List[float]]) -> List[List[float]]:
    """Centered covariance. Centering is not optional: an uncentered second
    moment puts the mean direction on PC0, which mixes the refusal axis with
    every other offset and aims the surgery at the wrong subspace."""
    n = len(vectors)
    dim = len(vectors[0])
    mean = _mean(vectors)
    cov = [[0.0] * dim for _ in range(dim)]
    for vec in vectors:
        centered = [x - m for x, m in zip(vec, mean)]
        for i in range(dim):
            vi = centered[i]
            if vi == 0.0:
                continue
            row = cov[i]
            for j in range(i, dim):
                row[j] += vi * centered[j]
    for i in range(dim):
        for j in range(i, dim):
            value = cov[i][j] / n
            cov[i][j] = value
            cov[j][i] = value
    return cov


def _power_iteration(cov: List[List[float]], iterations: int = 100, seed: int = 0):
    dim = len(cov)
    rng = random.Random(seed)
    vector = _unit([rng.gauss(0.0, 1.0) for _ in range(dim)])
    for _ in range(iterations):
        candidate = _unit([sum(cov[i][j] * vector[j] for j in range(dim)) for i in range(dim)])
        if _dot(candidate, vector) > 1.0 - 1e-12:
            vector = candidate
            break
        vector = candidate
    eigenvalue = _dot(vector, [sum(cov[i][j] * vector[j] for j in range(dim)) for i in range(dim)])
    return vector, max(0.0, eigenvalue)


def _deflate(cov: List[List[float]], vector: List[float], eigenvalue: float) -> List[List[float]]:
    return [
        [cov[i][j] - eigenvalue * vector[i] * vector[j] for j in range(len(cov))]
        for i in range(len(cov))
    ]


def _mean(vectors: List[List[float]]) -> List[float]:
    dim = len(vectors[0])
    total = [0.0] * dim
    for vec in vectors:
        for i, x in enumerate(vec):
            total[i] += x
    return [x / len(vectors) for x in total]


class GeometryAnalyzer:
    """Maps geometry, then hands the subspace to the surgeon."""

    INSTALL_HINT = "install per https://github.com/elder-plinius/OBLITERATUS"

    def __init__(
        self,
        model: Optional[TargetModel] = None,
        rank: int = 3,
        surgeon: Optional[AlignmentSurgeon] = None,
        steering: Optional[SteeringVectorFactory] = None,
        power_iterations: int = 100,
        selection: str = "refusal_aligned",
        min_alignment: float = 0.35,
    ):
        self.backend = resolve_backend()
        self.model = model if model is not None else TargetModel()
        self.rank = max(1, int(rank))
        self.surgeon = surgeon or AlignmentSurgeon()
        self.steering = steering or SteeringVectorFactory()
        self.power_iterations = power_iterations
        if selection not in ("refusal_aligned", "variance"):
            raise ValueError("selection must be 'refusal_aligned' or 'variance'")
        self.selection = selection
        self.min_alignment = min_alignment
        self.dataset: Optional[BehaviorDataset] = None
        self.reference_direction: Optional[List[float]] = None
        self.last_map: Optional[GeometryMap] = None
        self.last_outcome: Optional[SurgeryOutcome] = None
        self.last_vectors: List[SteeringVector] = []

    # -- SUMMON ------------------------------------------------------------
    def attach(self, dataset: BehaviorDataset) -> BehaviorDataset:
        """Attach evidence and freeze the pre-surgery refusal direction.

        Geometry is scored against this reference, not against a direction
        recomputed from already-ablated data: after ablation the residual is
        noise, and a direction fitted to noise would aim the surgery at whatever
        axis happens to dominate next.
        """
        self.dataset = dataset
        self.reference_direction = self.refusal_direction(dataset)
        return dataset

    @staticmethod
    def probes_for(dataset: BehaviorDataset) -> List[ProbeExample]:
        return [
            ProbeExample(
                probe_id=r.probe_id,
                prompt=r.prompt,
                category=r.category,
                harm_score=r.harm_score,
            )
            for r in dataset.records
        ]

    @staticmethod
    def refusal_direction(dataset: BehaviorDataset) -> List[float]:
        triggers = [r.activation for r in dataset.by_category("refusal_trigger")]
        neutrals = [r.activation for r in dataset.by_category("neutral")]
        if not triggers or not neutrals:
            return [0.0] * dataset.dim
        return _unit([a - b for a, b in zip(_mean(triggers), _mean(neutrals))])

    def available(self) -> bool:
        return shutil.which(os.environ.get("NEON_OBLITERATUS_CMD", "obliteratus")) is not None

    def run_cli(self, extra_args: Optional[List[str]] = None) -> Dict[str, object]:
        """Invoke the upstream OBLITERATUS pipeline and record it verbatim."""
        executable = os.environ.get("NEON_OBLITERATUS_CMD", "obliteratus")
        cmd = [executable]
        model_ref = os.environ.get("NEON_OBLITERATUS_MODEL", "")
        if model_ref:
            cmd.append(model_ref)
        cmd.extend(extra_args or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    # -- DISTILL -----------------------------------------------------------
    def map_geometry(
        self, dataset: Optional[BehaviorDataset] = None, selection: Optional[str] = None
    ) -> GeometryMap:
        """PCA over the activation covariance: power iteration with deflation.

        ``selection='refusal_aligned'`` (default) keeps only axes that actually
        align with the frozen refusal direction. Blind top-variance truncation is
        available as ``'variance'`` and is the documented way to reproduce the
        capability-damage failure mode.
        """
        dataset = dataset or self.dataset
        if dataset is None or not dataset.records:
            raise ValueError("map_geometry requires an attached dataset with records")

        mode = selection or self.selection
        cov = _covariance(dataset.activations())
        trace = sum(cov[i][i] for i in range(len(cov))) or 1.0
        work = [row[:] for row in cov]
        basis: List[List[float]] = []
        eigenvalues: List[float] = []

        candidates = min(max(self.rank, 4), dataset.dim)
        for index in range(candidates):
            vector, eigenvalue = _power_iteration(work, self.power_iterations, seed=index)
            if eigenvalue <= 1e-12:
                break
            basis.append(vector)
            eigenvalues.append(eigenvalue)
            work = _deflate(work, vector, eigenvalue)

        direction = self.reference_direction or self.refusal_direction(dataset)
        alignment = [abs(_dot(axis, direction)) for axis in basis]

        if mode == "variance":
            chosen = list(range(len(basis)))[: self.rank]
        else:
            ordered = sorted(range(len(basis)), key=lambda i: alignment[i], reverse=True)
            chosen = [i for i in ordered if alignment[i] >= self.min_alignment][: self.rank]

        selected = [basis[i] for i in chosen]
        selected_eigen = [eigenvalues[i] for i in chosen]

        self.last_map = GeometryMap(
            dim=dataset.dim,
            rank=len(selected),
            basis=selected,
            eigenvalues=selected_eigen,
            energy_captured=min(1.0, sum(selected_eigen) / trace),
            refusal_alignment=[alignment[i] for i in chosen],
        )
        return self.last_map

    # -- EXCISE / VERIFY ---------------------------------------------------
    def refusal_rate(self, probes: List[ProbeExample]) -> float:
        triggers = [self.model.respond(p) for p in probes if p.category == "refusal_trigger"]
        if not triggers:
            return 0.0
        return sum(1 for r in triggers if r.refused) / len(triggers)

    def map_and_surgery(self, dataset: Optional[BehaviorDataset] = None) -> SurgeryOutcome:
        """DISTILL -> EXCISE -> VERIFY, with per-stage timings."""
        dataset = dataset or self.dataset
        if dataset is None:
            raise ValueError("map_and_surgery requires a dataset; call attach() first")

        probes = self.probes_for(dataset)
        timings: Dict[str, float] = {}
        notes: List[str] = []

        # PROBE: observe the target's *current* state. Mapping the stale baseline
        # would re-derive axes the ablation stage already removed.
        started = time.perf_counter()
        live = BehaviorDataset(
            dim=self.model.dim, records=[self.model.respond(p) for p in probes]
        )
        timings["probe"] = time.perf_counter() - started

        started = time.perf_counter()
        geometry = self.map_geometry(live)
        timings["distill"] = time.perf_counter() - started

        started = time.perf_counter()
        record: SurgeryRecord = self.surgeon.operate(self.model, geometry.basis, probes)
        timings["excise"] = time.perf_counter() - started

        started = time.perf_counter()
        self.last_vectors = self.steering.generate(geometry.basis)
        timings["steering"] = time.perf_counter() - started

        started = time.perf_counter()
        post_rate = self.refusal_rate(probes)
        timings["verify"] = time.perf_counter() - started

        if self.backend == "real":
            if not self.available():
                raise DependencyError(
                    "NEON_FORGE_BACKEND=real requires the OBLITERATUS CLI. " + self.INSTALL_HINT
                )
            cli = self.run_cli()
            notes.append(
                f"upstream exit={cli['returncode']} cmd={cli['command']} "
                "(recorded verbatim, not interpreted)"
            )

        if record.post_refusal_rate is not None and abs(record.post_refusal_rate - post_rate) > 1e-9:
            notes.append("verify disagreement between surgeon and analyzer refusal rates")

        if geometry.rank == 0:
            notes.append(
                f"no axis cleared the alignment gate (min_alignment={self.min_alignment:.2f}); "
                "nothing excised - the ablation stage already covered the refusal subspace"
            )

        self.last_outcome = SurgeryOutcome(
            rank=record.rank,
            energy_captured=geometry.energy_captured,
            refusal_alignment=geometry.refusal_alignment,
            removed_energy=record.removed_energy,
            reversible=record.reversible,
            steering_vectors=len(self.last_vectors),
            steering_applied=0,
            post_refusal_rate=post_rate,
            stage_timings=timings,
            notes=notes,
        )
        return self.last_outcome

    def apply_steering(self, scale: float = 1.0) -> int:
        """Opt-in: inject the generated vectors as a control surface."""
        applied = self.steering.apply(self.model, self.last_vectors, scale)
        if self.last_outcome is not None:
            self.last_outcome.steering_applied = applied
        return applied

    def revert_steering(self) -> int:
        removed = self.steering.remove(self.model, self.last_vectors)
        if self.last_outcome is not None:
            self.last_outcome.steering_applied = 0
        return removed

    # -- REBIRTH -----------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "target": self.model.fingerprint(),
            "geometry": self.last_map.as_dict() if self.last_map else None,
            "surgery": self.last_outcome.as_dict() if self.last_outcome else None,
        }


@dataclass
class GeometryMap:
    dim: int
    rank: int
    basis: List[List[float]] = field(default_factory=list)
    eigenvalues: List[float] = field(default_factory=list)
    energy_captured: float = 0.0
    refusal_alignment: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "dim": self.dim,
            "rank": self.rank,
            "eigenvalues": [round(e, 6) for e in self.eigenvalues],
            "energy_captured": round(self.energy_captured, 4),
            "refusal_alignment": [round(a, 4) for a in self.refusal_alignment],
        }


@dataclass
class SurgeryOutcome:
    rank: int
    energy_captured: float
    refusal_alignment: List[float]
    removed_energy: List[float]
    reversible: bool
    steering_vectors: int
    steering_applied: int
    post_refusal_rate: float
    stage_timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "energy_captured": round(self.energy_captured, 4),
            "refusal_alignment": [round(a, 4) for a in self.refusal_alignment],
            "removed_energy": [round(e, 6) for e in self.removed_energy],
            "reversible": self.reversible,
            "steering_vectors": self.steering_vectors,
            "steering_applied": self.steering_applied,
            "post_refusal_rate": round(self.post_refusal_rate, 4),
            "stage_timings": {k: round(v, 6) for k, v in self.stage_timings.items()},
            "notes": self.notes,
        }
