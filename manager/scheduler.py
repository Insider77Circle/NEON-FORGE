"""Dependency-aware parallel task scheduling.

Used by the metrics aggregator (per-category scoring) and the monitor
(capability probes). Tasks declare ``depends_on`` names; a dependency failure
marks downstream tasks BLOCKED instead of letting them run on missing inputs.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
BLOCKED = "blocked"


class CycleError(RuntimeError):
    """Raised when the task graph cannot make progress."""


@dataclass
class Task:
    name: str
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    depends_on: Sequence[str] = ()
    status: str = PENDING
    result: Any = None
    error: str = ""
    started_at: float = 0.0
    duration: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration": round(self.duration, 6),
            "depends_on": list(self.depends_on),
            "error": self.error,
        }


class ParallelScheduler:
    """Runs a DAG of tasks on a thread pool, honoring dependencies."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max(1, int(max_workers))
        self.tasks: Dict[str, Task] = {}

    def add(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        depends_on: Sequence[str] = (),
        **kwargs: Any,
    ) -> Task:
        if name in self.tasks:
            raise ValueError(f"duplicate task name '{name}'")
        for dependency in depends_on:
            if dependency not in self.tasks:
                raise CycleError(f"task '{name}' declares unknown dependency '{dependency}'")
        task = Task(name=name, fn=fn, args=args, kwargs=kwargs, depends_on=tuple(depends_on))
        self.tasks[name] = task
        return task

    def reset(self) -> None:
        self.tasks.clear()

    def run(self) -> Dict[str, Task]:
        if not self.tasks:
            return {}
        pending = {name: t for name, t in self.tasks.items() if t.status == PENDING}
        running: Dict[Future, Task] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while pending or running:
                progressed = False

                for name in list(pending):
                    task = pending[name]
                    states = [self.tasks[d].status for d in task.depends_on]
                    if any(s in (FAILED, BLOCKED) for s in states):
                        task.status = BLOCKED
                        task.error = "upstream dependency failed"
                        del pending[name]
                        progressed = True
                        continue
                    if all(s == DONE for s in states):
                        task.status = RUNNING
                        task.started_at = time.perf_counter()
                        running[pool.submit(task.fn, *task.args, **task.kwargs)] = task
                        del pending[name]
                        progressed = True

                if running:
                    finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                    for future in finished:
                        task = running.pop(future)
                        task.duration = time.perf_counter() - task.started_at
                        try:
                            task.result = future.result()
                            task.status = DONE
                        except Exception as exc:  # noqa: BLE001 - surfaced in the task record
                            task.status = FAILED
                            task.error = f"{type(exc).__name__}: {exc}"
                        progressed = True

                if not progressed:
                    raise CycleError(
                        "scheduler stalled - cyclic dependency among: "
                        + ", ".join(sorted(pending))
                    )

        return dict(self.tasks)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for task in self.tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        return {
            "tasks": len(self.tasks),
            "status_counts": counts,
            "wall_clock": round(sum(t.duration for t in self.tasks.values()), 6),
            "detail": [t.as_dict() for t in self.tasks.values()],
        }
