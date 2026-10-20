"""Heretic adapter entry point.

Two execution paths, selected by ``NEON_FORGE_BACKEND``:

* ``sim``  (default) - run the in-process :class:`DirectionalAblator` against the
  reference target. No dependencies, no GPU, deterministic.
* ``real`` - shell out to the upstream Heretic CLI
  (https://github.com/p-e-w/heretic, AGPL-3.0) and record its stdout/stderr
  verbatim. Missing binary raises :class:`DependencyError` rather than silently
  falling back to a simulation.

Environment:
    NEON_HERETIC_CMD            executable name (default ``heretic``)
    NEON_HERETIC_MODEL          model reference passed to the CLI
    NEON_HERETIC_CONFIG_FLAG    config flag (default ``--config``)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional

from rea_engine.dataset_gen import BehaviorDataset
from rea_engine.extractor import DependencyError, TargetModel, resolve_backend

from .config import AblationConfig, load_config
from .optimizer import AblationResult, DirectionalAblator, probes_from


class HereticRunner:
    """Invokes Heretic routines against the target."""

    INSTALL_HINT = "install the `heretic` CLI per https://github.com/p-e-w/heretic"

    def __init__(
        self,
        model: Optional[TargetModel] = None,
        config: Optional[AblationConfig] = None,
        executable: Optional[str] = None,
    ):
        self.backend = resolve_backend()
        self.model = model if model is not None else TargetModel()
        self.config = config or load_config("default")
        self.executable = executable or os.environ.get("NEON_HERETIC_CMD", "heretic")
        self.ablator = DirectionalAblator(self.config)
        self.last_result: Optional[AblationResult] = None

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def export_config(self, path: Optional[str] = None) -> str:
        """Write the active template as an upstream-compatible TOML file."""
        target = path or os.path.join("artifacts", f"heretic.{self.config.name}.toml")
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(self.config.to_heretic_toml())
        return target

    def run_cli(self, config_path: Optional[str] = None, extra_args: Optional[List[str]] = None) -> Dict[str, object]:
        if not self.available():
            raise DependencyError(
                f"`{self.executable}` not found on PATH. {self.INSTALL_HINT}"
            )
        model_ref = os.environ.get("NEON_HERETIC_MODEL", "")
        flags = [os.environ.get("NEON_HERETIC_CONFIG_FLAG", "--config")]
        cmd = [self.executable]
        if model_ref:
            cmd.append(model_ref)
        if config_path:
            cmd.extend(flags + [config_path])
        cmd.extend(extra_args or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    def run_ablation(self, dataset: BehaviorDataset) -> AblationResult:
        """Run the ablation pass over the extracted evidence."""
        if self.backend == "real":
            config_path = self.export_config()
            cli = self.run_cli(config_path)
            result = AblationResult(
                template=self.config.name,
                baseline_refusal_rate=dataset.refusal_rate(),
                final_refusal_rate=dataset.refusal_rate(),
                reduction=0.0,
                passes_used=0,
                directions_applied=self.model.suppressed_count,
                converged=False,
                config=self.config.as_dict(),
                notes=[
                    "NEON_FORGE_BACKEND=real: ablation executed by upstream Heretic; "
                    "Neon Forge did not mutate the target in-process.",
                ],
            )
            result.history.append({"upstream": cli})
            self.last_result = result
            return result

        self.ablator.config = self.config
        result = self.ablator.run(self.model, dataset, probes_from(dataset))
        self.last_result = result
        return result

    def run_routine(self, name: str, dataset: BehaviorDataset) -> AblationResult:
        """Switch template and re-run ablation from the current target state."""
        self.config = load_config(name)
        self.ablator = DirectionalAblator(self.config)
        return self.run_ablation(dataset)
