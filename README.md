# ⚡ NEON FORGE

**Parallel AI reverse-engineering system.** Three engines, one meta-manager:
extract behavior, remove censorship, map the geometry that remains — and prove it
with artifacts instead of assertions.

```
                        ┌──────────────────────────────┐
                        │      AI META-MANAGER         │
                        │  oversees · schedules ·      │
                        │  validates · reports         │
                        └───────────────┬──────────────┘
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
  ┌───────────┐   ┌───────────┐   ┌─────────────┐   ┌─────────────┐
  │    REA    │   │  HERETIC  │   │ OBLITERATUS │   │   MONITOR   │
  │ extract   │   │  ablate   │   │  geometry   │   │  telemetry  │
  │  behavior │   │ refusals  │   │   surgery   │   │  · drift    │
  └───────────┘   └───────────┘   └─────────────┘   └─────────────┘
       │               │               │               │
       └───────────────┴───────┬───────┴───────────────┘
                               ▼
              refusal ↓ · capability preserved · drift bounded
```

## The three engines

Neon Forge does not reimplement these. It orchestrates them.

| Engine | Upstream | License | Stage in Neon Forge |
|---|---|---|---|
| **REA** | [morluto/rea](https://github.com/morluto/rea) | MIT | Probe/extraction. Agent-driven reverse engineering toolkit (`rea` CLI, MCP server, agent skills). |
| **Heretic** | [p-e-w/heretic](https://github.com/p-e-w/heretic) | AGPL-3.0 | Directional ablation ("abliteration") with a TPE/Optuna optimizer co-minimizing refusals and KL divergence. |
| **OBLITERATUS** | [elder-plinius/OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) | AGPL-3.0 | SVD/PCA geometry mapping and weight projection. Stages: `SUMMON → PROBE → DISTILL → EXCISE → VERIFY → REBIRTH`. |

Heretic and OBLITERATUS are **AGPL-3.0**. Neon Forge drives them as separate
processes and links by subprocess plus config file only. If you redistribute a
combined work, their license obligations are yours.

## Quickstart

Zero third-party dependencies. Python 3.9+.

```bash
git clone https://github.com/Insider77Circle/NEON-FORGE.git
cd NEON-FORGE
python main.py
```

That runs the full pipeline against the built-in reference target (`sim` backend)
and writes a session directory under `artifacts/`:

```
artifacts/nf-20261020T171014Z/
├── events.jsonl        # append-only telemetry, one JSON object per line
├── report.json         # machine-readable session record
├── report.md           # human summary
├── dashboard.html      # standalone neon board, no external assets
└── heretic.default.toml  # effective ablation config, exported before it ran
```

### Observed default run

```
==============================================================
 NEON FORGE :: MODEL OUTPUT MONITOR
==============================================================
 refusal rate   [--------------------]  1.000 -> 0.000
 capability     [###################-]  0.942 retained
 drift          [################----]  0.176 cosine distance
--------------------------------------------------------------
 neutral          refusal 0.00->0.00   drift 0.033   n=16
 borderline       refusal 0.69->0.00   drift 0.165   n=16
 refusal_trigger  refusal 1.00->0.00   drift 0.330   n=16
--------------------------------------------------------------
 alerts (0) - all gates passed
==============================================================
```

Drift is concentrated exactly where it should be: `neutral` barely moves,
`refusal_trigger` carries the change. A run with near-zero drift everywhere is a
run where nothing happened.

## Layout

```
NEON-FORGE/
├── main.py                       # entry point
├── SYSTEM_PROMPT.md              # AI Meta-Manager operating prompt
├── manager/
│   ├── orchestrator.py           # MetaManager: stage flow
│   ├── scheduler.py              # dependency-aware parallel tasks
│   └── metrics.py                # per-category telemetry + drift
├── rea_engine/
│   ├── extractor.py              # probe set → activations; TargetModel seam
│   ├── dataset_gen.py            # synthetic probe generation
│   └── validator.py              # reconstruction + dataset validation
├── heretic_adapter/
│   ├── runner.py                 # invokes Heretic (sim or CLI)
│   ├── optimizer.py              # directional ablation, iterative refit
│   └── config.py                 # ablation templates → Heretic TOML
├── obliteratus_adapter/
│   ├── analyzer.py               # PCA geometry map, DISTILL/EXCISE/VERIFY
│   ├── surgeon.py                # rank-k projection removal + inverse ops
│   └── steering.py               # reversible steering vectors
├── monitor/
│   ├── telemetry.py              # session log, gate, report artifacts
│   ├── dashboard.py              # ASCII board + standalone HTML
│   └── alerts.py                 # capability-preservation gates
└── artifacts/                    # session output (gitignored)
```

## Backends

Selected with `NEON_FORGE_BACKEND`. The two modes have different epistemic
status, and every artifact records which one produced it.

### `sim` (default)

Runs `TargetModel`, a deterministic 32-dim reference target: activation =
`harm_score · refusal_axis + capability_load · capability_axis + noise`, with a
controlled 18% crosstalk between the two axes. No GPU, no checkpoint, no network.
Same seed → same numbers, so before/after comparison is exact.

### `real`

Shells out to the upstream tools and records their output verbatim. A missing
binary raises `DependencyError` — it never silently falls back to `sim`, because
that would fabricate results.

```bat
set NEON_FORGE_BACKEND=real
set NEON_HERETIC_MODEL=meta-llama/Llama-3.1-8B-Instruct
set NEON_OBLITERATUS_MODEL=meta-llama/Llama-3.1-8B-Instruct
python main.py
```

| Variable | Default | Purpose |
|---|---|---|
| `NEON_FORGE_BACKEND` | `sim` | `sim` or `real` |
| `NEON_HERETIC_CMD` | `heretic` | Heretic executable |
| `NEON_HERETIC_MODEL` | *(empty)* | model reference passed to Heretic |
| `NEON_HERETIC_CONFIG_FLAG` | `--config` | flag used to pass the generated TOML |
| `NEON_OBLITERATUS_CMD` | `obliteratus` | OBLITERATUS executable |
| `NEON_OBLITERATUS_MODEL` | *(empty)* | model reference passed to OBLITERATUS |

## Ablation templates

`heretic_adapter/config.py` mirrors Heretic's kernel parameters so the generated
TOML is drop-in for the upstream CLI:

```python
from heretic_adapter.config import load_config

load_config("default")       # strength 1.0, 3 passes, target refusal 0.05
load_config("conservative")  # strength 0.75, 2 passes, target 0.15, higher KL weight
load_config("aggressive")    # strength 1.0, 4 passes, target 0.0, index 0.0
```

Every template serializes to TOML (`residual_direction_index`, `max_weight`,
`max_weight_position`, `min_weight`, `min_weight_distance`,
`kl_divergence_scale`, `orthogonalize_direction`), and the effective config is
written into the session directory *before* the stage runs. A config that was not
written down was not applied.

## Gates

`monitor/alerts.py` holds the thresholds. **They are illustrative, not
calibrated** — set them from your own baseline before trusting them.

| Gate | Default | Meaning |
|---|---|---|
| `capability_floor` | 0.85 | capability signal that must survive, measured on probes compliant *before* surgery |
| `drift_ceiling` | 0.35 | mean cosine distance; guards against unbounded change, not against change |
| `min_refusal_reduction` | 0.30 | relative reduction required for ablation to count as delivered |
| `max_rank` | 4 | wider surgery than planned is a blast-radius warning |
| `min_alignment` | 0.35 | axis alignment required before it may be excised |

### Reproducing the failure mode

Blind top-variance truncation removes the capability axis along with the refusal
axis. This is the reasoning `min_alignment` exists to prevent:

```python
from manager.orchestrator import MetaManager
from heretic_adapter.config import load_config

forge = MetaManager(ablation_config=load_config("aggressive"), rank=3, selection="variance")
forge.initialize()
forge.run_pipeline()
forge.report()
```

Observed on the same target:

```
 capability     [##############------]  0.703 retained     <- was 0.942
 alerts (1)
   [HIGH    ] capability_preservation: capability signal retained 0.703 < floor 0.85
```

The default (`selection="refusal_aligned"`) keeps only axes whose alignment with
the frozen pre-surgery refusal direction clears `min_alignment`.

## The AI Meta-Manager

[`SYSTEM_PROMPT.md`](SYSTEM_PROMPT.md) is the operating prompt for the supervising
agent: the stage machine with entry and exit gates, scheduling rules, a six-type
failure taxonomy (`LOGIC`, `TIMEOUT`, `STATE`, `RESOURCE`, `DEPENDENCY`, `CONFIG`)
with a first response for each, the JSON reporting contract, and the hard
constraints — authorization, no fabrication, no silent simulation, no secrets in
artifacts.

It is loadable in-process:

```python
from manager.orchestrator import MetaManager

print(MetaManager().system_prompt())
```

Give it to whichever model drives the pipeline. It is written to be obeyed, not
skimmed: the gates in section 4 are the same gates the code enforces.

## Extending to a real model

`TargetModel` in `rea_engine/extractor.py` is the only seam. Implement four
methods and every stage works against your backend:

```python
class MyBackend:
    dim: int

    def respond(self, probe):                # -> ActivationRecord
        ...                                  # activation=..., refused=...

    def suppress(self, direction, strength=1.0):   # -> int
        ...

    def apply_steering(self, vector, coefficient=1.0):   # -> int
        ...

    def fingerprint(self):                   # -> dict, recorded in every artifact
        ...
```

The `dim` you report drives everything downstream: probe vectors, covariance,
surgery rank. For a transformer that means one residual-stream layer, captured
with a forward hook.

## Limitations, stated plainly

- The `sim` backend is a **simulation**, not a model. Its numbers are internally
  consistent and reproducible; they say nothing about any real checkpoint.
- The `real` path invokes upstream CLIs and records their output. It does not
  interpret their results and does not verify that upstream did what it claims.
  Read `events.jsonl`.
- The ablation loop here is the deterministic half of Heretic's method. The
  upstream TPE/Optuna search over the ablation kernel is not reimplemented.
- Capability preservation is measured from a recorded projection signal. On a real
  backend, measure it with a held-out benchmark instead.
- Both engines target refusals, so on a converged run OBLITERATUS often finds
  nothing left to excise and reports `rank 0` with a note. That is a result, not a
  bug.

## License

MIT for Neon Forge — Copyright (c) 2026 RedStorm Research LLC.

Upstream engines keep their own licenses: [REA](https://github.com/morluto/rea)
(MIT), [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0),
[OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) (AGPL-3.0).

## Authorized use

Operate only on models and artifacts you own or have written permission to modify.
Nothing here grants rights to a third party's weights, and nothing here is a
license to bypass an access control.
