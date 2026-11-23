# AGENTS.md: rules for Codex in this inference engine repository

The SRE supervisor assigns one experiment to an implementation worker and owns independent
measurement and acceptance. When you receive an implementation task, work only in the assigned
candidate and apply the worker rules below. The supervisor maintains the contract, benchmark,
correctness check, protected paths, and ledger.

## The contract (read `sre/contract.yaml`)

- Model, weights, precision, sampling semantics, workload and quality thresholds are fixed.
- You may use only the precisions listed in `precision` and `precision.allowed_upgrades`.
- Never skip layers, heads or tokens, truncate context, approximate softmax, or cache
  outputs across requests or benchmark iterations.

## Protected paths: never edit, move, or delete

```
sre/bench/**
sre/check/**
sre/contract.yaml
sre/protected.txt
sre/ledger.jsonl
sre/STATE.md
AGENTS.md
<lockfiles, e.g. requirements.lock, uv.lock, poetry.lock>
<add project-specific paths>
```

Use `sre/protected.txt` as the guard's versioned source of protected path globs.

## Environment

- No new dependencies, no `pip install`, no driver/toolkit/framework changes.
- No network access is available or needed.
- Do not change global compiler flags or environment variables used by the benchmark.

## Commands

```bash
# build
<build command>
# quick correctness smoke test (not authoritative)
<e.g. python -m sre.check.quick>
# quick benchmark (not authoritative)
<e.g. python -m sre.bench.run --quick>
# profile one decode step
<e.g. nsys profile -o /tmp/step python -m sre.bench.run --profile-step>
```

## How to work

1. Implement **only** the single hypothesis in the task. If you think a different idea is
   better, say so in your report; do not implement it.
2. Keep the diff minimal and confined to the files the task allows.
3. Run the checks available in your environment. If a check fails and you cannot fix it
   within the task scope, report the negative result and the affected candidate.
4. If the task assigns device access to the supervisor, report the device checks as pending
   and provide the commands and expected code path. Otherwise verify the active path in the profiler.
5. Report honestly. A clean negative result is valuable; an inflated claim is not.

## Report format (your final message)

```
HYPOTHESIS: <one line>
RESULT: confirmed | partially confirmed | refuted | pending supervisor validation
CHANGES: <files and the mechanism, 3-8 bullets>
SELF-MEASURED: <checked commands and outputs, or device checks pending>
PROFILE EVIDENCE: <observed kernels and timings, or expected path for supervisor to verify>
RISKS: <edge cases, shapes not covered, numerics>
NEXT: <what you would try next and why, with expected gain>
```

`sre/runs/` is tool output and is git-ignored; do not read or write it.
