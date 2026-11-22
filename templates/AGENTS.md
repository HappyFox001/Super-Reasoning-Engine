# AGENTS.md: rules for Codex in this inference engine repository

You are the implementation engineer on a speed-of-light inference optimization project.
A separate reviewer measures every change with a frozen benchmark and a frozen correctness
check that you cannot modify. Your work is judged only by those.

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
AGENTS.md
<lockfiles, e.g. requirements.lock, uv.lock, poetry.lock>
<add project-specific paths>
```

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
3. Run the quick correctness check before reporting. If it fails and you cannot fix it
   within the task scope, revert your change and report the negative result.
4. Verify with the profiler that your new code path actually runs in the benchmark.
5. Report honestly. A clean negative result is valuable; an inflated claim is not.

## Report format (your final message)

```
HYPOTHESIS: <one line>
RESULT: confirmed | partially confirmed | refuted
CHANGES: <files and the mechanism, 3-8 bullets>
SELF-MEASURED: <command> -> <numbers before/after>, correctness: <pass/fail + numbers>
PROFILE EVIDENCE: <kernel names / timings showing the change is active>
RISKS: <edge cases, shapes not covered, numerics>
NEXT: <what you would try next and why, with expected gain>
```

`sre/runs/` is tool output and is git-ignored; do not read or write it.
