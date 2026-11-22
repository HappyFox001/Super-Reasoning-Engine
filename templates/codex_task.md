# Task <NNN>: <short imperative title>

You are implementing ONE optimization in a performance-critical LLM inference engine.
Follow AGENTS.md. A separate reviewer will verify your work with a frozen benchmark and a
frozen correctness check.

## Kind
<optimization | microbench | harness | investigation>

## Speed-of-light context
- Workload: <model>, <GPU xN>, <precision>, decode B=<B>, ctx=<ctx>
- Binding resource at operating point: <memory | compute | interconnect>
- Achievable SOL: <value unit>. Current best: <value> = <X>% of SOL.
- Gap waterfall (largest bars):
  - <term A>: <ms or %> of step time
  - <term B>: ...
- This task attacks: **<term A>**, max payoff ≈ <Y>% of step time.

## Hypothesis
<One paragraph. The mechanism: which bytes, launches, syncs or idle time disappear, and why.>

**Predicted gain:** <e.g. +6% decode tok/s at B=64>, because <short derivation from the numbers above>.

## Scope
- You MAY edit: <explicit paths/globs>
- You MUST NOT edit: protected paths in AGENTS.md, plus <any task-specific ones>
- Keep the diff under ~<N> changed lines unless strictly necessary; explain if exceeded.

## Constraints
- Precision contract: <restate>. Sampling semantics unchanged.
- No new dependencies or environment changes.
- Must work for all batch buckets <list> and context lengths up to <max>.

## Evidence to start from
```
<paste: relevant profiler lines, kernel timings, ledger rows, previous attempts>
```

## Known dead ends (do not repeat)
- <approach>: <why it failed>

## Commands
- Build: `<cmd>`
- Quick correctness: `<cmd>`
- Quick bench: `<cmd>`
- Profile: `<cmd>`

## Acceptance criteria
1. Quick correctness passes within contract tolerance.
2. The new path is active in the profiled decode step (show kernel names).
3. Self-measured gain on the quick bench, stated with command and numbers.

## Report
Use the report format from AGENTS.md exactly. If the hypothesis is refuted, revert your
changes and report what you learned.
