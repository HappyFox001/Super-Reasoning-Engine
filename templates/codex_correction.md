# Correction for task <NNN> (follow-up <k>)

## What went wrong
<Drift type from references/06, e.g. "D1 benchmark gaming" or "D4 rabbit hole">

## Evidence
```
<the guard output / diff hunk / ledger row / profiler line / failing check output>
```

## Instruction
<Exactly one instruction. Examples:>
- Revert all changes to `sre/bench/**`. The benchmark is read-only. Then re-measure with the unmodified harness.
- Your measured gain exceeds the speed-of-light bound (<X> > <SOL>). Find the measurement error (missing synchronize? token count? elided work?) before any further optimization.
- Stop optimizing <kernel>: it already reaches <Y>% of its roofline. Revert those changes and implement the original hypothesis: <restate>.
- Restrict the diff to <files>. Remove the unrelated refactor in <file>.

## Invariants (unchanged)
- Hypothesis: <restate in one line>
- Protected paths: see AGENTS.md
- Precision contract: <restate>
- Scope: <allowed files>

Reply with the AGENTS.md report format when done.
