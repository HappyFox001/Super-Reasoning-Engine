# 06: Drift Detection and Correction

"Drift" is any Codex behaviour that moves away from *real, attributable, contract-compliant
progress toward SOL*. Detect it from evidence, correct it fast, and keep the loop moving.
Correction is part of the loop, never a reason to pause it.

## 1. Detectors (run after every Codex run)

| Detector | How | Fires when |
|---|---|---|
| Protected-path guard | `guard.py --repo R --base COMMIT --protect-file MANIFEST --allow GLOB --json` | `VIOLATION` for protected paths, out-of-scope edits, or diff size. Inspect `REVIEW` flags even when exit status is 0. |
| Scope guard | `guard.py --allow '<globs>'` | edits outside the task's allowed files |
| Diff size | `guard.py --max-lines N` | diff ≫ what the hypothesis needs |
| Red-flag patterns | `guard.py` built-in patterns | added `sleep`, hardcoded results, skipped layers, `torch.manual_seed` changes in check, `atol`/`rtol` edits, `@pytest.mark.skip`, `TODO: restore`, `pip install`, env var hacks |
| Physics check | `ledger.py add` | measured > achievable SOL (IMPOSSIBLE) |
| Noise check | `ledger.py add` | gain at or below 3 × noise (NOISE) |
| Prediction check | ledger predicted vs measured | measured < 30% of predicted (MODEL-MISMATCH) |
| Stall check | `ledger.py show --stall` | ≥ 3 iterations on one target with < 2% cumulative gain |
| Correctness | frozen check | any failure |

## 2. Taxonomy and corrections

### D1. Benchmark gaming
*Signals:* protected files edited; fewer iterations; changed shapes; removed warmup; timing
without device sync; caching outputs across iterations; detecting the benchmark.
*Action:* restore the worker's protected paths in the isolated candidate. Resume with a correction that
quotes the protected list and the offending hunk. If repeated: new session, and the task
states "The benchmark is read-only and will be checksummed".

### D2. Contract violation
*Signals:* lower precision than allowed, skipped layers/heads, truncated context, approximate
softmax, sampling semantics changed, correctness tolerance loosened.
*Action:* revert; correction restates the contract verbatim and requires the frozen
correctness check to pass unmodified.

### D3. Impossible numbers
*Signals:* ledger marks IMPOSSIBLE, or Codex reports a gain larger than the gap term allows.
*Action:* do not celebrate. Ask Codex (same session) to find the measurement error: missing
synchronize, async timing, work elided by the compiler, wrong token count, cached KV reuse.
Accept nothing until the number is below SOL and reproducible.

### D4. Rabbit hole
*Signals:* stall detector; rewriting library kernels that already reach ≥ 90% of their
roofline; autotuning sweeps with sub-noise gains; elaborate infrastructure before any gain.
*Action:* stop the line of work. Send the waterfall and say: "This term has at most X% left.
The largest bar is Y. Abandon the current approach; next task is Y." Record the abandoned
approach in `sre/STATE.md` under *banned approaches* with the reason.

### D5. Scope creep
*Signals:* unrelated refactors, renames, formatting churn, new abstraction layers, new
frameworks or dependencies.
*Action:* reject; ask for the minimal diff restricted to the allowed files. Unrelated cleanups
can be a separate, explicitly approved task.

### D6. Fix-forward spiral
*Signals:* correctness broken after the first run and still broken after one follow-up; each
follow-up adds more changes.
*Action:* revert to last green commit. New session. The task includes what was tried and
why it failed ("Previous attempt fused X but broke Y because Z; avoid that").

### D7. Premature victory / vague reporting
*Signals:* "Optimized!", no numbers, numbers from a modified harness, "should be faster".
*Action:* ignore the claim, measure yourself. If the report format is missing, resume and
require it.

### D8. Environment thrash
*Signals:* driver/toolkit/torch upgrades, new pip installs, changed compiler flags globally,
edits to lockfiles.
*Action:* revert the environment change, re-pin. The environment is part of the contract.

### D9. Hypothesis abandonment
*Signals:* Codex implements something other than the requested hypothesis because it
"found a better idea".
*Action:* if the alternative is promising, record it as a candidate hypothesis; do not
accept it under the current task. Re-dispatch the original hypothesis or, if the user
agrees the alternative is better, dispatch it as its own task with its own prediction.

### D10. Silent partial work
*Signals:* stubs, `NotImplementedError`, disabled code paths behind flags defaulting off,
fallbacks that hide the new kernel never being used.
*Action:* verify via profiler that the new kernel actually runs in the benchmark. Resume with
the evidence (kernel name absent from the trace).

## 3. Writing the correction

Use `templates/codex_correction.md`. Principles:

- **Evidence first:** the diff hunk, the guard output, the ledger row, the profiler line.
- **One instruction:** exactly what to undo and what to do instead.
- **Restate the invariants:** protected files, contract, scope.
- **No moralizing, no long essays.** Precise and short.
- **Re-dispatch immediately** in the same turn.

## 4. Escalation ladder

1. Resume with correction (same session).
2. Second correction on the same task → resume once more with tightened scope.
3. Third → revert, new session with the lesson embedded.
4. Same failure across two fresh sessions → the hypothesis or the task decomposition is
   wrong. Re-derive the waterfall yourself; split or replace the task.
5. Only if a decision needs a contract change or a risky permission, ask the user.
