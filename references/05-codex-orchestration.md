# 05: Dispatch, acceptance, and recovery

The supervisor chooses each hypothesis, owns the workload contract, and independently measures
and judges the candidate. The implementation worker receives one task at a time.

## Choose a worker

| Available host | Dispatch | Follow-up |
|---|---|---|
| Codex with native subagents | Spawn a worker and wait for it to report | Send the correction to the same worker |
| Claude Code or Codex with a local CLI | Run `codex_task.sh` | Run `codex_correct.sh` with the recorded run directory |

The CLI path requires a working `codex --version` and authenticated `codex login status`.
Use `SRE_CODEX_BIN`, `SRE_CODEX_MODEL`, and `SRE_CODEX_EFFORT` to configure the CLI worker.
Native workers inherit the host session's permissions. Both paths use the same task template,
candidate isolation, guard, and measurement protocol.

## Prepare a candidate

1. Start at the last accepted commit. If there are existing user changes, preserve them and
   make a separate Git worktree from the accepted commit.
2. Resolve absolute paths. In the examples below, `SRE_SKILL_DIR` is the installed skill,
   `SRE_ENGINE_REPO` is the supervisor's inference checkout, and `SRE_CANDIDATE` is the worker's
   worktree. Keep task files and run logs in the supervisor checkout.
3. Fill in [the task template](../templates/codex_task.md) with the pinned base, single hypothesis,
   predicted mechanism and gain, editable paths, protected paths, and quick check commands.

```bash
git -C "$SRE_ENGINE_REPO" worktree add --detach "$SRE_CANDIDATE" "$SRE_BASE_SHA"
```

Set `SRE_BASE_SHA` to the full commit of the supervisor-approved engine snapshot.

For a native worker, pass the completed task, the absolute candidate directory, and the base
SHA. Ask it to return the changed paths, quick-check evidence, and blockers, then wait for it
to finish. Keep a record of its task, result, and worker identifier under the run directory.

For a CLI worker, dispatch from any directory using the same candidate and task:

```bash
"$SRE_SKILL_DIR/scripts/codex_task.sh" \
  --repo "$SRE_CANDIDATE" \
  --task "$SRE_ENGINE_REPO/sre/tasks/007-fuse-rmsnorm-qkv.md" \
  --runs-dir "$SRE_ENGINE_REPO/sre/runs" \
  --timeout-min 90
```

The wrapper prints an absolute `RUN_DIR`. It records the task, `events.jsonl`, `stderr.log`,
the final message, and metadata including the base and session identifier. `diff.patch` and
`diffstat.txt` cover tracked changes, and `untracked.json` lists new paths for review.
The guard inspects relevant new files without changing the Git index.

## GPU access and permissions

The CLI wrapper defaults to `workspace-write`. Check accelerator visibility from the actual
worker environment before dispatch. If device visibility is restricted, keep the worker on
code edits and have the supervisor run builds, profiles, or measurements with device access.
Where the user explicitly authorizes full filesystem access on an isolated GPU runner, use
`--sandbox danger-full-access`. This mode bypasses sandbox and approval prompts. Choose the
worker mode at dispatch. A follow-up uses the recorded mode, and a different mode requires a
fresh worker. A Git worktree provides a known base and edit attribution.

## Independently accept or return the candidate

After the worker stops writing, review the candidate against the pinned base. Use the
supervisor checkout's protected manifest and the task's actual allow list:

```bash
python3 "$SRE_SKILL_DIR/scripts/guard.py" --repo "$SRE_CANDIDATE" \
  --base "$SRE_BASE_SHA" --protect-file "$SRE_ENGINE_REPO/sre/protected.txt" \
  --allow 'engine/kernels/*' --max-lines 600 --json
```

- Require the JSON verdict `CLEAN`. A `REVIEW` finding has a successful process exit code.
  Inspect it and rerun with a specific `--ignore-flag` if the contract justifies that exception.
  A `VIOLATION` requires correction or rejection.
- Inspect the diff and new files. Capture the exact candidate as a commit or a portable patch
  including new files. If the worker returns a patch from another machine, apply it to a
  supervisor-owned worktree at the pinned base and rerun the guard there.
- Run the frozen correctness check, then the authoritative benchmark and SLO checks against
  the captured candidate. Keep other jobs off the device. Save the raw results, commands,
  hardware and model revisions, and repetition counts with the run. [Measurement protocol](07-measurement-protocol.md)
- Record the candidate commit, repetitions, noise, and run identifier with `ledger.py`.
  Submit `--verdict accept` only after the checks pass. A ledger refusal leaves acceptance
  unwritten. Acceptance also requires a valid baseline with recorded repetitions or an
  observed `--noise` floor set at initialization. Record a rejection or probe when needed.
- Promote the tested candidate and update the waterfall and `sre/STATE.md`. If promotion
  fails, reconcile the tested commit and acceptance record before dispatching another worker.

A worker's final report is a claim. Check that its stated files match the diff, its active
code path matches the profile, and its reported measurements reproduce with the frozen harness.

## Correct and recover

Use the same worker for a bounded correction to the same hypothesis. Cite the concrete
failure, restate the pinned base and scope, and rerun the full acceptance sequence after the
worker returns.

```bash
"$SRE_SKILL_DIR/scripts/codex_correct.sh" \
  --run "$SRE_RUN_DIR" --message "$SRE_ENGINE_REPO/sre/tasks/007-correction-1.md"
```

Each CLI follow-up records its own events and diff against the original base in
`followup-NN/`. On timeout or failure, inspect the artifacts and candidate worktree before
retrying. Exit status 124 indicates timeout. If a worker/session is unavailable or a correction
requires a fresh approach, send a new worker the last accepted base, the diagnosis, and the
relevant state. Preserve the original artifacts. Keep a single active writer per worktree and
serialize authoritative measurements on each GPU.

On context restart, read `sre/STATE.md` and the ledger. Verify the accepted commit, current
candidate base and worker status, and remaining iteration/time budget before resuming.
