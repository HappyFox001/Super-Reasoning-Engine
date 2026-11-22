# 05: Orchestrating the Local Codex CLI

Codex is the implementation engineer. You (the orchestrating agent) are the principal
engineer: you set the hypothesis, review the diff, run the truth, and decide. This file
covers how to call Codex reliably and how to keep it productive for long runs.

## 1. Preconditions

```bash
codex --version                     # CLI installed
codex login status || codex login   # authenticated (user does this interactively)
grep -E '^model|reasoning' ~/.codex/config.toml
```

- **Model:** set `SRE_CODEX_MODEL` to pin a model (e.g. an Astra-class model such as
  `gpt-6.2-astra`); otherwise the user's `config.toml` default is used. Use the strongest
  available reasoning effort for kernel work (`SRE_CODEX_EFFORT`, passed as
  `-c model_reasoning_effort=<level>`); a lower effort is fine for scaffolding tasks.
- **Repo:** the target engine is a git repository with a clean working tree at the last
  accepted commit. Every Codex task starts from a green, committed state.
- **AGENTS.md:** present at the repo root (from `templates/AGENTS.md`), stating the contract,
  protected paths, allowed commands and the report format. Codex loads it automatically.

## 2. The wrappers

### Dispatch a new task

```bash
scripts/codex_task.sh --repo <engine-repo> --task <task.md> \
  [--model <m>] [--effort <e>] [--sandbox workspace-write|danger-full-access] \
  [--runs-dir <engine-repo>/sre/runs] [--timeout-min 90]
```

It runs:

```bash
codex exec --json -C <repo> -s <sandbox> --skip-git-repo-check \
  [-m <model>] [-c model_reasoning_effort=<e>] \
  -o <run>/last_message.md - < <task.md>   > <run>/events.jsonl
```

and writes into `<run>/`: `task.md` (the exact prompt), `events.jsonl` (full event stream),
`last_message.md` (Codex's final report), `session_id`, `meta.json` (start/end, exit code,
base commit), `diff.patch` (the working-tree diff vs the base commit) and `diffstat.txt`.

### Correct or follow up in the same session

```bash
scripts/codex_correct.sh --run <run-dir> --message <correction.md>
```

It resumes the recorded session (`codex exec resume <session_id> -`), so Codex keeps its
understanding of the code. Each follow-up appends a numbered sub-run (`followup-01/`, …) with
its own events, last message and diff.

### When to use which

| Situation | Action |
|---|---|
| New hypothesis | new session (`codex_task.sh`) |
| Same hypothesis, needs a fix or a measured follow-up | resume (`codex_correct.sh`) |
| Drift correction, first or second time | resume with a correction |
| Third correction on the same task, or a fix-forward spiral | revert, then **new session** with the lesson embedded in the task |
| Context grew huge (very long session) | new session with a compact summary |

## 3. Writing the task prompt

Always start from `templates/codex_task.md`. Non-negotiable elements:

1. **Role and stakes:** "You are implementing one optimization in a performance-critical
   inference engine. A separate reviewer will measure your work with a frozen benchmark."
2. **SOL context:** binding resource, achievable SOL, current value and percent of SOL,
   the gap term this task attacks and its maximum payoff.
3. **Single hypothesis** with the *mechanism* (which bytes or microseconds disappear).
4. **Scope:** files Codex may edit. **Protected:** files it must not touch (bench, check,
   contract, AGENTS.md, lockfiles).
5. **Constraints:** precision contract, no new dependencies, no environment changes, no
   changes to shapes/iterations/warmup of any benchmark.
6. **Commands:** build, quick correctness smoke test, quick benchmark (Codex may run these
   to iterate; they are not authoritative).
7. **Acceptance criteria:** correctness within tolerance; expected measurable gain; no
   regression in other phases.
8. **Report format:** what changed (files, mechanism), the self-measured numbers with the
   command used, known risks, and what Codex would try next. Ask Codex to say plainly if
   the hypothesis turned out wrong. A clean negative result is valuable.

Keep prompts concrete and short enough to read in one pass; paste the relevant profile
excerpt or ledger rows instead of describing them.

## 4. Sandbox and GPU access

- `workspace-write` (default) lets Codex edit the repo and run commands without network.
  On some Linux setups the sandbox may block GPU device access or the compiler cache; detect
  this from the first task's events (errors opening `/dev/nvidia*`, `CUDA error: no CUDA-
  capable device`).
- Options if GPU access is blocked: (a) Codex builds and runs CPU-side checks only, and *you*
  run all GPU commands; (b) use `danger-full-access` only on a dedicated, disposable machine
  and only with the user's explicit approval. Never choose (b) on your own.
- Network is off by default. Codex must not need it: vendor any dependency ahead of time.

## 5. Cadence for long runs

- Typical task length: 10–60 minutes of Codex time. Use `--timeout-min` to cap it; a timeout
  is a signal the task was too big. Split it.
- Run the wrapper in the background when your harness supports it, and wait for completion
  rather than polling.
- After every Codex run, in order: read `last_message.md`, run `scripts/guard.py`, read the
  diff, run correctness, run the authoritative benchmark, `ledger.py add`, verdict, next task.
- Keep `sre/STATE.md` updated (current best, SOL fraction, open hypotheses, banned
  approaches with reasons). Paste its relevant part into every new task so fresh sessions
  do not repeat dead ends.
- Never run two Codex sessions that write to the same files concurrently. Parallel sessions
  are fine for independent microbenchmarks or for separate git worktrees (`codex exec
  --worktree`), each verified separately before merging.

## 6. Reading Codex's output critically

Codex's final message is a *claim*. Cross-check:

| Claim | Check |
|---|---|
| "X% faster" | your benchmark, your machine, ≥ 5 repeats |
| "numerically identical" | frozen correctness check |
| "only touched kernel files" | `diffstat.txt`, `guard.py` |
| "tests pass" | rerun them; check they were not edited |
| "this is the limit" | compare to SOL; if < target, it is not the limit |
