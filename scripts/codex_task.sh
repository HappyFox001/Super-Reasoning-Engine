#!/usr/bin/env bash
# Dispatch one optimization task to the local Codex CLI and record the run.
#
#   codex_task.sh --repo <engine-repo> --task <task.md>
#                 [--model <m>] [--effort <level>]
#                 [--sandbox danger-full-access|workspace-write|read-only] (default: workspace-write)
#                 [--runs-dir <dir>] [--timeout-min <minutes>] [--name <slug>]
#
# Env defaults: SRE_CODEX_MODEL, SRE_CODEX_EFFORT, SRE_CODEX_SANDBOX, SRE_CODEX_BIN.
# Relative paths are resolved against the caller's current directory.
# Prints the absolute run directory on the last line of stdout ("RUN_DIR=<path>").

# Shared with codex_correct.sh. These helpers cannot launch a task when sourced.
resolve_codex() {
  local executable
  executable="$(command -v -- "${SRE_CODEX_BIN:-codex}")" || {
    echo "codex CLI not found (${SRE_CODEX_BIN:-codex})" >&2
    return 69
  }
  case "$executable" in
    /*) ;;
    *) executable="$PWD/$executable" ;;
  esac
  if [[ ! -x "$executable" || -d "$executable" ]]; then
    echo "codex CLI is not executable: $executable" >&2
    return 69
  fi
  printf '%s\n' "$executable"
}

validate_timeout() {
  if [[ -z "$1" ]]; then return 0; fi
  if ! [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! python3 -c '
import sys
sys.exit(not 0 < float(sys.argv[1]) * 60 < float("inf"))
' "$1"; then
    echo "--timeout-min must be a positive number of minutes" >&2
    return 64
  fi
}

run_codex() {
  local repo="$1" log_dir="$2" prompt="$3" timeout_min="$4"
  local runner="" received_signal="" status=0
  shift 4
  if [[ -z "$timeout_min" ]]; then
    (cd -- "$repo" && "$@") < "$prompt" > "$log_dir/events.jsonl" 2> "$log_dir/stderr.log"
    return
  fi

  # Wait on an asynchronous supervisor so Bash handles signals to the wrapper immediately.
  trap 'received_signal="${received_signal:-TERM}"; if [[ -n "$runner" ]]; then kill -TERM "$runner" 2>/dev/null || true; fi' TERM
  trap 'received_signal="${received_signal:-INT}"; if [[ -n "$runner" ]]; then kill -INT "$runner" 2>/dev/null || true; fi' INT
  python3 - "$repo" "$log_dir" "$prompt" "$timeout_min" "$@" <<'PY' &
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

repo, log_dir, prompt, minutes, *command = sys.argv[1:]
seconds = float(minutes) * 60
stop_signal = None


def signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def request_stop(signum, _frame):
    global stop_signal
    if stop_signal is None:
        stop_signal = signum


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
output = Path(log_dir)
events_path = output / "events.jsonl"
stderr_path = output / "stderr.log"
with open(prompt, "rb") as source, events_path.open("wb") as events, stderr_path.open("wb") as errors:
    process = subprocess.Popen(
        command, cwd=repo, stdin=source, stdout=events, stderr=errors, start_new_session=True
    )
    deadline = time.monotonic() + seconds
    while stop_signal is None and process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue

    if stop_signal is None and process.returncode is not None:
        status = process.returncode
    else:
        signal_group(process, stop_signal or signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        signal_group(process, signal.SIGKILL)
        process.wait()
        status = 128 + stop_signal if stop_signal is not None else 124
sys.exit(status if status >= 0 else 128 - status)
PY
  runner=$!
  if [[ -n "$received_signal" ]]; then kill -s "$received_signal" "$runner" 2>/dev/null || true; fi
  wait "$runner" || status=$?
  if [[ -n "$received_signal" ]]; then
    # A trapped signal interrupts wait before the supervisor has finished cleaning up.
    while kill -0 "$runner" 2>/dev/null; do wait "$runner" 2>/dev/null || true; done
    if [[ "$received_signal" == "INT" ]]; then status=130; else status=143; fi
  fi
  trap - TERM INT
  return "$status"
}

record_git_evidence() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

repo, base, output, run_dir = sys.argv[1:]
git = ["git", "--no-optional-locks", "-C", repo]
output = Path(output)

if base != "none":
    for filename, options in (
        ("diff.patch", ["--binary"]),
        ("diffstat.txt", ["--stat"]),
    ):
        with (output / filename).open("wb") as target:
            subprocess.run(
                git + ["diff", "--no-ext-diff", "--no-textconv", *options, base, "--"],
                stdout=target,
                check=True,
            )

root = subprocess.run(git + ["rev-parse", "--show-toplevel"], capture_output=True, check=False)
if root.returncode == 0:
    git_root = Path(root.stdout.decode("utf-8", "surrogateescape").rstrip("\n"))
    files = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(git_root), "ls-files", "--others", "--exclude-standard", "-z"]
    )
    untracked = []
    for raw_name in files.split(b"\0"):
        if raw_name:
            name = raw_name.decode("utf-8", "surrogateescape")
            if Path(run_dir) not in (git_root / name).parents:
                untracked.append(name)
    (output / "untracked.json").write_text(json.dumps(untracked, indent=2) + "\n", encoding="utf-8")
PY
}

write_run_metadata() {
  python3 - "$@" <<'PY'
import json
from pathlib import Path
import sys

out, repo, base, model, effort, sandbox, code, start, end, session = sys.argv[1:]
data = {
    "repo": repo,
    "base_commit": base,
    "model": model or "config-default",
    "effort": effort or "config-default",
    "sandbox": sandbox,
    "exit_code": int(code),
    "start": int(start),
    "end": int(end),
    "duration_s": int(end) - int(start),
    "session_id": session,
}
(Path(out) / "meta.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

dispatch_task() {
  local CODEX MODEL EFFORT SANDBOX
  local REPO="" TASK="" RUNS_DIR="" TIMEOUT_MIN="" NAME=""
  local RUN BASE RC START END SID
  local -a ARGS
  MODEL="${SRE_CODEX_MODEL:-}"
  EFFORT="${SRE_CODEX_EFFORT:-}"
  SANDBOX="${SRE_CODEX_SANDBOX:-workspace-write}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) sed -n '2,11p' "$0"; return 0 ;;
      --repo|--task|--model|--effort|--sandbox|--runs-dir|--timeout-min|--name)
        if [[ $# -lt 2 || -z "$2" ]]; then echo "missing value for $1" >&2; return 64; fi ;;
      *) echo "unknown arg: $1" >&2; return 64 ;;
    esac
    case "$1" in
      --repo) REPO="$2" ;;
      --task) TASK="$2" ;;
      --model) MODEL="$2" ;;
      --effort) EFFORT="$2" ;;
      --sandbox) SANDBOX="$2" ;;
      --runs-dir) RUNS_DIR="$2" ;;
      --timeout-min) TIMEOUT_MIN="$2" ;;
      --name) NAME="$2" ;;
    esac
    shift 2
  done
  [[ -n "$REPO" && -n "$TASK" ]] || { echo "need --repo and --task" >&2; return 64; }
  [[ -d "$REPO" ]] || { echo "repo directory not found: $REPO" >&2; return 66; }
  [[ -f "$TASK" ]] || { echo "task file not found: $TASK" >&2; return 66; }
  validate_timeout "$TIMEOUT_MIN"
  case "$SANDBOX" in
    danger-full-access) echo "note: Codex runs with full filesystem and device access. Use a dedicated worktree or machine." >&2 ;;
    workspace-write|read-only) ;;
    *) echo "bad --sandbox $SANDBOX" >&2; return 64 ;;
  esac
  CODEX="$(resolve_codex)"
  REPO="$(cd -- "$REPO" && pwd -P)"

  NAME="${NAME:-$(basename -- "$TASK" .md)}"
  case "$NAME" in
    */*|.|..) echo "--name must be a single path component" >&2; return 64 ;;
  esac
  BASE="$(git --no-optional-locks -C "$REPO" rev-parse --verify HEAD 2>/dev/null || echo none)"
  if [[ "$BASE" != "none" && -n "$(git --no-optional-locks -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
    echo "warning: working tree has pre-existing tracked changes that will appear in the diff" >&2
  fi
  RUNS_DIR="${RUNS_DIR:-$REPO/sre/runs}"
  mkdir -p -- "$RUNS_DIR"
  RUNS_DIR="$(cd -- "$RUNS_DIR" && pwd -P)"
  RUN="$(mktemp -d "$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-$NAME.XXXXXX")"
  cp -- "$TASK" "$RUN/task.md"

  ARGS=(exec --json -C "$REPO" --skip-git-repo-check -o "$RUN/last_message.md")
  if [[ "$SANDBOX" == "danger-full-access" ]]; then
    ARGS+=(--dangerously-bypass-approvals-and-sandbox)
  else
    ARGS+=(-s "$SANDBOX")
  fi
  if [[ -n "$MODEL" ]]; then ARGS+=(-m "$MODEL"); fi
  if [[ -n "$EFFORT" ]]; then ARGS+=(-c "model_reasoning_effort=\"$EFFORT\""); fi
  ARGS+=(-)

  START="$(date +%s)"
  RC=0
  run_codex "$REPO" "$RUN" "$RUN/task.md" "$TIMEOUT_MIN" "$CODEX" "${ARGS[@]}" || RC=$?
  END="$(date +%s)"
  SID="$(python3 - "$RUN/events.jsonl" <<'PY'
import json
from pathlib import Path
import re
import sys

events_path = Path(sys.argv[1])
if not events_path.is_file():
    sys.exit(0)
with events_path.open(encoding="utf-8", errors="replace") as events:
    for line in events:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", thread_id
        ):
            print(thread_id)
            break
PY
)"
  if [[ -z "$SID" ]]; then
    echo "no valid Codex session ID in $RUN/events.jsonl; start a new task to retry" >&2
    if [[ "$RC" -eq 0 ]]; then RC=65; fi
  fi
  printf '%s\n' "$SID" > "$RUN/session_id"
  if ! record_git_evidence "$REPO" "$BASE" "$RUN" "$RUN"; then
    echo "could not record complete Git evidence in $RUN" >&2
    if [[ "$RC" -eq 0 ]]; then RC=74; fi
  fi
  write_run_metadata "$RUN" "$REPO" "$BASE" "$MODEL" "$EFFORT" "$SANDBOX" "$RC" "$START" "$END" "$SID"

  echo "codex exit=$RC  duration=$((END - START))s  session=$SID"
  if [[ -s "$RUN/diffstat.txt" ]]; then tail -n 1 "$RUN/diffstat.txt"; fi
  if [[ "$RC" -eq 124 ]]; then echo "TIMEOUT: Codex reached the requested time limit" >&2; fi
  printf 'RUN_DIR=%s\n' "$RUN"
  return "$RC"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  dispatch_task "$@"
fi
