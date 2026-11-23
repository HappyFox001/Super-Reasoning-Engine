#!/usr/bin/env bash
# Resume the Codex session of an earlier run with a correction or follow-up.
#
#   codex_correct.sh --run <run-dir> --message <correction.md> [--model <m>] [--effort <e>] [--timeout-min <minutes>]
#
# Creates <run-dir>/followup-NN/ with the prompt, worker logs and Git evidence.
# Uses the original run's sandbox mode and base commit.
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/codex_task.sh"

MODEL="${SRE_CODEX_MODEL:-}"
EFFORT="${SRE_CODEX_EFFORT:-}"
RUN="" MSG="" TIMEOUT_MIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    --run|--message|--model|--effort|--timeout-min)
      if [[ $# -lt 2 || -z "$2" ]]; then echo "missing value for $1" >&2; exit 64; fi ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
  case "$1" in
    --run) RUN="$2" ;;
    --message) MSG="$2" ;;
    --model) MODEL="$2" ;;
    --effort) EFFORT="$2" ;;
    --timeout-min) TIMEOUT_MIN="$2" ;;
  esac
  shift 2
done
[[ -d "$RUN" && -f "$MSG" ]] || { echo "need --run <dir> and --message <file>" >&2; exit 64; }
validate_timeout "$TIMEOUT_MIN"
CODEX="$(resolve_codex)"
RUN="$(cd -- "$RUN" && pwd -P)"

if [[ ! -f "$RUN/session_id" ]]; then
  echo "missing Codex session ID at $RUN/session_id; start a new task instead" >&2
  exit 65
fi
SID="$(cat -- "$RUN/session_id")"
if ! [[ "$SID" =~ ^[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$ ]]; then
  echo "invalid Codex session ID at $RUN/session_id; start a new task instead" >&2
  exit 65
fi

if ! RUN_META="$(python3 - "$RUN/meta.json" "$SID" <<'PY'
import json
from pathlib import Path
import re
import sys

file, session = sys.argv[1:]
try:
    meta = json.loads(Path(file).read_text(encoding="utf-8"))
    repo = Path(meta["repo"])
    base = meta["base_commit"]
    sandbox = meta["sandbox"]
    if not repo.is_absolute() or not repo.is_dir():
        raise ValueError(f"target repo is not an existing absolute directory: {repo}")
    if base != "none" and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base):
        raise ValueError("invalid base commit")
    if sandbox not in ("danger-full-access", "workspace-write", "read-only"):
        raise ValueError("missing or invalid sandbox")
    if meta.get("session_id") != session:
        raise ValueError("session ID does not match session_id file")
except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
    sys.exit(f"cannot resume from {file}: {error}")
print(base)
print(sandbox)
print(repo, end="")
PY
)"; then
  exit 65
fi
BASE="${RUN_META%%$'\n'*}"
RUN_META="${RUN_META#*$'\n'}"
ORIG_SANDBOX="${RUN_META%%$'\n'*}"
REPO="${RUN_META#*$'\n'}"

SANDBOX="${SRE_CODEX_SANDBOX:-$ORIG_SANDBOX}"
if [[ "$SANDBOX" != "$ORIG_SANDBOX" ]]; then
  echo "session was created with sandbox=$ORIG_SANDBOX; start a new task to use $SANDBOX" >&2
  exit 78
fi
if [[ "$BASE" != "none" ]] && ! git --no-optional-locks -C "$REPO" cat-file -e "$BASE^{commit}"; then
  echo "original base commit $BASE is unavailable in $REPO" >&2
  exit 65
fi

LOCK="$RUN/.followup.lock"
if ! mkdir -- "$LOCK"; then
  echo "could not lock $RUN for a follow-up; check for an active correction" >&2
  echo "After a forcibly killed correction, remove $LOCK only after its worker has exited." >&2
  exit 75
fi
trap 'rmdir -- "$LOCK"' EXIT

N=1
while [[ -e "$RUN/$(printf 'followup-%02d' "$N")" ]]; do N=$((N + 1)); done
OUT="$RUN/$(printf 'followup-%02d' "$N")"
mkdir -- "$OUT"
cp -- "$MSG" "$OUT/message.md"
if [[ "$N" -ge 3 ]]; then
  echo "note: follow-up #$N on one task; consider a fresh session per the correction policy" >&2
fi

ARGS=(exec resume "$SID" --json --skip-git-repo-check -o "$OUT/last_message.md")
if [[ "$SANDBOX" == "danger-full-access" ]]; then
  ARGS+=(--dangerously-bypass-approvals-and-sandbox)
else
  ARGS+=(-c "sandbox_mode=\"$SANDBOX\"")
fi
if [[ -n "$MODEL" ]]; then ARGS+=(-m "$MODEL"); fi
if [[ -n "$EFFORT" ]]; then ARGS+=(-c "model_reasoning_effort=\"$EFFORT\""); fi
ARGS+=(-)

START="$(date +%s)"
RC=0
run_codex "$REPO" "$OUT" "$OUT/message.md" "$TIMEOUT_MIN" "$CODEX" "${ARGS[@]}" || RC=$?
END="$(date +%s)"
if ! record_git_evidence "$REPO" "$BASE" "$OUT" "$RUN"; then
  echo "could not record complete Git evidence in $OUT" >&2
  if [[ "$RC" -eq 0 ]]; then RC=74; fi
fi
write_run_metadata "$OUT" "$REPO" "$BASE" "$MODEL" "$EFFORT" "$SANDBOX" "$RC" "$START" "$END" "$SID"

echo "codex resume exit=$RC  session=$SID  followup=$N"
if [[ "$RC" -eq 124 ]]; then echo "TIMEOUT: Codex reached the requested time limit" >&2; fi
printf 'RUN_DIR=%s\n' "$OUT"
exit "$RC"
