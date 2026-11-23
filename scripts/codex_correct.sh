#!/usr/bin/env bash
# Resume the Codex session of an earlier run with a correction or follow-up.
#
#   codex_correct.sh --run <run-dir> --message <correction.md> [--model <m>] [--effort <e>] [--timeout-min N]
#
# Creates <run-dir>/followup-NN/ with message.md, events.jsonl, last_message.md, diff.patch.
set -euo pipefail

CODEX="${SRE_CODEX_BIN:-codex}"
MODEL="${SRE_CODEX_MODEL:-}"
EFFORT="${SRE_CODEX_EFFORT:-}"
RUN="" MSG="" TIMEOUT_MIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --message) MSG="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done
[ -d "$RUN" ] && [ -f "$MSG" ] || { echo "need --run <dir> and --message <file>" >&2; exit 64; }
SID="$(tr -d '\n' < "$RUN/session_id" 2>/dev/null || true)"
[ -n "$SID" ] || { echo "no session id recorded in $RUN; start a new task instead" >&2; exit 65; }
REPO="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["repo"])' "$RUN/meta.json")"

N=1; while [ -d "$RUN/$(printf 'followup-%02d' $N)" ]; do N=$((N + 1)); done
OUT="$RUN/$(printf 'followup-%02d' $N)"
mkdir -p "$OUT"
cp "$MSG" "$OUT/message.md"
[ $N -ge 3 ] && echo "note: this is follow-up #$N on one task; escalation ladder says revert + fresh session" >&2

BASE="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
ORIG_SANDBOX="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("sandbox","danger-full-access"))' "$RUN/meta.json")"
SANDBOX="${SRE_CODEX_SANDBOX:-$ORIG_SANDBOX}"
if [ "$SANDBOX" != "$ORIG_SANDBOX" ]; then
  # Observed: a resumed session keeps the sandbox it was created with (e.g. still 0 Metal devices).
  echo "session was created with sandbox=$ORIG_SANDBOX; resume cannot change it to $SANDBOX." >&2
  echo "Start a new session with codex_task.sh --sandbox $SANDBOX and embed the lessons in the task." >&2
  exit 78
fi
ARGS=(exec resume "$SID" --json --skip-git-repo-check -o "$OUT/last_message.md")
if [ "$SANDBOX" = "danger-full-access" ]; then
  ARGS+=(--dangerously-bypass-approvals-and-sandbox)
else
  ARGS+=(-c "sandbox_mode=\"$SANDBOX\"")
fi
[ -n "$MODEL" ] && ARGS+=(-m "$MODEL")
[ -n "$EFFORT" ] && ARGS+=(-c "model_reasoning_effort=\"$EFFORT\"")
ARGS+=(-)

TO=()
if [ -n "$TIMEOUT_MIN" ]; then
  if command -v timeout >/dev/null; then TO=(timeout "$((TIMEOUT_MIN * 60))");
  elif command -v gtimeout >/dev/null; then TO=(gtimeout "$((TIMEOUT_MIN * 60))"); fi
fi

set +e
(cd "$REPO" && "${TO[@]+"${TO[@]}"}" "$CODEX" "${ARGS[@]}" < "$OUT/message.md" > "$OUT/events.jsonl" 2> "$OUT/stderr.log")
RC=$?
set -e

if [ "$BASE" != "none" ]; then
  git -C "$REPO" add -N . >/dev/null 2>&1 || true
  git -C "$REPO" diff "$BASE" > "$OUT/diff.patch" || true
  git -C "$REPO" diff --stat "$BASE" > "$OUT/diffstat.txt" || true
fi
echo "codex resume exit=$RC  session=$SID  followup=$N"
echo "RUN_DIR=$OUT"
exit $RC
