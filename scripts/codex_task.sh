#!/usr/bin/env bash
# Dispatch one optimization task to the local Codex CLI and record everything.
#
#   codex_task.sh --repo <engine-repo> --task <task.md>
#                 [--model <m>] [--effort <low|medium|high|xhigh|ultra>]
#                 [--sandbox danger-full-access|workspace-write|read-only]   (default: danger-full-access)
#                 [--runs-dir <dir>] [--timeout-min <minutes>] [--name <slug>]
#
# Env defaults: SRE_CODEX_MODEL, SRE_CODEX_EFFORT, SRE_CODEX_SANDBOX, SRE_CODEX_BIN.
# Prints the run directory on the last line of stdout (machine-readable: "RUN_DIR=<path>").
set -euo pipefail

CODEX="${SRE_CODEX_BIN:-codex}"
MODEL="${SRE_CODEX_MODEL:-}"
EFFORT="${SRE_CODEX_EFFORT:-}"
SANDBOX="${SRE_CODEX_SANDBOX:-danger-full-access}"  # GPU work needs device access; see references/05 §4
REPO="" TASK="" RUNS_DIR="" TIMEOUT_MIN="" NAME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --sandbox) SANDBOX="$2"; shift 2 ;;
    --runs-dir) RUNS_DIR="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done
[ -n "$REPO" ] && [ -n "$TASK" ] || { echo "need --repo and --task" >&2; exit 64; }
command -v "$CODEX" >/dev/null || { echo "codex CLI not found ($CODEX)" >&2; exit 69; }
REPO="$(cd "$REPO" && pwd)"
[ -f "$TASK" ] || { echo "task file not found: $TASK" >&2; exit 66; }
case "$SANDBOX" in danger-full-access|workspace-write|read-only) ;; *) echo "bad --sandbox $SANDBOX" >&2; exit 64 ;; esac
if [ "$SANDBOX" = "danger-full-access" ]; then
  echo "note: Codex runs WITHOUT a sandbox in $REPO (full disk + GPU access). Use a dedicated branch/machine." >&2
fi

[ -n "$NAME" ] || NAME="$(basename "$TASK" .md)"
RUNS_DIR="${RUNS_DIR:-$REPO/sre/runs}"
RUN="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-$NAME"
mkdir -p "$RUN"
cp "$TASK" "$RUN/task.md"

BASE="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)"
if [ "$BASE" != "none" ] && [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "warning: working tree not clean at dispatch; diff attribution will include pre-existing changes" >&2
fi

ARGS=(exec --json -C "$REPO" --skip-git-repo-check -o "$RUN/last_message.md")
if [ "$SANDBOX" = "danger-full-access" ]; then
  ARGS+=(--dangerously-bypass-approvals-and-sandbox)   # no sandbox, no approval prompts (non-interactive)
else
  ARGS+=(-s "$SANDBOX")
fi
[ -n "$MODEL" ] && ARGS+=(-m "$MODEL")
[ -n "$EFFORT" ] && ARGS+=(-c "model_reasoning_effort=\"$EFFORT\"")
ARGS+=(-)

TO=()
if [ -n "$TIMEOUT_MIN" ]; then
  if command -v timeout >/dev/null; then TO=(timeout "$((TIMEOUT_MIN * 60))");
  elif command -v gtimeout >/dev/null; then TO=(gtimeout "$((TIMEOUT_MIN * 60))");
  else echo "warning: no timeout/gtimeout binary; running without a time cap" >&2; fi
fi

START="$(date +%s)"
set +e
"${TO[@]+"${TO[@]}"}" "$CODEX" "${ARGS[@]}" < "$RUN/task.md" > "$RUN/events.jsonl" 2> "$RUN/stderr.log"
RC=$?
set -e
END="$(date +%s)"

python3 - "$RUN/events.jsonl" "$RUN/session_id" <<'PY'
import json, sys
sid = None
def walk(o):
    global sid
    if sid: return
    if isinstance(o, dict):
        for k in ("thread_id", "session_id", "conversation_id"):
            if isinstance(o.get(k), str):
                sid = o[k]; return
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
for line in open(sys.argv[1], errors="replace"):
    try: walk(json.loads(line))
    except Exception: pass
    if sid: break
open(sys.argv[2], "w").write((sid or "") + "\n")
PY

if [ "$BASE" != "none" ]; then
  git -C "$REPO" add -N . >/dev/null 2>&1 || true   # include new files in the diff
  git -C "$REPO" diff "$BASE" > "$RUN/diff.patch" || true
  git -C "$REPO" diff --stat "$BASE" > "$RUN/diffstat.txt" || true
fi

cat > "$RUN/meta.json" <<JSON
{"repo": "$REPO", "base_commit": "$BASE", "model": "${MODEL:-config-default}", "effort": "${EFFORT:-config-default}",
 "sandbox": "$SANDBOX", "exit_code": $RC, "start": $START, "end": $END, "duration_s": $((END - START)),
 "session_id": "$(tr -d '\n' < "$RUN/session_id")"}
JSON

echo "codex exit=$RC  duration=$((END - START))s  session=$(cat "$RUN/session_id")"
[ -f "$RUN/diffstat.txt" ] && tail -n 1 "$RUN/diffstat.txt" || true
[ $RC -eq 124 ] && echo "TIMEOUT: task too large; split it" >&2
echo "RUN_DIR=$RUN"
exit $RC
