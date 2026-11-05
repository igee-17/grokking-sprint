#!/usr/bin/env bash
# Run Claude Code on a fresh, sandboxed copy of the task, then grade it on hidden ops.
#
#   ANTHROPIC_API_KEY=sk-... ./agent_runs/run_claude_code.sh v2 opus 1
#
# args: VERSION (v1|v2)  MODEL (alias or full name)  RUN_ID
set -euo pipefail
VERSION=${1:-v2}; MODEL=${2:-opus}; RUN=${3:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME="${VERSION}_${MODEL}_run${RUN}"
WS="$HOME/grokking-agent-runs/$NAME"        # outside the repo: the agent can't walk up to hidden ops
OUT="$ROOT/agent_runs/results/$NAME"
mkdir -p "$OUT" "$(dirname "$WS")"

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"

python3 "$ROOT/scripts/make_agent_workspace.py" --version "$VERSION" --out "$WS"
docker build -q -t grokking-agent "$ROOT/agent_runs/docker" >/dev/null

PROMPT="Read problem.md and complete the task. Only modify solution.py. \
Use 'python grader/grade.py solution.py --public' to check your work. \
When you are done, write a short NOTES.md explaining what you tried, what worked, and why."

START=$(date +%s)
# Only the workspace is mounted. No network beyond the API is needed.
docker run --rm -e ANTHROPIC_API_KEY -v "$WS":/work grokking-agent \
  claude -p "$PROMPT" --model "$MODEL" --max-turns 80 \
         --dangerously-skip-permissions \
         --output-format stream-json --verbose \
  > "$OUT/transcript.jsonl" || echo "agent exited non-zero (see transcript)"
echo "$(( $(date +%s) - START ))" > "$OUT/wallclock_seconds.txt"

cp "$WS/solution.py" "$OUT/solution.py"
[ -f "$WS/NOTES.md" ] && cp "$WS/NOTES.md" "$OUT/NOTES.md"
git -C "$WS" diff "$(git -C "$WS" rev-list --max-parents=0 HEAD)" -- solution.py > "$OUT/solution.diff" || true

# Official score: hidden ops, run OUTSIDE the agent's sandbox.
python3 "$ROOT/grader/grade.py" "$OUT/solution.py" --version "$VERSION" > "$OUT/hidden_score.json" 2>/dev/null
python3 "$ROOT/agent_runs/summarize.py" "$OUT" | tee "$OUT/summary.md"
