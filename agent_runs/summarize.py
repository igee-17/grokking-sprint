"""Summarize one agent run: score, effort, and what the agent actually did.

  python agent_runs/summarize.py agent_runs/results/<run>
"""
import json
import os
import sys
from collections import Counter

run = sys.argv[1]
events = []
for line in open(os.path.join(run, "transcript.jsonl")):
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass

tools, grade_calls, edits = Counter(), [], 0
final = {}
for e in events:
    if e.get("type") == "assistant":
        for c in (e.get("message") or {}).get("content", []):
            if c.get("type") == "tool_use":
                tools[c.get("name")] += 1
                cmd = (c.get("input") or {}).get("command", "")
                if "grade.py" in cmd:
                    grade_calls.append(cmd)
                if c.get("name") in ("Edit", "Write", "MultiEdit"):
                    edits += 1
    if e.get("type") == "result":
        final = e

score = json.load(open(os.path.join(run, "hidden_score.json")))
print(f"# {os.path.basename(run)}\n")
print(f"- **Hidden-op score:** {score['score']:.2f} (mean held-out acc {score.get('mean_test_acc', 0)})")
if score.get("static_check_errors"):
    print(f"- **Rejected by static check:** {score['static_check_errors']}")
for r in score.get("runs", []):
    print(f"  - {r['op']} seed {r['seed']}: {r['test_acc']}" + (f"  ERROR {r['error']}" if r.get("error") else ""))
print(f"- **Turns:** {final.get('num_turns', '?')}, **cost (USD):** {final.get('total_cost_usd', '?')}, "
      f"**wall clock:** {open(os.path.join(run, 'wallclock_seconds.txt')).read().strip()} s")
print(f"- **Tool calls:** {dict(tools)}; edits to files: {edits}; public grader runs: {len(grade_calls)}")
notes = os.path.join(run, "NOTES.md")
if os.path.exists(notes):
    print("\n## Agent's own notes\n\n" + open(notes).read().strip())
print("\n## Final diff vs starter\n\n```diff\n" + open(os.path.join(run, "solution.diff")).read().strip() + "\n```")
