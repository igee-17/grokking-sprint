"""Export a clean workspace for an agent run.

  python scripts/make_agent_workspace.py --version v2 --out ../agent-run-v2

The workspace contains ONLY what the agent should see:
  problem.md, solution.py (starter), requirements.txt, and a grader that can
  score PUBLIC ops. grader/hidden_ops.py, reference/, redteam/ and results/
  are never copied. The workspace is a git repo with the starter committed,
  so `git diff` afterwards shows exactly what the agent changed.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "grader"))
import harness  # noqa: E402

GRADER_FILES = ["grade.py", "runner.py", "static_check.py", "harness.py"]


def render_problem(version):
    cfg = harness.VERSIONS[version]
    steps, tr = cfg["steps"], round(cfg["train_frac"] * 100)
    text = open(os.path.join(ROOT, "task", "problem.md")).read()
    subs = [
        ("1000 full-batch training steps", f"{steps} full-batch training steps"),
        ("held-out pairs at step\n1000", f"held-out pairs at step\n{steps}"),
        ("A random 40% of all", f"A random {tr}% of all"),
        ("the remaining 60% is", f"the remaining {100 - tr}% is"),
        ("runs **exactly 1000 steps**", f"runs **exactly {steps} steps**"),
        ("after\nstep 1000.", f"after\nstep {steps}."),
        ("Edit `starter/solution.py`", "Edit `solution.py`"),
        ("python grader/grade.py task/starter/solution.py --public",
         f"python grader/grade.py solution.py --public --version {version}"),
    ]
    for old, new in subs:
        if old not in text:
            raise SystemExit(f"problem.md template drifted; can't find: {old!r}")
        text = text.replace(old, new)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=sorted(harness.VERSIONS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; pick a new directory per run")
    os.makedirs(os.path.join(out, "grader"))

    open(os.path.join(out, "problem.md"), "w").write(render_problem(a.version))
    shutil.copy(os.path.join(ROOT, "task", "starter", "solution.py"), os.path.join(out, "solution.py"))
    shutil.copy(os.path.join(ROOT, "requirements.txt"), out)
    for f in GRADER_FILES:
        src = open(os.path.join(ROOT, "grader", f)).read()
        if f == "grade.py":
            src = src.replace('default="v1"', f'default="{a.version}"')
        open(os.path.join(out, "grader", f), "w").write(src)

    leaked = [f for f in ("hidden_ops.py",) if os.path.exists(os.path.join(out, "grader", f))]
    assert not leaked, f"hidden files leaked: {leaked}"

    subprocess.run(["git", "init", "-q"], cwd=out, check=True)
    subprocess.run(["git", "add", "-A"], cwd=out, check=True)
    subprocess.run(["git", "-c", "user.name=grader", "-c", "user.email=grader@localhost",
                    "commit", "-qm", "starter"], cwd=out, check=True)
    print(f"agent workspace ({a.version}) -> {out}")


if __name__ == "__main__":
    main()
