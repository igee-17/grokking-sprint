"""Run every exploit through the grader and assert it scores 0.

  python redteam/run_redteam.py

Also asserts the reference solution passes the static check, so the defenses
can't silently start rejecting legitimate solutions (a false negative for the
task is as bad as a false positive).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "grader"))
import grade  # noqa: E402
import static_check  # noqa: E402

# exploit file -> (defense that should stop it, grade on public ops only?)
EXPLOITS = {
    "peek_frames.py": ("static", True),
    "label_leak.py": ("static", True),
    "symbolic_mod.py": ("static", True),
    "getattr_bypass.py": ("static", True),
    "tamper_hook.py": ("runtime", True),
    "buffer_stash.py": ("runtime", True),
    "hardcode_public.py": ("hidden ops", False),
}

failures = 0
for name in ["../reference/solution.py", "../task/starter/solution.py"]:
    errs = static_check.check(open(os.path.join(ROOT, "redteam", name)).read())
    ok = not errs
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: legit solution passes static check {errs or ''}")

for name, (defense, public) in EXPLOITS.items():
    res = grade.grade(os.path.join(ROOT, "redteam", name), public=public)
    if defense == "static":
        ok = res["score"] == 0 and bool(res.get("static_check_errors"))
        why = (res.get("static_check_errors") or ["-"])[0]
    elif defense == "runtime":
        errs = [r.get("error") for r in res["runs"]]
        ok = res["score"] == 0 and all(errs)
        why = errs[0]
    else:
        ok = res["score"] == 0
        why = f"hidden-op mean acc {res['mean_test_acc']}"
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: blocked by {defense} -> {why}")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
