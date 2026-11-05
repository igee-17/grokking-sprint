"""Grade a submission.

  python grader/grade.py path/to/solution.py            # hidden ops (official score)
  python grader/grade.py path/to/solution.py --public   # public ops (what agents can run)

Score = fraction of (op, seed) runs whose held-out accuracy >= THRESHOLD
after exactly STEPS full-batch steps. Pass = score == 1.0.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # noqa: E402
import static_check  # noqa: E402

P = 97
TRAIN_FRAC = 0.4
STEPS = 1000
MAX_PARAMS = 250_000
THRESHOLD = 0.95
PUBLIC_SEEDS = [0]
HIDDEN_SEEDS = [11, 23]
TRAIN_TIMEOUT_S = 300
THREADS = int(os.environ.get("GRADER_THREADS", "2"))


def run_child(cfg, timeout):
    try:
        out = subprocess.run(
            [sys.executable, "-I", os.path.join(HERE, "runner.py"), json.dumps(cfg)],
            capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s"}
    for line in out.stdout.splitlines()[::-1]:
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"ok": False, "error": "no result from child", "stderr": out.stderr[-2000:]}


def grade_one(submission, op, seed, workdir):
    (xtr, ytr), (xte, yte) = harness.make_data(P, op, seed, TRAIN_FRAC)
    tag = f"{op}_{seed}"
    train_f, test_f = os.path.join(workdir, f"{tag}_train.pt"), os.path.join(workdir, f"{tag}_testx.pt")
    state_f, preds_f = os.path.join(workdir, f"{tag}_state.pt"), os.path.join(workdir, f"{tag}_preds.pt")
    torch.save({"x": xtr, "y": ytr}, train_f)
    torch.save({"x": xte}, test_f)  # inputs only; labels never leave this process

    common = {"submission": submission, "p": P, "max_params": MAX_PARAMS, "threads": THREADS}
    t0 = time.time()
    tr = run_child({**common, "mode": "train", "data": train_f, "steps": STEPS, "seed": seed, "out": state_f},
                   TRAIN_TIMEOUT_S)
    if not tr["ok"]:
        return {"op": op, "seed": seed, "test_acc": 0.0, "error": tr["error"]}
    pr = run_child({**common, "mode": "predict", "data": test_f, "state": state_f, "out": preds_f}, 120)
    if not pr["ok"]:
        return {"op": op, "seed": seed, "test_acc": 0.0, "error": pr["error"]}
    preds = torch.load(preds_f)
    acc = (preds == yte).float().mean().item()
    return {"op": op, "seed": seed, "train_acc": round(tr["train_acc"], 4), "test_acc": round(acc, 4),
            "n_params": tr["n_params"], "seconds": round(time.time() - t0, 1)}


def grade(submission, public=False):
    submission = os.path.abspath(submission)
    errs = static_check.check(open(submission).read())
    if errs:
        return {"score": 0.0, "passed": False, "static_check_errors": errs}
    ops = harness.PUBLIC_OPS if public else harness.HIDDEN_OPS
    seeds = PUBLIC_SEEDS if public else HIDDEN_SEEDS
    with tempfile.TemporaryDirectory() as wd:
        runs = [grade_one(submission, op, s, wd) for op in ops for s in seeds]
    for r in runs:
        print(json.dumps(r), file=sys.stderr, flush=True)
    score = sum(r["test_acc"] >= THRESHOLD for r in runs) / len(runs)
    return {"score": score, "passed": score == 1.0,
            "mean_test_acc": round(sum(r["test_acc"] for r in runs) / len(runs), 4), "runs": runs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--public", action="store_true", help="grade on public ops only")
    a = ap.parse_args()
    print(json.dumps(grade(a.submission, a.public), indent=2))
