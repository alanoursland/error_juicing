"""Cross-platform runner for all 18 E4 runs (6 arms x 3 seeds). Windows-safe.

  python src/run_e4_all.py               # sequential
  python src/run_e4_all.py --parallel 2  # two runs share the GPU

Resumable: a killed run loses at most one epoch; rerunning this script skips
completed runs via their checkpoints in seconds. Commit results/e4/*.json
afterwards; figures regenerate via: python src/fig_e4.py
"""

import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ARMS = ["baseline", "weight_decay", "label_smoothing", "logitnorm",
        "focal", "temperature"]
SEEDS = [0, 1, 2]


def run(job):
    arm, seed = job
    cmd = [sys.executable, "src/e4_fixes.py", "--config",
           f"configs/e4/{arm}.yaml", "--seed", str(seed), "--out", "results/e4/"]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()
    jobs = list(itertools.product(ARMS, SEEDS))
    if args.parallel <= 1:
        codes = [run(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            codes = list(ex.map(run, jobs))
    bad = [j for j, c in zip(jobs, codes) if c != 0]
    if bad:
        print(f"FAILED: {bad}")
        sys.exit(1)
    print("all 18 runs complete")


if __name__ == "__main__":
    main()
