"""Generate the labeled dataset.

Runs the agent on many tasks. In about half of them we secretly break one random
step. Every run gets saved by the tracer, with the answer key stored separately.

Examples (PowerShell):
  python generate.py --n 300 --db data_mock.db          # instant, free (mock mode)

  $env:BLACKBOX_MODE = "real"                           # real Claude calls
  $env:ANTHROPIC_API_KEY = "your-key-here"
  python generate.py --n 200 --db data_real.db
"""
import argparse
import collections
import random

from agent import ALL_FAULTS, MODE, run_agent
from tasks import make_tasks
from tracer import Tracer


def summarize(tracer):
    runs = tracer.list_runs()
    clean = [r for r in runs if r["fault_kind"] is None]
    faulty = [r for r in runs if r["fault_kind"] is not None]
    failed = [r for r in faulty if not r["success"]]
    print("\n=== DATASET SUMMARY ===")
    print(f"total runs:                    {len(runs)}")
    print(f"clean runs:                    {len(clean)}  (succeeded: {sum(r['success'] for r in clean)})")
    print(f"runs with an injected fault:   {len(faulty)}")
    print(f"  ...that caused a failure:    {len(failed)}   <- usable labeled failures")
    print(f"  ...that still succeeded:     {len(faulty) - len(failed)}   <- not usable as failures")
    counts = collections.Counter((r["culprit_step"], r["fault_kind"]) for r in failed)
    print("labeled failures by fault:")
    for (step, kind), n in sorted(counts.items()):
        print(f"  step {step}  {kind:<20} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", default="blackbox.db", help="database file to write")
    ap.add_argument("--fault_rate", type=float, default=0.5, help="fraction of runs to break")
    ap.add_argument("--append", action="store_true", help="add to an existing database")
    args = ap.parse_args()

    tracer = Tracer(args.db)
    if tracer.list_runs() and not args.append:
        raise SystemExit(f"{args.db} already has runs. Use a new --db name (keeps mock and "
                         "real data from mixing), or pass --append.")

    rng = random.Random(args.seed)
    print(f"Generating {args.n} runs in '{MODE}' mode -> {args.db}")
    for i, task in enumerate(make_tasks(args.n, seed=args.seed), 1):
        fault = rng.choice(ALL_FAULTS) if rng.random() < args.fault_rate else None
        run_agent(task, tracer, fault)
        if i % 25 == 0:
            print(f"  {i}/{args.n} runs done")
    summarize(tracer)


if __name__ == "__main__":
    main()