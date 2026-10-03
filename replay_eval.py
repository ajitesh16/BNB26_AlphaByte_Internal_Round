"""The CAUSAL CHECK: does patching the step Black Box blames actually fix the run?

  python replay_eval.py data_mock.db

For every failed test run we patch each step in turn and replay. That tells us exactly
which steps, when fixed, flip the run to success. Then we ask: how often does each
method (Black Box, random, last step, first error) blame a step that really fixes it?
"""
import json
import random
import sys

from diagnose import Diagnoser, baseline_scores, load_runs, split_clean_failed
from replay import Patch, ReplaySession


def order(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def attempts(flips, ordering):
    """How many replays until one fixes the run, following this method's ranking."""
    for pos, i in enumerate(ordering, 1):
        if flips[i]:
            return pos
    return len(flips)


def main(db, seed=0):
    runs = load_runs(db)
    rng = random.Random(seed)
    rng.shuffle(runs)                                    # same split as evaluate.py
    cut = int(0.7 * len(runs))
    clean_tr, failed_tr = split_clean_failed(runs[:cut])
    _, failed_te = split_clean_failed(runs[cut:])
    print(f"Training on {len(clean_tr)} clean + {len(failed_tr)} failed runs; "
          f"testing on {len(failed_te)} failed runs...")
    model = Diagnoser("gbm").fit(clean_tr, failed_tr)

    acc = {k: {"fixed": 0, "attempts": 0.0} for k in
           ["Random step", "Always blame last step", "First logged error", "Black Box", "Oracle (true culprit)"]}
    eff = {"steps_saved": 0.0, "tokens_saved": 0.0, "executed": 0.0}

    for run in failed_te:
        sess = ReplaySession(run)
        n = len(run["steps"])
        branches = [sess.fork(Patch(i, "fix")) for i in range(n)]
        flips = [b["success"] for b in branches]
        m = sum(flips)
        bb = order([s["score"] for s in model.score_run(run)])
        orders = {"Always blame last step": order(baseline_scores("last_step", run)),
                  "First logged error": order(baseline_scores("first_error", run)),
                  "Black Box": bb,
                  "Oracle (true culprit)": [run["culprit_step"]]}
        for name, o in orders.items():
            acc[name]["fixed"] += flips[o[0]]
            acc[name]["attempts"] += attempts(flips, o) if name != "Oracle (true culprit)" else 1
        acc["Random step"]["fixed"] += m / n                           # exact expectation
        acc["Random step"]["attempts"] += (n + 1) / (m + 1) if m else n
        top = branches[bb[0]]
        eff["steps_saved"] += top["steps_saved_pct"]
        eff["tokens_saved"] += top["tokens_saved_pct"]
        eff["executed"] += top["executed"]

    k = len(failed_te)
    print("\nCAUSAL CHECK: patch the step each method blames, replay, did the run get fixed?")
    print(f"  {'method':<26}{'fixed on 1st try':>18}{'avg replays needed':>22}")
    out = {"n_failed_runs": k, "methods": {}}
    for name, a in acc.items():
        out["methods"][name] = {"fixed_first_try": a["fixed"] / k, "avg_replays": a["attempts"] / k}
        print(f"  {name:<26}{a['fixed'] / k:>18.1%}{a['attempts'] / k:>22.2f}")

    out["replay_efficiency"] = {"avg_steps_executed": eff["executed"] / k,
                                "steps_saved_pct": eff["steps_saved"] / k,
                                "tokens_saved_pct": eff["tokens_saved"] / k}
    print(f"\nREPLAY EFFICIENCY (applying Black Box's top fix, versus re-running the whole agent)")
    print(f"  steps re-run on average: {eff['executed'] / k:.1f} of 6   "
          f"steps saved: {eff['steps_saved'] / k:.0%}   tokens saved: {eff['tokens_saved'] / k:.0%}")
    with open("results_causal.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results_causal.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_mock.db")