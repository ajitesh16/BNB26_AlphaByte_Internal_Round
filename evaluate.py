"""Measure how well Black Box finds the culprit step, versus simple baselines.

  python evaluate.py data_mock.db

Experiment 1: train on 70% of runs, test on the other 30%.
Experiment 2: leave one FAULT TYPE out of training entirely, then see if the model
              can still find that kind of fault (generalization to unseen failures).
"""
import json
import random
import sys
import time

from diagnose import (Diagnoser, baseline_scores, evaluate, evaluate_random,
                      load_runs, split_clean_failed)


def row(name, m):
    print(f"  {name:<28} top-1 {m['top1']:>6.1%}   top-3 {m['top3']:>6.1%}   "
          f"MRR {m['mrr']:.3f}   AUROC {m['auroc']:.3f}")


def main(db, seed=0):
    runs = load_runs(db)
    rng = random.Random(seed)
    rng.shuffle(runs)
    cut = int(0.7 * len(runs))
    train, test = runs[:cut], runs[cut:]
    clean_tr, failed_tr = split_clean_failed(train)
    _, failed_te = split_clean_failed(test)
    print(f"{len(runs)} runs: train has {len(clean_tr)} clean + {len(failed_tr)} failed; "
          f"test has {len(failed_te)} failed runs to diagnose\n")
    results = {"split": {}, "unseen_fault_types": {}}

    # ---------------- Experiment 1: standard split
    print("EXPERIMENT 1: find the culprit step in failed runs the model has never seen")
    t0 = time.time()
    models = {"Black Box (boosted trees)": Diagnoser("gbm").fit(clean_tr, failed_tr),
              "Black Box (logistic)": Diagnoser("logistic").fit(clean_tr, failed_tr)}
    print(f"  (mined + trained in {time.time() - t0:.1f}s)")
    table = {"Random guess": evaluate_random(failed_te),
             "Always blame last step": evaluate(lambda r: baseline_scores("last_step", r), failed_te),
             "First logged error": evaluate(lambda r: baseline_scores("first_error", r), failed_te)}
    for name, d in models.items():
        table[name] = evaluate(lambda r, d=d: [s["score"] for s in d.score_run(r)], failed_te)
    for name, m in table.items():
        row(name, m)
    results["split"] = table

    main_model = models["Black Box (boosted trees)"]
    n_inv = sum(len(v) for v in main_model.miner.invariants.values())
    print(f"\n  Rules learned from successful runs: {n_inv} "
          f"(across {len(main_model.miner.invariants)} steps)")
    imp = main_model.importances()
    if imp:
        print("  What the model relies on most: " +
              ", ".join(f"{k} ({v:.0%})" for k, v in imp[:5]))

    # ---------------- Experiment 2: unseen fault types
    print("\nEXPERIMENT 2: trained WITHOUT this fault type, tested on it (never seen before)")
    all_failed = split_clean_failed(runs)[1]
    kinds = sorted({r["fault_kind"] for r in all_failed})
    for kind in kinds:
        held = [r for r in all_failed if r["fault_kind"] == kind]
        d = Diagnoser("gbm").fit(clean_tr, [r for r in failed_tr if r["fault_kind"] != kind])
        m = evaluate(lambda r: [s["score"] for s in d.score_run(r)], held)
        results["unseen_fault_types"][kind] = m
        print(f"  held out {kind:<20} n={m['n']:<4} top-1 {m['top1']:>6.1%}   top-3 {m['top3']:>6.1%}")
    avg = sum(m["top1"] for m in results["unseen_fault_types"].values()) / len(kinds)
    print(f"  average top-1 on unseen fault types: {avg:.1%}")

    # ---------------- One example of evidence
    ex = failed_te[0]
    scored = sorted(main_model.score_run(ex), key=lambda s: -s["score"])
    print(f"\nEXAMPLE DIAGNOSIS\n  task: {ex['task']}\n  expected ${ex['expected']:.2f}, got {ex['final_answer']!r}")
    print(f"  true culprit: step {ex['culprit_step']} ({ex['fault_kind']})")
    for s in scored[:2]:
        print(f"  suspect #{scored.index(s) + 1}: step {s['step_idx']} {s['name']}  (score {s['score']:.2f})")
        for e in s["evidence"][:3]:
            print(f"      - broke rule: {e}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json (use these numbers on your slides)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_mock.db")