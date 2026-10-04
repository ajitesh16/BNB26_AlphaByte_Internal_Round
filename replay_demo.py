"""The demo story in your terminal: diagnose a failure, patch it, replay, compare.

  python replay_demo.py data_mock.db               (picks a failed run for you)
  python replay_demo.py data_mock.db stale_price   (picks a failed run with that fault)
"""
import sys

from diagnose import Diagnoser, load_runs, split_clean_failed
from replay import Patch, ReplaySession, compare, format_comparison


def main(db, kind=None):
    runs = load_runs(db)
    clean, failed = split_clean_failed(runs)
    target = next(r for r in failed if kind is None or r["fault_kind"] == kind)
    print("Training Black Box on all the other runs (takes a few seconds)...")
    model = Diagnoser("gbm").fit(clean, [r for r in failed if r is not target])

    print(f"\n=== 1. A FAILED RUN ===\n  task: {target['task']}")
    print(f"  expected ${target['expected']:.2f}, agent said {target['final_answer']!r}")

    scored = sorted(model.score_run(target), key=lambda s: -s["score"])
    top = scored[0]
    print(f"\n=== 2. BLACK BOX DIAGNOSIS ===")
    print(f"  most suspicious: step {top['step_idx']} ({top['name']})   suspicion {top['score']:.0%}")
    for e in top["evidence"][:3]:
        print(f"    evidence: {e}")
    print(f"  (answer key says the real culprit was step {target['culprit_step']}: {target['fault_kind']})")

    session = ReplaySession(target)
    wrong = scored[-1]["step_idx"]
    print(f"\n=== 3. CONTROL: patch a step Black Box did NOT blame (step {wrong}) ===")
    b = session.fork(Patch(wrong, "fix"))
    print(format_comparison(compare(target, b), b))

    print(f"\n=== 4. PATCH THE SUSPECT (step {top['step_idx']}) AND REPLAY ===")
    b = session.fork(Patch(top["step_idx"], "fix"))
    print(format_comparison(compare(target, b), b))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_mock.db",
         sys.argv[2] if len(sys.argv) > 2 else None)
