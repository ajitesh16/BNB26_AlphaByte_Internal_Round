"""Guard rail: compare a new results file against the FROZEN baseline.

  python baseline_check.py                       (uses baseline_v1/ and ./results*.json)
  python baseline_check.py <old_dir> <new_dir>

Prints every headline number side by side. Differences within 3 points are shown as OK
(different library versions and the timer fix can move a boosted-tree model slightly);
anything bigger is flagged CHECK so a new feature can never silently change the baseline.
"""
import json
import os
import sys

TOL = 0.03


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p}. Copy your known-good results.json and results_causal.json into {d}/ first.")
    return json.load(open(p))


def main(old_dir="baseline_v1", new_dir="."):
    o, n = load(old_dir, "results.json"), load(new_dir, "results.json")
    oc, nc = load(old_dir, "results_causal.json"), load(new_dir, "results_causal.json")
    rows = []
    for m in ["Random guess", "Always blame last step", "First logged error",
              "Black Box (boosted trees)", "Black Box (logistic)"]:
        rows.append((f"{m}: top-1", o["split"][m]["top1"], n["split"][m]["top1"]))
    ua = lambda r: sum(v["top1"] for v in r["unseen_fault_types"].values()) / len(r["unseen_fault_types"])
    rows.append(("Unseen fault types: avg top-1", ua(o), ua(n)))
    rows.append(("Causal: Black Box fixed on 1st try", oc["methods"]["Black Box"]["fixed_first_try"],
                 nc["methods"]["Black Box"]["fixed_first_try"]))
    rows.append(("Replay: steps saved", oc["replay_efficiency"]["steps_saved_pct"], nc["replay_efficiency"]["steps_saved_pct"]))
    rows.append(("Replay: tokens saved", oc["replay_efficiency"]["tokens_saved_pct"], nc["replay_efficiency"]["tokens_saved_pct"]))
    bad = 0
    print(f"{'metric':<40}{'frozen':>9}{'now':>9}{'delta':>9}  status")
    for name, a, b in rows:
        ok = abs(a - b) <= TOL
        bad += not ok
        print(f"{name:<40}{a:>9.1%}{b:>9.1%}{b - a:>+9.1%}  {'OK' if ok else 'CHECK'}")
    print("\nfeature set now:", (n.get("meta") or {}).get("feature_set", "?"),
          "| features hash:", (n.get("meta") or {}).get("features_hash", "?"),
          "| frozen:", (o.get("meta") or {}).get("features_hash", "(not recorded in the old file)"))
    print("BASELINE INTACT" if not bad else f"{bad} metric(s) moved more than {TOL:.0%}: investigate before continuing")


if __name__ == "__main__":
    main(*(sys.argv[1:3]))