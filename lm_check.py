"""Evaluate the optional pretrained local-LM signal against the frozen base-v1 split.

Usage:
    python lm_check.py data_mock.db

This does not replace the frozen baseline. It reports a separate base+LM ablation.
"""
import json, sys, random, time
from diagnose import Diagnoser, load_runs, split_clean_failed, evaluate

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data_mock.db"
    runs = load_runs(db)
    rng = random.Random(0); rng.shuffle(runs)
    cut = int(0.7 * len(runs))
    clean_tr, failed_tr = split_clean_failed(runs[:cut])
    _, failed_te = split_clean_failed(runs[cut:])
    print(f"DB: {db} | train: {len(failed_tr)} failed | test: {len(failed_te)} failed")
    t0=time.perf_counter()
    base = Diagnoser("gbm", use_lm=False).fit(clean_tr, failed_tr)
    base_m = evaluate(lambda r: [x["score"] for x in base.score_run(r)], failed_te)
    print("BASE-v1:", json.dumps(base_m, indent=2))
    t1=time.perf_counter()
    lm = Diagnoser("gbm", use_lm=True).fit_with_base(base, clean_tr, failed_tr)
    lm_m = evaluate(lambda r: [x["score"] for x in lm.score_run(r)], failed_te)
    print("BASE+LOCAL-LM:", json.dumps(lm_m, indent=2))
    print(f"Base fit/eval seconds: {t1-t0:.1f}")
    print(f"LM total seconds: {time.perf_counter()-t1:.1f}")
    print("LM feature importances:", lm.importances()[:10])
