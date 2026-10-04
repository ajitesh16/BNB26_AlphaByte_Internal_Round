"""The diagnosis model: given a FAILED run, rank its steps by 'how likely is this the root cause?'

Pipeline:  mine invariants from successful runs  ->  turn every step into numbers
           ->  train a classifier on labeled failures  ->  rank steps in new failures.
"""
import json
import math
from collections import Counter

import numpy as np

from invariants import InvariantMiner
from tracer import Tracer

KINDS = ["known_values", "copied_from_input", "contains_input_value", "substring", "formula"]
FEATURES = (["idx", "rel_pos", "is_llm", "is_retrieval", "is_tool", "has_error",
             "n_viol", "frac_viol"] + [f"viol_{k}" for k in KINDS] +
            ["viol_before", "viol_after", "first_viol", "len_ratio", "latency_z"])


FEATURE_SET_ID = "base-v1"        # the frozen feature set behind the 97.6% baseline
LM_FEATURES = ["lm_nll", "lm_nll_z", "lm_max_nll", "lm_token_count"]


def run_meta(db_path, n_runs, model):
    """Stamp saved with every results file: which data, which features, which model backend."""
    import hashlib
    import os
    import time
    meta = {"db": os.path.basename(db_path), "n_runs": n_runs, "feature_set": FEATURE_SET_ID,
            "features_hash": hashlib.sha1(",".join(FEATURES).encode()).hexdigest()[:8],
            "model_backend": type(model.model).__name__,
            "evaluated": time.strftime("%Y-%m-%d %H:%M:%S")}
    side = db_path + ".meta.json"
    if os.path.exists(side):
        with open(side) as f:
            meta["data_mode"] = json.load(f).get("mode", "unknown")
    return meta


def load_runs(db_path):
    t = Tracer(db_path)
    runs = []
    for r in t.list_runs():
        steps = t.get_steps(r["run_id"])
        for s in steps:
            s["input"], s["output"] = json.loads(s["input"]), json.loads(s["output"])
        runs.append({**r, "steps": steps})
    return runs


def split_clean_failed(runs):
    clean = [r for r in runs if r["fault_kind"] is None and r["success"]]
    failed = [r for r in runs if r["culprit_step"] is not None and not r["success"]]
    return clean, failed


def make_model(name="gbm"):
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=8,
                                  min_child_samples=5, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                              max_leaf_nodes=8, min_samples_leaf=5)


class Diagnoser:
    def __init__(self, model="gbm", use_lm=False, lm_model=None):
        self.model_name = model
        self.use_lm = bool(use_lm)
        self.lm_model_name = lm_model
        self.lm = None

    def fit(self, clean_runs, failed_runs):
        self.miner = InvariantMiner().fit(clean_runs)
        self._fit_reference(clean_runs)
        return self._fit_with_reference(clean_runs, failed_runs)

    def fit_with_base(self, base, clean_runs, failed_runs):
        """Fit an optional-signal model while reusing the frozen base miner/reference.

        This avoids mining the same invariants twice and keeps the base-v1 benchmark
        exactly unchanged while running optional LM ablations.
        """
        self.miner = base.miner
        self.ref = base.ref
        return self._fit_with_reference(clean_runs, failed_runs)

    def _fit_with_reference(self, clean_runs, failed_runs):
        if self.use_lm:
            from lm_signal import LocalLMSignal
            self.lm = LocalLMSignal(self.lm_model_name)
            self.lm.fit_clean_reference(clean_runs)
        self.feature_names = FEATURES + (LM_FEATURES if self.use_lm else [])
        X, y = [], []
        for run in failed_runs:
            rows, _ = self.featurize(run)
            X += [[r[f] for f in self.feature_names] for r in rows]
            y += [int(r["idx"] == run["culprit_step"]) for r in rows]
        self.model = make_model(self.model_name).fit(np.array(X), np.array(y))
        return self

    def _fit_reference(self, clean_runs):
        lens, lats = {}, {}
        for run in clean_runs:
            for s in run["steps"]:
                lens.setdefault(s["name"], []).append(len(json.dumps(s["output"])))
                lats.setdefault(s["name"], []).append(s["latency_ms"])
        self.ref = {}
        for name in lens:
            med = float(np.median(lats[name]))
            self.ref[name] = {"len": float(np.median(lens[name])), "lat": med,
                              "mad": float(np.median(np.abs(np.array(lats[name]) - med)))}

    def featurize(self, run):
        steps = run["steps"]
        checks = [self.miner.check(s) for s in steps]
        nviol = [len(v) for v, _ in checks]
        rows = []
        for i, s in enumerate(steps):
            viol, ninv = checks[i]
            kinds = Counter(v.kind for v in viol)
            ref = self.ref.get(s["name"], {"len": 1, "lat": 0, "mad": 0})
            before, after = sum(nviol[:i]), sum(nviol[i + 1:])
            row = {"idx": i, "rel_pos": i / max(len(steps) - 1, 1),
                   "is_llm": s["type"] == "llm", "is_retrieval": s["type"] == "retrieval",
                   "is_tool": s["type"] == "tool", "has_error": bool(s["error"]),
                   "n_viol": nviol[i], "frac_viol": nviol[i] / ninv if ninv else 0,
                   "viol_before": before, "viol_after": after,
                   "first_viol": int(nviol[i] > 0 and before == 0),
                   "len_ratio": min(len(json.dumps(s["output"])) / max(ref["len"], 1), 5),
                   "latency_z": float(np.clip((s["latency_ms"] - ref["lat"]) / (1.4826 * ref["mad"] + 1e-3), -5, 5))}
            for k in KINDS:
                row[f"viol_{k}"] = kinds.get(k, 0)
            if self.use_lm and self.lm is not None:
                lm_row = self.lm.features([s])[0]
                row.update(lm_row)
            rows.append({k: float(v) for k, v in row.items()})
        return rows, checks

    def score_run(self, run):
        """Return one dict per step: its suspicion score and the broken rules (evidence)."""
        rows, checks = self.featurize(run)
        p = self.model.predict_proba(np.array([[r[f] for f in self.feature_names] for r in rows]))[:, 1]
        return [{"step_idx": i, "name": s["name"], "type": s["type"], "score": float(p[i]),
                 "evidence": [v.explain() for v in checks[i][0]],
                 "violations": self.miner.details(s, checks[i][0]) if checks[i][0] else [],
                 "features": rows[i]}
                for i, s in enumerate(run["steps"])]

    def importances(self):
        m = self.model
        if hasattr(m, "feature_importances_"):
            vals = np.array(m.feature_importances_, dtype=float)
        elif hasattr(m, "steps"):                     # logistic pipeline
            vals = np.abs(m.steps[-1][1].coef_[0])
        else:
            return None
        vals = vals / (vals.sum() or 1)
        return sorted(zip(self.feature_names, vals), key=lambda t: -t[1])


# ------------------------------------------------------------------ baselines
def baseline_scores(name, run):
    n = len(run["steps"])
    if name == "last_step":
        return [float(i) for i in range(n)]
    if name == "first_error":      # steps with a logged error first; otherwise just the earliest step
        return [100.0 * bool(s["error"]) - i for i, s in enumerate(run["steps"])]
    raise ValueError(name)


# -------------------------------------------------------------------- metrics
def rank_of(scores, culprit):
    """1 = best. Ties are counted against us, to stay honest."""
    return 1 + sum(1 for j, s in enumerate(scores) if j != culprit and s >= scores[culprit])


def evaluate(score_fn, runs):
    from sklearn.metrics import roc_auc_score
    ranks, ys, ss = [], [], []
    for run in runs:
        sc = score_fn(run)
        ranks.append(rank_of(sc, run["culprit_step"]))
        ys += [int(i == run["culprit_step"]) for i in range(len(sc))]
        ss += list(sc)
    r = np.array(ranks)
    return {"n": len(runs), "top1": float(np.mean(r <= 1)), "top3": float(np.mean(r <= 3)),
            "mrr": float(np.mean(1 / r)), "auroc": float(roc_auc_score(ys, ss))}


def evaluate_random(runs):
    """Exact expected metrics for guessing at random (no simulation noise)."""
    top1 = top3 = mrr = 0.0
    for run in runs:
        n = len(run["steps"])
        top1 += 1 / n
        top3 += min(3, n) / n
        mrr += sum(1 / r for r in range(1, n + 1)) / n
    k = len(runs)
    return {"n": k, "top1": top1 / k, "top3": top3 / k, "mrr": mrr / k, "auroc": 0.5}
