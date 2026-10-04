"""Learns what 'normal' looks like by studying SUCCESSFUL runs.

For every step, we automatically discover rules ("invariants") that held in
(almost) every successful run, for example:
   - "pick_price is always one of the prices in retrieve_catalog"
   - "calculate.total = quantity x price x (1 - discount/100)"
   - "check_stock.in_stock is always True"
No one writes these rules by hand. When a new run fails, the steps that break
rules are our suspects, and the broken rules are the EVIDENCE shown to the user.

Only successful runs are used here, so this part never sees the answer key.
"""
import itertools
import re
from collections import defaultdict

NUM = re.compile(r"\d+(?:\.\d+)?")


def key(v):
    """Canonical form so 7.25 and 7.250 match, and True is never confused with 1."""
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, float)):
        return ("n", round(float(v), 2))
    return ("s", str(v).strip().lower())


def leaves(obj, prefix):
    """Flatten JSON-like data into {path: [values]}. Lists use '[]' in the path.
    Text that contains digits also gets a '<path>#nums' entry listing those numbers."""
    out = defaultdict(list)

    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}.{k}" if p else k)
        elif isinstance(o, list):
            for v in o:
                walk(v, p + "[]")
        elif o is not None:
            out[p].append(o)
            if isinstance(o, str):
                nums = [float(n) for n in NUM.findall(o)]
                if nums:
                    out[p + "#nums"].extend(nums)
    walk(obj, prefix)
    return out


def _scalar_num(d, path):
    v = d.get(path)
    if v and len(v) == 1 and not isinstance(v[0], (bool, str)):
        return float(v[0])
    return None


def _fmt(v, n=12):
    """Compact JSON-style text for a value or a list of values."""
    import json
    if isinstance(v, list) and len(v) > n:
        return json.dumps(v[:n], default=str)[:-1] + ", ...]"
    return json.dumps(v, default=str)


class Invariant:
    def __init__(self, step, kind, desc, fn, info=None, out_path=None, in_paths=()):
        self.step, self.kind, self.desc, self.fn = step, kind, desc, fn
        self.info, self.out_path, self.in_paths = info, out_path, tuple(in_paths)
        self.hold_rate, self.support = 1.0, 0

    def violated(self, inp, out):
        return self.fn(inp, out) is not True        # False, or evidence missing

    def explain(self):
        return f"{self.desc}  (held in {self.hold_rate:.0%} of {self.support} successful runs)"

    def detail(self, inp, out):
        """What exactly was observed, and what the rule says it should have been."""
        d = {"kind": self.kind, "rule": self.desc, "out_path": self.out_path,
             "in_paths": list(self.in_paths), "hold_rate": self.hold_rate, "support": self.support,
             "held_in": f"{self.hold_rate:.0%} of {self.support} successful runs",
             "observed": list(out.get(self.out_path, [])) if self.out_path else None,
             "expected_text": None, "expected_values": None, "expected_value": None, "inputs": {}}
        if self.info:
            try:
                d.update(self.info(inp, out))
            except Exception:
                pass
        return d


class InvariantMiner:
    def __init__(self, min_support=20, min_hold=0.98):
        self.min_support, self.min_hold = min_support, min_hold
        self.invariants = {}

    def fit(self, clean_runs):
        per_step = defaultdict(list)
        for run in clean_runs:
            for s in run["steps"]:
                if s["output"] is not None:
                    per_step[s["name"]].append(
                        (leaves(s["input"], ""), leaves(s["output"], s["name"])))
        self.invariants = {n: self._mine(n, inst) for n, inst in per_step.items()
                           if len(inst) >= self.min_support}
        return self

    def details(self, step, violated):
        """Observed-vs-expected description for each violated rule of this step."""
        inp = leaves(step["input"], "")
        out = leaves(step["output"], step["name"]) if step["output"] is not None else {}
        return [iv.detail(inp, out) for iv in violated]

    def check(self, step):
        """Return (list of violated invariants, number of invariants checked)."""
        invs = self.invariants.get(step["name"], [])
        inp = leaves(step["input"], "")
        out = leaves(step["output"], step["name"]) if step["output"] is not None else {}
        return [iv for iv in invs if iv.violated(inp, out)], len(invs)

    # ------------------------------------------------------------ mining
    def _mine(self, step, inst):
        n = len(inst)
        out_paths = sorted({p for _, o in inst for p in o})
        in_paths = sorted({p for i, _ in inst for p in i})
        is_bool = {p for _, o in inst for p, vs in o.items() if any(isinstance(v, bool) for v in vs)}
        cands = []

        # 1) "this value is always one of a small known set"
        for P in out_paths:
            vals, raw = [], {}
            for _, o in inst:
                for v in o.get(P, []):
                    vals.append(key(v))
                    raw.setdefault(key(v), v)
            distinct = set(vals)
            if len(distinct) <= 25 and len(distinct) <= 0.3 * len(vals):
                cands.append(self._known(step, P, distinct, raw))

        # 2) relations between the step's output and what it was given
        for P in out_paths:
            for Q in in_paths:
                if P == Q:
                    continue
                if P not in is_bool:
                    cands.append(self._copied(step, P, Q))
                    cands.append(self._substring(step, P, Q))
                cands.append(self._contains(step, P, Q))

        # 3) arithmetic: output = b*c, or b*c*(1-d/100)
        scal = [Q for Q in in_paths if "#" not in Q and all(_scalar_num(i, Q) is not None for i, _ in inst)]
        for P in out_paths:
            if "#" in P or not all(_scalar_num(o, P) is not None for _, o in inst):
                continue
            for b, c in itertools.combinations(scal, 2):
                cands.append(self._formula(step, P, b, c, None))
                for d in scal:
                    if d not in (b, c):
                        cands.append(self._formula(step, P, b, c, d))

        kept = []
        for inv in cands:
            res = [inv.fn(i, o) for i, o in inst]
            present = [r for r in res if r is not None]
            if len(present) >= 0.98 * n:
                rate = sum(present) / len(present)
                if rate >= self.min_hold:
                    inv.hold_rate, inv.support = rate, n
                    kept.append(inv)
        return kept

    @staticmethod
    def _known(step, P, seen, raw):
        shown = sorted(str(v) for _, v in seen)[:6]
        more = ", ..." if len(seen) > 6 else ""
        allowed = [raw[k] for k in sorted(seen, key=str)][:12]

        def fn(i, o):
            return all(key(v) in seen for v in o[P]) if o.get(P) else None

        def info(i, o):
            vals = list(o.get(P, []))
            bad = [v for v in vals if key(v) not in seen]
            return {"observed": bad or vals, "expected_text": f"one of {_fmt(allowed)}",
                    "expected_values": allowed}
        return Invariant(step, "known_values", f"`{P}` is always one of {{{', '.join(shown)}{more}}}",
                         fn, info, P)

    @staticmethod
    def _copied(step, P, Q):
        def fn(i, o):
            if not o.get(P) or not i.get(Q):
                return None
            pool = {key(w) for w in i[Q]}
            return all(key(v) in pool for v in o[P])

        def info(i, o):
            pool = list(i.get(Q, []))
            keys = {key(w) for w in pool}
            vals = list(o.get(P, []))
            bad = [v for v in vals if key(v) not in keys]
            return {"observed": bad or vals, "expected_text": f"a value found in `{Q}`: {_fmt(pool)}",
                    "expected_values": pool[:12]}
        return Invariant(step, "copied_from_input", f"every value in `{P}` can be found in `{Q}`",
                         fn, info, P, [Q])

    @staticmethod
    def _substring(step, P, Q):
        def fn(i, o):
            if not o.get(P) or not i.get(Q) or len(i[Q]) != 1:
                return None
            if not all(isinstance(v, str) and len(v) <= 40 for v in o[P]) or not isinstance(i[Q][0], str):
                return None
            return all(v.lower() in i[Q][0].lower() for v in o[P])

        def info(i, o):
            text = i[Q][0] if i.get(Q) else ""
            return {"expected_text": f"text quoted from `{Q}`: {_fmt(text[:80])}", "expected_value": text}
        return Invariant(step, "substring", f"`{P}` is quoted from the text of `{Q}`", fn, info, P, [Q])

    @staticmethod
    def _contains(step, P, Q):
        def fn(i, o):
            if not o.get(P) or not i.get(Q) or len(i[Q]) != 1 or isinstance(i[Q][0], bool):
                return None
            return key(i[Q][0]) in {key(v) for v in o[P]}

        def info(i, o):
            need = i[Q][0] if i.get(Q) else None
            return {"expected_text": f"`{P}` should include {_fmt(need)} (the value of `{Q}`)",
                    "expected_value": need, "inputs": {Q: need}}
        return Invariant(step, "contains_input_value", f"`{Q}` shows up in `{P}`", fn, info, P, [Q])

    @staticmethod
    def _formula(step, P, b, c, d):
        def fn(i, o):
            a, vb, vc = _scalar_num(o, P), _scalar_num(i, b), _scalar_num(i, c)
            if None in (a, vb, vc):
                return None
            if d is None:
                return abs(a - vb * vc) <= 0.011
            vd = _scalar_num(i, d)
            return None if vd is None else abs(a - vb * vc * (1 - vd / 100)) <= 0.011

        def info(i, o):
            vb, vc = _scalar_num(i, b), _scalar_num(i, c)
            vd = _scalar_num(i, d) if d else None
            if None in (vb, vc) or (d and vd is None):
                return {}
            exp = round(vb * vc * (1 - vd / 100), 2) if d else round(vb * vc, 2)
            ins = {b: vb, c: vc, **({d: vd} if d else {})}
            txt = (f"{b} ({vb:g}) x {c} ({vc:g}) x (1 - {d} ({vd:g})/100) = {exp:g}" if d
                   else f"{b} ({vb:g}) x {c} ({vc:g}) = {exp:g}")
            return {"expected_text": txt, "expected_value": exp, "inputs": ins}
        desc = f"`{P}` = `{b}` x `{c}`" if d is None else f"`{P}` = `{b}` x `{c}` x (1 - `{d}`/100)"
        return Invariant(step, "formula", desc, fn, info, P, [b, c] + ([d] if d else []))
