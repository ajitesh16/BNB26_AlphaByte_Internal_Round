"""One-command judge-readiness check. Needs no browser and starts no server.

    python judge_selfcheck.py

It runs the real app.py code against a recording stand-in for Streamlit, clicks the buttons,
and checks: every tab renders, a failed run is diagnosed and explained, the What-If Lab and replay
produce SUCCESS for the predicted step, no answer key is visible, displayed numbers equal the
saved results files, and the wording makes no false claims. It does NOT test browser rendering:
also open the app once (see the launch command in the README of your hand-off notes).
"""
import functools
import json
import os
import re
import sys
from unittest.mock import MagicMock

import pandas as pd

for need in ("app.py", "data_mock.db", "results.json", "results_causal.json"):
    if not os.path.exists(need):
        raise SystemExit(f"Run this from the blackbox folder: {need} not found in {os.getcwd()}")


class SS(dict):
    __getattr__ = dict.get

    def __setattr__(s, k, v):
        s[k] = v


st = MagicMock()
st.session_state = SS()
CLICK, RADIO, SELECT = set(), {}, {}
FRAMES = []


class Col:
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def __getattr__(s, n): return getattr(st, n)


st.cache_resource = lambda *a, **k: (functools.lru_cache(maxsize=None)(a[0]) if a and callable(a[0])
                                     else (lambda f: functools.lru_cache(maxsize=None)(f)))
st.columns.side_effect = lambda spec, **k: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.tabs.side_effect = lambda labels: [Col() for _ in labels]
st.sidebar = Col()
st.spinner.side_effect = lambda *a, **k: Col()
st.expander.side_effect = lambda *a, **k: Col()
st.radio.side_effect = lambda label, options, **k: RADIO.get(label, options[0])
st.selectbox.side_effect = lambda label, options, index=0, **k: SELECT.get(label, options[index])
st.checkbox.return_value = False
st.text_input.side_effect = lambda label, value="", **k: value
st.number_input.side_effect = lambda *a, value=1, **k: value
st.text_area.side_effect = lambda label, value="", **k: value
st.button.side_effect = lambda label, **k: label in CLICK
st.stop.side_effect = SystemExit
st.dataframe.side_effect = lambda df, **k: FRAMES.append(df)
st.table.side_effect = lambda df, **k: FRAMES.append(df)
sys.modules["streamlit"] = st
sys.path.insert(0, os.getcwd())
import app  # noqa: E402
from agent import ALL_FAULTS  # noqa: E402
from diagnose import FEATURES, FEATURE_SET_ID  # noqa: E402
from replay import ReplaySession, Patch, compare  # noqa: E402

TEXT = ["markdown", "caption", "info", "warning", "success", "error", "write", "subheader", "header", "title", "code"]
FAILS, ALLTEXT = [], []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def rerun():
    for m in [getattr(st, f) for f in TEXT] + [st.metric, st.checkbox, st.download_button]:
        m.reset_mock()
    FRAMES.clear()
    app.main()
    out = [str(a) for f in TEXT for c in getattr(st, f).call_args_list for a in c.args]
    out += [str(a) for c in st.metric.call_args_list for a in c.args]
    ALLTEXT.extend(out)
    return " ".join(out)


def frame(col):
    got = [f for f in FRAMES if isinstance(f, pd.DataFrame) and col in f.columns]
    return got[-1] if got else None


R, C = json.load(open("results.json")), json.load(open("results_causal.json"))
LM = json.load(open("results_lm.json")) if os.path.exists("results_lm.json") else None

print("1. Every tab renders; a failed run is selected, diagnosed and explained")
t = rerun()
for n in ["Mission Control", "1. Where did it go wrong?", "2. Why does Black Box think so?", "Evidence chain",
          "Impact: how the failure propagates", "Recommended corrective action", "Finding the culprit step",
          "Where do agent failures start?", "Grounding check passed"]:
    check(n, n in t)

print("2. No developer answer key in the normal UI")
dev = os.environ.get("BLACKBOX_DEV") == "1"
check("answer-key checkbox hidden", dev or not st.checkbox.call_args_list)
check("no 'TRUE CULPRIT' tag rendered", dev or "TRUE CULPRIT" not in t)

print("3. Displayed numbers equal the saved results files")
mets = {str(c.args[0]): str(c.args[1]) for c in st.metric.call_args_list}
bb, u = R["split"]["Black Box (boosted trees)"], R["unseen_fault_types"]
eff = C["replay_efficiency"]
exp = {"Top-1 culprit localization": f"{bb['top1']:.1%}",
       "Unseen-fault Top-1": f"{sum(v['top1'] for v in u.values()) / len(u):.1%}",
       "First-try repair": f"{C['methods']['Black Box']['fixed_first_try']:.1%}",
       "Steps avoided": f"{1 - eff['avg_steps_executed'] / 6:.1%}",
       "Tokens saved vs full re-run": f"{eff['tokens_saved_pct']:.1%}",
       "Average steps replayed": f"{eff['avg_steps_executed']:.2f} of 6"}
if LM:
    exp["Base + local LM"] = f"{LM['base_plus_local_lm']['top1']:.1%}"
for k, v in exp.items():
    check(f"{k} = {v}", mets.get(k) == v, f"shown {mets.get(k)}")
tbl = next(f for f in FRAMES if isinstance(f, pd.DataFrame) and "AUROC" in f.columns)
ok = all(row["Top-1"] == f"{R['split'][n]['top1']:.1%}" and row["Top-3"] == f"{R['split'][n]['top3']:.1%}"
         and row["AUROC"] == f"{R['split'][n]['auroc']:.3f}" for n, row in tbl.iterrows())
check("evaluation table rows match results.json", ok)
check("no empty tables", all(len(f) > 0 for f in FRAMES if isinstance(f, pd.DataFrame)))
check("frozen feature set is base-v1", FEATURE_SET_ID == "base-v1" and
      __import__("hashlib").sha1(",".join(FEATURES).encode()).hexdigest()[:8] == (R.get("meta") or {}).get("features_hash"))

print("4. What-If Lab and replay on the default failed run")
CLICK.add("Run 3-path investigation")
rerun()
wi = frame("Path")
res = dict(zip(wi["Path"], wi["Result"])) if wi is not None else {}
check("unchanged retry fails", res.get("Unchanged retry") == "FAIL", str(res))
check("predicted-step fix succeeds", res.get("Predicted-step fix") == "SUCCESS")
check("runner-up fix fails", res.get("Runner-up fix") == "FAIL")
cmp_t = frame("Predicted fix")
check("trace comparison has all 6 steps", cmp_t is not None and len(cmp_t) == 6)
CLICK.clear()
CLICK.add("Replay with this fix")
t = rerun()
check("manual replay at the suspect flips FAILED -> SUCCESS", "Fixed. The run flipped" in t)
check("verdict says 'supports the diagnosis (not proof)'", "supports the diagnosis (not proof)" in t)
CLICK.clear()

print("5. All held-out failures: retry vs predicted vs runner-up")
data = app.load_all("data_mock.db", os.path.getmtime("data_mock.db"), use_lm=False)
n = ok_pred = ok_retry = ok_ru = complete = 0
for r in data["failed_te"]:
    sc = sorted(data["model"].score_run(r), key=lambda x: -x["score"])
    s = ReplaySession(r)
    pred, ru, rt = s.fork(Patch(sc[0]["step_idx"], "fix")), s.fork(Patch(sc[1]["step_idx"], "fix")), s.retry()
    n += 1; ok_pred += pred["success"]; ok_ru += ru["success"]; ok_retry += rt["success"]
    complete += len(compare(r, pred)["rows"]) == 6 and all("latency_ms" in x and "tokens" in x for x in pred["steps"])
print(f"       {n} held-out failures: predicted fix SUCCESS {ok_pred}/{n}, runner-up fix SUCCESS {ok_ru}/{n}, "
      f"unchanged retry SUCCESS {ok_retry}/{n}, complete comparisons {complete}/{n}")
want = C["methods"]["Black Box"]["fixed_first_try"] * n
check("predicted-fix successes agree with results_causal.json (within 1 run)", abs(ok_pred - want) <= 1,
      f"{ok_pred} vs {want:.1f}")
check("unchanged retry never succeeds", ok_retry == 0)
check("every comparison complete", complete == n)

print("6. 'Break it live' for every fault type")
RADIO["Which failure do you want to investigate?"] = "Break it live"
bad = []
for j, (step, kind) in enumerate(ALL_FAULTS):
    SELECT["Break which step, and how?"] = j
    CLICK.add("Run the agent"); rerun(); CLICK.clear()
    CLICK.add("Run 3-path investigation"); rerun(); CLICK.clear()
    wi = frame("Path")
    r2 = dict(zip(wi["Path"], wi["Result"])) if wi is not None else {}
    if r2.get("Predicted-step fix") != "SUCCESS":
        bad.append(f"{step}:{kind}")
check("predicted fix succeeds for all 7 fault types", not bad, ", ".join(bad))
RADIO.clear(); SELECT.clear()

print("7. Wording")
blob = " ".join(ALLTEXT)
check("no claim that proof is established ('proves')", not re.search(r"\bproves?\b", blob, re.I))
scrubbed = blob.replace("no language-model API calls", "")      # the disclosure that none were used
check("no Claude/API/LLM mention in mock mode", not re.search(r"claude|\bapi\b|anthropic|\bgpt\b", scrubbed, re.I))
check("benchmark scope disclosed (controlled, mock, not arbitrary agents)",
      "controlled fault-injection benchmark" in blob and "arbitrary real-world agents" in blob)

print()
print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: " + "; ".join(FAILS))
sys.exit(1 if FAILS else 0)
