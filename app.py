import html
import json
import os
import random
import re
from collections import Counter

import pandas as pd
import streamlit as st

import agent
from agent import ALL_FAULTS, call_llm, run_agent
from diagnose import Diagnoser, load_runs, split_clean_failed
from replay import HOW, Patch, ReplaySession, compare
from tasks import make_tasks
from tracer import Tracer

CSS = """
<style>
.bb-row {display:flex; align-items:stretch; gap:6px; flex-wrap:wrap; margin:8px 0 4px 0;}
.bb-step {border-radius:10px; padding:10px 12px; min-width:128px; text-align:center; color:#1b1b1b;}
.bb-arrow {align-self:center; font-size:22px; color:#90a4ae;}
.bb-name {font-weight:700; font-size:14px;}
.bb-type {font-size:11px; opacity:.75;}
.bb-score {font-size:20px; font-weight:800;}
.bb-tag {font-size:11px; font-weight:700; color:#b71c1c; text-align:center; margin-top:2px;}
.bb-table {width:100%; border-collapse:collapse; font-size:13px; color:#1b1b1b; background:#ffffff;}
.bb-table th {background:#263238; color:#ffffff; text-align:left; padding:6px 8px;}
.bb-table td {padding:6px 8px; border-bottom:1px solid #cfd8dc; vertical-align:top;}
</style>
"""

TASK_RE = re.compile(r"(\d+) units of (.+?) cost with a (\d+)% discount")


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner=False)
def load_all(db_path, mtime):
    """Load recorded runs, hold out 30% as 'never seen', and train Black Box on the rest."""
    runs = load_runs(db_path)
    rng = random.Random(0)
    rng.shuffle(runs)
    cut = int(0.7 * len(runs))
    clean_tr, failed_tr = split_clean_failed(runs[:cut])
    _, failed_te = split_clean_failed(runs[cut:])
    model = Diagnoser("gbm").fit(clean_tr, failed_tr)
    return {"model": model, "failed_te": failed_te, "clean": split_clean_failed(runs)[0]}


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_live(task, fault):
    """Run the agent right now, with a fault secretly injected, and return it like a recorded run."""
    tr = Tracer(":memory:")
    rid = run_agent(task, tr, fault)
    run, steps = tr.get_run(rid), tr.get_steps(rid)
    for s in steps:
        s["input"], s["output"] = json.loads(s["input"]), json.loads(s["output"])
    return {**run, "steps": steps}


def params(task):
    m = TASK_RE.search(task)
    return (m.group(2).lower(), int(m.group(1)), int(m.group(3))) if m else None


def nearest_clean(run, clean):
    """The most similar SUCCESSFUL run (same product, similar quantity and discount)."""
    p, best = params(run["task"]), None
    if not p:
        return None
    for c in clean:
        q = params(c["task"])
        if not q or len(c["steps"]) != len(run["steps"]):
            continue
        d = (0 if q[0] == p[0] else 100) + abs(q[1] - p[1]) + abs(q[2] - p[2]) / 5
        if best is None or d < best[0]:
            best = (d, c)
    return best[1] if best else None


# ------------------------------------------------------------- rendering
def esc(x):
    return html.escape(str(x))


def short(x, n=70):
    s = x if isinstance(x, str) else json.dumps(x)
    return s if len(s) <= n else s[:n - 3] + "..."


def blend(p):
    lo, hi = (232, 245, 233), (239, 83, 80)
    return "rgb(%d,%d,%d)" % tuple(int(a + (b - a) * p) for a, b in zip(lo, hi))


def timeline_html(scored, top_idx, truth=None):
    cells = []
    for s in scored:
        hot = s["step_idx"] == top_idx
        edge = ("border:3px solid #b71c1c; box-shadow:0 0 16px 4px rgba(229,57,53,.7);" if hot
                else "border:1px solid #b0bec5;")
        tag = '<div class="bb-tag">TRUE CULPRIT</div>' if truth == s["step_idx"] else ""
        cells.append(
            f'<div><div class="bb-step" style="background:{blend(s["score"])};{edge}">'
            f'<div class="bb-type">step {s["step_idx"]}</div><div class="bb-name">{esc(s["name"])}</div>'
            f'<div class="bb-score">{s["score"]:.0%}</div></div>{tag}</div>')
    return '<div class="bb-row">' + '<div class="bb-arrow">&rarr;</div>'.join(cells) + "</div>"


def diff_html(cmp):
    rows = []
    for r in cmp["rows"]:
        bg = "background:#fff8e1;" if not r["same"] else ""
        new = "(same)" if r["same"] else esc(short(r["new_output"]))
        rows.append(f"<tr style='{bg}'><td>{r['idx']}</td><td><b>{esc(r['name'])}</b></td>"
                    f"<td>{esc(short(r['orig_output']))}</td><td>{new}</td><td>{esc(HOW[r['status']])}</td></tr>")
    return ("<table class='bb-table'><tr><th>#</th><th>step</th><th>original run</th>"
            "<th>branch with fix</th><th>how it was produced</th></tr>" + "".join(rows) + "</table>")


def template_explanation(scored, top):
    ev = top["evidence"]
    later = [s["step_idx"] for s in scored if s["step_idx"] > top["step_idx"] and s["evidence"]]
    txt = f"Black Box blames step {top['step_idx']} ({top['name']}) for this failure. "
    if ev:
        txt += (f"That step broke {len(ev)} pattern(s) that held in almost every successful run, "
                f"for example: {ev[0].split('  (held')[0]}. ")
    else:
        txt += "No learned rule was broken there, so the suspicion comes from the step's position, size and timing. "
    if later:
        txt += ("Later steps that also look off (" + ", ".join(f"step {i}" for i in later) +
                ") most likely just inherited the bad output. ")
    return txt + "Patch that step and replay to confirm."


def llm_explanation(run, scored, top):
    facts = "\n".join(f"step {s['step_idx']} {s['name']}: output={short(r['output'], 120)}"
                      for s, r in zip(scored, run["steps"]))
    prompt = (f"An AI agent failed. Task: {run['task']}. Expected ${run['expected']:.2f}, "
              f"got {run['final_answer']!r}.\nTrace:\n{facts}\n"
              f"A detector blames step {top['step_idx']} ({top['name']}). Broken rules: {top['evidence']}.\n"
              "In 3 short sentences, explain the root cause for a non-expert. "
              "Cite steps like [step 1]. Use only the facts above.")
    fallback = template_explanation(scored, top)
    try:
        return call_llm(prompt, lambda: fallback)[0]
    except Exception:
        return fallback


def report_markdown(run, top, cmp, branch):
    lines = ["# Black Box diagnosis report", "", f"**Task:** {run['task']}",
             f"**Expected:** ${run['expected']:.2f}   **Agent said:** {run['final_answer']}", "",
             f"**Suspected root cause:** step {top['step_idx']} ({top['name']}), suspicion {top['score']:.0%}", ""]
    lines += [f"- evidence: {e}" for e in top["evidence"]] or ["- (no broken rule; other signals)"]
    if cmp:
        lines += ["", f"**Patch tested:** step {branch['patch'].step_idx}, {branch['patch'].describe()}",
                  f"**Result:** {'FIXED' if cmp['flipped'] else 'still failing'} -> agent now says {cmp['new_answer']!r}",
                  f"**Replay cost:** {branch['executed']} of {branch['n_steps']} steps re-run "
                  f"({branch['steps_saved_pct']:.0%} steps saved)"]
    return "\n".join(lines)


# ------------------------------------------------------------------ tab 1
def pick_run(data):
    mode = st.radio("Which failure do you want to investigate?",
                    ["Browse recorded failures", "Break it live"], horizontal=True)
    if mode == "Browse recorded failures":
        fails = data["failed_te"]
        if not fails:
            st.warning("No failed runs in the held-out set. Generate more data first.")
            return None
        i = st.selectbox("Failed run (held out: the model never saw it in training)", list(range(len(fails))),
                         format_func=lambda j: f"#{j + 1}   {fails[j]['task']}   ->   {short(fails[j]['final_answer'], 38)}")
        return fails[i]
    c1, c2, c3 = st.columns([3, 1, 1])
    labels = [f"{s}: {k}" for s, k in ALL_FAULTS]
    pick = c1.selectbox("Break which step, and how?", list(range(len(ALL_FAULTS))), format_func=lambda j: labels[j])
    seed = c2.number_input("Question #", min_value=0, value=1, step=1)
    if c3.button("Run the agent", type="primary"):
        st.session_state["live"] = run_live(make_tasks(1, seed=int(seed))[0], ALL_FAULTS[pick])
    run = st.session_state.get("live")
    if run is None:
        st.info("Choose a fault, then press Run the agent. Black Box is not told what you broke.")
        return None
    if run["success"]:
        st.warning("The agent still got this one right, so there is nothing to diagnose. Try another question.")
        return None
    return run


def debug_tab(data):
    run = pick_run(data)
    if run is None:
        return
    st.markdown(f"**Task:** {run['task']}  \n**Expected:** ${run['expected']:.2f} &nbsp;|&nbsp; "
                f"**Agent said:** {run['final_answer']}")
    scored = data["model"].score_run(run)
    top = max(scored, key=lambda s: s["score"])

    st.subheader("1. Where did it go wrong?")
    reveal = st.checkbox("Reveal the true culprit (answer key)", value=False)
    st.markdown(timeline_html(scored, top["step_idx"], run["culprit_step"] if reveal else None),
                unsafe_allow_html=True)
    st.caption("Each box is one step of the run, shaded by how suspicious Black Box finds it. "
               "The glowing box is its top suspect.")

    st.subheader("2. Why does Black Box think so?")
    left, right = st.columns(2)
    with left:
        st.markdown("**Rules this step broke** (learned from successful runs)")
        if top["evidence"]:
            for e in top["evidence"]:
                st.markdown(f"- {e}")
        else:
            st.markdown("- No learned rule was broken; the suspicion comes from other signals.")
        st.markdown("**In plain English**")
        st.write(template_explanation(scored, top))
        if agent.MODE == "real" and st.button("Ask Claude to write it up"):
            st.session_state["expl"] = (run["run_id"], llm_explanation(run, scored, top))
        ex = st.session_state.get("expl")
        if ex and ex[0] == run["run_id"]:
            st.info(ex[1])
    with right:
        st.markdown(f"**Contrast with a similar successful run** (step {top['step_idx']}, {top['name']})")
        twin = nearest_clean(run, data["clean"])
        if twin:
            st.caption(f"Closest success: {twin['task']}")
            st.markdown("This run's output:")
            st.code(short(run["steps"][top["step_idx"]]["output"], 400), language="json")
            st.markdown("Successful run's output:")
            st.code(short(twin["steps"][top["step_idx"]]["output"], 400), language="json")
        else:
            st.caption("No comparable successful run was recorded.")

    st.subheader("3. Try a fix and replay")
    n = len(run["steps"])
    a, b = st.columns(2)
    step_i = a.selectbox("Patch which step? (try a wrong one too)", list(range(n)), index=top["step_idx"],
                         format_func=lambda j: f"{j}  {run['steps'][j]['name']}")
    how = b.radio("How?", ["Fix the step (re-run it without the bug)", "Override its output with my own value"])
    override = ""
    if how.startswith("Override"):
        override = st.text_area("New output (JSON)", value=json.dumps(run["steps"][step_i]["output"]))
    if st.button("Replay with this fix", type="primary"):
        try:
            patch = Patch(step_i, "override", json.loads(override)) if how.startswith("Override") else Patch(step_i, "fix")
        except json.JSONDecodeError:
            st.error("That is not valid JSON. For text values, wrap them in double quotes.")
        else:
            sessions = st.session_state.setdefault("sessions", {})
            if run["run_id"] not in sessions:
                sessions[run["run_id"]] = ReplaySession(run)
            st.session_state["branch"] = (run["run_id"], sessions[run["run_id"]].fork(patch))

    saved = st.session_state.get("branch")
    cmp = branch = None
    if saved and saved[0] == run["run_id"]:
        branch = saved[1]
        cmp = compare(run, branch)
        st.subheader("4. What changed?")
        if cmp["flipped"]:
            st.success(f"Fixed. The run flipped from FAILED to SUCCESS. The agent now says: {cmp['new_answer']}")
        elif cmp["new_success"]:
            st.success("The run succeeds.")
        else:
            st.error("Still failing. That step was not the root cause, or this patch was not enough.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Skipped (checkpoint)", f"{branch['restored']} steps")
        m2.metric("Reused (cache)", f"{branch['cached']} steps")
        m3.metric("Re-run", f"{branch['executed']} of {branch['n_steps']}")
        m4.metric("Tokens saved", f"{branch['tokens_saved_pct']:.0%}")
        st.markdown(diff_html(cmp), unsafe_allow_html=True)
        if cmp["first_divergence"] is not None:
            st.caption(f"The two runs first differ at step {cmp['first_divergence']}. "
                       "Highlighted rows are steps whose output changed.")
    st.download_button("Download diagnosis report", report_markdown(run, top, cmp, branch),
                       file_name="blackbox_report.md")


# ------------------------------------------------------------------ tab 2
def eval_tab():
    split, causal = load_json("results.json"), load_json("results_causal.json")
    if not split and not causal:
        st.info("Run `python evaluate.py <db>` and `python replay_eval.py <db>` first; "
                "their numbers will appear here.")
        return
    if split:
        st.subheader("Finding the culprit step (held-out failed runs)")
        rows = [{"Method": k, "Top-1": f"{v['top1']:.1%}", "Top-3": f"{v['top3']:.1%}",
                 "MRR": f"{v['mrr']:.3f}", "AUROC": f"{v['auroc']:.3f}"} for k, v in split["split"].items()]
        st.table(pd.DataFrame(rows).set_index("Method"))
        st.subheader("Generalization: fault types the model never saw in training")
        rows = [{"Held-out fault": k, "Runs": v["n"], "Top-1": f"{v['top1']:.1%}", "Top-3": f"{v['top3']:.1%}"}
                for k, v in split["unseen_fault_types"].items()]
        st.table(pd.DataFrame(rows).set_index("Held-out fault"))
    if causal:
        st.subheader("Causal check: does patching the blamed step really fix the run?")
        rows = [{"Method": k, "Fixed on 1st try": f"{v['fixed_first_try']:.1%}",
                 "Avg replays needed": f"{v['avg_replays']:.2f}"} for k, v in causal["methods"].items()]
        st.table(pd.DataFrame(rows).set_index("Method"))
        e = causal["replay_efficiency"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Steps re-run (avg)", f"{e['avg_steps_executed']:.1f} of 6")
        c2.metric("Steps saved vs full re-run", f"{e['steps_saved_pct']:.0%}")
        c3.metric("Tokens saved vs full re-run", f"{e['tokens_saved_pct']:.0%}")
    st.caption("These numbers come from the most recent run of evaluate.py and replay_eval.py.")


# ------------------------------------------------------------------ tab 3
def patterns_tab(data):
    fails = data["failed_te"]
    if not fails:
        st.info("No held-out failures to analyse yet.")
        return
    tops, rules = [], Counter()
    for r in fails:
        sc = data["model"].score_run(r)
        t = max(sc, key=lambda s: s["score"])
        tops.append(t["name"])
        for e in t["evidence"]:
            rules[e.split("  (held")[0]] += 1
    st.subheader("Where do agent failures start?")
    st.caption(f"Diagnosed by Black Box across {len(fails)} failed runs it had never seen (no answer key used).")
    st.bar_chart(pd.Series(Counter(tops)).sort_values(ascending=False))
    st.subheader("Most commonly broken rules")
    st.table(pd.DataFrame([{"Rule": k, "Failures": v} for k, v in rules.most_common(6)]).set_index("Rule"))


# ------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Black Box", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Black Box")
    st.markdown("**A flight recorder for AI agents.** It finds the step that broke a run, "
                "then proves it by replaying a fix.")
    with st.sidebar:
        st.header("Setup")
        default = "data_real.db" if os.path.exists("data_real.db") else "data_mock.db"
        db = st.text_input("Recorded runs (database file)", value=default)
        st.caption(f"Agent mode: {agent.MODE}")
        st.markdown("**How it works**  \n1. Record every step of every run  \n"
                    "2. Learn what normal looks like from successes  \n"
                    "3. Rank the steps of a failed run  \n4. Replay a fix from that step only")
    if not os.path.exists(db):
        st.error(f"Cannot find {db}. Run: python generate.py --n 300 --db {db}")
        st.stop()
    with st.spinner("Learning from recorded runs (the first load takes a few seconds)..."):
        data = load_all(db, os.path.getmtime(db))
    t1, t2, t3 = st.tabs(["Debug a run", "How well does it work?", "Failure patterns"])
    with t1:
        debug_tab(data)
    with t2:
        eval_tab()
    with t3:
        patterns_tab(data)


if __name__ == "__main__":
    main()