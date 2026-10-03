"""Black Box: the demo app.   Run with:   python -m streamlit run app.py

Tabs:
  1. Debug a run          pick (or break live) a failed run -> diagnosis + evidence -> patch -> replay -> diff
  2. How well does it work?   evaluation numbers saved by evaluate.py and replay_eval.py
  3. Failure patterns     where do failures start, across many runs?
"""
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
from explain import (attach_counterfactual, build_explanation, check_grounding, nearest_clean,
                     render_text, to_agent_prompt, to_agent_report)
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

DEV = os.environ.get("BLACKBOX_DEV") == "1"


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner=False)
def load_all(db_path, mtime, use_lm=False):
    """Load recorded runs, hold out 30% as 'never seen', and train Black Box on the rest."""
    runs = load_runs(db_path)
    rng = random.Random(0)
    rng.shuffle(runs)
    cut = int(0.7 * len(runs))
    clean_tr, failed_tr = split_clean_failed(runs[:cut])
    _, failed_te = split_clean_failed(runs[cut:])
    lm_error = None
    try:
        model = Diagnoser("gbm", use_lm=use_lm).fit(clean_tr, failed_tr)
    except RuntimeError as e:
        if use_lm:
            lm_error = str(e)
            model = Diagnoser("gbm", use_lm=False).fit(clean_tr, failed_tr)
        else:
            raise
    return {"model": model, "failed_te": failed_te, "clean": split_clean_failed(runs)[0],
            "lm_enabled": bool(use_lm and model.use_lm), "lm_error": lm_error}


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


def llm_explanation(expl):
    """Optional (real mode): plain-language rewrite. It may only use the fact sheet below."""
    fallback = expl["summary"]
    prompt = ("An AI agent failed. Below is a fact sheet built from its recorded trace. In 3 short sentences, "
              "explain the root cause for a non-expert. Use ONLY facts from the sheet and cite evidence ids "
              "like [E1].\n\n" + render_text(expl))
    try:
        return call_llm(prompt, lambda: fallback)[0]
    except Exception:
        return fallback


def explanation_panel(expl, run, twin):
    rc, c, b = expl["root_cause"], expl["confidence"], expl["blast_radius"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Root cause", f"step {rc['step_idx']}: {rc['step_name']}")
    m2.metric("Ranker score", f"{c['ranker_score']:.0%}", f"{c['band']} confidence", delta_color="off")
    m3.metric("Blast radius (potential)", f"{b['potential_count']} step(s)",
              "reaches the final answer" if b["reaches_final_output"] else "contained", delta_color="off")
    st.info(expl["summary"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Observed**")
        for o in expl["observed"] or [None]:
            st.markdown(f"- `{o['field']}` = {short(o['value'], 90)}  [{o['fact']}]" if o else "- no rule-based observation")
        st.markdown("**Expected** (learned from successful runs)")
        for e in expl["expected"] or [None]:
            st.markdown(f"- {short(e['text'], 120)}  [{e['fact']}]" if e else "- none")
        st.markdown("**Evidence chain**")
        for e in expl["evidence"]:
            st.markdown(f"- **[{e['id']}]** " + (e.get("statement") or f"Rule broken: {e['rule']} (held in {e['held_in']})"))
    with right:
        st.markdown("**Impact: how the failure propagates**")
        if b["edges"]:
            st.markdown("  \n".join(f"`{a}` &rarr; `{z}`" for a, z in b["edges"]))
        st.caption("The error reaches the final answer." if b["reaches_final_output"]
                   else "No later step reads this result.")
        st.markdown("**Recommended corrective action**")
        ra = expl["recommended_action"]
        st.markdown(f"- {ra['primary']['text']}")
        for x in ra["also"]:
            st.markdown(f"- also: {x['text']}")
        p = ra["replay_plan"]
        st.markdown("**Alternative execution (replay plan)**")
        st.markdown(f"- Restore the checkpoint before step {p['restore_checkpoint_before_step']} "
                    f"(keeps {len(p['reuse_steps'])} earlier step(s)).")
        st.markdown(f"- Re-run only what depends on it: {', '.join(p['rerun_if_output_changes']) or 'nothing else'}.")
        if p["reuse_from_cache_after_root"]:
            st.markdown(f"- Reuse unchanged: {', '.join(p['reuse_from_cache_after_root'])}.")
    g = check_grounding(expl, run)
    if g["ok"]:
        st.caption(f"Grounding check passed: {g['checked']} claims in this explanation were re-verified "
                   "against the recorded trace. Nothing here is generated by a language model.")
    else:
        st.warning("Grounding check FAILED: " + "; ".join(g["problems"][:3]))
    if twin:
        with st.expander(f"Side by side with the closest successful run ({twin['task']})"):
            k = rc["step_idx"]
            st.markdown("This run's output at the suspect step:")
            st.code(short(run["steps"][k]["output"], 400), language="json")
            st.markdown("Successful run's output at the same step:")
            st.code(short(twin["steps"][k]["output"], 400), language="json")
    if agent.MODE == "real" and st.button("Ask Claude to put it in plain words"):
        st.session_state["expl_text"] = (run["run_id"], llm_explanation(expl))
    ex = st.session_state.get("expl_text")
    if ex and ex[0] == run["run_id"]:
        st.info(ex[1])


# ------------------------------------------------------------------ tab 1
def pick_run(data):
    mode = st.radio("Which failure do you want to investigate?",
                    ["Browse recorded failures", "Break it live"], horizontal=True)
    if mode == "Browse recorded failures":
        fails = data["failed_te"]
        if not fails:
            st.warning("No failed runs in the held-out set. Generate more data first.")
            return None
        # Choose a representative failure for the first screen: one where the model has
        # a clear separation between its top suspect and runner-up. This does not use
        # the hidden culprit label; it is purely a judge/demo usability choice.
        demo_scores = []
        for j, r in enumerate(fails):
            sc = data["model"].score_run(r)
            vals = sorted((x["score"] for x in sc), reverse=True)
            gap = vals[0] - vals[1] if len(vals) > 1 else vals[0]
            evidence = sum(len(x["evidence"]) for x in sc)
            demo_scores.append((gap, evidence, -j, j))
        demo_idx = max(demo_scores)[-1]
        st.caption("Demo view starts on a failure with clear diagnostic evidence; you can choose any held-out failure.")
        i = st.selectbox("Failed run (held out: the model never saw it in training)", list(range(len(fails))),
                         index=demo_idx,
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


def what_if_panel(run, scored, session):
    ranked = sorted(scored, key=lambda x: -x["score"])
    top_i = ranked[0]["step_idx"]
    runner_i = ranked[1]["step_idx"] if len(ranked) > 1 else top_i
    key = run["run_id"]
    all_paths = st.session_state.setdefault("what_if", {})
    if st.button("Run 3-path investigation", type="primary", key=f"whatif-{key}"):
        all_paths[key] = {"retry": session.retry(), "predicted": session.fork(Patch(top_i, "fix")), "runner_up": session.fork(Patch(runner_i, "fix"))}
    paths = all_paths.get(key)
    if not paths:
        st.caption("Compare an unchanged retry, the predicted fix, and a runner-up fix.")
        return
    labels = {"retry": "Unchanged retry", "predicted": "Predicted-step fix", "runner_up": "Runner-up fix"}
    rows = [{"Path": labels[k], "Result": "SUCCESS" if b["success"] else "FAIL", "Re-run": f"{b['executed']}/{b['n_steps']}", "Cached": b["cached"], "Tokens": b["tokens_used"], "Latency": f"{b['latency_ms_used']:.2f} ms"} for k,b in paths.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    pred_cmp, run_cmp = compare(run, paths["predicted"]), compare(run, paths["runner_up"])
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted intervention", "SUCCESS" if paths["predicted"]["success"] else "FAIL")
    p2.metric("Runner-up intervention", "SUCCESS" if paths["runner_up"]["success"] else "FAIL")
    p3.metric("First divergence",
              f"step {pred_cmp['first_divergence']}" if pred_cmp["first_divergence"] is not None else "none")
    matrix = [{"Step": a["name"], "Predicted fix": "CHANGED" if not a["same"] else "same", "Runner-up fix": "CHANGED" if not b["same"] else "same", "Predicted status": a["status"], "Runner-up status": b["status"]} for a,b in zip(pred_cmp["rows"], run_cmp["rows"])]
    st.markdown("**Trace comparison: predicted fix vs runner-up**")
    st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)
    if paths["predicted"]["success"] and not paths["runner_up"]["success"]:
        st.success("The predicted intervention fixes the failure while the runner-up does not — counterfactual evidence supporting the diagnosis.")
    elif paths["predicted"]["success"]:
        st.info("The predicted intervention succeeds. Compare the traces above to inspect what changed.")
    else:
        st.warning("This is a non-repair case: the predicted intervention did not fix the run. Try another held-out failure to demonstrate a successful counterfactual.")
    st.caption("Counterfactual replay supports the diagnosis when the predicted intervention succeeds; it is not proof.")


def debug_tab(data):
    run = pick_run(data)
    if run is None:
        return
    st.markdown(f"**Task:** {run['task']}  \n**Expected:** ${run['expected']:.2f} &nbsp;|&nbsp; "
                f"**Agent said:** {run['final_answer']}")
    scored = data["model"].score_run(run)
    top = max(scored, key=lambda s: s["score"])
    if data.get("lm_enabled"):
        lm_rank = sorted(scored, key=lambda s: -s["features"].get("lm_nll_z", -999))
        st.caption("Pretrained-model signal (auxiliary): output-surprise z-score; higher means more surprising than clean runs for this step. It is not a failure probability.")
        st.dataframe(pd.DataFrame([{
            "Step": s["name"], "Ranker": f"{s['score']:.0%}",
            "LM surprise z": f"{s['features'].get('lm_nll_z', 0):.2f}",
            "LM max-token surprise": f"{s['features'].get('lm_max_nll', 0):.2f}"
        } for s in lm_rank]), use_container_width=True, hide_index=True)
    twin = nearest_clean(run, data["clean"])
    expl = build_explanation(run, scored, twin)

    st.subheader("1. Where did it go wrong?")
    # The answer key exists for development only; it is never shown in the judge-facing app.
    reveal = st.checkbox("DEV: reveal the true culprit (answer key)", value=False) if DEV else False
    st.markdown(timeline_html(scored, top["step_idx"], run["culprit_step"] if reveal else None),
                unsafe_allow_html=True)
    st.caption("Each box is one step of the run, shaded by how suspicious Black Box finds it. "
               "The glowing box is its top suspect.")

    st.subheader("2. Why does Black Box think so?")
    explanation_panel(expl, run, twin)

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
        attach_counterfactual(expl, cmp, branch)
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
        m4.metric("Latency saved", f"{branch.get('latency_saved_pct', 0):.0%}")
        st.markdown(diff_html(cmp), unsafe_allow_html=True)
        v = expl["verification"]
        st.markdown(f"**Verdict:** {v['statement']}")
        st.caption(f"Blast radius: {expl['blast_radius']['potential_count']} step(s) could be affected "
                   f"(from the dependency graph); {expl['blast_radius']['observed_count']} actually changed "
                   "after the fix.")
        if cmp["first_divergence"] is not None:
            st.caption(f"The two runs first differ at step {cmp['first_divergence']}. "
                       "Highlighted rows are steps whose output changed.")
    st.subheader("4. What-If Lab: alternative executions")
    sessions = st.session_state.setdefault("sessions", {})
    what_if_panel(run, scored, sessions.get(run["run_id"], ReplaySession(run)))
    st.subheader("5. Reports")
    report = to_agent_report(expl)
    with st.expander("Agent-readable report (what Black Box hands back to the agent)"):
        st.code(to_agent_prompt(report))
        st.json(report)
    c1, c2 = st.columns(2)
    c1.download_button("Download human explanation (.txt)", render_text(expl),
                       file_name="blackbox_failure_explanation.txt")
    c2.download_button("Download agent report (.json)", json.dumps(report, indent=2, default=str),
                       file_name="blackbox_agent_report.json")


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
    lm_ab = load_json("results_lm.json")
    if lm_ab:
        st.subheader("Ablation: optional pretrained-model signal")
        lm_rows = [
            {"System": "Frozen base-v1", **{k: f"{v:.1%}" if k in ("top1", "top3") else f"{v:.3f}" for k, v in lm_ab["base_v1"].items()}},
            {"System": "Base + local pretrained LM", **{k: f"{v:.1%}" if k in ("top1", "top3") else f"{v:.3f}" for k, v in lm_ab["base_plus_local_lm"].items()}},
        ]
        st.table(pd.DataFrame(lm_rows).set_index("System"))
        st.caption(f"Base evaluation: {lm_ab['base_seconds']:.1f}s; base+LM evaluation: {lm_ab['lm_total_seconds']:.1f}s. " + lm_ab["conclusion"])
    if causal:
        st.subheader("Causal check: does patching the blamed step really fix the run?")
        rows = [{"Method": k, "Fixed on 1st try": f"{v['fixed_first_try']:.1%}",
                 "Avg replays needed": f"{v['avg_replays']:.2f}"} for k, v in causal["methods"].items()]
        st.table(pd.DataFrame(rows).set_index("Method"))
        e = causal["replay_efficiency"]
        # Canonical replay-efficiency display: derive steps avoided from the
        # average replayed-step count so the two displayed numbers can never
        # contradict each other. The persisted tokens-saved measurement is
        # taken directly from results_causal.json.
        avg_replayed = float(e.get("avg_steps_executed", 0.0))
        total_steps = 6
        steps_avoided = 1.0 - (avg_replayed / total_steps) if total_steps else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Average steps replayed", f"{avg_replayed:.2f} of {total_steps}")
        c2.metric("Steps avoided vs full re-run", f"{steps_avoided:.1%}")
        c3.metric("Tokens saved vs full re-run", f"{e['tokens_saved_pct']:.1%}")
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
    if rules:
        st.table(pd.DataFrame([{"Rule": k, "Failures": v} for k, v in rules.most_common(6)]).set_index("Rule"))
    else:
        st.caption("No learned rule was broken at the top suspects of these runs.")



# ---------------------------------------------------------------- Mission Control
def mission_control_tab():
    split = load_json("results.json")
    causal = load_json("results_causal.json")
    lm_ab = load_json("results_lm.json")

    st.markdown("## 🛰 Mission Control")
    st.caption("A judge-facing overview of what Black Box observes, diagnoses, changes, and verifies.")

    if not split or not causal:
        st.warning("Benchmark results are not available yet. Run evaluate.py and replay_eval.py first.")
        return

    # Headline metrics: descriptive benchmark measurements only.
    bb = split["split"].get("Black Box (boosted trees)", {})
    unseen = split.get("unseen_fault_types", {})
    unseen_top1 = sum(v["top1"] for v in unseen.values()) / len(unseen) if unseen else 0.0
    repair = causal["methods"].get("Black Box", {})
    eff = causal.get("replay_efficiency", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top-1 culprit localization", f"{bb.get('top1', 0):.1%}")
    m2.metric("Unseen-fault Top-1", f"{unseen_top1:.1%}")
    m3.metric("First-try repair", f"{repair.get('fixed_first_try', 0):.1%}")
    avg_replayed = float(eff.get("avg_steps_executed", 0.0))
    total_steps = 6
    steps_avoided = 1.0 - (avg_replayed / total_steps) if total_steps else 0.0
    m4.metric("Steps avoided", f"{steps_avoided:.1%}")

    st.markdown("### Observe → Diagnose → Explain → Intervene → Replay → Verify")
    stages = [
        ("1", "Observe", "Record every step, input, output, timing and execution metadata."),
        ("2", "Diagnose", "Rank suspicious steps using learned normal-behavior signals."),
        ("3", "Explain", "Ground the suspected cause in observed vs expected trace evidence."),
        ("4", "Intervene", "Try the predicted step and a runner-up as controlled alternatives."),
        ("5", "Replay", "Restore a checkpoint and rerun only affected downstream work."),
        ("6", "Verify", "Compare branch traces and check whether the final result changes."),
    ]
    cols = st.columns(6)
    for col, (num, title, body) in zip(cols, stages):
        with col:
            st.markdown(f"**{num}. {title}**")
            st.caption(body)

    st.markdown("### What the benchmark measures")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Diagnosis**")
        st.write(f"Held-out failed runs: **{bb.get('n', 0)}**")
        st.write(f"Top-3 localization: **{bb.get('top3', 0):.1%}**")
        st.write(f"MRR: **{bb.get('mrr', 0):.3f}**")
        st.write(f"AUROC: **{bb.get('auroc', 0):.3f}**")
        st.markdown("**Replay efficiency**")
        st.write(f"Average steps replayed: **{avg_replayed:.2f} / {total_steps}**")
        st.write(f"Steps avoided: **{steps_avoided:.1%}**")
        st.write(f"Tokens saved: **{eff.get('tokens_saved_pct', 0):.1%}**")
    with c2:
        st.markdown("**Generalization**")
        if unseen:
            unseen_rows = [{"Fault type": k, "Runs": v["n"], "Top-1": f"{v['top1']:.1%}"}
                           for k, v in unseen.items()]
            st.dataframe(pd.DataFrame(unseen_rows), use_container_width=True, hide_index=True)
        st.markdown("**Counterfactual verification**")
        st.write("The benchmark compares the predicted intervention against unchanged and runner-up paths.")
        st.write("A successful intervention is treated as supporting evidence, not proof of causality.")

    if lm_ab:
        st.markdown("### Pretrained-model ablation")
        lm1, lm2, lm3 = st.columns(3)
        lm1.metric("Base top-1", f"{lm_ab['base_v1']['top1']:.1%}")
        lm2.metric("Base + local LM", f"{lm_ab['base_plus_local_lm']['top1']:.1%}")
        lm3.metric("LM evaluation time", f"{lm_ab['lm_total_seconds']:.1f}s")
        st.caption(lm_ab["conclusion"])

    st.info(
        "Benchmark scope: this is a controlled fault-injection benchmark on a deterministic 6-step shopping agent. "
        "The results demonstrate the debugging workflow and measured replay behavior; they are not a claim of "
        "performance on arbitrary real-world agents."
    )

    st.markdown("### Live demo path")
    st.code("Mission Control → Investigation → Run 3-path investigation → Trace comparison → Evaluation", language="text")
    st.caption("Use data_mock.db for the reproducible benchmark demo. Keep data_real.db for separate live-mode testing.")

# ------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Black Box", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Black Box")
    st.markdown("**A flight recorder for AI agents.** It finds the step that broke a run, "
                "then proves it by replaying a fix.")
    with st.sidebar:
        st.header("Setup")
        default = "data_mock.db" if os.path.exists("data_mock.db") else "data_real.db"
        db = st.text_input("Recorded runs (database file)", value=default)
        st.caption(f"Agent mode: {agent.MODE}")
        use_lm = os.environ.get("BLACKBOX_LM") == "1"
        st.caption("Local pretrained-model signal: ON (experimental auxiliary ranker)" if use_lm else "Local pretrained-model signal: OFF (frozen base-v1 benchmark)")
        st.markdown("**How it works**  \n1. Record every step of every run  \n"
                    "2. Learn what normal looks like from successes  \n"
                    "3. Rank the steps of a failed run  \n4. Replay a fix from that step only")
    if not os.path.exists(db):
        st.error(f"Cannot find {db}. Run: python generate.py --n 300 --db {db}")
        st.stop()
    db_run_count = None
    try:
        import sqlite3
        with sqlite3.connect(db) as conn:
            db_run_count = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    except Exception:
        pass
    if db_run_count is not None and db_run_count < 30:
        st.warning(
            f"{db} contains only {db_run_count} recorded runs. That is too small for a stable held-out diagnosis demo. "
            "For the benchmark/demo, use data_mock.db (300 runs). Real-mode runs can be kept separately for live testing."
        )
    with st.spinner("Learning from recorded runs (the first load takes a few seconds)..."):
        data = load_all(db, os.path.getmtime(db), use_lm=use_lm)
    if data.get("lm_error"):
        st.warning("Local LM was requested but is unavailable; Black Box is using the frozen base-v1 model. " + data["lm_error"])
    t1, t2, t3, t4 = st.tabs(["🛰 Mission Control", "🔬 Investigation", "📊 Evaluation", "🧩 Failure patterns"])
    with t1:
        mission_control_tab()
    with t2:
        debug_tab(data)
    with t3:
        eval_tab()
    with t4:
        patterns_tab(data)


if __name__ == "__main__":
    main()
