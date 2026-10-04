"""Structured, evidence-grounded failure explanation and blast radius.

Every statement in an explanation is built from one of three sources, and nothing else:
  1. the recorded trace (what each step actually received and produced),
  2. rules learned from successful runs (what normally holds),
  3. the declared step-dependency graph (replay.READS: which step reads which result).
No language model writes any of it, so it cannot invent facts. check_grounding() re-checks
every quoted value against the trace to prove it.

One explanation object serves two readers:
  - render_text()        the version for a human who is monitoring the agent
  - to_agent_report()    the same facts as a machine-readable report an agent can act on
                         (to_agent_prompt() turns that report into plain text)

Workflow:  hypothesis (build_explanation)  ->  experiment (a replay)  ->
           attach_counterfactual() records whether the experiment supports the diagnosis.
"""
import json
import re

from invariants import _fmt as fmt
from invariants import key, leaves
from replay import READS

TASK_RE = re.compile(r"(\d+) units of (.+?) cost with a (\d+)% discount")
# When a step breaks several rules, lead with the one that says most about the fix.
PRIORITY = ["formula", "contains_input_value", "copied_from_input", "substring", "known_values"]
LIMITS = [
    "Rules are learned from successful runs of this agent; a failure no rule covers is ranked by weaker signals.",
    "The ranker score is a ranking signal, not a calibrated probability.",
    "Potential blast radius comes from the declared dependency graph; observed blast radius needs a replay.",
]


# ------------------------------------------------------- closest successful run
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


# ----------------------------------------------------------- dependency graph
def downstream(root, names):
    """Steps that read the root step's result, directly or through other steps."""
    hit, frontier, grew = set(), {root}, True
    while grew:
        grew = False
        for n in names:
            if n not in hit and n != root and any(r in frontier for r in READS.get(n, [])):
                hit.add(n)
                frontier.add(n)
                grew = True
    return [n for n in names if n in hit]


def edges_within(root, down):
    group = {root, *down}
    return [[src, dst] for dst in down for src in READS.get(dst, []) if src in group]


# ----------------------------------------------------------------- the builder
def _action_for(v, step_name):
    """Turn one broken rule into a concrete corrective action plus a machine-readable hint."""
    P, kind, src = v["out_path"], v["kind"], (v["in_paths"][0] if v["in_paths"] else None)
    if kind == "formula" and v.get("expected_value") is not None:
        return {"kind": "recompute",
                "text": f"Recompute `{P}` from its inputs: {v['expected_text']}.",
                "hint": {"field": P, "set_to": v["expected_value"], "inputs": v.get("inputs", {})}}
    if kind == "contains_input_value":
        return {"kind": "re_run_with_requirement",
                "text": f"Re-run `{step_name}` so that `{P}` includes {fmt(v['expected_value'])} "
                        f"(the value of `{src}`).",
                "hint": {"field": P, "must_include": v["expected_value"], "source": src}}
    if kind == "copied_from_input":
        return {"kind": "choose_from_source",
                "text": f"Choose `{P}` from the values available in `{src}`: {fmt(v['expected_values'])}.",
                "hint": {"field": P, "choose_from": v["expected_values"], "source": src}}
    if kind == "substring":
        return {"kind": "quote_from_source", "text": f"Quote `{P}` from the text of `{src}`.",
                "hint": {"field": P, "source": src}}
    return {"kind": "check_allowed_values",
            "text": f"Check `{P}`: successful runs only ever produced {fmt(v.get('expected_values'))}.",
            "hint": {"field": P, "allowed": v.get("expected_values")}}


def _cut(text, n):
    return text if len(text) <= n else text[:n - 3] + "..."


def _obs(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def build_explanation(run, scored, twin=None):
    """scored = Diagnoser.score_run(run).  twin = closest successful run (optional)."""
    names = [s["name"] for s in run["steps"]]
    ranked = sorted(scored, key=lambda s: -s["score"])
    top, runner = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    k, name = top["step_idx"], top["name"]
    viols = sorted(top["violations"], key=lambda v: PRIORITY.index(v["kind"]) if v["kind"] in PRIORITY else 99)[:3]

    facts, input_facts, observed, expected, evidence = [], [], [], [], []
    for n, v in enumerate(viols, 1):
        facts.append({"id": f"F{n}", "step_idx": k, "path": v["out_path"], "value": v["observed"]})
        for p, val in (v.get("inputs") or {}).items():
            input_facts.append({"step_idx": k, "path": p, "value": val})
        observed.append({"fact": f"F{n}", "field": v["out_path"], "value": _obs(v["observed"])})
        expected.append({"fact": f"F{n}", "text": v["expected_text"], "values": v["expected_values"],
                         "value": v["expected_value"], "rule_kind": v["kind"], "held_in": v["held_in"]})
        evidence.append({"id": f"E{len(evidence) + 1}", "type": "learned_rule_violation", "step_idx": k,
                         "rule": v["rule"], "kind": v["kind"], "held_in": v["held_in"],
                         "observed": _obs(v["observed"]), "expected": v["expected_text"],
                         "fields": [v["out_path"], *v["in_paths"]]})

    def add(kind, **kw):
        evidence.append({"id": f"E{len(evidence) + 1}", "type": kind, **kw})

    # where does the first rule break?
    first = min((s["step_idx"] for s in scored if s["violations"]), default=None)
    if first is None:
        add("no_rule_broken", step_idx=k, statement="No learned rule is broken anywhere in this run; "
            "the ranking rests on step position, output size and timing signals only.")
    elif first == k:
        add("first_divergence", step_idx=k, statement=f"Step {k} ({name}) is the earliest step in this run "
            f"that breaks a learned rule; no earlier step does.")
    else:
        add("earlier_violation", step_idx=first, statement=f"Step {first} ({names[first]}) also breaks "
            f"learned rules and comes before step {k}; the ranker still puts step {k} first.")

    # contrast with the closest successful run
    if twin is not None and k < len(twin["steps"]):
        t = twin["steps"][k]
        add("contrast_with_success", step_idx=k, twin_run=twin["run_id"], twin_task=twin["task"],
            this_output=run["steps"][k]["output"], twin_output=t["output"],
            differs=run["steps"][k]["output"] != t["output"],
            statement=f"Closest successful run ({twin['task']}) produced {_cut(fmt(t['output']), 90)} at this step.")

    # dependency graph: how far can the error travel?
    down = downstream(name, names)
    flagged = [n for n in down if scored[names.index(n)]["violations"]]
    for n in down:
        add("downstream_dependency", step_idx=names.index(n),
            statement=f"{n} reads the result of {name} ({'also breaks learned rules' if n in flagged else 'no rule broken, but it depends on the suspect'}).")
    final_step = names[-1]
    blast = {"root": name, "potential_steps": down, "potential_count": len(down),
             "flagged_steps": flagged, "flagged_count": len(flagged),
             "reaches_final_output": final_step == name or final_step in down,
             "edges": edges_within(name, down), "observed_steps": None, "observed_count": None}

    # weaker model signals, quoted only when they are unusual
    ratio = top["features"].get("len_ratio", 1.0)
    if abs(ratio - 1) > 0.25:
        add("output_size", step_idx=k, ratio=ratio,
            statement=f"This step's output is {ratio:.2f}x the usual size for {name} in successful runs.")

    # corrective action + replay plan
    actions = [_action_for(v, name) for v in viols]
    primary = actions[0] if actions else {"kind": "inspect", "hint": None, "text":
        f"No learned rule explains the suspicion. Inspect the input and output of `{name}` directly."}
    reuse_after = [n for n in names[k + 1:] if n not in down]
    plan = {"restore_checkpoint_before_step": k, "reuse_steps": names[:k], "apply_correction_to": name,
            "rerun_if_output_changes": down, "reuse_from_cache_after_root": reuse_after}

    margin = top["score"] - runner["score"] if runner else None
    band = "high" if top["score"] >= 0.8 and (margin is None or margin >= 0.3) else \
           "medium" if top["score"] >= 0.5 else "low"
    clause = (f"produced {fmt(observed[0]['value'])[:70]} where successful runs show {expected[0]['text'][:90]}"
              if observed else "is ranked highest by weaker signals")
    summary = (f"Step {k} ({name}) is the likeliest root cause: it {clause}. "
               + (f"The error can travel to {len(down)} downstream step(s)"
                  + (", including the final answer." if blast["reaches_final_output"] else ".")
                  if down else "No later step reads its result directly."
                  if not blast["reaches_final_output"] else "It is the final step, so the answer itself is wrong."))
    return {"schema": "blackbox.explanation.v1",
            "run": {"run_id": run["run_id"], "task": run["task"], "expected": run["expected"],
                    "actual": run["final_answer"]},
            "root_cause": {"step_idx": k, "step_name": name, "step_type": top["type"]},
            "confidence": {"ranker_score": top["score"], "band": band, "margin_over_runner_up": margin,
                           "runner_up": ({"step_idx": runner["step_idx"], "step_name": runner["name"],
                                          "score": runner["score"]} if runner else None)},
            "summary": summary, "observed": observed, "expected": expected, "evidence": evidence,
            "facts": facts, "input_facts": input_facts, "blast_radius": blast,
            "recommended_action": {"primary": primary, "also": actions[1:], "replay_plan": plan},
            "verification": None, "limits": list(LIMITS)}


# ----------------------------------------------------- the experiment's verdict
def attach_counterfactual(expl, cmp, branch):
    """Record what a replay showed. Counterfactual EVIDENCE, not proof."""
    k = expl["root_cause"]["step_idx"]
    patched = branch["patch"].step_idx
    changed = [r["name"] for r in cmp["rows"] if not r["same"] and r["idx"] != patched]
    expl["blast_radius"]["observed_steps"] = changed
    expl["blast_radius"]["observed_count"] = len(changed)
    if patched == k and cmp["flipped"]:
        verdict = (f"Patching step {k} and replaying turned the run from FAILED to SUCCESS. "
                   "This is counterfactual evidence that supports the diagnosis (not proof).")
    elif patched == k:
        verdict = f"Patching step {k} did not fix the run, so this patch alone does not support the diagnosis."
    elif cmp["flipped"]:
        verdict = f"Patching step {patched} (not the suspect) also fixed the run, so the diagnosis is not unique."
    else:
        verdict = (f"Patching step {patched} (not the suspect) did not fix the run, "
                   "which is consistent with the diagnosis but does not confirm it.")
    expl["verification"] = {
        "patched_step": patched, "patch": branch["patch"].describe(), "patched_the_suspect": patched == k,
        "outcome_before": "FAILED" if not cmp["orig_success"] else "SUCCESS",
        "outcome_after": "SUCCESS" if cmp["new_success"] else "FAILED", "flipped": cmp["flipped"],
        "first_divergence": cmp["first_divergence"], "steps_changed": changed,
        "steps_skipped_by_checkpoint": branch["restored"], "steps_reused_from_cache": branch["cached"],
        "steps_rerun": branch["executed"], "steps_saved_pct": branch["steps_saved_pct"],
        "tokens_saved_pct": branch["tokens_saved_pct"], "new_answer": cmp["new_answer"], "statement": verdict}
    return expl


# -------------------------------------------------------------- grounding check
def check_grounding(expl, run):
    """Re-check every quoted value against the recorded trace. ok=True means nothing was invented."""
    problems, checked = [], 0
    for f in expl["facts"]:
        st = run["steps"][f["step_idx"]]
        have = leaves(st["output"], st["name"]) if st["output"] is not None else {}
        _compare(f, have.get(f["path"], []), problems)
        checked += 1
    for f in expl["input_facts"]:
        st = run["steps"][f["step_idx"]]
        _compare(f, leaves(st["input"], "").get(f["path"], []), problems)
        checked += 1
    n = len(run["steps"])
    for e in expl["evidence"]:
        checked += 1
        if not 0 <= e["step_idx"] < n:
            problems.append(f"{e['id']}: refers to a step that does not exist")
    return {"checked": checked, "problems": problems, "ok": not problems}


def _compare(fact, vals, problems):
    want = fact["value"] if isinstance(fact["value"], list) else [fact["value"]]
    pool = {key(v) for v in vals}
    if not want:
        if vals:
            problems.append(f"{fact['path']}: expected no value but the trace has {vals!r}")
        return
    for w in want:
        if key(w) not in pool:
            problems.append(f"{fact['path']}: {w!r} is not in the trace at step {fact['step_idx']}")


# -------------------------------------------------------------------- rendering
def render_text(expl):
    r, rc, c = expl["run"], expl["root_cause"], expl["confidence"]
    L = ["FAILURE EXPLANATION", "",
         f"Root cause: step {rc['step_idx']} - {rc['step_name']}   "
         f"(ranker score {c['ranker_score']:.0%}, confidence band: {c['band']})",
         f"Task: {r['task']}", f"Expected result: ${r['expected']:.2f}   Agent produced: {r['actual']!r}", "",
         expl["summary"], "", "OBSERVED"]
    L += [f"  - `{o['field']}` = {fmt(o['value'])[:90]}   [{o['fact']}]" for o in expl["observed"]] or ["  - (no rule-based observation)"]
    L += ["EXPECTED"]
    L += [f"  - {e['text']}   (rule held in {e['held_in']})   [{e['fact']}]" for e in expl["expected"]] or ["  - (none)"]
    L += ["", "EVIDENCE"]
    for e in expl["evidence"]:
        body = e.get("statement") or f"Rule broken: {e['rule']} (held in {e['held_in']})"
        L.append(f"  [{e['id']}] {body}")
    b = expl["blast_radius"]
    L += ["", f"IMPACT  (potential blast radius: {b['potential_count']} downstream step(s)"
          + (f"; observed after replay: {b['observed_count']}" if b["observed_count"] is not None else "") + ")"]
    if b["edges"]:
        L += [f"  {a} -> {z}" for a, z in b["edges"]]
    L.append("  The error reaches the final answer." if b["reaches_final_output"] else "  No later step reads this result.")
    ra = expl["recommended_action"]
    L += ["", "RECOMMENDED CORRECTIVE ACTION", f"  {ra['primary']['text']}"]
    L += [f"  Also: {a['text']}" for a in ra["also"]]
    p = ra["replay_plan"]
    L += ["", "ALTERNATIVE EXECUTION (replay plan)",
          f"  Restore the checkpoint taken before step {p['restore_checkpoint_before_step']} "
          f"(reuses {len(p['reuse_steps'])} earlier step(s)), apply the correction to {p['apply_correction_to']},",
          f"  then re-run only what depends on it: {', '.join(p['rerun_if_output_changes']) or 'nothing else'}."]
    if p["reuse_from_cache_after_root"]:
        L.append(f"  Reused unchanged from cache: {', '.join(p['reuse_from_cache_after_root'])}.")
    v = expl["verification"]
    L += ["", "VERIFICATION", f"  {v['statement']}" if v else "  Not run yet. Replay a fix to test this diagnosis.", "", "LIMITS"]
    L += [f"  - {x}" for x in expl["limits"]]
    return "\n".join(L)


def to_agent_report(expl):
    """The same facts, shaped for an agent: what failed, where, what to change, what to reuse."""
    ra, b, rc = expl["recommended_action"], expl["blast_radius"], expl["root_cause"]
    plan = ra["replay_plan"]
    return {"schema": "blackbox.agent_report.v1", "run_id": expl["run"]["run_id"],
            "task": expl["run"]["task"], "expected_result": expl["run"]["expected"],
            "actual_result": expl["run"]["actual"],
            "failed_step": {"index": rc["step_idx"], "name": rc["step_name"], "type": rc["step_type"]},
            "confidence": {"ranker_score": round(expl["confidence"]["ranker_score"], 3),
                           "band": expl["confidence"]["band"]},
            "problems": [{"field": o["field"], "observed": o["value"], "expected": e["text"],
                          "rule_kind": e["rule_kind"], "evidence_id": o["fact"]}
                         for o, e in zip(expl["observed"], expl["expected"])],
            "affected_downstream_steps": b["potential_steps"], "reaches_final_output": b["reaches_final_output"],
            "recovery_plan": {"restore_checkpoint_before_step": plan["restore_checkpoint_before_step"],
                              "reuse_steps": plan["reuse_steps"],
                              "apply_correction": {"step": rc["step_name"], "action": ra["primary"]["kind"],
                                                   "instruction": ra["primary"]["text"], "hint": ra["primary"]["hint"]},
                              "rerun_if_output_changes": plan["rerun_if_output_changes"],
                              "reuse_from_cache": plan["reuse_from_cache_after_root"]},
            "alternative_corrections": [{"action": a["kind"], "instruction": a["text"], "hint": a["hint"]}
                                        for a in ra["also"]],
            "verification": expl["verification"],
            "caveats": ["This diagnosis is a hypothesis. Re-run from the checkpoint and check the final "
                        "result before trusting it."]}


def to_agent_prompt(report):
    """Plain-text form of the agent report, ready to hand to an agent."""
    rp = report["recovery_plan"]
    L = ["BLACK BOX FAILURE REPORT",
         f"Your run failed. Expected result: ${report['expected_result']:.2f}. You produced: {report['actual_result']!r}",
         f"Likely failing step: [{report['failed_step']['index']}] {report['failed_step']['name']} "
         f"(confidence: {report['confidence']['band']}).", "Problem:"]
    L += [f"  - {p['field']} = {fmt(p['observed'])[:90]}; expected {p['expected']} [{p['evidence_id']}]"
          for p in report["problems"]] or ["  - no rule-based problem found; inspect the step manually"]
    L += ["Do this:",
          f"  1. Restore your state from before step {rp['restore_checkpoint_before_step']} "
          f"(keep steps: {', '.join(rp['reuse_steps']) or 'none'}).",
          f"  2. {rp['apply_correction']['instruction']}",
          f"  3. Re-run only: {', '.join(rp['rerun_if_output_changes']) or 'nothing else'}. "
          f"Reuse unchanged results of: {', '.join(rp['reuse_from_cache']) or 'none'}.",
          "  4. Check the final answer. " + report["caveats"][0]]
    return "\n".join(L)
