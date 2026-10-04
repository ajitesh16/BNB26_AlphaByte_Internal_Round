"""Checkpointed replay, alternative execution and trace comparison.

IDEA
  Every logged step stores the exact state the agent had BEFORE it ran (a checkpoint).
  To investigate a failure we do not re-run everything from the start:
    1. restore the checkpoint of the suspect step      -> earlier steps are skipped
    2. apply a patch to that step                       -> "alternative execution"
    3. re-run forward, but reuse any later step whose inputs did not change (cache hit)
  Only steps that really depend on the patch are recomputed. That is the saving.

THE SIMULATED BUG
  Our agent's bug was injected, so a replay must keep the bug until it is patched
  (a real agent would simply behave the same way again). That is why ReplaySession
  reads the run's fault label. The DIAGNOSIS model never sees that label.

THREE KINDS OF PATCH
  "fix"        re-run the suspect step with its bug removed (e.g. repaired tool or prompt)
  "override"   replace the step's output with a value you choose
  "edit_input" change what the step receives, then re-run it as-is
"""
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from agent import FAULTS, STEP_NAMES, STEPS

# Which earlier results does each step actually read? Only these go into its cache key,
# so a change somewhere else does not force an unnecessary re-run.
READS = {
    "parse_question":   ["question"],
    "retrieve_catalog": ["parse_question"],
    "pick_price":       ["parse_question", "retrieve_catalog"],
    "check_stock":      ["parse_question"],
    "calculate":        ["parse_question", "pick_price"],
    "write_answer":     ["calculate", "check_stock"],
}


@dataclass
class Patch:
    step_idx: int
    mode: str = "fix"                     # "fix" | "override" | "edit_input"
    value: Any = None                     # override: the new output
    edits: Optional[dict] = None          # edit_input: {state_key: new_value}

    def describe(self):
        return {"fix": "re-run with the bug fixed",
                "override": f"output replaced by {self.value!r}",
                "edit_input": f"input edited ({list((self.edits or {}))})"}[self.mode]


def grade(expected, final):
    """Same success check as agent.py."""
    return f"${expected:.2f}" in str(final).replace(",", "")


class ReplaySession:
    """Everything needed to investigate ONE failed run. Forks share one cache."""

    def __init__(self, run):
        self.run = run
        self.n = len(STEPS)
        c = run["culprit_step"]
        self.fault = (STEP_NAMES[c], run["fault_kind"]) if c is not None else None
        self.cache = {}                                       # key -> (output, tokens, error)
        for s in run["steps"]:                                # seed with the original trace
            self.cache[self._key(s["name"], s["input"], False)] = (s["output"], s["tokens"], s["error"], s.get("latency_ms", 0.0))
        self.full_tokens = sum(s["tokens"] for s in run["steps"])

    def _key(self, name, state, fixed):
        deps = {k: state[k] for k in READS[name]}
        is_fixed = bool(fixed and self.fault and self.fault[0] == name)
        blob = json.dumps([name, deps, is_fixed], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _execute(self, i, state, fixed):
        name, _, fn = STEPS[i]
        out, tokens = fn(state)
        if self.fault and self.fault[0] == name and not fixed:
            out = FAULTS[name][self.fault[1]](out, state)     # the bug is still there
        return out, tokens

    def retry(self):
        """Full unchanged retry: bypass cache and keep the injected fault."""
        state = copy.deepcopy(self.run["steps"][0]["input"]) if self.run["steps"] else {"question": self.run.get("task", "")}
        log, final, crashed = [], "", False
        tokens_used, latency_used = 0, 0.0
        for i, (name, type_, _) in enumerate(STEPS):
            error = None
            try:
                t0 = time.perf_counter()
                out, tok = self._execute(i, state, fixed=False)
                step_latency_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                step_latency_ms = (time.perf_counter() - t0) * 1000
                out, tok, error = None, 0, f"{type(e).__name__}: {e}"
            state[name] = out
            tokens_used += tok
            latency_used += step_latency_ms
            log.append({"idx": i, "name": name, "type": type_, "status": "recomputed", "output": copy.deepcopy(out), "error": error, "latency_ms": step_latency_ms, "tokens": tok})
            if error:
                crashed = True
                break
            if name == "write_answer":
                final = out
        ok = (not crashed) and grade(self.run["expected"], final)
        return {"patch": None, "steps": log, "final_answer": str(final), "success": bool(ok), "n_steps": self.n, "restored": 0, "executed": len(log), "cached": 0, "tokens_used": tokens_used, "tokens_full_rerun": self.full_tokens, "tokens_saved_pct": 0.0, "latency_ms_used": latency_used, "latency_ms_full": sum(float(x.get("latency_ms", 0.0)) for x in self.run["steps"]), "latency_saved_pct": 0.0}

    def fork(self, patch):
        """Create a new branch of this run with `patch` applied, and return its full result."""
        old, k = self.run["steps"], patch.step_idx
        state = copy.deepcopy(old[k]["input"])                # CHECKPOINT: state before step k
        log = [{"idx": i, "name": old[i]["name"], "type": old[i]["type"], "status": "restored",
                "output": copy.deepcopy(old[i]["output"]), "error": old[i]["error"], "latency_ms": 0.0, "tokens": 0} for i in range(k)]
        final, crashed, tokens_used, latency_used = "", False, 0, 0.0

        for i in range(k, self.n):
            name, type_, _ = STEPS[i]
            error = None
            step_latency_ms, tok = 0.0, 0
            if i == k and patch.mode == "edit_input" and patch.edits:
                state.update(copy.deepcopy(patch.edits))
            if i == k and patch.mode == "override":
                out, status = copy.deepcopy(patch.value), "patched"
            else:
                forced = i == k                                # the patched step always really runs
                fixed = forced and patch.mode == "fix"
                key = self._key(name, state, fixed)
                if not forced and key in self.cache:
                    out, _, error, _ = copy.deepcopy(self.cache[key])
                    status = "cached"
                else:
                    try:
                        t0 = time.perf_counter()
                        out, tok = self._execute(i, state, fixed)
                        step_latency_ms = (time.perf_counter() - t0) * 1000
                    except Exception as e:
                        step_latency_ms = (time.perf_counter() - t0) * 1000
                        out, tok, error = None, 0, f"{type(e).__name__}: {e}"
                    tokens_used += tok
                    latency_used += step_latency_ms
                    self.cache[key] = (copy.deepcopy(out), tok, error, step_latency_ms)
                    status = "patched-rerun" if forced else "recomputed"
            state[name] = out
            log.append({"idx": i, "name": name, "type": type_, "status": status,
                        "output": copy.deepcopy(out), "error": error,
                        "latency_ms": (step_latency_ms if status in ("recomputed", "patched-rerun") else 0.0),
                        "tokens": (tok if status in ("recomputed", "patched-rerun") else 0)})
            if error:
                crashed = True
                break
            if name == "write_answer":
                final = out

        executed = sum(1 for s in log if s["status"] in ("recomputed", "patched-rerun"))
        ok = (not crashed) and grade(self.run["expected"], final)
        return {"patch": patch, "steps": log, "final_answer": str(final), "success": bool(ok),
                "n_steps": self.n, "restored": k, "executed": executed,
                "latency_ms_used": latency_used, "latency_ms_full": sum(float(s.get("latency_ms", 0.0)) for s in old),
                "cached": sum(1 for s in log if s["status"] == "cached"),
                "steps_saved_pct": 1 - executed / self.n,
                "tokens_used": tokens_used, "tokens_full_rerun": self.full_tokens,
                "tokens_saved_pct": (1 - tokens_used / self.full_tokens) if self.full_tokens else 0.0,
                "latency_saved_pct": (1 - latency_used / sum(float(s.get("latency_ms", 0.0)) for s in old)) if sum(float(s.get("latency_ms", 0.0)) for s in old) else 0.0}


# ------------------------------------------------------------ trace comparison
def compare(run, branch):
    """Line the original trace up with the branch (same pipeline, so step by step)."""
    rows, first = [], None
    for b in branch["steps"]:
        o = run["steps"][b["idx"]] if b["idx"] < len(run["steps"]) else None
        same = o is not None and o["output"] == b["output"]
        if not same and first is None:
            first = b["idx"]
        rows.append({"idx": b["idx"], "name": b["name"], "same": same, "status": b["status"],
                     "orig_output": o["output"] if o else None, "new_output": b["output"]})
    return {"rows": rows, "first_divergence": first, "expected": run["expected"],
            "orig_success": bool(run["success"]), "new_success": branch["success"],
            "orig_answer": run["final_answer"], "new_answer": branch["final_answer"],
            "flipped": (not run["success"]) and branch["success"]}


HOW = {"restored": "skipped (checkpoint)", "cached": "reused (cache hit)", "recomputed": "re-run",
       "patched-rerun": ">> PATCHED <<", "patched": ">> PATCHED <<"}


def _short(x, n=34):
    s = json.dumps(x) if not isinstance(x, str) else x
    return s if len(s) <= n else s[:n - 3] + "..."


def format_comparison(cmp, branch):
    lines = [f"  {'step':<20}{'original':<36}{'branch':<36}how", "  " + "-" * 110]
    for r in cmp["rows"]:
        new = "(same)" if r["same"] else _short(r["new_output"])
        mark = "  " if r["same"] else "* "
        lines.append(f"{mark}{r['idx']} {r['name']:<17}{_short(r['orig_output']):<36}{new:<36}{HOW[r['status']]}")
    if cmp["first_divergence"] is not None:
        lines.append(f"\n  First divergence: step {cmp['first_divergence']} "
                     f"({cmp['rows'][cmp['first_divergence']]['name']})")
    else:
        lines.append("\n  The branch never diverged from the original run.")
    lines.append(f"  OUTCOME: {'SUCCESS' if cmp['orig_success'] else 'FAILED'} ({cmp['orig_answer']!r})  ->  "
                 f"{'SUCCESS' if cmp['new_success'] else 'FAILED'} ({cmp['new_answer']!r})   "
                 f"[expected ${cmp['expected']:.2f}]")
    b = branch
    lines.append(f"  WORK: skipped {b['restored']} step(s) by checkpoint, reused {b['cached']} from cache, "
                 f"re-ran {b['executed']} of {b['n_steps']}  ->  {b['steps_saved_pct']:.0%} of steps and "
                 f"{b['tokens_saved_pct']:.0%} of tokens saved vs a full re-run")
    return "\n".join(lines)
