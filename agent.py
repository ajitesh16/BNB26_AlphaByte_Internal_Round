"""A small order-pricing agent made of 6 steps.

  0 parse_question   (LLM)       read the question -> product, quantity, discount
  1 retrieve_catalog (retrieval) look up catalog rows for that product
  2 pick_price       (LLM)       choose the right unit price from the rows
  3 check_stock      (tool)      is the quantity in stock?
  4 calculate        (tool)      quantity x price x (1 - discount)
  5 write_answer     (LLM)       write the final sentence

Modes (set the BLACKBOX_MODE environment variable):
  mock  (default) no API needed; instant, free, deterministic. Use this to build.
  real            calls the Claude API (needs ANTHROPIC_API_KEY). Use this for the demo.

Fault injection: run_agent(..., fault=("retrieve_catalog", "missing_row")) secretly
corrupts that step's output. The run then (almost always) fails, and we KNOW which
step is to blame. That is our free, perfectly labeled training data.
"""
import json
import os
import re
import time

from tasks import CATALOG
from tracer import Tracer

MODE = os.environ.get("BLACKBOX_MODE", "mock")
MODEL = "claude-haiku-4-5-20251001"
_client = None


def call_llm(prompt, mock_fn):
    """Return (text, tokens_used). Real Claude call, or a mock for fast offline work."""
    global _client
    if MODE == "mock":
        return mock_fn(), len(prompt) // 4
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    r = _client.messages.create(
        model=MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip(), r.usage.input_tokens + r.usage.output_tokens


def _json(text):
    return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())


# ---------------------------------------------------------------- the 6 steps
# Each step takes the run's `state` (a dict of everything so far) and returns
# (output, tokens_used).

def parse_question(s):
    q = s["question"]
    prompt = ('Extract the order details from this question. Reply with ONLY JSON like '
              '{"product": "...", "quantity": 1, "discount_pct": 0}.\nQuestion: ' + q)

    def mock():
        m = re.search(r"(\d+) units of (.+?) cost with a (\d+)% discount", q)
        return json.dumps({"product": m.group(2), "quantity": int(m.group(1)),
                           "discount_pct": int(m.group(3))})
    text, tokens = call_llm(prompt, mock)
    return _json(text), tokens


def retrieve_catalog(s):
    """Search the catalog: matching rows plus 2 look-alike distractors (deterministic)."""
    product = s["parse_question"]["product"].lower()
    match = [r for r in CATALOG if product in r["name"].lower() or r["name"].lower() in product]
    others = sorted((r for r in CATALOG if r not in match), key=lambda r: hash_str(r["name"] + product))
    rows = [dict(r) for r in (match + others[:2])]
    return sorted(rows, key=lambda r: hash_str(r["name"])), 0


def hash_str(x):
    return sum(ord(c) * (i + 7) for i, c in enumerate(x)) % 1009


def pick_price(s):
    product = s["parse_question"]["product"]
    rows = s["retrieve_catalog"]
    prompt = (f"Here are catalog rows: {json.dumps(rows)}\n"
              f"What is the unit price of '{product}'? Reply with ONLY the number.")

    def mock():
        for r in rows:
            if r["name"].lower() == product.lower():
                return str(r["price"])
        return str(rows[0]["price"])  # a careless model guesses when the item is missing
    text, tokens = call_llm(prompt, mock)
    return float(re.sub(r"[^0-9.]", "", text)), tokens


def check_stock(s):
    p = s["parse_question"]
    item = next((r for r in CATALOG if r["name"].lower() == p["product"].lower()), None)
    ok = bool(item) and item["stock"] >= p["quantity"]
    return {"in_stock": ok, "stock": item["stock"] if item else 0}, 0


def calculate(s):
    p = s["parse_question"]
    total = round(p["quantity"] * s["pick_price"] * (1 - p["discount_pct"] / 100), 2)
    return {"total": total}, 0


def write_answer(s):
    total, in_stock = s["calculate"]["total"], s["check_stock"]["in_stock"]
    prompt = (f"Write ONE short sentence for a customer. Total cost: ${total:.2f}. "
              f"In stock: {in_stock}. If not in stock, say it is out of stock instead. "
              "Write the total as $12.34.")

    def mock():
        return f"The total cost is ${total:.2f}." if in_stock else "Sorry, that item is out of stock."
    return call_llm(prompt, mock)


# name, type, function
STEPS = [
    ("parse_question", "llm", parse_question),
    ("retrieve_catalog", "retrieval", retrieve_catalog),
    ("pick_price", "llm", pick_price),
    ("check_stock", "tool", check_stock),
    ("calculate", "tool", calculate),
    ("write_answer", "llm", write_answer),
]
STEP_NAMES = [s[0] for s in STEPS]


# ------------------------------------------------------------ fault injection
def _bump_digit(text):
    return re.sub(r"\$(\d)", lambda m: "$" + str((int(m.group(1)) + 1) % 10), text, count=1)


FAULTS = {
    "parse_question":   {"wrong_quantity":  lambda o, s: {**o, "quantity": o["quantity"] + 1}},
    "retrieve_catalog": {
        "missing_row": lambda o, s: [r for r in o if r["name"].lower() != s["parse_question"]["product"].lower()],
        "stale_price": lambda o, s: [{**r, "price": round(r["price"] * 1.3, 2)}
                                     if r["name"].lower() == s["parse_question"]["product"].lower() else r for r in o],
    },
    "pick_price":       {"hallucinated_price": lambda o, s: round(o * 1.5, 2)},
    "check_stock":      {"false_out_of_stock": lambda o, s: {**o, "in_stock": False}},
    "calculate":        {"forgot_discount": lambda o, s: {"total": round(
        s["parse_question"]["quantity"] * s["pick_price"], 2)}},
    "write_answer":     {"garbled_number": lambda o, s: _bump_digit(o)},
}
ALL_FAULTS = [(step, kind) for step, kinds in FAULTS.items() for kind in kinds]


# ------------------------------------------------------------------ the runner
def run_agent(task, tracer, fault=None):
    """Run the agent on one task, log every step, label success/failure.

    fault: None for a clean run, or (step_name, kind) to secretly break one step.
    """
    run_id = tracer.start_run(task["question"], task["expected"])
    state = {"question": task["question"]}
    final, failed_hard = "", False

    for idx, (name, type_, fn) in enumerate(STEPS):
        t0, error, tokens = time.perf_counter(), None, 0
        inp = {k: v for k, v in state.items()}          # what this step could see
        try:
            out, tokens = fn(state)
            latency_ms = (time.perf_counter() - t0) * 1000   # timed BEFORE any fault is applied,
            if fault and fault[0] == name:                   # so the injected corruption cannot
                out = FAULTS[name][fault[1]](out, state)     # leak into the latency feature
        except Exception as e:                            # a crash is observable evidence too
            latency_ms = (time.perf_counter() - t0) * 1000
            out, error, failed_hard = None, f"{type(e).__name__}: {e}", True
        state[name] = out
        tracer.log_step(run_id, idx, name, type_, inp, out, state,
                        round(latency_ms, 3), tokens, error)
        if failed_hard:
            break
        final = out if name == "write_answer" else final

    ok = (not failed_hard) and f"${task['expected']:.2f}" in str(final).replace(",", "")
    culprit = STEP_NAMES.index(fault[0]) if fault else None
    tracer.end_run(run_id, str(final), ok, culprit, fault[1] if fault else None)
    return run_id
