"""Optional pretrained causal-LM surprise signal for Black Box.

This is an auxiliary signal, not a failure probability.  It scores each recorded
step output with a pretrained causal language model and reports length-normalized
negative log-likelihood (NLL), a clean-run z-score, and the most surprising token.

Dependency is optional: install transformers + torch and set BLACKBOX_LM=1.
The default model is the tiny GPT-2 checkpoint so CPU demos stay practical.
"""
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


class LocalLMSignal:
    def __init__(self, model_name=None, device=None, max_length=256):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Optional local LM requires 'transformers' and 'torch'. "
                "Install them with: pip install transformers torch"
            ) from e

        self.torch = torch
        self.model_name = model_name or os.environ.get("BLACKBOX_LM_MODEL", "sshleifer/tiny-gpt2")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.cache = {}
        self.clean_ref = {}

    @staticmethod
    def _text(value):
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    def _prompt(self, step, examples):
        inp = self._text(step.get("input", {}))
        name = step.get("name", "step")
        prefix = [f"STEP: {name}", "SUCCESSFUL EXAMPLES:"]
        for ex in examples[:2]:
            prefix.append(self._text(ex))
        prefix += [f"INPUT: {inp}", "OUTPUT:"]
        return "\n".join(prefix)

    def _score_batch(self, items):
        # Score only the generated/output portion; prompt tokens are masked.
        prompts, fulls, prefix_lens = [], [], []
        for step, examples in items:
            p = self._prompt(step, examples)
            out = self._text(step.get("output"))
            full = p + " " + out
            prompts.append(p)
            fulls.append(full)
            prefix_lens.append(len(self.tokenizer(p, add_special_tokens=False)["input_ids"]))

        enc = self.tokenizer(
            fulls, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length, add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        labels = input_ids.clone()
        for i, n in enumerate(prefix_lens):
            n = min(n, labels.shape[1])
            labels[i, :n] = -100
        labels[attn == 0] = -100

        with self.torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        # causal shift
        logp = self.torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = labels[:, 1:]
        valid = target != -100
        safe_target = target.masked_fill(~valid, 0)
        tok_nll = -logp.gather(2, safe_target.unsqueeze(-1)).squeeze(-1)
        vals = []
        for i in range(len(items)):
            v = tok_nll[i][valid[i]].detach().cpu().numpy()
            if len(v) == 0:
                vals.append({"nll": 0.0, "max_nll": 0.0, "token_count": 0})
            else:
                vals.append({"nll": float(np.mean(v)), "max_nll": float(np.max(v)), "token_count": int(len(v))})
        return vals

    def score_steps(self, steps, examples_by_name=None):
        examples_by_name = examples_by_name or defaultdict(list)
        out = []
        pending, keys = [], []
        for step in steps:
            ex = examples_by_name.get(step.get("name"), [])
            key_blob = json.dumps([self.model_name, step.get("name"), step.get("input"), step.get("output"), ex[:2]], sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(key_blob.encode()).hexdigest()
            if key in self.cache:
                out.append(dict(self.cache[key]))
            else:
                out.append(None)
                pending.append((step, ex[:2]))
                keys.append(key)
        if pending:
            scored = self._score_batch(pending)
            it = iter(scored)
            for i, key in zip([i for i, x in enumerate(out) if x is None], keys):
                self.cache[key] = next(it)
                out[i] = dict(self.cache[key])
        return out

    def fit_clean_reference(self, clean_runs, max_reference_runs=32):
        by_name = defaultdict(list)
        for run in clean_runs:
            for s in run["steps"]:
                by_name[s["name"]].append(s["output"])
        # Keep the calibration set bounded: the LM is an auxiliary signal, not the
        # main learner, and CPU-only judging should stay practical.
        examples = {k: v[:2] for k, v in by_name.items()}
        ref_runs = clean_runs[:max_reference_runs]
        values = defaultdict(list)
        for run in ref_runs:
            scores = self.score_steps(run["steps"], examples)
            for s, sc in zip(run["steps"], scores):
                values[s["name"]].append(sc["nll"])
        self.clean_ref = {}
        for name, vals in values.items():
            a = np.asarray(vals, dtype=float)
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med)))
            self.clean_ref[name] = {"median": med, "mad": mad}
        self.examples = examples
        return self

    def features(self, steps):
        scores = self.score_steps(steps, getattr(self, "examples", {}))
        out = []
        for s, sc in zip(steps, scores):
            ref = self.clean_ref.get(s["name"], {"median": sc["nll"], "mad": 0.0})
            z = (sc["nll"] - ref["median"]) / (1.4826 * ref["mad"] + 1e-3)
            out.append({
                "lm_nll": float(sc["nll"]),
                "lm_nll_z": float(np.clip(z, -8, 8)),
                "lm_max_nll": float(sc["max_nll"]),
                "lm_token_count": float(sc["token_count"]),
            })
        return out
