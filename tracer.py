"""The 'flight recorder': saves every step of every agent run to a SQLite file.

Two tables:
  runs  - one row per agent run (the task, the final answer, success or fail)
  steps - one row per step inside a run (what went in, what came out, timing...)

IMPORTANT: culprit_step / fault_kind in `runs` are the ANSWER KEY used only to
train and grade the diagnosis model. They are never shown to the model as input.
"""
import json
import sqlite3
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task TEXT, expected REAL, final_answer TEXT,
    success INTEGER, culprit_step INTEGER, fault_kind TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT, step_idx INTEGER, name TEXT, type TEXT,
    input TEXT, output TEXT, state TEXT,
    latency_ms REAL, tokens INTEGER, error TEXT,
    PRIMARY KEY (run_id, step_idx)
);
"""


class Tracer:
    def __init__(self, path="blackbox.db"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def start_run(self, task, expected):
        run_id = uuid.uuid4().hex[:8]
        self.db.execute(
            "INSERT INTO runs (run_id, task, expected, created_at) VALUES (?,?,?,?)",
            (run_id, task, expected, time.time()),
        )
        self.db.commit()
        return run_id

    def log_step(self, run_id, idx, name, type_, inp, out, state, latency_ms, tokens, error=None):
        self.db.execute(
            "INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, name, type_, json.dumps(inp), json.dumps(out),
             json.dumps(state), latency_ms, tokens, error),
        )
        self.db.commit()

    def end_run(self, run_id, final_answer, success, culprit_step=None, fault_kind=None):
        self.db.execute(
            "UPDATE runs SET final_answer=?, success=?, culprit_step=?, fault_kind=? WHERE run_id=?",
            (final_answer, int(success), culprit_step, fault_kind, run_id),
        )
        self.db.commit()

    def get_run(self, run_id):
        row = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_steps(self, run_id):
        rows = self.db.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY step_idx", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_runs(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM runs ORDER BY created_at")]
