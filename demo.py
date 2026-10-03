"""Run one clean and one broken agent run, then print their traces side by side."""
from agent import run_agent, STEP_NAMES
from tasks import make_tasks
from tracer import Tracer


def show(tracer, run_id):
    run = tracer.get_run(run_id)
    print(f"\nRUN {run_id}  success={bool(run['success'])}  "
          f"(answer key -> culprit step: {run['culprit_step']}, fault: {run['fault_kind']})")
    print(f"  task:     {run['task']}")
    print(f"  expected: ${run['expected']:.2f}   got: {run['final_answer']!r}")
    for s in tracer.get_steps(run_id):
        out = s["output"] if len(s["output"]) < 70 else s["output"][:67] + "..."
        print(f"  [{s['step_idx']}] {s['name']:<17} {s['type']:<9} {s['latency_ms']:>6}ms  {out}")


if __name__ == "__main__":
    tracer = Tracer("demo.db")
    task = make_tasks(1, seed=1)[0]
    show(tracer, run_agent(task, tracer))                                    # clean run
    show(tracer, run_agent(task, tracer, fault=("retrieve_catalog", "missing_row")))  # broken run
