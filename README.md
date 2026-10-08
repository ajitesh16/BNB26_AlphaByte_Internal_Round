# 🕵️ Black Box — AI Agent Forensics & Causal Debugging

> **Don't just detect the failure. Find the cause. Prove it. Replay it.**

Black Box is an **AI-agent forensics and causal debugging system** designed to investigate failures inside multi-step AI agent executions.

Modern AI agents can fail silently: one incorrect intermediate decision can corrupt everything that happens afterward, while the final error gives very little information about where the failure actually originated.

Black Box treats an agent execution like a **flight recorder**. It captures the execution trace, learns normal behavioral patterns, identifies the most likely root-cause step, explains the evidence behind that diagnosis, and then **validates the diagnosis through counterfactual replay**.

The key idea is simple:

> **A diagnosis is stronger when changing the suspected cause actually changes the outcome.**

---

# 🎯 Problem Statement

AI agents increasingly perform tasks through multiple sequential steps involving:

* Reasoning
* Retrieval
* Tool calls
* Data processing
* Validation
* Calculations
* Response generation

When something goes wrong, the final output may be incorrect even though the actual cause occurred several steps earlier.

Consider:

```text
User Request
     │
     ▼
Identify Product          ✓
     │
     ▼
Retrieve Catalog          ✓
     │
     ▼
Validate Candidate        ✓
     │
     ▼
Retrieve Price            ❌  ← Root Cause
     │
     ▼
Calculate Total           ⚠
     │
     ▼
Generate Response         ⚠
```

A conventional debugging system might simply report:

```text
Agent failed.
```

A better debugger might report:

```text
Step 6 produced an invalid result.
```

But Black Box asks a deeper question:

> **Which step actually caused the failure?**

And then:

> **Can we prove that by changing the suspected step and replaying the execution?**

---

# 💡 Our Solution

Black Box combines **execution tracing, behavioral analysis, machine learning, checkpointed replay, and causal intervention** into one investigation workflow.

```text
                 ┌───────────────────────┐
                 │       AI Agent        │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Flight Recorder     │
                 │       / Tracer        │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Execution Trace    │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Behavioral Invariants │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Root-Cause Diagnosis │
                 │       ML Model        │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Causal Intervention   │
                 │ & Counterfactual      │
                 │       Replay          │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │      Trace Diff       │
                 │ & Outcome Validation  │
                 └───────────────────────┘
```

---

# 🔬 How Black Box Works

Black Box follows a six-stage investigation pipeline.

## 1. Record

The agent execution is captured as a structured trace.

Each step can contain information such as:

* Step index
* Input
* Output
* Execution latency
* Error information
* State information
* Behavioral features
* Relationships between steps

This creates a **flight recorder** for the agent.

---

## 2. Learn

Black Box analyzes successful executions to learn what normal agent behavior looks like.

These patterns are represented through behavioral features and invariants.

Examples include:

* Expected output characteristics
* Input/output relationships
* Output length
* Execution latency
* Step ordering
* State transitions
* Presence or absence of expected information

---

## 3. Detect

When a failed execution is investigated, Black Box searches for deviations from normal behavior.

Instead of relying only on explicit errors, the system looks for **behavioral divergence**.

For example:

```text
Step 1   Normal
Step 2   Normal
Step 3   Normal
Step 4   Behavioral Violation  ← suspicious
Step 5   Downstream Effect
Step 6   Downstream Effect
```

---

## 4. Diagnose

The diagnosis system ranks execution steps according to their probability of being the root cause.

The model considers signals such as:

* First invariant violation
* Step position
* Input/output characteristics
* Output length
* Latency
* Trace structure
* Downstream effects

The result is a ranked root-cause prediction.

---

## 5. Intervene

Black Box does not stop after making a prediction.

It allows the investigator to modify the suspected step.

Possible interventions include:

```text
FIX
OVERRIDE
EDIT INPUT
```

This creates a **counterfactual version** of the execution.

---

## 6. Replay & Validate

The modified execution is replayed from a checkpoint.

Black Box then compares:

```text
Original Execution
        VS
Counterfactual Execution
```

If changing the suspected step causes the execution to recover, this provides causal evidence supporting the original diagnosis.

This is the central idea behind Black Box:

> **Don't just predict the cause — intervene on it and test the hypothesis.**

---

# 🧠 Causal Validation

A major distinction between Black Box and conventional debugging is the separation between:

### Correlation

```text
"This step looks suspicious."
```

and:

### Causal evidence

```text
"We changed this step,
replayed the execution,
and the failure disappeared."
```

Black Box therefore uses counterfactual replay as a practical causal validation mechanism within its controlled execution environment.

---

# ⏪ Checkpointed Replay

Replaying an entire agent execution from scratch can be expensive.

Black Box uses checkpoints so that previously computed portions of an execution can be reused.

Instead of:

```text
Step 1
  ↓
Step 2
  ↓
Step 3
  ↓
Step 4
  ↓
Step 5
  ↓
Step 6
```

every time, the system can reuse the unaffected prefix and recompute the relevant portion.

```text
Cached Steps
───────────────┐
               │
               ▼
          Intervention
               │
               ▼
        Recompute From
          Checkpoint
```

This reduces unnecessary replay computation.

---

# 🔎 Trace Diff

After a counterfactual replay, Black Box compares the original and modified traces.

The comparison identifies:

* Reused steps
* Recomputed steps
* Changed values
* First divergence
* Downstream effects
* Final outcome

Conceptually:

```text
ORIGINAL

Step 1   ✓
Step 2   ✓
Step 3   ✓
Step 4   ❌
Step 5   ⚠
Step 6   ⚠


COUNTERFACTUAL

Step 1   ✓  reused
Step 2   ✓  reused
Step 3   ✓  reused
Step 4   ✓  patched
Step 5   ✓  recovered
Step 6   ✓  recovered
```

This provides a visual explanation of **what changed and why the outcome changed**.

---

# 🧪 Controlled Fault-Injection Benchmark

To evaluate Black Box, we use a controlled benchmark where failures are intentionally injected into agent executions.

This allows us to know the true root cause and objectively evaluate the diagnosis system.

Current fault categories include:

```text
wrong_quantity
missing_row
stale_price
hallucinated_price
false_out_of_stock
forgot_discount
garbled_number
```

The benchmark also evaluates the system on **held-out / unseen fault types** to test whether the diagnosis approach generalizes beyond the exact failures observed during training.

---

# 📊 Evaluation Results

Current benchmark:

| Metric                              |             Result |
| ----------------------------------- | -----------------: |
| Generated Runs                      |            **300** |
| Clean Training Runs                 |            **108** |
| Failed Training Runs                |            **102** |
| Held-out Failed Test Runs           |             **41** |
| Black Box Top-1 Root-Cause Accuracy |          **97.6%** |
| Unseen Fault Types Top-1 Accuracy   |          **99.2%** |
| First-Try Causal Repair             |          **97.6%** |
| Average Replay                      | **3.22 / 6 steps** |
| Steps Saved                         |          **46.3%** |
| Token Usage Saved                   |          **43.3%** |

---

# 📈 Baseline Comparison

Black Box was compared against simple debugging strategies.

| Method             | Top-1 Accuracy |
| ------------------ | -------------: |
| Random Step        |          16.7% |
| Last Step          |          12.2% |
| First Logged Error |          19.5% |
| **Black Box**      |      **97.6%** |
| Oracle             |           100% |

The large gap between simple heuristics and Black Box demonstrates the value of combining behavioral evidence with learned root-cause localization.

---

# 🌐 Generalization

An important evaluation goal is avoiding a system that simply memorizes known fault labels.

Black Box therefore evaluates **unseen fault types**.

Current result:

> **99.2% average Top-1 localization accuracy on unseen fault types.**

This suggests that the system is using execution-level behavioral signals rather than relying only on fixed fault labels.

---

# 📐 Important Diagnostic Signals

The current diagnosis pipeline uses multiple trace-level features.

Examples include:

| Feature      | Approx. Importance |
| ------------ | -----------------: |
| `first_viol` |                28% |
| `idx`        |                25% |
| `len_ratio`  |                17% |
| `latency_z`  |                12% |

These signals capture different aspects of anomalous behavior:

* **`first_viol`** — how early a behavioral violation appears
* **`idx`** — where the suspicious step occurs
* **`len_ratio`** — relationship between input/output lengths
* **`latency_z`** — abnormal execution latency

The combination provides a richer signal than simply selecting the first error in the trace.

---

# 🏗️ System Architecture

```text
                       ┌───────────────┐
                       │   User Task   │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │    AI Agent   │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │     Tracer    │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │ Execution Log │
                       └───────┬───────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌────────────────┐          ┌─────────────────┐
        │   Invariants   │          │ Feature Engine  │
        └───────┬────────┘          └────────┬────────┘
                │                            │
                └────────────┬───────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Diagnosis Model  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Root Cause Step  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Intervention     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Checkpoint Replay│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Trace Diff    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Causal Evidence │
                    └──────────────────┘
```

---

# 🖥️ User Interface

The Black Box interface is designed around an **AI-agent investigation workflow**.

## Mission Control

Provides an overview of the system.

Example metrics:

```text
Runs
Failures
Detection Rate
Replay Savings
```

---

## Investigate

The investigation screen allows the user to:

1. Select a failed run.
2. Inspect the execution timeline.
3. Identify suspicious steps.
4. View root-cause evidence.
5. Review confidence.
6. Trigger a replay.

Example:

```text
RUN: BB-0247

STATUS: FAILED

Step 1   Parse Request              ✓
Step 2   Identify Product           ✓
Step 3   Retrieve Catalog            ✓
Step 4   Retrieve Price             🔴
Step 5   Calculate Total             ⚠
Step 6   Generate Response           ⚠

Suspected Root Cause:
Step 4 — Retrieve Price

Evidence:
• First behavioral divergence
• Invariant violation
• Downstream dependency
```

---

# 🧪 What-If Lab

The What-If Lab allows users to perform counterfactual experiments.

```text
Selected Step: Step 4

Intervention:
○ Fix
○ Override
○ Edit Input

             [ Replay ]
```

The system then shows the result of the intervention.

This turns debugging into an interactive experiment rather than a static prediction.

---

# 🔄 Investigation Workflow

A complete investigation looks like:

```text
Select Failed Run
       ↓
Inspect Timeline
       ↓
Identify Suspicious Step
       ↓
Review Evidence
       ↓
Choose Intervention
       ↓
Replay From Checkpoint
       ↓
Compare Traces
       ↓
Observe Final Outcome
       ↓
Validate / Reject Hypothesis
```

---

# 📁 Project Structure

```text
blackbox/
│
├── agent.py
│       └── Core toy AI-agent execution
│
├── tracer.py
│       └── Execution tracing / flight recorder
│
├── invariants.py
│       └── Behavioral invariant learning and checking
│
├── diagnose.py
│       └── Root-cause diagnosis models
│
├── replay.py
│       └── Checkpointed and counterfactual replay
│
├── evaluate.py
│       └── Benchmark evaluation
│
├── replay_eval.py
│       └── Replay / causal evaluation
│
├── replay_demo.py
│       └── Replay demonstration utilities
│
├── explain.py
│       └── Diagnostic explanation logic
│
├── app.py
│       └── Streamlit application
│
├── results.json
│       └── Evaluation results
│
├── results_causal.json
│       └── Causal replay results
│
├── requirements.txt
│       └── Python dependencies
│
└── README.md
        └── Project documentation
```

---

# 🛠️ Technology Stack

## Programming

* Python

## Machine Learning

* Logistic Regression
* Gradient-boosted decision trees
* Feature engineering
* Behavioral anomaly detection

## AI / Agent Infrastructure

* Agent execution tracing
* Structured execution logs
* Behavioral invariants
* Counterfactual intervention
* Checkpointed replay

## Frontend

* Streamlit

## Evaluation

* Controlled fault injection
* Held-out evaluation
* Unseen fault-type testing
* Baseline comparison
* Causal replay experiments

---

# 🚀 Installation

## Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd blackbox
```

## Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit interface:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# 🔑 API Configuration

If the application is configured to use an external AI API, the API key should be stored as an environment variable.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### Windows CMD

```cmd
set OPENAI_API_KEY=your_api_key_here
```

### macOS / Linux

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Important

**Never commit API keys to GitHub.**

Use environment variables or a secure secrets manager instead.

For Streamlit deployments, configure secrets through the platform's secrets management rather than placing the key directly inside the source code.

---

# 🧪 Running the Evaluation

The project contains separate evaluation components for diagnosis and replay.

Depending on the project configuration, evaluation scripts can be executed with Python:

```bash
python evaluate.py
```

and replay evaluation:

```bash
python replay_eval.py
```

These generate evaluation results that can be used to compare Black Box against baseline strategies.

---

# 🔬 Research Methodology

The system follows a controlled experimental methodology.

### Step 1 — Generate executions

Generate normal and fault-injected agent executions.

### Step 2 — Build traces

Record every execution as a structured trace.

### Step 3 — Learn normal behavior

Extract behavioral patterns from clean executions.

### Step 4 — Train diagnosis

Train a model to distinguish likely root-cause steps.

### Step 5 — Hold out failures

Evaluate on failed executions not used during training.

### Step 6 — Test unseen faults

Evaluate whether the model can localize fault patterns outside the training fault distribution.

### Step 7 — Perform intervention

Modify the predicted root-cause step.

### Step 8 — Replay

Replay the execution from the relevant checkpoint.

### Step 9 — Compare

Compare original and counterfactual traces.

### Step 10 — Validate

Determine whether the intervention changes the final outcome.

---

# ⚖️ Why Not Just Use the First Error?

A simple debugger could choose:

```text
first error = root cause
```

But this can fail because:

* Some failures do not produce explicit errors.
* A downstream step may report the first visible error.
* The actual behavioral divergence may occur earlier.
* Multiple downstream failures can originate from one upstream mistake.

Black Box therefore considers the **execution trajectory**, not just the final error message.

---

# 🆚 Black Box vs Conventional Debugging

| Capability                  | Traditional Debugging | Black Box |
| --------------------------- | --------------------- | --------- |
| Execution tracing           | Sometimes             | ✅         |
| Behavioral analysis         | Limited               | ✅         |
| Root-cause localization     | Manual                | ✅         |
| ML-based diagnosis          | Usually absent        | ✅         |
| Counterfactual intervention | Rare                  | ✅         |
| Checkpointed replay         | Rare                  | ✅         |
| Trace comparison            | Limited               | ✅         |
| Causal validation           | ❌                     | ✅         |
| Unseen fault evaluation     | Rare                  | ✅         |

---

# 🧩 Example

Suppose an AI shopping agent executes:

```text
1. Parse request
2. Identify product
3. Retrieve catalog
4. Retrieve price
5. Calculate total
6. Generate response
```

The agent produces an incorrect final price.

A conventional system might say:

```text
Final answer is incorrect.
```

Black Box investigates the trace:

```text
Step 1  Normal
Step 2  Normal
Step 3  Normal
Step 4  Behavioral divergence
Step 5  Dependent on Step 4
Step 6  Dependent on Step 5
```

Diagnosis:

```text
Root Cause = Step 4
```

Black Box then intervenes:

```text
Step 4 → corrected price
```

and replays:

```text
Step 5 → correct calculation
Step 6 → correct response
```

The original failure disappears.

Therefore, Step 4 receives **causal validation**.

---

## 🎨 Creativity

Instead of building another generic AI chatbot or agent dashboard, Black Box approaches AI reliability through **forensics and causal debugging**.

The flight-recorder metaphor provides an intuitive way to understand opaque agent behavior.

---

## 🧠 Technical Complexity

The system combines:

* Agent tracing
* Feature engineering
* Behavioral invariants
* Machine learning
* Root-cause localization
* Fault injection
* Checkpointing
* Counterfactual intervention
* Replay
* Trace comparison
* Evaluation on unseen faults

---

## 🏭 Practicality

As AI agents become more autonomous, developers need tools to understand why an agent made a wrong decision.

A system like Black Box could eventually support:

* AI debugging
* Agent observability
* Reliability engineering
* Tool-call debugging
* Retrieval debugging
* Production incident investigation
* Automated agent repair

---

## 🎤 Presentation

The project is designed to provide a strong live demonstration:

```text
Agent fails
     ↓
Black Box investigates
     ↓
Root cause identified
     ↓
"What-If" intervention
     ↓
Replay
     ↓
Failure disappears
     ↓
Causal evidence
```

The final demonstration shows not only **what the system predicts**, but **how the prediction is tested**.

---

# 🔮 Future Scope

Black Box can be extended beyond the current controlled benchmark.

### Longer Agent Traces

Move from short 6-step workflows to longer workflows such as:

```text
Parse Request
      ↓
Identify Product
      ↓
Retrieve Catalog
      ↓
Filter Candidates
      ↓
Validate Candidate
      ↓
Retrieve Price
      ↓
Check Stock
      ↓
Calculate
      ↓
Validate Calculation
      ↓
Generate Response
```

---

### Tool-Call Diagnosis

Detect failures such as:

* Wrong tool selected
* Incorrect tool arguments
* Incorrect API response interpretation
* Tool output ignored

---

### Retrieval Diagnosis

Detect:

* Wrong document retrieved
* Relevant information ignored
* Stale information
* Incorrect retrieval ranking

---

### Context-Loss Detection

Identify cases where important information disappears from the agent's working context during execution.

---

### Multi-Agent Debugging

Extend the flight recorder to systems where several AI agents communicate and collaborate.

---

### Automated Repair

After identifying a root cause, the system could automatically generate and test candidate repairs.

---

### Production Observability

A future version could integrate with real agent frameworks and production traces to provide continuous AI reliability monitoring.

---

# ⚠️ Limitations

The current system is a **research prototype**.

Its benchmark is based on controlled fault injection rather than arbitrary real-world AI-agent failures.

Therefore:

* Benchmark performance should not be interpreted as universal real-world accuracy.
* The current agent environment is intentionally controlled.
* Causal validation is performed through controlled interventions.
* Broader validation on real LLM agents and production workloads is future work.

These limitations are important because they make the evaluation more scientifically honest and reproducible.

---

# 📌 Key Takeaway

Most debugging systems tell you:

> **“The agent failed.”**

Black Box asks:

> **“Where did the failure begin?”**

Then it goes one step further:

> **“What happens if we change that step?”**

And finally:

> **“Did changing it actually fix the execution?”**

That is the difference between **error detection** and **causal debugging**.

---

# 📜 Disclaimer

Black Box is currently evaluated using a controlled fault-injection benchmark.

The reported metrics represent performance on this benchmark and should not be interpreted as a guarantee of performance across all possible AI agents, models, tools, or production environments.

---

# ⭐ Project Vision

> **AI agents are becoming increasingly autonomous.**
>
> **When they fail, we need more than an error message.**
>
> **We need a way to reconstruct what happened, identify why it happened, intervene, and prove whether the intervention worked.**
>
> **That is Black Box.**

---

## 🕵️ Black Box

### **Don't just detect the failure.**

### **Find the cause.**

### **Prove it.**

### **Replay it.**
