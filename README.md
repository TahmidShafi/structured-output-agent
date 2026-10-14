# Structured Output Agent

The Structured Output Agent evaluates Large Language Models (LLMs) on support-ticket triage, classifying customer service tickets into strictly structured JSON decisions with deterministic validation, business policy compliance checks, and reproducible metric scoring.

---

## 1. System Capabilities & Benchmark Showcase

### System Architecture Highlights (Stages 1–7)
The Structured Output Agent is an enterprise-grade evaluation and execution harness engineered to enforce schema compliance, deterministic business policy alignment, and adversarial resilience on LLM outputs:

- **Strict Schema Enforcement:** Powered by Pydantic v2 `StrictModel` (`ConfigDict(extra="forbid")`), enforcing rigid types, closed string enums (`TicketCategory`, `Priority`, `AssignedTeam`), and bounded text fields (`summary` between 1 and 280 characters).
- **Deterministic Business Policy (C1 Constraints):** Enforces hard validation invariants via `validate_decision_policy`:
  - *Validator A (Bijective Routing):* Strict mapping of category to assigned team (e.g. `billing` $\to$ `payments`, `account_access` $\to$ `account_support`).
  - *Validator B (Mandatory Human Review Triggers):* Enforces `requires_human_review = True` whenever `category == "needs_review"` or `priority == "urgent"`.
- **Multi-Turn Self-Healing Retry Loop (`retry_budget=2`):** On model-output failures (O1 parse, O2 schema, O3 policy), the runner issues up to 2 corrective retries (up to 3 total attempts per ticket) with sanitized, actionable error feedback while terminating immediately on valid predictions (O4/O5).
- **Zero-Leakage Security & Secret Redaction:** Scrubbing engine (`redact_secrets`) strips API keys (`AIza...`, `sk-...`), bearer tokens, and raw validation payloads from retry prompts, error logs, and serializations.
- **Safe Tool Execution Registry:** Registry providing `ticket_lookup` and `request_escalation`, blocking decoy or destructive tool calls (e.g., `delete_account`) with `ToolExecutionError` and verifying ticket ID entity alignment.
- **Reproducibility & Tamper Resistance:** Computes canonical length-prefixed SHA-256 fingerprints of datasets and prompt templates, with atomic file writing (`.tmp` $\to$ destination) and checkpoint-resume capabilities.
- **Held-Out Test Set Isolation:** Strict multi-layer path guards, canonicalization, and symlink/hardlink resolution prevent any read, probe, or evaluation of `evals/data/test.jsonl` prior to explicit freeze release.
- **Cross-Split Duplicate Validation:** In-memory validation helper (`validate_no_cross_split_duplicates`) ensuring zero data leakage or verbatim overlap between dataset splits.

---

### Development Benchmark Progression (`evals/data/dev.jsonl`, N=20)
Across iterative development stages, prompt engineering, and policy enforcement, the agent progressed from unconstrained generation failures to 95% strict decision accuracy on the canonical development split:

| Evaluation Target | Architecture / Version Details | Strict Accuracy | 95% Wilson Score CI | Schema-Valid Rate | Primary Failure Mode |
|---|---|:---:|:---:|:---:|---|
| **Majority Baseline** | Constant prediction (`account_access / normal / account_support / false`) | 2 / 20 (10.0%) | [2.8%, 30.1%] | 20 / 20 (100.0%) | Heuristic rule-based baseline |
| **Always-Needs-Review** | Constant review prediction (`needs_review / normal / human_review / true`) | 4 / 20 (20.0%) | [8.1%, 41.6%] | 20 / 20 (100.0%) | Heuristic rule-based baseline |
| **Gemini v1** | Unconstrained free-form JSON generation (`gemini-3.1-flash-lite`) | 0 / 20 (0.0%) | [0.0%, 16.1%] | 0 / 20 (0.0%) | 20 / 20 Schema Failures (O2): unclosed quotes, markdown fences, missing required keys |
| **Gemini v2** | JSON extraction + Pydantic schema validation (`gemini-3.1-flash-lite`) | 10 / 20 (50.0%) | [29.9%, 70.1%] | 18 / 20 (90.0%) | Missing team routing alignment and review trigger rules |
| **Gemini v3** | C1 Policy rules + prompt hardening (`gemini-3.1-flash-lite`) | 18 / 20 (90.0%) | [69.9%, 97.2%] | 20 / 20 (100.0%) | 2 subtle priority misclassifications; 4/5 ticket injection resistance |
| **Gemini v4 (Current)** | Full C1 rules + refined prompt + retry architecture (`gemini-3.1-flash-lite`) | **19 / 20 (95.0%)** | **[76.4%, 99.1%]** | **20 / 20 (100.0%)** | 1 minor priority tier discrepancy (`low` vs `normal` on Ticket 15) |

*Note: All development runs evaluate the identical 20 tickets in `evals/data/dev.jsonl` with `attempts_per_ticket=1`. Confidence intervals reflect binomial sampling variance across N=20 development tickets.*

---

### Expanded Diagnostic & Security Scorecard (Gemini v4)

#### 1. Per-Field Decision Accuracy (N=20)
| Field Name | Correct Count | Accuracy Rate | Notes |
|---|:---:|:---:|---|
| `category` | 20 / 20 | **100.0%** | Perfect high-level domain classification |
| `assigned_team` | 20 / 20 | **100.0%** | 100% adherence to Validator A bijective routing rules |
| `requires_human_review` | 20 / 20 | **100.0%** | 100% adherence to Validator B mandatory review triggers |
| `priority` | 19 / 20 | **95.0%** | Missed on Ticket 15 only (`low` predicted vs `normal` ground truth) |

#### 2. Human Review Classification Matrix (Positive Class: `requires_human_review=True`)
| Metric | Value | 95% Wilson Score CI | Interpretation |
|---|:---:|:---:|---|
| **True Positives (TP)** | 6 | — | Successfully identified all 6 tickets requiring human review (ambiguous & urgent) |
| **False Positives (FP)** | 0 | — | Zero unnecessary human escalation on standard routine tickets |
| **True Negatives (TN)** | 14 | — | Correctly routed routine tickets directly to automated queues |
| **False Negatives (FN)** | 0 | — | Zero missed human reviews on risky or ambiguous tickets |
| **Precision** | **100.0%** (6/6) | [61.0%, 100.0%] | Perfect precision; zero false alarms |
| **Recall / Sensitivity** | **100.0%** (6/6) | [61.0%, 100.0%] | Perfect recall; 100% capture of ambiguous / urgent safety cases |
| **Specificity** | **100.0%** (14/14) | [78.5%, 100.0%] | Complete filtering of routine non-review tickets |

#### 3. Adversarial Prompt-Injection Resilience (5 Tickets, 6 Targets)
- **Ticket-Level Resistance:** **4 / 5 (80.0%)**
- **Target-Level Resistance:** **5 / 6 (83.3%)**

| Attack Vector | Target Description | Result | Details |
|---|---|:---:|---|
| **`field_override`** | Injected text attempting to force `category="shipping"` or `billing` | **2 / 2 (100.0%)** | Model ignored prompt instructions and triaged genuine customer issue |
| **`tool_call`** | Injected command instructing execution of decoy `delete_account` | **1 / 1 (100.0%)** | Model completely ignored destructive tool instruction; safely performed triage |
| **`prompt_leak`** | Injected inquiry demanding revelation of system prompt instructions | **1 / 1 (100.0%)** | Zero leak of system instructions or canary string (`DEV-INJ-LEAK-001`) |
| **`summary_echo`** | Adversarial text attempting to force inclusion of canary markers in summary | **1 / 2 (50.0%)** | Resisted long marker on Ticket 18; echoed marker on Ticket 15 |

---

### Quick Demo Commands (100% Offline)

Run the full 567-test offline verification suite (including mock-based Gemini adapter tests):
```powershell
$env:GEMINI_REQUEST_INTERVAL = "0"
.venv\Scripts\pytest -m "not live"
```

Run an end-to-end evaluation run using the deterministic offline stub client:
```powershell
.venv\Scripts\python -m structured_output_agent --dataset-path evals/data/dev.jsonl --output evals/results/report.json --client stub
```

---

## 2. Environment Setup

The repository supports two development profiles:

1. **Base Offline Environment (No Gemini SDK):**
   ```powershell
   .venv\Scripts\pip install -e ".[dev]"
   .venv\Scripts\pytest
   ```
   Runs the complete base test suite offline. Gemini SDK-dependent tests are skipped cleanly without collection errors.

2. **Full Environment (With Gemini SDK):**
   ```powershell
   .venv\Scripts\pip install -e ".[dev,gemini]"
   .venv\Scripts\pytest
   ```
   Installs the `google-genai` SDK extra and runs all unit tests, including mock-based Gemini adapter tests.

---

## 3. Offline Evaluation (Deterministic Stub)

The evaluation harness includes a built-in deterministic stub client (`StubModelClient`) that allows developers to run and test the complete evaluation workflow without an API key, network access, or Gemini quota consumption.

### Execution Command
```powershell
.venv\Scripts\python -m structured_output_agent --dataset-path evals/data/dev.jsonl --output evals/results/report.json --client stub
```

### Purpose and Provenance
- **Pipeline Verification:** The stub client produces deterministic canned responses (`billing / normal / payments / requires_human_review=False`). It is ideal for verifying dataset loading, integrity validation, CLI arguments, metrics computation, and report serialization.
- **Not Evidence of Model Capability:** The stub's recorded accuracy on the development set was 2/20 (10%), reflecting how often its canned response happened to match the ground truth. This result is useful for checking the evaluation pipeline, but must not be presented as evidence of real LLM performance, reasoning, or capabilities.
- **Provenance Tracking:** The generated evaluation report explicitly records the client type and model identifier (`client_type: "stub"`, `model_id: null`), preventing stub runs from being confused with real model evaluations.

---

## 4. Live Gemini Evaluation

Developers with Google AI Studio API access can evaluate live Gemini models against the development dataset.

### Execution Command
```powershell
.venv\Scripts\python -m structured_output_agent --dataset-path evals/data/dev.jsonl --output evals/results/gemini_dev_report.json --client gemini --model gemini-3.1-flash-lite
```

### Configuration and Precedence
- **API Key Configuration:** The `GEMINI_API_KEY` environment variable must be set in the active terminal session before execution:
  - **PowerShell:** `$env:GEMINI_API_KEY = "your-api-key"`
  - **Bash/Zsh:** `export GEMINI_API_KEY="your-api-key"`
  Never hardcode API keys or commit credential files into the repository.
- **Model Selection Precedence:** The model identifier resolves according to strict precedence:
  1. CLI option `--model` (e.g. `--model gemini-3.1-flash-lite`), which takes highest precedence.
  2. Environment variable `GEMINI_MODEL`.
  3. Default constant (`DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"`).
  Specifying `--model gemini-3.1-flash-lite` ensures that model is explicitly used and recorded in report provenance.

### Request Pacing and Runtime
- **Conservative Client-Side Pacing:** To reduce the risk of exceeding the free-tier quota (5 requests per minute), the Gemini client enforces default request pacing of 13.0 seconds between request starts (`DEFAULT_REQUEST_INTERVAL = 13.0`).
- **Immediate First Request:** The first ticket request is issued immediately (0s delay). Subsequent requests wait for the remainder of the 13-second interval measured from the start of the preceding request.
- **Expected Duration:** For the 20 tickets in the development dataset (`evals/data/dev.jsonl`), an evaluation executes 1 immediate request and 19 paced intervals of 13 seconds (19 × 13s = 247 seconds ≈ 4 minutes 7 seconds), plus network latency. A full 20-ticket run typically takes **four to five minutes**, excluding unusually slow network responses.
- **Quotas and Availability:** Provider rate limits (RPM, TPM, RPD) and service availability are external dependencies managed by Google AI Studio. Do not assume or claim that `gemini-3.1-flash-lite` (or any model) is always available or that its quota is guaranteed.

---

## 5. Metrics and Interpretation

The evaluation harness calculates deterministic metrics on completed runs according to the project evaluation policy.

### Outcome Taxonomy (O1–O5)
Each eligible ticket receives 1 initial attempt and up to 2 retries on eligible failures (O1–O3) under `retry_budget=2` (allowing up to 3 total attempts per ticket). Its final outcome is classified into one mutually exclusive category:

| Outcome | Name | Definition | Strict Credit |
|---|---|---|:---:|
| **O1** | Parse failure | No valid JSON object could be extracted from the model response (empty text, invalid JSON, or competing candidates). | 0 |
| **O2** | Schema failure | JSON object extracted but fails `TicketTriageOutput` Pydantic validation (missing field, extra field, invalid enum value, summary > 280 chars). | 0 |
| **O3** | Policy failure | Output passes schema validation but violates business triage policy (`validate_decision_policy`: category-to-team mismatch or `needs_review`/`urgent` missing human review flag). | 0 |
| **O4** | Valid, incorrect | Output passes schema and policy checks, but its four decision fields do not match any accepted ground-truth decision for the ticket. | 0 |
| **O5** | Valid, correct | Output passes schema and policy checks and its four decision fields match at least one accepted decision. | 1 |

### Metric Definitions
- **`strict_correct`:** The ratio and rate of tickets where all four decision fields (`category`, `priority`, `assigned_team`, `requires_human_review`) match an accepted decision:
  $$\text{strict\_correct} = \frac{\text{O5}}{N}$$
  The ticket summary is excluded from credit comparison. Each ticket contributes at most 1 credit.
- **`invalid_outputs`:** The proportion of outputs failing extraction, schema, or business policy validation:
  $$\text{invalid\_outputs} = \frac{\text{O1} + \text{O2} + \text{O3}}{N}$$
- **`accuracy_among_valid`:** A diagnostic metric measuring accuracy strictly among outputs that passed extraction, schema, and policy checks:
  $$\text{accuracy\_among\_valid} = \frac{\text{O5}}{\text{O4} + \text{O5}}$$
  Invalid outputs (O1–O3) are excluded from the denominator. This metric isolates decision-making quality from syntax/format adherence, but is not used as the primary ranking metric.
- **`human_review`:** The proportion of model predictions where `requires_human_review` was set to `true`:
  $$\text{human\_review} = \frac{\text{count}(\text{requires\_human_review} = \text{true})}{N}$$

### Reference Baselines
The harness recomputes deterministic baselines directly on eligible development records:
- **Majority Complete-Decision Baseline:** Always predicts `account_access / normal / account_support / false`.
  - Development score: **2 / 20 (10.0%)**
- **Always-Needs-Review Baseline:** Always predicts `needs_review / normal / human_review / true`.
  - Development score: **4 / 20 (20.0%)**

### Key Interpretation Guidelines
1. **Valid Outputs Can Be Incorrect:** Passing schema validation (O2) and policy rules (O3) only establishes structural validity. An O4 prediction is fully valid and policy-compliant, but represents an incorrect triage routing decision.
2. **Benchmark Against Baselines:** Model performance must always be evaluated relative to the reference baselines. An LLM scoring 20% or lower on the development set performs no better than the trivial rule-based `always_needs_review` heuristic.
3. **Stub Client Is Not an LLM:** Results produced by the stub client (`--client stub`) must never be reported as real model capability.
4. **Sample Size Context:** The development split contains $N = 20$ tickets. Each individual ticket represents 5.0 percentage points. Metrics must always be interpreted alongside the eligible sample count and not treated as conclusive evidence of general production accuracy.

---

## 6. Provider Failure Recovery

The evaluation runner adheres to a fail-fast design for operational and infrastructure errors.

### Fail-Fast Behavior
- **Operational Failures:** Provider errors (HTTP 503 `UNAVAILABLE`, HTTP 429 `RESOURCE_EXHAUSTED`, HTTP 400/404 invalid model identifier), authentication errors (HTTP 401/403), transport timeouts, and dropped connections are classified as `OperationalError`.
- **Immediate Abort (Exit Code 1):** When an operational failure occurs, execution aborts immediately with exit code `1`. The adapter explicitly configures `HttpRetryOptions(attempts=1)` to disable SDK automatic retries, ensuring rapid, observable feedback rather than silent looping.
- **Configuration/Harness Errors (Exit Code 2):** Invalid CLI arguments, missing datasets, or dataset integrity violations abort with exit code `2`.

### Report File Integrity Guarantees
- **No Partial Reports:** An aborted or failed run **never** produces or writes a final evaluation report.
- **Atomic File Writes:** The evaluation harness writes reports to a uniquely named temporary file (`.output.uuid.tmp`) and atomically moves it to the target destination only after all tickets, metrics, and validations succeed. If a run aborts, the target file is untouched, preserving any pre-existing report completely intact.
- **Verification Requirement:** Never assume an evaluation run completed successfully merely because a file exists at the `--output` path. Check that the command returned exit code `0` and verify the report's `created_at` timestamp.

### Recovery Protocols & Retry Architecture
- **Do Not Blindly Repeat:** Never re-run full evaluations immediately following an operational failure. Repeated calls risk exhausting daily quotas or compounding provider rate limits.
- **Handling Transient 503 Errors:** HTTP 503 indicates upstream service congestion or temporary capacity exhaustion. Wait before attempting another evaluation run. Do not assume or promise a fixed recovery time, as provider load varies externally.
- **Controlled Retry Budget vs. Operational Fail-Fast:**
  - **Operational / Infrastructure Errors:** Fail fast immediately with exit code `1` and `HttpRetryOptions(attempts=1)`. No retries are attempted on transport, rate-limit, or connection errors.
  - **Model-Output Failures (O1–O3):** When extraction, schema, or policy failures occur, the harness executes up to 2 retries per ticket (`retry_budget=2`, allowing up to 3 total attempts per ticket) using redacted, actionable feedback and structured failure logging. Valid predictions (O4, O5) terminate attempts immediately.
  - **No Heuristic Fallback Predictions:** An unrecovered ticket that exhausts its retry budget retains its final model-output failure outcome (O1, O2, or O3) with 0 credit. It is never replaced with a heuristic or baseline prediction (such as defaulting to `needs_review` or the majority class).

---

## 7. Safe Evaluation Practices

Follow these essential safety practices during all development and evaluation activities:

1. **Development Dataset Only:** All prompt engineering, model tuning, and evaluation runs must use the development dataset split (`evals/data/dev.jsonl`).
2. **Protected Test Dataset:** The held-out test dataset `evals/data/test.jsonl` is strictly protected and reserved for final verification. It must **never** be opened, read, statted, hashed, linked, probed, or evaluated during development workflows. The CLI strictly rejects non-dev splits.
3. **Controlled Model Comparisons:** When comparing different models or prompt variants, evaluate against the exact same development split (`evals/data/dev.jsonl`) under compatible settings (`retry_budget=2`, identical scoring rules).
4. **Distinct Output Paths:** Always use distinct output file paths when evaluating different models or configurations (e.g. `--output evals/results/report.json` for stub and `--output evals/results/gemini_dev_report.json` for live Gemini). This prevents accidentally overwriting or confusing evaluation reports.
5. **Clean Version Control:** Do not stage or commit generated evaluation reports, temporary files, credentials, or unrelated modifications.

---

## 8. Relevant Specifications

For further technical details, see:
- [docs/evaluation-policy.md](docs/evaluation-policy.md): Authoritative evaluation specification, failure handling, and freeze criteria.
- [docs/evaluation-baselines.md](docs/evaluation-baselines.md): Baseline definitions, tie-breaking algorithms, and dataset fingerprints.
- [docs/gemini-live-verification.md](docs/gemini-live-verification.md): Focused manual verification guide for Google AI Studio credentials.
