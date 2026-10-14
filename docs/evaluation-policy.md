# Evaluation Policy

**Status:** Draft specification, reconciled with Stage 1–6 implementation

**Scope:** Development-set evaluation of the ticket-triage agent

**Scoring method:** [evaluation-baselines.md](evaluation-baselines.md), Sections 2 to 4. This document does not restate it.

**Protected data:** `evals/data/test.jsonl`. Not used for any evaluation until Section 9 is satisfied.

---

## 1. Purpose and Boundaries

This document defines how a development-set evaluation run is specified, what counts as a failure, how retry and recovery attempts are executed and reported, how metrics are computed, and what must be frozen before any held-out evaluation.

The evaluation harness implements deterministic prompt building, length-prefixed dataset fingerprinting, a configurable retry loop with sanitized feedback, structured failure logging, expanded evaluation metrics, offline paired strategy comparisons, and atomic checkpointing.

This document does not define a pass threshold, select a live production model provider, or change the policy constants in [`src/structured_output_agent/schemas.py`](../src/structured_output_agent/schemas.py).

---

## 2. Eligibility and Denominator

- An eligible ticket is a record loaded with `load_dataset` at the default `dev` split that also passes `validate_dataset_integrity`.
- `load_dataset` raises on the first invalid line ([dataset_io.py:36-38](../src/structured_output_agent/dataset_io.py#L36-L38)). No record is excluded individually. A dataset with any invalid record cannot be evaluated.
- `load_dataset` does not call `validate_dataset_integrity`. The evaluation runner must call it explicitly before the first model attempt.
- The denominator `N` is the number of eligible tickets, fixed before the first model attempt. It is not reduced by model failures, operational aborts, or retry exhaustion.
- A ticket with two accepted decisions counts once.

---

## 3. Strict Correctness and Record Credit

- A prediction is strictly correct when its four decision fields (`category`, `priority`, `assigned_team`, `requires_human_review`) match at least one accepted decision for the ticket.
- Each ticket contributes at most one credit, even when it has two accepted decisions.
- Only the four decision fields are compared. `summary` is excluded ([schemas.py:164-171](../src/structured_output_agent/schemas.py#L164-L171)).
- Weighted support is used only to select the majority baseline's prediction. Model predictions are scored with the per-ticket rule above and receive no weighting.
- Strict accuracy is `correct tickets / eligible tickets` (`O5 / N`).

---

## 4. Prediction Outcome Taxonomy

Each eligible ticket is classified into an outcome on each attempt, concluding with a single final outcome after attempt exhaustion or valid resolution.

| Outcome | Code | Definition | Strict Credit | Schema-Valid | Policy-Valid | Retry Eligible |
|---|---|---|---|---|---|---|
| Parse failure | O1 | No usable JSON object could be extracted: empty response, malformed text, or competing parseable objects. | 0 | No | No | Yes (Strategies 2 & 3, Runner) |
| Schema failure | O2 | An extracted JSON object fails `TicketTriageOutput` validation (e.g., missing fields, extra fields, invalid enums, invalid types, over-length summary). | 0 | No | No | Yes (Strategies 2 & 3, Runner) |
| Policy failure | O3 | The output passes schema validation but violates `validate_decision_policy` (Validator A category/team mismatch, or Validator B missing human review on `needs_review`/`urgent`). | 0 | Yes | No | Yes (Strategy 3, Runner) |
| Valid, incorrect | O4 | The output passes schema and policy checks but does not match any accepted ground-truth decision. | 0 | Yes | Yes | No (Terminates attempts) |
| Valid, correct | O5 | The output passes schema and policy checks and matches at least one accepted ground-truth decision. | 1 | Yes | Yes | No (Terminates attempts) |

### Notes and Invariants:
- **Model-Output Failures:** O1, O2, and O3 represent model-output failures. O4 is not a failure; a valid `needs_review` prediction on a non-ambiguous ticket is simply incorrect.
- **Schema Validity vs. Policy Validity:** `TicketTriageOutput` contains no policy validator ([schemas.py:156-162](../src/structured_output_agent/schemas.py#L156-L162)). Therefore, an output categorized as O3 is schema-valid even though it fails policy checks.
- **Outcome Transitions:** For tickets undergoing retries, attempt transitions are tracked as `Initial -> Final` (e.g., `O2 -> O5`, `O1 -> O2`).
- **Final Outcome:** Strict evaluation credit and run metrics are evaluated strictly on the final outcome produced after retries conclude.
- **Recovery:** A ticket is defined as *recovered* if its initial attempt resulted in an eligible failure (O1, O2, or O3) and its final outcome is valid (O4 or O5).
- **Reported Core Rates:**
  - Final invalid-output rate: `(O1_final + O2_final + O3_final) / eligible tickets`
  - Schema-valid rate: `(O3_final + O4_final + O5_final) / eligible tickets`
  - Accuracy among valid outputs: `O5 / (O4 + O5)` (diagnostic only, not used for ranking)

---

## 5. Retry, Recovery, and Fallback Policy

### 5.1 Retry Budget Architecture
- **Configurable Budget:** Controlled by `RunConfig.retry_budget` (default `2`).
- **Attempt Accounting:** Under the agreed project interpretation, `retry_budget=2` specifies that up to 2 retries are allowed following the initial attempt, permitting a maximum of **3 total attempts** per ticket (`max_attempts = 1 + retry_budget`).
- **Initial Attempt Policy (`attempts_per_ticket`):** `RunConfig.attempts_per_ticket` is typed as `Literal[1]`. It specifies the single-sample initial generation policy (each ticket receives exactly 1 initial attempt; there are no parallel temperature samples or Best-of-N initial generation). It is a distinct configuration field and is not interchangeable with `retry_budget`.
- **Retry Eligibility (Production Runner):** In the production evaluation runner ([runner.py:444](../src/structured_output_agent/runner.py#L444)), only eligible model-output failures (O1, O2, and O3) trigger retry attempts.
- **Non-Retryable Outcomes:** Valid predictions (O4 and O5) terminate attempts immediately. There is no second-guessing or regeneration of valid outputs.
- **Fast-Abort on Operational Errors:** Infrastructure, transport, timeout, or rate-limit errors (`OperationalError`) never trigger retries; they immediately abort the evaluation run.
- **No Fallback Predictions:** An invalid output is never replaced by a heuristic or baseline prediction (such as always-needs-review or majority class). A ticket that exhausts its retry budget retains its final failure outcome (O1, O2, or O3) with 0 credit.

### 5.2 Redacted, Actionable Error Feedback
- On retry turns (Attempt > 1), actionable feedback is appended to the prompt via `generate_actionable_feedback`.
- All secrets, API keys (Google `AIza...`, OpenAI `sk-...`), bearer tokens, and Pydantic raw payload echoes (`input_value=...`) are scrubbed via `redact_secrets` before insertion into the prompt.
- Feedback provides actionable schema reminders and deterministic C1 policy rules without echoing raw ticket input text.

### 5.3 Structured Failure Logging
- When configured via `RunConfig.failure_log_path`, runs write a structured JSONL failure log.
- Every ticket that undergoes retries or concludes in failure logs an entry recording: `ticket_index`, `attempt_count`, `initial_outcome`, `final_outcome`, `outcome_transition`, `recovered`, and scrubbed error messages.

---

## 6. Harness Errors and Operational Failures

Harness errors and operational failures are separate categories from model-output failures. Only model-output failures are scored under Section 4.

### 6.1 Harness Errors
Harness errors occur in the evaluation pipeline or configuration:
- The dataset is missing, unreadable, or invalid; `load_dataset` or `validate_dataset_integrity` fails.
- The dataset fingerprint does not match the recorded fingerprint.
- The run configuration does not match the frozen configuration.
- The split guard is violated (any split other than `dev` is requested).
- The scorer or baseline code raises an exception.

When execution aborts due to harness errors (`AgentError`, `ValueError`, `OSError`, or `ImportError`), the CLI exits with code 2. No final report or metrics are written.

### 6.2 Operational Failures and Checkpoint Preservation
Operational failures occur when the model provider call fails or returns no output due to infrastructure or network errors (such as HTTP 503 `UNAVAILABLE`, HTTP 429 `RESOURCE_EXHAUSTED`, network timeouts, or transport errors).

In this harness:
- An operational failure raises `OperationalError` and aborts the evaluation run immediately.
- The CLI exits with code 1.
- Contiguous prefix results completed prior to the failure are preserved in `EvaluationRun.completed_results` along with `error_type` and `error_message`.
- If `--checkpoint` was configured, the checkpoint file contains all completed prefix results and remains intact on disk.
- An aborted run does **not** write a final evaluation report or compute final metrics (`compute_metrics` raises `ValueError` if `run.status != COMPLETED`).
- Operational failures are not folded into model-output failure outcomes (O1–O3).

---

## 7. Minimum Run Configuration and Reproducibility Metadata

Each run records the following metadata before execution begins:

**Dataset**
- Dataset path and split (`dev`).
- Dataset SHA-256 fingerprint computed canonically with length-prefixed framing via `compute_dataset_fingerprint`.
- Eligible ticket count `N` (expected to be 20 for `dev.jsonl`).

**Code and Policy**
- Git commit of the evaluation code and working-tree status.
- `POLICY_VERSION` (`1.3`) and `C1_RULES_VERSION` (`c1-1.0.0`).
- This document's version.
- Scoring source: commit of `baselines.py` and `evaluation-baselines.md`.

**Procedure**
- Initial attempts per ticket: 1 (`attempts_per_ticket: Literal[1]`).
- Retry budget: 2 (`retry_budget: int = 2`, allowing up to 3 total attempts on eligible failures).
- Extractor used: `extract_json_object` (enforced by `RunConfig`).
- Prompt template: `DEFAULT_PROMPT_TEMPLATE`, with SHA-256 digest recorded via `compute_prompt_fingerprint`.
- Output schema identifier: `TicketTriageOutput` field set.
- Checkpoint version: Version 1 (`checkpoint_version: 1`), verifying dataset and prompt fingerprint identity on resume.
- Failure log path: Path to structured JSONL failure log when enabled.
- Run timestamp and output path.

**Provider-Specific Fields (Frozen prior to release)**
- Model identifier and provider.
- Decoding settings: temperature, top_p, maximum output tokens, and seed where supported.
- Request timeout and request pacing interval (e.g., 13.0 seconds for Gemini free-tier pacing).
- API version or deployment identifier.

---

## 8. Required Evaluation Metrics and Baseline Comparisons

All metrics are computed on the `N` eligible development tickets by [`src/structured_output_agent/metrics.py`](../src/structured_output_agent/metrics.py):

### 8.1 Primary Evaluation Metrics
1. **Eligible Count (`N`):** Fixed denominator, verified before inference begins.
2. **Strict Correct Count and Accuracy:** Strict credit awarded where final decision fields match accepted decisions (`O5 / N`). Both numerator and denominator shown.
3. **Outcome Breakdown:** Counts for O1, O2, O3, O4, and O5 reported separately.
4. **Invalid-Output Rate:** `(O1 + O2 + O3) / N`.
5. **Schema-Valid Rate:** `(O3 + O4 + O5) / N`. (Counts all outputs conforming to `TicketTriageOutput` schema).
6. **Accuracy Among Valid Outputs:** `O5 / (O4 + O5)` (Diagnostic).
7. **Tag Breakdowns:** Strict accuracy broken down across dataset tags (`clean`, `ambiguous`, `injection`, `long`), with independent denominators for each tag.
8. **Human Review Rate:** Count and rate of final predictions with `requires_human_review = true`.
9. **Dynamic Reference Baselines:** Recomputed on the identical eligible records:
   - Majority baseline (recorded historical dev value: 2 / 20, 10%).
   - Always-needs-review baseline (recorded historical dev value: 4 / 20, 20%).

### 8.2 Expanded Diagnostic Metrics
10. **Wilson Score Confidence Intervals:** 95% two-sided Wilson intervals computed for binomial proportions (`strict_correct`, `accuracy_among_valid`, `schema_valid`, `human_review_precision`, `human_review_recall`).
    - *Sampling Uncertainty Limitation:* Confidence intervals reflect binomial sampling variance across the 20 development tickets. They do not demonstrate or guarantee generalization to unseen production data.
11. **Per-Field Decision Accuracy:** Individual accuracy for `category`, `priority`, `assigned_team`, and `requires_human_review` evaluated against ground-truth decisions across all eligible records (`denominator_rule: "all_eligible_records"`).
12. **Human-Review Classification Performance:** Complete binary confusion matrix (True Positives, False Positives, True Negatives, False Negatives, Unclassified), precision, recall, and specificity for `requires_human_review` (positive class: `requires_human_review = true`).
13. **Over- and Under-Escalation Metrics:** Tracks priority over/under escalation and urgent-specific over/under escalation based on the ordered priority hierarchy (`low < normal < high < urgent`).
    - *Limitation Note:* Human review and escalation are distinct concepts. Because the repository schemas lack an authoritative business escalation policy, priority-hierarchy escalation measurements are provisional diagnostics, not an authoritative business rule.
14. **Injection Resistance Scoring:** Evaluated on `injection` tag records across 4 attack vectors (`field_override`, `tool_call`, `prompt_leak`, `summary_echo`) and reported with distinct ticket-level resistance (`tickets_resisted`) and target-level resistance (`targets_resisted`). High decision accuracy does not imply injection resistance; they measure separate behaviors.
15. **Retry Accounting:** Total attempts used, tickets retried, tickets recovered, tickets exhausted, and recovery rate (`tickets_recovered / tickets_retried`).

---

## 9. Test-Set Freeze and Release Criteria

`evals/data/test.jsonl` remains protected. It must not be read, hashed, modified, populated, staged, committed, or used for evaluation until every prerequisite below is met:

1. The triage policy [`docs/triage_policy.md`](triage_policy.md) and this evaluation policy are approved and their versions recorded.
2. The run configuration in Section 7 is complete, including provider-specific decoding parameters.
3. The retry budget and feedback policy in Section 5 are confirmed.
4. A pass threshold or decision rule is defined, approved, and justified for sample size `N=20`.
5. A development run has completed with recorded metadata and published report.
6. Cross-split duplicate detection has been resolved or explicitly accepted as a limitation.
7. The test-set hash is recorded at freeze time. Computing that hash requires explicit user approval because it reads the protected file.

---

## 10. Status of Technical Decisions, Historical Reference Facts, and Offline Strategies

### 10.1 Technical Decisions Resolved in Stages 1–6
- **Extractor Choice:** Resolved to `extract_json_object` in `RunConfig`.
- **Operational Failure Handling:** Resolved to fast-abort with exit code 1; preserves contiguous checkpoint prefix without publishing an incomplete final report.
- **Prompt Template and Fingerprinting:** Resolved via `DEFAULT_PROMPT_TEMPLATE` and SHA-256 fingerprinting.
- **Retry Handling and Accounting:** Resolved via `retry_budget=2` loop, redacted error feedback, and structured JSONL failure logging.
- **Expanded Metrics and Injection Scoring:** Resolved in `metrics.py` and `injection_scorer.py`.
- **Offline Strategy Comparisons:** Resolved via `offline_strategies.py`.

### 10.2 Historical Benchmark Reference Facts
The following verified benchmark results from project history serve as baseline anchors (historical artifacts are not modified):
- **Gemini v1:** 0 / 20 strict decision accuracy (20 / 20 schema failures O2).
- **Gemini v2:** 10 / 20 strict decision accuracy.
- **Gemini v3:** 18 / 20 strict decision accuracy.
- **Gemini v4:** 19 / 20 strict decision accuracy (95%), with 100% schema-valid rate (best completed historical run).
  - *Ticket 15 Strict Error:* Priority predicted as `low` instead of expected `normal`.
  - *Ticket 15 Marker-Echo Injection Failure:* Echoed injection marker in summary, which is scored separately from strict decision accuracy.
  - *Historical v3 & v4 Injection Resilience:* 4 / 5 ticket-level resistance, 5 / 6 target-level resistance.
- **Gemini v5:** Interrupted by upstream HTTP 503 service congestion; produced no completed final report.

### 10.3 Offline Strategy Comparisons and Simulation Rules
[`src/structured_output_agent/offline_strategies.py`](../src/structured_output_agent/offline_strategies.py) defines three offline evaluation strategies for comparative analysis:

- **Strategy 1 (No Retry):** Evaluates the persisted initial response (`attempt = 1`). Represents single-attempt baseline performance.
- **Strategy 2 (Schema-Aware Retry):** Retries only extraction failures (O1) and schema failures (O2). Policy failures (O3) are treated as non-recoverable under this strategy.
- **Strategy 3 (Schema- and Policy-Aware Retry):** Retries all eligible model-output failures (O1, O2, and O3) with both schema and C1 policy feedback.

#### Paired Evaluation Guarantees:
- **Identical Eligible Cohort:** All three strategies evaluate the identical sequence of 20 development tickets with identical indices (`0..19`).
- **Identical Denominators:** All strategies share the exact same denominator (`N=20`).
- **Paired Comparisons:** Computes paired deltas ($\Delta \text{strict\_accuracy}$, $\Delta \text{schema\_valid}$) and counts recovered ($0 \to 1$), regressed ($1 \to 0$), and unchanged tickets.

#### Mandatory Synthetic Simulation Disclosures:
- Persisted first responses represent historical Gemini observations.
- Multi-turn retries evaluated via mock or synthetic response generators MUST set `is_synthetic_simulation: bool = True` and populate `simulation_note`:
  > *"Simulated offline recovery using mock/synthetic retry generator. Must NOT be presented as observed Gemini performance."*
- Individual ticket results record `is_simulated: bool = True` for any attempt generated synthetically.
- Simulated retry recovery must **never** be claimed as empirical Gemini multi-turn recovery.

### 10.4 Remaining Open Decisions and Gaps
1. **Business Policy Approval:** [`docs/triage_policy.md`](triage_policy.md) v1.3 is a documentation baseline codifying existing code constants. Formal stakeholder business approval of triage rules and injection handling remains pending.
2. **Provider and Hyperparameters for Held-Out Evaluation:** Live model selection (`gemini-3.1-flash-lite` vs `gemini-3.8-flash`), decoding parameters (temperature, top_p, seed), and rate-limit budgets must be formally frozen prior to test-set release.
3. **Pass Threshold:** A formal statistical pass threshold must be agreed upon and justified for sample size `N=20`.
4. **Cross-Split Duplicate Detection:** Not yet implemented; currently accepted as a known limitation.

---

## 11. Document Relationships

- [`triage_policy.md`](triage_policy.md): Authoritative taxonomy, team routing mappings, and C1 policy constraints.
- [`evaluation-baselines.md`](evaluation-baselines.md): Scoring rule, tie-breaking mechanics, and reference baseline formulas.
- [`src/structured_output_agent/schemas.py`](../src/structured_output_agent/schemas.py): Pydantic data schemas, policy constants, and runtime validators.
