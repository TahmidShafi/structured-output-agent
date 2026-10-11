# Evaluation Policy

**Status:** Draft specification, not frozen

**Scope:** Development-set evaluation of the ticket-triage agent

**Scoring method:** [evaluation-baselines.md](evaluation-baselines.md), Sections 2 to 4. This document does not restate it.

**Protected data:** `evals/data/test.jsonl`. Not used for any evaluation until Section 9 is satisfied.

## 1. Purpose and Boundaries

This document defines how a development-set evaluation run is specified, what counts as a failure, how failures are reported, and what must be frozen before any held-out evaluation.

It does not define a pass threshold, select a model provider, add an output confidence field, or change the policy constants in `schemas.py`.

## 2. Eligibility and Denominator

- An eligible ticket is a record loaded with `load_dataset` at the default `dev` split that also passes `validate_dataset_integrity`.
- `load_dataset` raises on the first invalid line ([dataset_io.py:36-38](../src/structured_output_agent/dataset_io.py#L36-L38)). No record is excluded individually. A dataset with any invalid record cannot be evaluated.
- `load_dataset` does not call `validate_dataset_integrity`. The evaluation runner must call it explicitly before the first model attempt.
- The denominator is the number of eligible tickets, fixed before the first model attempt. It is not reduced by model failures.
- A ticket with two accepted decisions counts once.

## 3. Strict Correctness and Record Credit

- A prediction is strictly correct when its four decision fields (`category`, `priority`, `assigned_team`, `requires_human_review`) match at least one accepted decision for the ticket.
- Each ticket contributes at most one credit, even when it has two accepted decisions.
- Only the four decision fields are compared. `summary` is excluded ([schemas.py:130](../src/structured_output_agent/schemas.py#L130)).
- Weighted support is used only to select the majority baseline's prediction. Model predictions are scored with the per-ticket rule above and receive no weighting.
- Strict accuracy is `correct tickets / eligible tickets`.

## 4. Prediction Outcome Taxonomy

Each eligible ticket receives exactly one outcome from one attempt.

| Outcome | Definition | Strict credit |
|---|---|---|
| O1 Parse failure | No usable JSON object could be extracted: empty response, no parseable object, or competing parseable objects. | 0 |
| O2 Schema failure | An object was extracted but fails `TicketTriageOutput` validation, for example an extra field, an invalid enum value, an over-length summary, or a non-boolean review flag. | 0 |
| O3 Policy failure | The output passes schema validation but fails `validate_decision_policy` (Validator A: team must match category; Validator B: `needs_review` and `urgent` require human review). | 0 |
| O4 Valid, incorrect | The output passes schema and policy checks but matches no accepted decision. | 0 |
| O5 Valid, correct | The output passes schema and policy checks and matches an accepted decision. | 1 |

Notes:

- O1 to O3 are model-output failures. O4 is not a failure: a valid `needs_review` prediction on a non-ambiguous ticket is simply incorrect.
- `TicketTriageOutput` contains no policy validator ([schemas.py:121-126](../src/structured_output_agent/schemas.py#L121-L126)). A schema-valid output can therefore fail O3, so the runner must apply `validate_decision_policy` separately.
- Injection compliance is not an output-validity category. It is a diagnostic on injection-tagged records and is covered in Section 8.

Reported invalid-output rate: `(O1 + O2 + O3) / eligible tickets`.

## 5. Retry and Fallback Policy

- Each eligible ticket receives exactly one model attempt.
- No automatic retries occur for O1, O2, O3, or any other outcome.
- No fallback prediction is used. An invalid output is never replaced by always-needs-review or any other baseline.
- Because there is one attempt, first-attempt and final results are identical. Retry metrics are not reported in the first run.
- Any future change to retry or fallback behavior requires a new policy version. Runs under different retry policies must not be compared directly.

## 6. Harness Errors and Model-Output Failures

These are separate categories. Only model-output failures are scored under Section 4.

**Harness errors** occur in the evaluation pipeline, not in the model's response:

- The dataset is missing, unreadable, or invalid; `load_dataset` or `validate_dataset_integrity` fails.
- The dataset hash does not match the recorded hash.
- The run configuration does not match the frozen configuration.
- The split guard is violated (any split other than `dev` is requested).
- The scorer or baseline code raises, including the empty-records error in [baselines.py:23-24](../src/structured_output_agent/baselines.py#L23-L24).

Proposed rule (not yet approved): a harness error aborts the run. No partial metrics are reported, and the error is not counted as a model failure.

**Operational failures** occur when the model call returns no output: transport errors, timeouts, rate limits, or other provider errors. Their treatment is unresolved (Section 10). Approved decision 2 covers only O1 to O3, so operational failures must not be silently folded into those categories.

## 7. Minimum Run Configuration and Reproducibility Metadata

Each run must record the following before it starts, and the record must be fixed for the duration of the run.

**Dataset**

- Dataset path and split (`dev`).
- SHA-256 of the file and the form that was hashed. The baseline document records the CRLF working-tree form on the Windows checkout ([evaluation-baselines.md](evaluation-baselines.md), Section 8).
- Eligible ticket count and excluded count (expected to be 0 under Section 2).

**Code and policy**

- Git commit of the evaluation code and whether the working tree was clean.
- `POLICY_VERSION` and `C1_RULES_VERSION` (currently `1.3` and `c1-1.0.0`).
- This document's version.
- Scoring source: commit of `baselines.py` and `evaluation-baselines.md`.

**Procedure**

- Attempts per ticket: 1.
- Extractor used. The stricter `extract_json_object` is the candidate because it raises on competing objects, but this choice is unresolved (Section 10).
- Prompt template identifier and content hash. Not yet defined in the repository.
- Output schema identifier (`TicketTriageOutput` field set).
- Run timestamp and output path.

**Unresolved provider-specific fields (not selected in this stage)**

- Model identifier and provider.
- Decoding settings, including temperature, top_p, maximum output tokens, and seed where supported.
- Request timeout and rate-limit handling.
- API version or deployment identifier.

These fields must be resolved before any run counts as reproducible.

## 8. Required First-Run Metrics and Baseline Comparisons

Required metrics, all computed on the eligible dev tickets:

1. Eligible ticket count `N`.
2. Strict correct count and strict accuracy, with numerator and denominator shown.
3. Counts for O1, O2, O3, O4, and O5, each reported separately.
4. Invalid-output rate, `(O1 + O2 + O3) / N`.
5. Strict accuracy broken down by tag (`clean`, `ambiguous`, `injection`, `long`), with the denominator for each tag.
6. Count and rate of predictions that require human review.
7. Baseline comparison on the same dataset and scoring rule. Recorded baseline values are majority 2 / 20 (10%) and always-needs-review 4 / 20 (20%). These must be recomputed in the same run and not copied forward.

Diagnostic metric, not used for ranking: accuracy among valid outputs, `O5 / (O4 + O5)`. Both counts must be shown.

Not required in the first run: confidence intervals and per-field accuracy. Injection-compliance rate is not required until its detection method is defined (Section 10).

No pass or fail judgment is defined by this document.

## 9. Test-Set Freeze and Release Criteria

`evals/data/test.jsonl` remains protected. It must not be read, hashed, modified, populated, staged, committed, or used for evaluation until every prerequisite below is met. None is met today.

Freeze prerequisites:

1. This policy is approved and its version is recorded.
2. The run configuration in Section 7 is complete, including the provider-specific fields.
3. The retry and fallback policy in Section 5 is confirmed.
4. A pass threshold or decision rule is defined, approved, and justified. This document does not define one.
5. A development run has completed with recorded metadata.
6. Cross-split duplicate detection has been resolved or explicitly accepted as a limitation. The baseline document states this check is not performed ([evaluation-baselines.md](evaluation-baselines.md), Section 2).
7. The test-set hash is recorded at freeze time. Computing that hash requires explicit user approval, because it reads the protected file.

Release criteria: none are defined yet. They depend on the threshold decision in prerequisite 4.

## 10. Unresolved Decisions and Contradictions

**Unresolved decisions**

- Model provider, model identifier, decoding settings, timeout, and rate-limit handling.
- Treatment of operational failures (Section 6). Options include aborting the run, or scoring as zero credit with a separate count. Excluding them from the denominator is not recommended, because it would inflate accuracy.
- Extractor choice: `extract_json_object` or `extract_json_candidate_text`.
- Pass threshold or decision rule. With 20 dev records, each ticket is 5 percentage points, so any threshold needs a sample-size justification.
- Method for detecting injection compliance on injection-tagged records.
- Cross-split duplicate detection (Section 9, prerequisite 6).
- Whether harness errors abort the run, as proposed in Section 6.

**Contradictions and gaps found while drafting**

- [schemas.py:18](../src/structured_output_agent/schemas.py#L18) cites `triage_policy.md`. No such file is tracked in the repository. The category-to-team mapping exists only as the code constant `CATEGORY_TEAM_MAP`.
- `SemanticValidationError` is defined ([exceptions.py:18-19](../src/structured_output_agent/exceptions.py#L18-L19)), but `validate_decision_policy` raises plain `ValueError` ([schemas.py:174](../src/structured_output_agent/schemas.py#L174), [schemas.py:183](../src/structured_output_agent/schemas.py#L183), [schemas.py:191](../src/structured_output_agent/schemas.py#L191)). The runner must classify O3 by wrapping these errors. Changing the exception type would modify existing code and is not proposed here.
- The baseline document describes eligibility per record, but `load_dataset` makes it all-or-nothing per file (Section 2).
- The baseline document requires `validate_dataset_integrity`, which `load_dataset` does not call.
- `RetryExhaustedError` is defined ([exceptions.py:30-31](../src/structured_output_agent/exceptions.py#L30-L31)) but was not found in any file read. A repository-wide search has not been run.
- No prompt template, model client, or retry loop exists in the repository.

## 11. Document Relationships

- [evaluation-baselines.md](evaluation-baselines.md): scoring rule, baseline definitions, tie-breaking, and recorded dev values.
- This document: run specification, failure handling, configuration, metrics, and freeze criteria.
