# Development Evaluation Baselines



**Status:** Preliminary development baseline  

**Policy version:** `1.3`  

**C1 rules version:** `c1-1.0.0`  

**Dataset:** `evals/data/dev.jsonl`  

**Development records:** 20



## 1. Purpose



These baselines establish simple reference points for evaluating the structured-output ticket-triage agent.



Results in this document are calculated from the development dataset only. They are not test-set results and must not be presented as evidence of generalization to unseen data.



## 2. Evaluation Scoring Rule



A prediction is strictly correct when all four decision fields match at least one accepted expected decision for the ticket:



- `category`

- `priority`

- `assigned_team`

- `requires_human_review`



For tickets with multiple accepted expected decisions, a prediction matching either accepted decision receives credit for that ticket.



Strict accuracy is calculated as:



**Strict accuracy = correct tickets / total eligible tickets**



Each ticket contributes at most one correct result.



## 3. Majority Complete-Decision Baseline



This baseline always predicts the most frequently occurring complete expected decision.



| Field | Baseline prediction |

|---|---|

| Category | `account_access` |

| Priority | `normal` |

| Assigned team | `account_support` |

| Requires human review | `false` |



**Observed weighted support:** `2.0`  

**Correct tickets:** `2 / 20`  

**Strict accuracy:** `10%`



### Deterministic tie-breaking



If multiple complete decisions have equal weighted support, select the decision using this order:



1. Category alphabetically.

2. Priority in this order: `low`, `normal`, `high`, `urgent`.

3. Assigned team alphabetically.

4. Human-review flag, with `false` before `true`.



For records with multiple accepted decisions, each decision receives a weight of `1 / number of accepted decisions in that record`. Weighted support is used to select the majority prediction; strict accuracy is calculated separately using the ticket-level scoring rule above.



## 4. Always-Needs-Review Baseline



This baseline always predicts the following complete decision.



| Field | Baseline prediction |

|---|---|

| Category | `needs_review` |

| Priority | `normal` |

| Assigned team | `human_review` |

| Requires human review | `true` |



**Correct tickets:** `4 / 20`  

**Strict accuracy:** `20%`



## 5. Comparison



| Baseline | Correct tickets | Strict accuracy |

|---|---:|---:|

| Majority complete decision | 2 / 20 | 10% |

| Always needs review | 4 / 20 | 20% |



The always-needs-review baseline exceeds the majority baseline by **10 percentage points** on this development dataset.



These results are descriptive. They do not establish which strategy is safest or most useful in production.



## 6. Reproducibility and Limitations



- Dataset split: development (`dev`).

- Dataset size: 20 records.

- Policy version: `1.3`.

- C1 rules version: `c1-1.0.0`.

- The results depend on the current dataset labels and policy.

- The baselines do not measure model performance, schema-validation effectiveness, retry recovery, or resistance to prompt injection.

- The test dataset must remain untouched until the evaluation policy, scoring rules, and relevant configuration are frozen.



## 7. Status



These are preliminary development-set baselines. Before the formal evaluation, verify the dataset and policy versions, record the exact dataset SHA-256 hash, and freeze the evaluation configuration.



No test-set results are reported in this document.


## 8. Dataset Fingerprint

The development dataset was fingerprinted using SHA-256 over the exact file bytes.

- **File:** `evals/data/dev.jsonl`
- **SHA-256:** `23A7ED778F6D95D7909F36371093A6ED4290A837F71428746A4F4693705C4F1E`
- **Recorded on:** 2026-10-11

Recalculate this hash whenever the dataset changes. A different hash indicates that the file bytes differ; it does not, by itself, establish whether the changes are intentional or correct.
