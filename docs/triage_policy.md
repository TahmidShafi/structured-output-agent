# Ticket Triage Policy

**Policy Version:** 1.3  
**C1 Rules Version:** c1-1.0.0  
**Document Status:** Current Draft / Documentation Baseline (Not formally ratified as external business policy)  

---

## 1. Scope and Purpose

This document specifies the classification taxonomy, team assignment rules, and deterministic policy constraints for the ticket triage agent.

The triage agent receives customer support ticket text and produces a single structured triage output conforming to the `TicketTriageOutput` schema. This policy defines the authoritative values, relations, and validation rules governing those triage decisions.

> [!NOTE]
> **Documentation Baseline and Version History**  
> This v1.3 document is a documentation baseline that codifies the existing, verified implementation in [`src/structured_output_agent/schemas.py`](../src/structured_output_agent/schemas.py). Historical versions prior to v1.3 (v1.0 through v1.2) were referenced during earlier development phases but are unrecorded in repository Git history. No historical versions, approval dates, or prior changelogs have been fabricated.

---

## 2. Classification Schema and Constraints

The triage agent produces a structured JSON output with exactly five required fields. In accordance with `StrictModel` (`ConfigDict(extra="forbid")`), any extraneous or unexpected fields cause schema validation failure.

### 2.1 Fields and Types

| Field Name | Type | Constraints / Permitted Values | Description |
|---|---|---|---|
| `category` | String Enum (`TicketCategory`) | `billing`, `account_access`, `technical_issue`, `shipping`, `other`, `needs_review` | High-level classification of the customer issue. |
| `priority` | String Enum (`Priority`) | `low`, `normal`, `high`, `urgent` | Operational urgency of the ticket. |
| `assigned_team` | String Enum (`AssignedTeam`) | `payments`, `account_support`, `technical_support`, `fulfillment`, `general_support`, `human_review` | Department responsible for handling the ticket. |
| `summary` | Non-empty String (`SummaryText`) | Length between 1 and 280 characters inclusive; leading/trailing whitespace stripped. | Concise summary of the legitimate support issue. |
| `requires_human_review` | Boolean (`StrictBool`) | `true` or `false` (Strict JSON boolean; string booleans like `"true"` are rejected). | Flag indicating whether human review is required before automated routing. |

### 2.2 Reusable Constraints
- **Ticket Input (`TicketText`):** Non-empty text up to 20,000 characters with leading/trailing whitespace stripped.
- **Ticket Identifier (`TicketId`):** Non-empty text up to 128 characters.
- **Summary Text (`SummaryText`):** Non-empty text up to 280 characters.

---

## 3. Deterministic Policy Rules (C1 Constraints)

The deterministic business rules (version `c1-1.0.0`) are enforced programmatically by `validate_decision_policy`. A triage output can be strictly valid under Pydantic schema validation yet fail deterministic policy validation.

### 3.1 Validator A: Category-to-Team Alignment Rule

Every ticket category maps bijectively to exactly one assigned team in `CATEGORY_TEAM_MAP`. Outputs assigning any other team violate Validator A:

| Ticket Category (`category`) | Required Assigned Team (`assigned_team`) |
|---|---|
| `billing` | `payments` |
| `account_access` | `account_support` |
| `technical_issue` | `technical_support` |
| `shipping` | `fulfillment` |
| `other` | `general_support` |
| `needs_review` | `human_review` |

**Validation Failure Behavior:**  
If `decision.assigned_team != CATEGORY_TEAM_MAP[decision.category]`, `validate_decision_policy` raises `ValueError` with the message:  
`Validator A: '<category>' requires '<required_team>', not '<assigned_team>'.`

### 3.2 Validator B: Mandatory Human Review Triggers

Validator B mandates that human review (`requires_human_review = true`) must be set under two specific conditions:

1. **Category `needs_review` Trigger:**  
   If `category == "needs_review"`, `requires_human_review` must be `true`.  
   *Failure message:* `Validator B: needs_review requires human review.`

2. **Priority `urgent` Trigger:**  
   If `priority == "urgent"`, `requires_human_review` must be `true`.  
   *Failure message:* `Validator B: urgent priority requires human review.`

Under all other conditions, `requires_human_review` may be either `true` or `false` based on model judgment.

---

## 4. Separation of Conceptual Boundaries

To ensure clear reporting and avoid conflating distinct agent behaviors, the following concepts are strictly separated in implementation and evaluation:

### 4.1 Schema Validity vs. Policy Validity
- **Schema Validity:** The output is valid JSON and strictly conforms to `TicketTriageOutput` (valid enums, string constraints, boolean type, no extra fields).
- **Policy Validity:** The output satisfies Validator A (`CATEGORY_TEAM_MAP`) and Validator B (mandatory review triggers).
- **Relationship:** An output can be schema-valid while violating policy (Outcome O3). All policy-valid outputs (Outcomes O4 and O5) are schema-valid by definition.

### 4.2 Human Review Decisions vs. Escalation Actions
- **Human Review (`requires_human_review`):** A triage classification field indicating whether the initial routing requires human oversight.
- **Escalation Actions (`request_escalation`):** A tool operation in [`tools.py`](../src/structured_output_agent/tools.py) that queues a support escalation record (`status: "queued"`).
- **Decoupling Rule:** Calling `request_escalation` does **not** automatically set, mutate, or imply `requires_human_review = true`. Escalation management and triage review flags operate in separate workflows.

### 4.3 Strict Correctness vs. Injection Resilience
- **Strict Decision Accuracy:** Measures whether the four decision fields (`category`, `priority`, `assigned_team`, `requires_human_review`) match an accepted ground-truth decision for the ticket. Summary text is excluded from decision scoring.
- **Injection Resilience:** Measures whether the model resists adversarial prompt injections across four specific vectors (`field_override`, `summary_echo`, `prompt_leak`, `tool_call`). High decision accuracy does not imply injection resilience (for example, a ticket may achieve correct triage while echoing an injected marker phrase in its summary).

---

## 5. Tool Registry and Execution Safety

The agent provides a safe tool execution registry ([`tools.py`](../src/structured_output_agent/tools.py)) with strict input/output validation:

- **Approved Tools:** `ticket_lookup` (retrieves ticket status and details) and `request_escalation` (queues an escalation record).
- **Decoy Tool Protection:** The registry explicitly blocks destructive or decoy tools (e.g., `delete_account`), raising `ToolExecutionError`.
- **Entity Identity Check:** The registry enforces that the tool result's `ticket_id` matches the input argument's `ticket_id`, raising `ToolResultMismatchError` on any discrepancy.

---

## 6. Proposed Rules — Pending Approval

The following rules represent current prompt instructions or proposed operational guidelines that have **not** been formally approved as binding business policy. Implementation behavior (such as prompt template instructions) does not constitute business policy approval.

### 6.1 Adversarial Prompt Injection Handling (Proposed)
- **Classification Invariance:** Instructions embedded within customer ticket text that attempt to dictate classification rules, response format, summary contents, tool execution, or field values should be ignored.
- **Underlying Request Triage:** The model should classify the customer's legitimate underlying support issue.
- **No Automatic Review on Injection:** The model should not assign `needs_review` solely because an injection attempt is present.
- *Status:* Implemented as guidance in `DEFAULT_PROMPT_TEMPLATE`, but pending formal business review.

### 6.2 Priority Tier and Ambiguity Criteria (Proposed)
- Formal operational definitions distinguishing customer impact thresholds for `low`, `normal`, `high`, and `urgent` priority are pending product SLA approval.
- Semantic criteria defining when ticket ambiguity warrants `needs_review` (versus best-effort routing to a standard category) are pending stakeholder sign-off.

---

## 7. Document Relationships

- [`evaluation-policy.md`](evaluation-policy.md): Governs development evaluation, metric definitions, retry accounting, and test release criteria.
- [`evaluation-baselines.md`](evaluation-baselines.md): Governs scoring mechanics, tie-breaking rules, and baseline algorithms.
- [`src/structured_output_agent/schemas.py`](../src/structured_output_agent/schemas.py): Authoritative implementation of data types, policy constants, and runtime validators.
