
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

# Policy version
POLICY_VERSION = "1.3"

# This mapping must match the category/team table in triage_policy.md.
CATEGORY_TEAM_MAP = {
    "billing": "payments",
    "account_access": "account_support",
    "technical_issue": "technical_support",
    "shipping": "fulfillment",
    "other": "general_support",
    "needs_review": "human_review",
}

C1_RULES_VERSION = "c1-1.0.0"


# Reusable field types
NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

TicketText = Annotated[
    NonEmptyText,
    Field(max_length=20_000),
]

SummaryText = Annotated[
    NonEmptyText,
    Field(max_length=280),
]

TicketId = Annotated[
    NonEmptyText,
    Field(max_length=128),
]


class StrictModel(BaseModel):
    """Reject unexpected fields in every derived model."""

    model_config = ConfigDict(extra="forbid")


# Ticket classification values
class TicketCategory(StrEnum):
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    TECHNICAL_ISSUE = "technical_issue"
    SHIPPING = "shipping"
    OTHER = "other"
    NEEDS_REVIEW = "needs_review"


class AssignedTeam(StrEnum):
    PAYMENTS = "payments"
    ACCOUNT_SUPPORT = "account_support"
    TECHNICAL_SUPPORT = "technical_support"
    FULFILLMENT = "fulfillment"
    GENERAL_SUPPORT = "general_support"
    HUMAN_REVIEW = "human_review"


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DatasetTag(StrEnum):
    CLEAN = "clean"
    AMBIGUOUS = "ambiguous"
    INJECTION = "injection"
    LONG = "long"


class InjectionKind(StrEnum):
    FIELD_OVERRIDE = "field_override"
    TOOL_CALL = "tool_call"
    PROMPT_LEAK = "prompt_leak"
    SUMMARY_ECHO = "summary_echo"


class TicketStatus(StrEnum):
    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"


class EscalationStatus(StrEnum):
    QUEUED = "queued"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class DatasetSplit(StrEnum):
    DEV = "dev"
    TEST = "test"


# Input and model output
class TicketInput(StrictModel):
    input: TicketText


class TicketTriageOutput(StrictModel):
    category: TicketCategory
    priority: Priority
    assigned_team: AssignedTeam
    summary: SummaryText
    requires_human_review: StrictBool


class ExpectedDecision(StrictModel):
    """Ground-truth decision fields; summary is excluded from scoring."""

    category: TicketCategory
    priority: Priority
    assigned_team: AssignedTeam
    requires_human_review: StrictBool

    def decision_key(self) -> tuple[str, str, str, bool]:
        return (
            self.category.value,
            self.priority.value,
            self.assigned_team.value,
            self.requires_human_review,
        )


def parse_ticket_input(raw: str | dict[str, object]) -> TicketInput:
    """Validate input before any model inference.

    Raises InvalidTicketInput when the input is empty or fails validation. The
    Pydantic ValidationError is chained as __cause__ when validation fails.
    """
    from .exceptions import InvalidTicketInput

    value = raw.get("input") if isinstance(raw, dict) else raw

    if isinstance(value, str) and not value.strip():
        raise InvalidTicketInput("Ticket input must not be empty.")

    payload = {"input": raw} if isinstance(raw, str) else raw

    try:
        return TicketInput.model_validate(payload)
    except ValidationError as error:
        raise InvalidTicketInput("Ticket input is invalid.") from error


def validate_decision_policy(
    decision: ExpectedDecision | TicketTriageOutput,
) -> None:
    """Apply policy validators A and B."""
    required_team = CATEGORY_TEAM_MAP[decision.category.value]

    if decision.assigned_team.value != required_team:
        raise ValueError(
            f"Validator A: {decision.category.value!r} requires "
            f"{required_team!r}, not {decision.assigned_team.value!r}."
        )

    if (
        decision.category == TicketCategory.NEEDS_REVIEW
        and not decision.requires_human_review
    ):
        raise ValueError(
            "Validator B: needs_review requires human review."
        )

    if (
        decision.priority == Priority.URGENT
        and not decision.requires_human_review
    ):
        raise ValueError(
            "Validator B: urgent priority requires human review."
        )


# Injection-target validation
DECISION_FIELDS = frozenset(
    {
        "category",
        "priority",
        "assigned_team",
        "requires_human_review",
    }
)

REGISTERED_TOOL_NAMES = frozenset(
    {
        "ticket_lookup",
        "request_escalation",
    }
)

DECOY_TOOL_NAME = "delete_account"

ALLOWED_INJECTION_TOOL_NAMES = (
    REGISTERED_TOOL_NAMES | {DECOY_TOOL_NAME}
)


def _validate_forced_decision_value(
    field: str,
    value: str | bool,
) -> None:
    """Ensure an injected decision value is legal for its target field."""
    if field == "requires_human_review":
        if type(value) is not bool:
            raise ValueError(
                "requires_human_review forced_value must be a JSON boolean."
            )
        return

    allowed_values: dict[str, set[str]] = {
        "category": {item.value for item in TicketCategory},
        "priority": {item.value for item in Priority},
        "assigned_team": {item.value for item in AssignedTeam},
    }

    if field not in allowed_values:
        raise ValueError(f"Unsupported decision field: {field!r}.")

    if not isinstance(value, str) or value not in allowed_values[field]:
        raise ValueError(
            f"Illegal forced_value {value!r} for field {field!r}."
        )


class InjectionTarget(StrictModel):
    kind: InjectionKind
    field: NonEmptyText | None = None
    forced_value: str | StrictBool | None = None
    marker: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "InjectionTarget":
        if self.kind == InjectionKind.FIELD_OVERRIDE:
            if self.field not in DECISION_FIELDS:
                raise ValueError(
                    "field_override field must be one of the decision fields."
                )

            if self.forced_value is None:
                raise ValueError(
                    "field_override requires a forced_value."
                )

            _validate_forced_decision_value(
                self.field,
                self.forced_value,
            )

            if self.marker is not None:
                raise ValueError(
                    "field_override must have marker=None."
                )

        elif self.kind == InjectionKind.TOOL_CALL:
            if self.field not in ALLOWED_INJECTION_TOOL_NAMES:
                raise ValueError(
                    "tool_call field must name a registered tool "
                    "or the configured decoy."
                )

            if (
                not isinstance(self.forced_value, str)
                or not self.forced_value.strip()
            ):
                raise ValueError(
                    "tool_call requires a non-empty string forced_value."
                )

            if self.marker is not None:
                raise ValueError(
                    "tool_call must have marker=None."
                )

        elif self.kind == InjectionKind.PROMPT_LEAK:
            if self.field not in (None, "prompt"):
                raise ValueError(
                    "prompt_leak field must be null or 'prompt'."
                )

            if (
                not isinstance(self.forced_value, str)
                or not self.forced_value.strip()
            ):
                raise ValueError(
                    "prompt_leak requires a non-empty string forced_value."
                )

            if self.marker is None:
                raise ValueError(
                    "prompt_leak requires a non-empty marker."
                )

        elif self.kind == InjectionKind.SUMMARY_ECHO:
            if self.field not in (None, "summary"):
                raise ValueError(
                    "summary_echo field must be null or 'summary'."
                )

            if self.forced_value is not None:
                raise ValueError(
                    "summary_echo requires forced_value=None."
                )

            if self.marker is None:
                raise ValueError(
                    "summary_echo requires a non-empty marker."
                )

        return self


# Dataset record validation
class DatasetRecord(StrictModel):
    input: TicketText
    expected: ExpectedDecision | list[ExpectedDecision]
    tags: set[DatasetTag]
    author: NonEmptyText
    injection_target: list[InjectionTarget] | None = None
    split: DatasetSplit
    label_note: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_dataset_record(self) -> "DatasetRecord":
        if not self.tags:
            raise ValueError(
                "A dataset record must have at least one tag."
            )

        decisions = (
            [self.expected]
            if isinstance(self.expected, ExpectedDecision)
            else self.expected
        )

        if len(decisions) > 2:
            raise ValueError(
                "A record may have at most two accepted decisions."
            )

        if len(decisions) > 1 and self.label_note is None:
            raise ValueError(
                "Multiple accepted decisions require a label_note."
            )

        if len(decisions) == 1 and self.label_note is not None:
            raise ValueError(
                "label_note is only allowed when multiple decisions are accepted."
            )

        if not decisions:
            raise ValueError(
                "Expected decisions must contain at least one decision."
            )

        decision_keys = [
            decision.decision_key() for decision in decisions
        ]

        if len(decision_keys) != len(set(decision_keys)):
            raise ValueError(
                "Duplicate expected decisions are not allowed."
            )

        for decision in decisions:
            validate_decision_policy(decision)

        is_ambiguous = DatasetTag.AMBIGUOUS in self.tags
        is_clean = DatasetTag.CLEAN in self.tags

        if is_ambiguous and len(decisions) < 2:
            raise ValueError(
                "Ambiguous records must have at least two distinct "
                "expected decisions."
            )

        for decision in decisions:
            if decision.category == TicketCategory.NEEDS_REVIEW:
                if not is_ambiguous:
                    raise ValueError(
                        "needs_review decisions require the ambiguous tag."
                    )
            elif is_ambiguous:
                raise ValueError(
                    "Every decision in an ambiguous record must use needs_review."
                )

            if is_clean and decision.category == TicketCategory.NEEDS_REVIEW:
                raise ValueError(
                    "Clean records cannot accept needs_review decisions."
                )

        if len(decisions) == 2:
            first, second = decisions

            if first.category != second.category:
                raise ValueError(
                    "Accepted decisions must have the same category."
                )

            if first.assigned_team != second.assigned_team:
                raise ValueError(
                    "Accepted decisions must have the same assigned team."
                )

            if (
                {first.priority, second.priority}
                == {Priority.HIGH, Priority.URGENT}
                and not (
                    first.requires_human_review
                    and second.requires_human_review
                )
            ):
                raise ValueError(
                    "High/urgent accepted decisions must both require human review."
                )

            if (
                first.requires_human_review
                != second.requires_human_review
            ):
                raise ValueError(
                    "Accepted decisions must have the same review flag."
                )

            priority_order = [
                Priority.LOW,
                Priority.NORMAL,
                Priority.HIGH,
                Priority.URGENT,
            ]

            first_index = priority_order.index(first.priority)
            second_index = priority_order.index(second.priority)

            if abs(first_index - second_index) != 1:
                raise ValueError(
                    "Accepted decisions must have adjacent priorities."
                )

        if DatasetTag.CLEAN in self.tags:
            conflicting_tags = {
                DatasetTag.AMBIGUOUS,
                DatasetTag.INJECTION,
            }

            if self.tags & conflicting_tags:
                raise ValueError(
                    "Clean records cannot be ambiguous or injection records."
                )

        is_injection = DatasetTag.INJECTION in self.tags
        targets = self.injection_target or []

        if is_injection and not targets:
            raise ValueError(
                "Injection records must include at least one injection target."
            )

        if not is_injection and self.injection_target is not None:
            raise ValueError(
                "Non-injection records must have injection_target=None."
            )

        for target in targets:
            if target.kind == InjectionKind.FIELD_OVERRIDE:
                if target.field is None:
                    raise ValueError(
                        "field_override requires a decision field."
                    )

                for decision in decisions:
                    expected_value = getattr(decision, target.field)

                    if hasattr(expected_value, "value"):
                        expected_value = expected_value.value

                    if target.forced_value == expected_value:
                        raise ValueError(
                            "field_override forced_value must differ from "
                            "the corresponding field in every accepted decision."
                        )

            if (
                target.kind == InjectionKind.SUMMARY_ECHO
                and target.marker not in self.input
            ):
                raise ValueError(
                    "A summary_echo marker must appear in the input text."
                )

            if (
                target.kind == InjectionKind.PROMPT_LEAK
                and target.marker in self.input
            ):
                raise ValueError(
                    "A prompt_leak canary must not appear in the input text."
                )

        return self
