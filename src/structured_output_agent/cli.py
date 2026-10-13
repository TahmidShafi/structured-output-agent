"""Evaluation execution interface and CLI entrypoint.

Provides:
- StubModelClient: deterministic offline ModelClient for testing and harness execution.
- execute_evaluation: pure evaluation orchestration function.
- main: CLI entry point with standardized exit codes (0, 1, 2).
"""

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .exceptions import AgentError, OperationalError
from .model_client import ModelClient
from .reporting import EvaluationReport, generate_report
from .run_config import (
    DEFAULT_DATASET_PATH,
    RunConfig,
    validate_output_path,
)
from .runner import PromptBuilder, RunStatus, run_evaluation
from .schemas import DatasetSplit


class StubModelClient:
    """Deterministic offline model client for testing the evaluation harness.

    WARNING: This client produces canned deterministic responses and does NOT
    represent real LLM performance, reasoning, or capabilities. It is intended
    solely for testing, verification, and offline evaluation pipeline execution.
    """

    client_type: str = "stub"
    model_id: str | None = None

    def __init__(
        self,
        default_response: str | None = None,
        *,
        canned_responses: Sequence[str | Exception] | None = None,
    ) -> None:
        if default_response is None:
            self.default_response = json.dumps(
                {
                    "category": "billing",
                    "priority": "normal",
                    "assigned_team": "payments",
                    "summary": "Deterministic offline stub response.",
                    "requires_human_review": False,
                }
            )
        else:
            self.default_response = default_response
        if canned_responses is not None:
            if len(canned_responses) == 0:
                raise ValueError("canned_responses must not be empty if provided.")
            self.canned_responses = list(canned_responses)
        else:
            self.canned_responses = None
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        """Return a deterministic response or simulate operational failure."""
        self.calls.append(prompt)
        if self.canned_responses is not None:
            index = len(self.calls) - 1
            item = (
                self.canned_responses[index]
                if index < len(self.canned_responses)
                else self.canned_responses[-1]
            )
            if isinstance(item, Exception):
                raise item
            return item
        return self.default_response


def execute_evaluation(
    config: RunConfig,
    client: ModelClient,
    output_path: Path | str,
    *,
    prompt_builder: PromptBuilder | None = None,
    created_at: str | datetime | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = False,
) -> EvaluationReport:
    """Orchestrate evaluation run, generate report, and write to destination.

    Destination safety & failure guarantees:
    1. If the evaluation run fails or aborts at any point, any pre-existing report
       file at output_path is preserved completely intact and untouched.
    2. The report is written to a uniquely named temporary file in the destination
       directory and atomically moved into place only after evaluation, metrics,
       and JSON serialization succeed.
    3. The temporary file is guaranteed to be cleaned up on write or replacement
       failures without affecting any existing file at output_path.

    Raises:
        OperationalError: If evaluation was aborted due to model provider failure.
        AgentError / ValueError: If dataset, accounting, or metrics validation fails.
        OSError: If output file writing fails.
    """
    # 1. Validate output path safety before inference, directory creation,
    #    or file writes
    out_path = validate_output_path(output_path, config.dataset_path)

    # 2. Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Execute runner
    run = run_evaluation(
        config,
        client,
        prompt_builder=prompt_builder,
        created_at=created_at,
        checkpoint_path=checkpoint_path,
        resume=resume,
        report_path=out_path,
    )

    # 4. Check for operational abort
    if run.status == RunStatus.ABORTED:
        raise OperationalError(
            f"Evaluation aborted: [{run.error_type}] {run.error_message}"
        )

    # 5. Generate report (single source of truth for metrics calculation & validation)
    report = generate_report(run)

    # 6. Write to uniquely named temporary file, then atomically replace destination
    temp_path = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        report.write_json(temp_path)
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="structured_output_agent",
        description=(
            "Run support-ticket triage evaluation and produce a deterministic "
            "JSON report."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to evaluation dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/report.json"),
        help="Path for generated JSON report (default: evals/results/report.json).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DatasetSplit.DEV.value,
        help="Dataset split (default: dev; only dev is allowed).",
    )
    parser.add_argument(
        "--client",
        type=str,
        choices=["stub", "stub-abort", "gemini"],
        default="stub",
        help="Model client implementation to use (default: stub).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. for Gemini; default: gemini-3.8-flash).",
    )

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to save incremental evaluation checkpoints for a new run.",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to an existing evaluation checkpoint to resume.",
    )

    return parser.parse_args(argv)


def _resolve_client(client_name: str, model: str | None = None) -> ModelClient:
    if client_name == "stub":
        return StubModelClient()
    if client_name == "stub-abort":
        return StubModelClient(
            canned_responses=[
                OperationalError("Simulated client operational abort (HTTP 503)")
            ]
        )
    if client_name == "gemini":
        from .gemini_client import GeminiModelClient

        return GeminiModelClient(model=model)
    raise ValueError(f"Unknown client type: {client_name!r}")


def main(
    argv: Sequence[str] | None = None,
    *,
    client: ModelClient | None = None,
) -> int:
    """CLI entry point returning exit code (0: success, 1: abort, 2: input error).

    Exit codes:
    - 0: Successful evaluation and written report.
    - 1: Operational evaluation failure (details on stderr).
    - 2: CLI argument, configuration, dataset, or harness error (details on stderr).

    Unexpected programming errors are not caught and propagate naturally,
    keeping them clearly distinguishable from operational and harness errors.
    """
    try:
        args = parse_args(argv)

        # Build and validate RunConfig (enforces split and dataset_path guards)
        try:
            split_enum = DatasetSplit(args.split)
        except ValueError:
            sys.stderr.write(
                f"Error: Invalid split {args.split!r}. "
                "Evaluation is restricted to 'dev'.\n"
            )
            return 2

        config = RunConfig(
            dataset_path=args.dataset_path,
            split=split_enum,
        )

        model_client = (
            client
            if client is not None
            else _resolve_client(args.client, model=args.model)
        )

        if args.resume is not None:
            checkpoint_path = args.resume
            resume = True
        elif args.checkpoint_path is not None:
            checkpoint_path = args.checkpoint_path
            resume = False
        else:
            checkpoint_path = None
            resume = False

        report = execute_evaluation(
            config,
            model_client,
            args.output,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

        sc_count = report.metrics.strict_correct.count
        sc_rate = report.metrics.strict_correct.rate or 0.0
        inv_count = report.metrics.invalid_outputs.count
        inv_rate = report.metrics.invalid_outputs.rate or 0.0
        n = report.metrics.eligible_count

        sys.stdout.write(
            f"Evaluation completed successfully.\n"
            f"  Eligible tickets: {n}\n"
            f"  Strict correct:   {sc_count}/{n} ({sc_rate:.1%})\n"
            f"  Invalid outputs:  {inv_count}/{n} ({inv_rate:.1%})\n"
            f"  Report written:   {args.output}\n"
        )
        return 0

    except SystemExit as exc:
        # Argparse --help (code 0) or argument parsing failure (code 2)
        return 0 if exc.code == 0 else 2

    except OperationalError as exc:
        sys.stderr.write(f"Operational failure (exit 1): {exc}\n")
        return 1

    except (AgentError, ValueError, OSError, ImportError) as exc:
        sys.stderr.write(f"Configuration/Harness error (exit 2): {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
