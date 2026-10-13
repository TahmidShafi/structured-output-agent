import importlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from structured_output_agent.checkpoint_io import load_checkpoint
from structured_output_agent.cli import StubModelClient, execute_evaluation, main
from structured_output_agent.exceptions import OperationalError
from structured_output_agent.model_client import ModelClient
from structured_output_agent.reporting import EvaluationReport
from structured_output_agent.run_config import RunConfig
from structured_output_agent.runner import RunStatus, run_evaluation
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    ExpectedDecision,
    Priority,
    TicketCategory,
)


def _make_valid_dataset_file(path: Path) -> None:
    category_teams = {
        TicketCategory.BILLING: AssignedTeam.PAYMENTS,
        TicketCategory.ACCOUNT_ACCESS: AssignedTeam.ACCOUNT_SUPPORT,
        TicketCategory.TECHNICAL_ISSUE: AssignedTeam.TECHNICAL_SUPPORT,
        TicketCategory.SHIPPING: AssignedTeam.FULFILLMENT,
        TicketCategory.OTHER: AssignedTeam.GENERAL_SUPPORT,
        TicketCategory.NEEDS_REVIEW: AssignedTeam.HUMAN_REVIEW,
    }
    records = []
    for cat, team in category_teams.items():
        for pri in (Priority.NORMAL, Priority.HIGH):
            if cat == TicketCategory.NEEDS_REVIEW:
                other_pri = Priority.HIGH if pri == Priority.NORMAL else Priority.NORMAL
                expected = [
                    ExpectedDecision(
                        category=cat,
                        priority=pri,
                        assigned_team=team,
                        requires_human_review=True,
                    ),
                    ExpectedDecision(
                        category=cat,
                        priority=other_pri,
                        assigned_team=team,
                        requires_human_review=True,
                    ),
                ]
                tags = {DatasetTag.AMBIGUOUS}
                label_note = "Coverage test needs_review"
            else:
                expected = ExpectedDecision(
                    category=cat,
                    priority=pri,
                    assigned_team=team,
                    requires_human_review=False,
                )
                tags = {DatasetTag.CLEAN}
                label_note = None

            rec = DatasetRecord(
                input=f"Ticket regarding {cat.value} with priority {pri.value}.",
                expected=expected,
                tags=tags,
                author="cli_test_author",
                split=DatasetSplit.DEV,
                label_note=label_note,
            )
            records.append(rec)

    lines = [r.model_dump_json() for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stub_model_client_protocol_conformance() -> None:
    client = StubModelClient()
    assert isinstance(client, ModelClient)

    # Calling complete records the prompt and returns valid default JSON
    prompt = "Triage this ticket"
    response = client.complete(prompt)
    assert client.calls == [prompt]

    parsed = json.loads(response)
    assert parsed["category"] == "billing"
    assert parsed["assigned_team"] == "payments"
    assert parsed["requires_human_review"] is False


def test_stub_model_client_canned_responses_and_exceptions() -> None:
    shipping_json = json.dumps(
        {
            "category": "shipping",
            "priority": "high",
            "assigned_team": "fulfillment",
            "summary": "Shipment delayed",
            "requires_human_review": False,
        }
    )
    canned = [
        shipping_json,
        OperationalError("Simulated provider outage (HTTP 503)"),
    ]
    client = StubModelClient(canned_responses=canned)

    resp1 = client.complete("Prompt 1")
    assert "shipping" in resp1

    with pytest.raises(OperationalError, match="Simulated provider outage"):
        client.complete("Prompt 2")


def test_stub_model_client_default_response_omitted_returns_default_json() -> None:
    client = StubModelClient()
    response = client.complete("Test prompt")
    parsed = json.loads(response)
    assert parsed["category"] == "billing"
    assert parsed["priority"] == "normal"
    assert parsed["assigned_team"] == "payments"
    assert parsed["requires_human_review"] is False


def test_stub_model_client_explicit_empty_response_returns_empty_string() -> None:
    client = StubModelClient(default_response="")
    response = client.complete("Test prompt")
    assert response == ""


def test_stub_model_client_custom_non_empty_response_returned_unchanged() -> None:
    custom_payload = '{"category": "shipping", "summary": "custom test"}'
    client = StubModelClient(default_response=custom_payload)
    response = client.complete("Test prompt")
    assert response == custom_payload


def test_execute_evaluation_with_empty_stub_response(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "reports" / "report.json"

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient(default_response="")

    report = execute_evaluation(config, client, output_file)

    assert isinstance(report, EvaluationReport)
    assert report.metrics.eligible_count == 12
    assert report.metrics.strict_correct.count == 0
    assert report.metrics.invalid_outputs.count == 12
    assert output_file.exists()


def test_execute_evaluation_success(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "reports" / "report.json"

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    report = execute_evaluation(config, client, output_file)

    assert isinstance(report, EvaluationReport)
    assert report.metrics.eligible_count == 12
    assert report.dataset_fingerprint is not None
    assert output_file.exists()

    # Verify content was written properly
    content = output_file.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert content == report.to_json() + "\n"


def test_execute_evaluation_aborted_preserves_existing_report(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"

    # Pre-create a previous valid report at destination
    output_file.write_text('{"previous_report": true}\n', encoding="utf-8")
    assert output_file.exists()

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient(canned_responses=[OperationalError("Connection reset")])

    with pytest.raises(OperationalError, match="Evaluation aborted"):
        execute_evaluation(config, client, output_file)

    # Previous valid report must be preserved intact and untouched
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == '{"previous_report": true}\n'


def test_execute_evaluation_creates_parent_directory(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    nested_output = tmp_path / "a" / "b" / "c" / "report.json"
    assert not nested_output.parent.exists()

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    report = execute_evaluation(config, client, nested_output)

    assert nested_output.exists()
    assert report.metrics.eligible_count == 12


def test_main_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "output.json"

    exit_code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--client",
            "stub",
        ]
    )

    assert exit_code == 0
    assert output_file.exists()

    captured = capsys.readouterr()
    assert "Evaluation completed successfully" in captured.out
    assert "Eligible tickets: 12" in captured.out
    assert captured.err == ""


def test_main_operational_abort_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "output.json"

    # Seed with a previous report
    output_file.write_text('{"previous_report": true}\n', encoding="utf-8")

    exit_code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--client",
            "stub-abort",
        ]
    )

    assert exit_code == 1
    # Ensure previous report is preserved intact
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == '{"previous_report": true}\n'

    captured = capsys.readouterr()
    assert "Operational failure (exit 1)" in captured.err
    assert "Simulated client operational abort" in captured.err


def test_main_invalid_split_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_file = tmp_path / "output.json"

    exit_code = main(["--split", "test", "--output", str(output_file)])

    assert exit_code == 2
    assert not output_file.exists()

    captured = capsys.readouterr()
    assert (
        "restricted to 'dev'" in captured.err
        or "Evaluation is restricted to the development split" in captured.err
    )


def test_main_protected_dataset_path_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_file = tmp_path / "output.json"

    exit_code = main(
        [
            "--dataset-path",
            "evals/data/test.jsonl",
            "--output",
            str(output_file),
        ]
    )

    assert exit_code == 2
    assert not output_file.exists()

    captured = capsys.readouterr()
    assert "protected test dataset" in captured.err


def test_main_missing_dataset_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_file = tmp_path / "output.json"
    missing_path = tmp_path / "nonexistent.jsonl"

    exit_code = main(
        [
            "--dataset-path",
            str(missing_path),
            "--output",
            str(output_file),
        ]
    )

    assert exit_code == 2
    assert not output_file.exists()

    captured = capsys.readouterr()
    assert "Configuration/Harness error (exit 2)" in captured.err


def test_main_invalid_cli_arg_exit_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--completely-invalid-argument"])
    assert exit_code == 2


def test_main_custom_injected_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "output.json"

    injected_client = StubModelClient()
    exit_code = main(
        ["--dataset-path", str(dataset_file), "--output", str(output_file)],
        client=injected_client,
    )

    assert exit_code == 0
    assert len(injected_client.calls) == 12
    assert output_file.exists()


def test_deterministic_report_content(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"

    exit_code = main(
        ["--dataset-path", str(dataset_file), "--output", str(output_file)]
    )
    assert exit_code == 0

    report = EvaluationReport.from_json(output_file.read_text(encoding="utf-8"))
    assert report.schema_version == "1.1"
    assert report.metrics.eligible_count == 12
    assert report.dataset_fingerprint is not None
    assert len(report.results) == 12

    # Check key ordering in JSON
    raw = output_file.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert list(data.keys()) == sorted(data.keys())


def test_stub_model_client_rejects_empty_canned_responses() -> None:
    with pytest.raises(
        ValueError, match="canned_responses must not be empty if provided"
    ):
        StubModelClient(canned_responses=[])


def test_main_unexpected_exception_propagates(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "output.json"

    class _BuggyClient:
        def complete(self, prompt: str) -> str:
            raise TypeError("Unexpected bug in model client")

    with pytest.raises(TypeError, match="Unexpected bug in model client"):
        main(
            ["--dataset-path", str(dataset_file), "--output", str(output_file)],
            client=_BuggyClient(),
        )
    assert not output_file.exists()


def test_execute_evaluation_write_failure_preserves_dest_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    output_file.write_text('{"existing_report": true}\n', encoding="utf-8")

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    def mock_write_json(
        self: EvaluationReport, path: Path | str, *, indent: int | None = 2
    ) -> None:
        Path(path).write_text("partial corrupted data", encoding="utf-8")
        raise OSError("Disk full simulation")

    monkeypatch.setattr(EvaluationReport, "write_json", mock_write_json)

    with pytest.raises(OSError, match="Disk full simulation"):
        execute_evaluation(config, client, output_file)

    # Existing report is preserved intact
    assert output_file.read_text(encoding="utf-8") == '{"existing_report": true}\n'

    # Uniquely named temporary file was cleaned up
    temp_files = list(output_file.parent.glob(".*.tmp"))
    assert temp_files == []


def test_execute_evaluation_replacement_failure_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    output_file.write_text('{"existing_report": true}\n', encoding="utf-8")

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    def mock_replace(self: Path, target: Path) -> Path:
        raise OSError("Permission denied replacing file")

    monkeypatch.setattr(Path, "replace", mock_replace)

    with pytest.raises(OSError, match="Permission denied replacing file"):
        execute_evaluation(config, client, output_file)

    # Existing report is preserved intact
    assert output_file.read_text(encoding="utf-8") == '{"existing_report": true}\n'

    # Uniquely named temporary file was cleaned up
    temp_files = list(output_file.parent.glob(".*.tmp"))
    assert temp_files == []


def test_execute_evaluation_loads_dataset_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"

    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    import structured_output_agent.runner as runner_module

    load_count = 0
    original_load = runner_module.load_dataset

    def spy_load(*args: object, **kwargs: object) -> list[DatasetRecord]:
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_module, "load_dataset", spy_load)

    report = execute_evaluation(config, client, output_file)

    assert load_count == 1
    assert report.metrics.eligible_count == 12
    assert output_file.exists()


def test_execute_evaluation_immune_to_dataset_mutation_or_deletion_after_inference(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"

    config = RunConfig(dataset_path=dataset_file)

    class MutatingClient(StubModelClient):
        def complete(self, prompt: str) -> str:
            res = super().complete(prompt)
            # On the final call, delete the dataset file on disk
            if len(self.calls) == 12:
                dataset_file.unlink()
            return res

    client = MutatingClient()
    report = execute_evaluation(config, client, output_file)

    assert not dataset_file.exists()
    assert output_file.exists()
    assert report.metrics.eligible_count == 12
    assert report.dataset_fingerprint is not None


def test_execute_evaluation_rejects_protected_output_path_before_inference(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)

    class FailingClient(StubModelClient):
        def complete(self, prompt: str) -> str:
            raise AssertionError("Inference must not be invoked on invalid output path")

    client = FailingClient()

    with pytest.raises(ValueError, match="protected test dataset"):
        execute_evaluation(config, client, "evals/data/test.jsonl")

    assert len(client.calls) == 0


def test_execute_evaluation_rejects_output_identical_to_dataset_before_inference(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    initial_content = dataset_file.read_text(encoding="utf-8")
    config = RunConfig(dataset_path=dataset_file)

    class FailingClient(StubModelClient):
        def complete(self, prompt: str) -> str:
            raise AssertionError("Inference must not be invoked on invalid output path")

    client = FailingClient()

    with pytest.raises(ValueError, match="input dataset"):
        execute_evaluation(config, client, dataset_file)

    assert len(client.calls) == 0
    assert dataset_file.read_text(encoding="utf-8") == initial_content


def test_execute_evaluation_rejects_symlink_alias_to_dataset_path(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)
    alias_path = tmp_path / "dataset_symlink.json"

    try:
        alias_path.symlink_to(dataset_file)
    except OSError:
        pytest.skip("Symlink creation is not supported on this platform/environment")

    client = StubModelClient()
    with pytest.raises(ValueError, match="input dataset"):
        execute_evaluation(config, client, alias_path)


@pytest.mark.parametrize(
    "protected_variant",
    [
        "test.jsonl",
        "TEST.JSONL",
        "test.jsonl.",
        "test.jsonl ",
        "test.jsonl::$DATA",
        "TEST~1.JSO",
        "evals/data/test.jsonl",
        "evals/data/TEST~1.JSO",
    ],
)
def test_execute_evaluation_rejects_windows_protected_path_variant_syntactically(
    tmp_path: Path, protected_variant: str
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)

    class FailingClient(StubModelClient):
        def complete(self, prompt: str) -> str:
            raise AssertionError("Inference must not be invoked on invalid output path")

    client = FailingClient()
    with pytest.raises(ValueError, match="protected test dataset"):
        execute_evaluation(config, client, protected_variant)

    assert len(client.calls) == 0


def test_execute_evaluation_preserves_existing_report_on_invalid_output(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)

    existing_report = tmp_path / "prior_report.json"
    existing_report.write_text('{"preserved": true}\n', encoding="utf-8")

    client = StubModelClient()
    # Attempting to overwrite the dataset as output
    with pytest.raises(ValueError, match="input dataset"):
        execute_evaluation(config, client, dataset_file)

    assert existing_report.read_text(encoding="utf-8") == '{"preserved": true}\n'


def test_execute_evaluation_creates_no_parent_dir_or_temp_file_on_invalid_output(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    nested_dir = tmp_path / "never_created_dir"
    target_output = nested_dir / "test.jsonl"

    with pytest.raises(ValueError, match="protected test dataset"):
        execute_evaluation(config, client, target_output)

    assert not nested_dir.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_main_rejects_protected_output_path_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)

    exit_code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            "evals/data/test.jsonl",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "protected test dataset" in captured.err


def test_main_rejects_output_identical_to_dataset_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)

    exit_code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(dataset_file),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "input dataset" in captured.err


def test_execute_evaluation_rejects_existing_directory_as_output(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    dest_dir = tmp_path / "existing_dir"
    dest_dir.mkdir()

    with pytest.raises(ValueError, match="existing directory"):
        execute_evaluation(config, client, dest_dir)


def test_pyproject_scripts_entry_point() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject_path.exists()

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    scripts = data.get("project", {}).get("scripts", {})
    assert "structured-output-agent" in scripts
    entry_point = scripts["structured-output-agent"]
    assert entry_point == "structured_output_agent.cli:main"

    module_name, func_name = entry_point.split(":")
    mod = importlib.import_module(module_name)
    target_func = getattr(mod, func_name)
    assert callable(target_func)


def test_offline_module_execution_subprocess(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"

    # 1. Successful execution -> exit code 0
    res_success = subprocess.run(
        [
            sys.executable,
            "-m",
            "structured_output_agent",
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--client",
            "stub",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_success.returncode == 0
    assert output_file.exists()
    assert "Evaluation completed successfully." in res_success.stdout

    # 2. Operational abort -> exit code 1
    abort_output = tmp_path / "abort_report.json"
    res_abort = subprocess.run(
        [
            sys.executable,
            "-m",
            "structured_output_agent",
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(abort_output),
            "--client",
            "stub-abort",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_abort.returncode == 1
    assert not abort_output.exists()
    assert "Operational failure (exit 1)" in res_abort.stderr

    # 3. Invalid split / configuration error -> exit code 2
    res_err = subprocess.run(
        [
            sys.executable,
            "-m",
            "structured_output_agent",
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--split",
            "test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_err.returncode == 2
    assert (
        "restricted to 'dev'" in res_err.stderr
        or "Evaluation is restricted to the development split" in res_err.stderr
    )


def test_report_provenance_stub_cli(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report_stub.json"

    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--client",
            "stub",
        ]
    )
    assert code == 0
    assert output_file.exists()

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1"
    assert data["client_type"] == "stub"
    assert data["model_id"] is None

    # created_at is valid ISO 8601 with timezone
    dt = datetime.fromisoformat(data["created_at"])
    assert dt.tzinfo is not None
    assert dt.utcoffset() is not None


def test_report_provenance_stub_with_model_flag(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report_stub_model.json"

    # Even if --model is passed to stub client, stub does not invent a model name
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--client",
            "stub",
            "--model",
            "ignored-stub-model",
        ]
    )
    assert code == 0

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["client_type"] == "stub"
    assert data["model_id"] is None


def test_report_provenance_gemini_model_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)

    # Mock the SDK client for offline execution (no network, no credentials)
    mock_sdk_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {
            "category": "billing",
            "priority": "normal",
            "assigned_team": "payments",
            "summary": "Mock Gemini triage",
            "requires_human_review": False,
        }
    )
    mock_sdk_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr("google.genai.Client", MagicMock(return_value=mock_sdk_client))
    monkeypatch.setenv("GEMINI_API_KEY", "offline-mock-key")

    # 1. Default fallback: CLI --client gemini (no --model, no GEMINI_MODEL env)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    out_default = tmp_path / "report_gemini_default.json"
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(out_default),
            "--client",
            "gemini",
        ]
    )
    assert code == 0
    data_default = json.loads(out_default.read_text(encoding="utf-8"))
    assert data_default["client_type"] == "gemini"
    assert data_default["model_id"] == "gemini-3.8-flash"
    dt = datetime.fromisoformat(data_default["created_at"])
    assert dt.tzinfo is not None

    # 2. Environment variable: GEMINI_MODEL overrides default
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    out_env = tmp_path / "report_gemini_env.json"
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(out_env),
            "--client",
            "gemini",
        ]
    )
    assert code == 0
    data_env = json.loads(out_env.read_text(encoding="utf-8"))
    assert data_env["client_type"] == "gemini"
    assert data_env["model_id"] == "gemini-2.5-flash"

    # 3. Explicit CLI argument: --model overrides environment variable
    out_cli = tmp_path / "report_gemini_cli.json"
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(out_cli),
            "--client",
            "gemini",
            "--model",
            "gemini-3.8-flash-custom",
        ]
    )
    assert code == 0
    data_cli = json.loads(out_cli.read_text(encoding="utf-8"))
    assert data_cli["client_type"] == "gemini"
    assert data_cli["model_id"] == "gemini-3.8-flash-custom"


def test_runner_provenance_on_abort(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    config = RunConfig(dataset_path=dataset_file)
    abort_client = StubModelClient(
        canned_responses=[OperationalError("Transport timeout")]
    )

    run = run_evaluation(config, abort_client)
    assert run.status == RunStatus.ABORTED
    assert run.client_type == "stub"
    assert run.model_id is None
    assert run.created_at is not None
    dt = datetime.fromisoformat(run.created_at)
    assert dt.tzinfo is not None


def test_execute_evaluation_with_explicit_created_at(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report_explicit_time.json"
    config = RunConfig(dataset_path=dataset_file)
    client = StubModelClient()

    report = execute_evaluation(
        config, client, output_file, created_at="2026-10-12T16:00:00+00:00"
    )
    assert report.created_at == "2026-10-12T16:00:00+00:00"
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["created_at"] == "2026-10-12T16:00:00+00:00"
    assert data["client_type"] == "stub"
    assert data["model_id"] is None


def test_cli_checkpoint_new_run_creates_checkpoint_and_report(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    checkpoint_file = tmp_path / "checkpoint.json"

    client = StubModelClient()
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--checkpoint-path",
            str(checkpoint_file),
        ],
        client=client,
    )
    assert code == 0
    assert output_file.exists()
    assert checkpoint_file.exists()

    cp = load_checkpoint(checkpoint_file, dataset_path=dataset_file)
    assert cp.eligible_count == 12
    assert len(cp.completed_results) == 12
    assert len(client.calls) == 12


def test_cli_checkpoint_new_run_rejects_existing_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    checkpoint_file = tmp_path / "checkpoint.json"
    checkpoint_file.write_text("{}", encoding="utf-8")

    client = StubModelClient()
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--checkpoint-path",
            str(checkpoint_file),
        ],
        client=client,
    )
    assert code == 2
    assert not output_file.exists()
    assert checkpoint_file.read_text(encoding="utf-8") == "{}"
    assert len(client.calls) == 0
    captured = capsys.readouterr()
    assert "already exists" in captured.err


def test_cli_resume_partial_checkpoint_skips_completed_and_produces_report(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    checkpoint_file = tmp_path / "checkpoint.json"

    canned = [
        json.dumps(
            {
                "category": "billing",
                "priority": "normal",
                "assigned_team": "payments",
                "summary": "Ticket 0",
                "requires_human_review": False,
            }
        ),
        json.dumps(
            {
                "category": "billing",
                "priority": "normal",
                "assigned_team": "payments",
                "summary": "Ticket 1",
                "requires_human_review": False,
            }
        ),
        OperationalError("HTTP 503 Service Unavailable"),
    ]
    abort_client = StubModelClient(canned_responses=canned)
    code1 = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--checkpoint-path",
            str(checkpoint_file),
        ],
        client=abort_client,
    )
    assert code1 == 1
    assert not output_file.exists()
    assert checkpoint_file.exists()

    cp_partial = load_checkpoint(checkpoint_file, dataset_path=dataset_file)
    assert len(cp_partial.completed_results) == 2

    resume_client = StubModelClient()
    code2 = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--resume",
            str(checkpoint_file),
        ],
        client=resume_client,
    )
    assert code2 == 0
    assert output_file.exists()
    assert len(resume_client.calls) == 10

    cp_final = load_checkpoint(checkpoint_file, dataset_path=dataset_file)
    assert len(cp_final.completed_results) == 12

    report_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert report_data["metrics"]["eligible_count"] == 12


def test_cli_both_checkpoint_flags_rejected_by_argparse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    cp1 = tmp_path / "cp1.json"
    cp2 = tmp_path / "cp2.json"

    client = StubModelClient()
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--checkpoint-path",
            str(cp1),
            "--resume",
            str(cp2),
        ],
        client=client,
    )
    assert code == 2
    assert not output_file.exists()
    assert len(client.calls) == 0
    captured = capsys.readouterr()
    assert "not allowed with argument" in captured.err


def test_cli_checkpoint_report_collision_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    target_path = tmp_path / "colliding.json"

    client = StubModelClient()
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(target_path),
            "--checkpoint-path",
            str(target_path),
        ],
        client=client,
    )
    assert code == 2
    assert not target_path.exists()
    assert len(client.calls) == 0
    captured = capsys.readouterr()
    assert "cannot target the final report path" in captured.err


def test_cli_operational_abort_preserves_checkpoint_prefix_no_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    checkpoint_file = tmp_path / "checkpoint.json"

    canned = [
        json.dumps(
            {
                "category": "billing",
                "priority": "normal",
                "assigned_team": "payments",
                "summary": "Ticket 0",
                "requires_human_review": False,
            }
        ),
        OperationalError("HTTP 503 Provider Outage"),
    ]
    abort_client = StubModelClient(canned_responses=canned)
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--checkpoint-path",
            str(checkpoint_file),
        ],
        client=abort_client,
    )
    assert code == 1
    assert not output_file.exists()
    assert checkpoint_file.exists()

    cp = load_checkpoint(checkpoint_file, dataset_path=dataset_file)
    assert len(cp.completed_results) == 1
    assert cp.completed_results[0].ticket_index == 0

    captured = capsys.readouterr()
    assert "Operational failure (exit 1)" in captured.err


def test_cli_neither_checkpoint_flag_preserves_default_behavior(
    tmp_path: Path,
) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"

    client = StubModelClient()
    code = main(
        [
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
        ],
        client=client,
    )
    assert code == 0
    assert output_file.exists()
    assert len(client.calls) == 12
    report_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert report_data["metrics"]["eligible_count"] == 12


def test_cli_subprocess_checkpoint_and_resume(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev_data.jsonl"
    _make_valid_dataset_file(dataset_file)
    output_file = tmp_path / "report.json"
    checkpoint_file = tmp_path / "cp.json"

    res_cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "structured_output_agent",
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file),
            "--client",
            "stub",
            "--checkpoint-path",
            str(checkpoint_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_cp.returncode == 0
    assert output_file.exists()
    assert checkpoint_file.exists()

    output_file2 = tmp_path / "report2.json"
    res_resume = subprocess.run(
        [
            sys.executable,
            "-m",
            "structured_output_agent",
            "--dataset-path",
            str(dataset_file),
            "--output",
            str(output_file2),
            "--client",
            "stub",
            "--resume",
            str(checkpoint_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_resume.returncode == 0
    assert output_file2.exists()
