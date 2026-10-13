import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from structured_output_agent.run_config import (
    DEFAULT_ATTEMPTS_PER_TICKET,
    DEFAULT_DATASET_PATH,
    DEFAULT_EXTRACTOR,
    RunConfig,
    validate_output_path,
)
from structured_output_agent.schemas import DatasetSplit


def test_default_run_config() -> None:
    config = RunConfig()

    assert config.dataset_path == DEFAULT_DATASET_PATH
    assert config.split == DatasetSplit.DEV
    assert config.extractor == DEFAULT_EXTRACTOR
    assert config.attempts_per_ticket == DEFAULT_ATTEMPTS_PER_TICKET


def test_explicit_valid_run_config() -> None:
    config = RunConfig(
        dataset_path=Path("evals/data/dev.jsonl"),
        split=DatasetSplit.DEV,
        extractor="extract_json_object",
        attempts_per_ticket=1,
    )

    assert config.dataset_path == Path("evals/data/dev.jsonl")
    assert config.split == DatasetSplit.DEV
    assert config.extractor == "extract_json_object"
    assert config.attempts_per_ticket == 1


def test_string_inputs_coerced_properly() -> None:
    config = RunConfig(
        dataset_path="evals/data/dev.jsonl",
        split="dev",
    )

    assert isinstance(config.dataset_path, Path)
    assert config.dataset_path == Path("evals/data/dev.jsonl")
    assert isinstance(config.split, DatasetSplit)
    assert config.split == DatasetSplit.DEV


def test_custom_valid_dataset_path() -> None:
    custom_path = Path("custom/path/dev_data.jsonl")
    config = RunConfig(dataset_path=custom_path)

    assert config.dataset_path == custom_path


def test_accepts_similar_legitimate_dataset_names() -> None:
    for name in (
        "contest.jsonl",
        "my_test.jsonl",
        "evals/data/test_run.jsonl",
        "evals/data/speed_test.jsonl",
    ):
        config = RunConfig(dataset_path=name)
        assert config.dataset_path == Path(name)


def test_rejects_test_split_enum() -> None:
    with pytest.raises(ValidationError, match="Test-split evaluation is protected"):
        RunConfig(split=DatasetSplit.TEST)


def test_rejects_test_split_string() -> None:
    with pytest.raises(ValidationError, match="Test-split evaluation is protected"):
        RunConfig(split="test")


def test_rejects_invalid_split_string() -> None:
    with pytest.raises(ValidationError):
        RunConfig(split="train")


def test_rejects_test_dataset_path_edge_cases() -> None:
    test_path_edge_cases = (
        "test.jsonl",
        Path("test.jsonl"),
        "evals/data/test.jsonl",
        r"evals\data\test.jsonl",
        "evals/data/TEST.JSONL",
        Path("Test.Jsonl"),
        "TEST.JSONL",
        "./test.jsonl",
        r".\test.jsonl",
        "../test.jsonl",
        "evals/data/sub/../test.jsonl",
        "evals/data/test.jsonl/",
    )

    for edge_case in test_path_edge_cases:
        with pytest.raises(ValidationError, match="protected test dataset"):
            RunConfig(dataset_path=edge_case)


def test_rejects_empty_or_whitespace_dataset_path() -> None:
    invalid_paths = ("", "   ", Path(""), Path("."))

    for path in invalid_paths:
        with pytest.raises(
            ValidationError,
            match="dataset_path must specify a valid file path",
        ):
            RunConfig(dataset_path=path)


def test_rejects_unapproved_extractor() -> None:
    with pytest.raises(ValidationError):
        RunConfig(extractor="extract_json_candidate_text")

    with pytest.raises(ValidationError):
        RunConfig(extractor="regex_extractor")


def test_rejects_invalid_attempts_per_ticket() -> None:
    with pytest.raises(ValidationError):
        RunConfig(attempts_per_ticket=0)

    with pytest.raises(ValidationError):
        RunConfig(attempts_per_ticket=2)

    with pytest.raises(ValidationError):
        RunConfig(attempts_per_ticket=-1)


def test_rejects_unresolved_settings_extra_fields() -> None:
    extra_field_cases = (
        {"model_id": "gemini-1.5-pro"},
        {"temperature": 0.0},
        {"prompt_template": "triage_v1"},
        {"timeout": 30},
        {"policy_version": "1.3"},
    )

    for extra_payload in extra_field_cases:
        with pytest.raises(ValidationError):
            RunConfig.model_validate(extra_payload)


def test_deterministic_json_serialization() -> None:
    config = RunConfig()

    json_str_1 = config.to_json()
    json_str_2 = config.to_json()

    assert json_str_1 == json_str_2

    parsed = json.loads(json_str_1)
    keys = list(parsed.keys())
    assert keys == sorted(keys)

    # Cross-platform posix path representation
    assert parsed["dataset_path"] == "evals/data/dev.jsonl"
    assert parsed["split"] == "dev"
    assert parsed["extractor"] == "extract_json_object"
    assert parsed["attempts_per_ticket"] == 1


def test_json_round_trip_reconstruction() -> None:
    original = RunConfig(dataset_path=Path("evals/data/dev.jsonl"))
    serialized = original.to_json()

    reconstructed = RunConfig.from_json(serialized)

    assert reconstructed == original
    assert reconstructed.dataset_path == original.dataset_path
    assert reconstructed.split == original.split
    assert reconstructed.extractor == original.extractor
    assert reconstructed.attempts_per_ticket == original.attempts_per_ticket


def test_pydantic_model_validate_json_round_trip() -> None:
    original = RunConfig()
    serialized = original.to_json()

    reconstructed = RunConfig.model_validate_json(serialized)

    assert reconstructed == original


def test_reconstruction_rejects_malformed_json() -> None:
    malformed_inputs = (
        "{not valid json",
        "",
        "   ",
        "{",
        '{"dataset_path": }',
        "[]",
        "[1, 2, 3]",
        "null",
        "123",
        '"just a string"',
        "true",
    )

    for bad_input in malformed_inputs:
        with pytest.raises(ValidationError):
            RunConfig.from_json(bad_input)


def test_reconstruction_rejects_tampered_json() -> None:
    valid_data = json.loads(RunConfig().to_json())

    # Tamper split
    tampered_split = dict(valid_data, split="test")
    with pytest.raises(ValidationError, match="Test-split evaluation is protected"):
        RunConfig.from_json(json.dumps(tampered_split))

    # Tamper attempts
    tampered_attempts = dict(valid_data, attempts_per_ticket=2)
    with pytest.raises(ValidationError):
        RunConfig.from_json(json.dumps(tampered_attempts))

    # Tamper extractor
    tampered_extractor = dict(valid_data, extractor="other_extractor")
    with pytest.raises(ValidationError):
        RunConfig.from_json(json.dumps(tampered_extractor))

    # Tamper path
    tampered_path = dict(valid_data, dataset_path="evals/data/test.jsonl")
    with pytest.raises(ValidationError, match="protected test dataset"):
        RunConfig.from_json(json.dumps(tampered_path))

    # Tamper path with casing
    tampered_case = dict(valid_data, dataset_path="evals/data/TEST.JSONL")
    with pytest.raises(ValidationError, match="protected test dataset"):
        RunConfig.from_json(json.dumps(tampered_case))

    # Extra unresolved field
    tampered_extra = dict(valid_data, temperature=0.7)
    with pytest.raises(ValidationError):
        RunConfig.from_json(json.dumps(tampered_extra))


def test_is_protected_component_isolated() -> None:
    from structured_output_agent.run_config import _is_protected_component

    # Direct and casing
    assert _is_protected_component("test.jsonl") is True
    assert _is_protected_component("Test.Jsonl") is True
    assert _is_protected_component("TEST.JSONL") is True

    # Windows trailing dot and space normalization
    assert _is_protected_component("test.jsonl.") is True
    assert _is_protected_component("test.jsonl ") is True
    assert _is_protected_component("test.jsonl. . .") is True

    # Windows Alternate Data Stream (ADS)
    assert _is_protected_component("test.jsonl::$DATA") is True
    assert _is_protected_component("test.jsonl:stream") is True
    assert _is_protected_component("test.jsonl:$DATA") is True

    # Windows 8.3 short filename alias
    assert _is_protected_component("TEST~1.JSO") is True
    assert _is_protected_component("TEST~99.JSO") is True

    # Legitimate non-protected components
    assert _is_protected_component("dev.jsonl") is False
    assert _is_protected_component("contest.jsonl") is False
    assert _is_protected_component("my_test.jsonl") is False
    assert _is_protected_component("test_run.jsonl") is False
    assert _is_protected_component("speed_test.jsonl") is False
    assert _is_protected_component("dev.jsonl:stream") is False
    assert _is_protected_component("C:") is False
    assert _is_protected_component(".") is False
    assert _is_protected_component("..") is False
    assert _is_protected_component("") is False


def test_rejects_windows_path_aliases_and_stream_syntax() -> None:
    windows_aliases = (
        "test.jsonl.",
        "test.jsonl ",
        "test.jsonl. . .",
        "evals/data/test.jsonl.",
        r"evals\data\test.jsonl.",
        "evals/data/test.jsonl ",
        "evals/data/test.jsonl::$DATA",
        "evals/data/test.jsonl:stream",
        "evals/data/test.jsonl:$DATA",
        "test.jsonl::$DATA",
        "test.jsonl:stream",
        "TEST~1.JSO",
        "evals/data/TEST~1.JSO",
        r"evals\data\TEST~1.JSO",
        "TEST~2.JSO",
    )
    for alias in windows_aliases:
        with pytest.raises(ValidationError, match="protected test dataset"):
            RunConfig(dataset_path=alias)


def test_rejects_symlink_alias_to_test_dataset(tmp_path: Path) -> None:
    dummy_test = tmp_path / "test.jsonl"
    dummy_test.write_text('{"split": "test"}\n', encoding="utf-8")
    alias_path = tmp_path / "dev_alias.jsonl"

    try:
        alias_path.symlink_to(dummy_test)
    except OSError:
        pytest.skip("Symlink creation is not supported on this platform/environment")

    with pytest.raises(ValidationError, match="protected test dataset"):
        RunConfig(dataset_path=alias_path)


def test_rejects_resolved_symlink_path_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_path = tmp_path / "custom_alias.jsonl"
    # Verify resolve fallback rejection even when OS symlink creation is unavailable
    monkeypatch.setattr(
        Path, "resolve", lambda self, *args, **kwargs: tmp_path / "test.jsonl"
    )
    with pytest.raises(ValidationError, match="protected test dataset"):
        RunConfig(dataset_path=fake_path)


def test_rejects_hardlink_alias_to_test_dataset(tmp_path: Path) -> None:
    dummy_test = tmp_path / "test.jsonl"
    dummy_test.write_text('{"split": "test"}\n', encoding="utf-8")
    alias_path = tmp_path / "dev_alias.jsonl"

    try:
        os.link(dummy_test, alias_path)
    except OSError:
        pytest.skip("Hardlink creation is not supported on this platform/filesystem")

    with pytest.raises(ValidationError, match="protected test dataset"):
        RunConfig(dataset_path=alias_path)


def test_cross_directory_hardlink_demonstrates_boundary_limitation(
    tmp_path: Path,
) -> None:
    dummy_test = tmp_path / "test.jsonl"
    dummy_test.write_text('{"split": "test"}\n', encoding="utf-8")
    sub_dir = tmp_path / "subdir"
    sub_dir.mkdir()
    alias_path = sub_dir / "dev_alias.jsonl"

    try:
        os.link(dummy_test, alias_path)
    except OSError:
        pytest.skip("Hardlink creation is not supported on this platform/filesystem")

    # Boundary limitation: Because subdir does not contain test.jsonl and broad
    # volume scanning is intentionally excluded, RunConfig cannot detect the
    # cross-directory link.
    config = RunConfig(dataset_path=alias_path)
    assert config.dataset_path == alias_path


def test_error_policy_fails_closed_on_resolution_symlink_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_path = tmp_path / "cyclic_alias.jsonl"

    def mock_resolve_loop(self: Path, strict: bool = False) -> Path:
        raise RuntimeError("Symlink loop encountered during resolution")

    monkeypatch.setattr(Path, "resolve", mock_resolve_loop)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=fake_path)


def test_error_policy_fails_closed_on_resolution_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_path = tmp_path / "restricted_alias.jsonl"

    def mock_resolve_perm(self: Path, strict: bool = False) -> Path:
        raise PermissionError("Access is denied")

    monkeypatch.setattr(Path, "resolve", mock_resolve_perm)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=fake_path)


def test_error_policy_fails_closed_on_stat_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_file = tmp_path / "existing_dev.jsonl"
    existing_file.write_text('{"split": "dev"}\n', encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat_perm(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == existing_file:
            raise PermissionError("Access is denied reading metadata")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat_perm)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=existing_file)


def test_error_policy_fails_closed_on_resolution_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_path = tmp_path / "custom_alias.jsonl"

    def mock_resolve_err(self: Path, strict: bool = False) -> Path:
        raise OSError("Network drive connection dropped")

    monkeypatch.setattr(Path, "resolve", mock_resolve_err)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=fake_path)


def test_error_policy_fails_closed_on_stat_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_file = tmp_path / "existing_dev.jsonl"
    existing_file.write_text('{"split": "dev"}\n', encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat_err(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == existing_file:
            raise OSError("Hardware device I/O failure")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat_err)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=existing_file)


def test_error_policy_fails_closed_on_sibling_stat_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_file = tmp_path / "existing_dev.jsonl"
    dev_file.write_text('{"split": "dev"}\n', encoding="utf-8")
    dummy_sibling = tmp_path / "test.jsonl"

    can_hardlink = True
    try:
        os.link(dev_file, tmp_path / "extra_link.jsonl")
    except OSError:
        can_hardlink = False

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == dummy_sibling:
            raise PermissionError("Access denied reading sibling metadata")
        if not can_hardlink and self == dev_file:
            real_st = orig_stat(self, *args, **kwargs)

            class _MockStat:
                def __getattr__(self, name: str) -> object:
                    return getattr(real_st, name)

                @property
                def st_nlink(self) -> int:
                    return 2

            return _MockStat()  # type: ignore[return-value]
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=dev_file)


def test_error_policy_fails_closed_on_sibling_stat_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_file = tmp_path / "existing_dev.jsonl"
    dev_file.write_text('{"split": "dev"}\n', encoding="utf-8")
    dummy_sibling = tmp_path / "test.jsonl"

    can_hardlink = True
    try:
        os.link(dev_file, tmp_path / "extra_link.jsonl")
    except OSError:
        can_hardlink = False

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == dummy_sibling:
            raise OSError("I/O failure reading sibling metadata")
        if not can_hardlink and self == dev_file:
            real_st = orig_stat(self, *args, **kwargs)

            class _MockStat:
                def __getattr__(self, name: str) -> object:
                    return getattr(real_st, name)

                @property
                def st_nlink(self) -> int:
                    return 2

            return _MockStat()  # type: ignore[return-value]
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=dev_file)


def test_error_policy_fails_closed_on_samefile_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_file = tmp_path / "existing_dev.jsonl"
    dev_file.write_text('{"split": "dev"}\n', encoding="utf-8")
    dummy_sibling = tmp_path / "test.jsonl"
    dummy_sibling.write_text('{"split": "test"}\n', encoding="utf-8")

    # Ensure st_nlink > 1 so the co-located samefile check is invoked
    try:
        os.link(dev_file, tmp_path / "extra_link.jsonl")
    except OSError:
        real_stat = dev_file.stat()

        class _MockStat:
            def __getattr__(self, name: str) -> object:
                return getattr(real_stat, name)

            @property
            def st_nlink(self) -> int:
                return 2

        monkeypatch.setattr(Path, "stat", lambda self: _MockStat())

    def mock_samefile_perm(
        path1: os.PathLike[str] | str, path2: os.PathLike[str] | str
    ) -> bool:
        raise PermissionError("Access is denied comparing file identity")

    monkeypatch.setattr(os.path, "samefile", mock_samefile_perm)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=dev_file)


def test_error_policy_fails_closed_on_samefile_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_file = tmp_path / "existing_dev.jsonl"
    dev_file.write_text('{"split": "dev"}\n', encoding="utf-8")
    dummy_sibling = tmp_path / "test.jsonl"
    dummy_sibling.write_text('{"split": "test"}\n', encoding="utf-8")

    # Ensure st_nlink > 1 so the co-located samefile check is invoked
    try:
        os.link(dev_file, tmp_path / "extra_link.jsonl")
    except OSError:
        real_stat = dev_file.stat()

        class _MockStat:
            def __getattr__(self, name: str) -> object:
                return getattr(real_stat, name)

            @property
            def st_nlink(self) -> int:
                return 2

        monkeypatch.setattr(Path, "stat", lambda self: _MockStat())

    def mock_samefile_err(
        path1: os.PathLike[str] | str, path2: os.PathLike[str] | str
    ) -> bool:
        raise OSError("I/O error inspecting filesystem identity")

    monkeypatch.setattr(os.path, "samefile", mock_samefile_err)
    with pytest.raises(ValidationError, match="Cannot verify path safety"):
        RunConfig(dataset_path=dev_file)


def test_hardlink_with_missing_sibling_target_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev_file = tmp_path / "existing_dev.jsonl"
    dev_file.write_text('{"split": "dev"}\n', encoding="utf-8")

    can_hardlink = True
    try:
        os.link(dev_file, tmp_path / "extra_link.jsonl")
    except OSError:
        can_hardlink = False

    if not can_hardlink:
        orig_stat = Path.stat

        def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
            if self == dev_file:
                real_st = orig_stat(self, *args, **kwargs)

                class _MockStat:
                    def __getattr__(self, name: str) -> object:
                        return getattr(real_st, name)

                    @property
                    def st_nlink(self) -> int:
                        return 2

                return _MockStat()  # type: ignore[return-value]
            return orig_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", mock_stat)

    # Note: tmp_path does NOT contain test.jsonl
    config = RunConfig(dataset_path=dev_file)
    assert config.dataset_path == dev_file


def test_accepts_harmless_nonexistent_dataset_path() -> None:
    nonexistent = Path("custom/path/future_dev_data.jsonl")
    config = RunConfig(dataset_path=nonexistent)
    assert config.dataset_path == nonexistent


def test_accepts_valid_dev_dataset_path_without_accessing_protected(
    tmp_path: Path,
) -> None:
    # Verify existing default dev dataset
    config_default = RunConfig()
    assert config_default.dataset_path == Path("evals/data/dev.jsonl")

    # Verify custom temporary dev dataset file in tmp_path
    dev_file = tmp_path / "custom_dev.jsonl"
    dev_file.write_text('{"split": "dev"}\n', encoding="utf-8")
    config_custom = RunConfig(dataset_path=dev_file)
    assert config_custom.dataset_path == dev_file


def test_validate_output_path_accepts_valid_path(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dev.jsonl"
    output_file = tmp_path / "report.json"
    result = validate_output_path(output_file, dataset_file)
    assert isinstance(result, Path)
    assert result == output_file


def test_validate_output_path_fails_closed_on_resolution_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    dataset_file = tmp_path / "dev.jsonl"

    def mock_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        raise PermissionError("Access denied resolving output path")

    monkeypatch.setattr(Path, "resolve", mock_resolve)
    with pytest.raises(ValueError, match="Cannot verify path safety for output_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_resolution_symlink_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    dataset_file = tmp_path / "dev.jsonl"

    def mock_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        raise RuntimeError("Symlink loop encountered")

    monkeypatch.setattr(Path, "resolve", mock_resolve)
    with pytest.raises(ValueError, match="Cannot verify path safety for output_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_samefile_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    output_file.write_text("{}", encoding="utf-8")
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    def mock_samefile_err(
        path1: os.PathLike[str] | str, path2: os.PathLike[str] | str
    ) -> bool:
        raise PermissionError("Access denied inspecting identity")

    monkeypatch.setattr(os.path, "samefile", mock_samefile_err)
    with pytest.raises(ValueError, match="Cannot verify path safety for output_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_candidate_stat_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == output_file:
            raise PermissionError("Access denied statting output file")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValueError, match="Cannot verify path safety for output_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_candidate_stat_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == output_file:
            raise OSError("I/O error on candidate")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValueError, match="Cannot verify path safety for output_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_directory_stat_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "subdir"
    output_dir.mkdir()
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == output_dir:
            raise PermissionError("Access denied statting directory")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValueError, match="Cannot verify path safety for output_path"):
        validate_output_path(output_dir, dataset_file)


def test_validate_output_path_fails_closed_on_dataset_stat_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == dataset_file:
            raise PermissionError("Access denied statting dataset file")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValueError, match="Cannot verify path safety for dataset_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_dataset_stat_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_file = tmp_path / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == dataset_file:
            raise OSError("I/O error on dataset")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(ValueError, match="Cannot verify path safety for dataset_path"):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_preserves_nonexistent_dataset_target(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "report.json"
    output_file.write_text("{}", encoding="utf-8")
    nonexistent_dataset = tmp_path / "nonexistent_dev.jsonl"
    result = validate_output_path(output_file, nonexistent_dataset)
    assert result == output_file


def test_validate_output_path_preserves_nonexistent_output_path(
    tmp_path: Path,
) -> None:
    nonexistent_output = tmp_path / "new_report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")
    result = validate_output_path(nonexistent_output, dataset_file)
    assert result == nonexistent_output


def test_validate_output_path_rejects_protected_dataset_path_syntactically(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "report.json"
    with pytest.raises(
        ValueError, match="dataset_path cannot target the protected test dataset"
    ):
        validate_output_path(output_file, "evals/data/test.jsonl")


def test_validate_output_path_accepts_new_output_file_in_existing_directory(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "new_report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")
    result = validate_output_path(out_file, dataset_file)
    assert result == out_file


def test_validate_output_path_with_missing_parent_under_policies(
    tmp_path: Path,
) -> None:
    missing_parent_output = tmp_path / "missing_dir" / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    # Under strict require_parent_exists policy: rejected
    with pytest.raises(
        ValueError, match="Parent directory of output_path does not exist"
    ):
        validate_output_path(
            missing_parent_output, dataset_file, require_parent_exists=True
        )

    # Under default ancestor-verified policy: accepted (ancestor tmp_path is safe)
    result = validate_output_path(
        missing_parent_output, dataset_file, require_parent_exists=False
    )
    assert result == missing_parent_output


def test_validate_output_path_rejects_output_parent_that_is_a_file(
    tmp_path: Path,
) -> None:
    existing_file = tmp_path / "file.txt"
    existing_file.write_text("hello", encoding="utf-8")
    invalid_output = existing_file / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError, match="Parent path of output_path is not a directory"
    ):
        validate_output_path(invalid_output, dataset_file)


def test_validate_output_path_rejects_output_ancestor_that_is_a_file(
    tmp_path: Path,
) -> None:
    existing_file = tmp_path / "file.txt"
    existing_file.write_text("hello", encoding="utf-8")
    invalid_output = existing_file / "sub" / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError, match="Ancestor path '.*' of output_path is not a directory"
    ):
        validate_output_path(invalid_output, dataset_file)


def test_validate_output_path_fails_closed_on_parent_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_dir = tmp_path / "secure_dir"
    parent_dir.mkdir()
    output_file = parent_dir / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == parent_dir:
            raise PermissionError("Access denied on parent directory")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(
        ValueError, match="Cannot verify path safety for output_path"
    ):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_fails_closed_on_parent_unexpected_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_dir = tmp_path / "broken_dir"
    parent_dir.mkdir()
    output_file = parent_dir / "report.json"
    dataset_file = tmp_path / "dev.jsonl"
    dataset_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == parent_dir:
            raise OSError("I/O error reading parent directory")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(
        ValueError, match="Cannot verify path safety for output_path"
    ):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_preserves_missing_dataset_file_in_existing_directory(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "report.json"
    output_file.write_text("{}", encoding="utf-8")
    nonexistent_dataset = tmp_path / "nonexistent_dev.jsonl"
    result = validate_output_path(output_file, nonexistent_dataset)
    assert result == output_file


def test_validate_output_path_handles_missing_dataset_parent_consistently(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "report.json"
    output_file.write_text("{}", encoding="utf-8")
    nested_nonexistent_dataset = tmp_path / "nonexistent_sub" / "dev.jsonl"
    result = validate_output_path(output_file, nested_nonexistent_dataset)
    assert result == output_file


def test_validate_output_path_fails_closed_on_dataset_parent_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_dir = tmp_path / "secure_ds_dir"
    parent_dir.mkdir()
    dataset_file = parent_dir / "dev.jsonl"
    output_file = tmp_path / "report.json"
    output_file.write_text("{}", encoding="utf-8")

    orig_stat = Path.stat

    def mock_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == parent_dir:
            raise PermissionError("Access denied on dataset parent directory")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", mock_stat)
    with pytest.raises(
        ValueError, match="Cannot verify path safety for dataset_path"
    ):
        validate_output_path(output_file, dataset_file)


def test_validate_output_path_rejects_dataset_parent_that_is_a_file(
    tmp_path: Path,
) -> None:
    existing_file = tmp_path / "file.txt"
    existing_file.write_text("hello", encoding="utf-8")
    invalid_dataset = existing_file / "dev.jsonl"
    output_file = tmp_path / "report.json"
    output_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError, match="Parent path of dataset_path is not a directory"
    ):
        validate_output_path(output_file, invalid_dataset)




