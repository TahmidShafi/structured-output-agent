import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_serializer, field_validator

from .schemas import DatasetSplit, StrictModel

DEFAULT_DATASET_PATH = Path("evals/data/dev.jsonl")
DEFAULT_EXTRACTOR = "extract_json_object"
DEFAULT_ATTEMPTS_PER_TICKET = 1
DEFAULT_RETRY_BUDGET = 2
PROTECTED_DATASET_NAME = "test.jsonl"


def _is_protected_component(comp: str) -> bool:
    """Check if a path component targets the protected test dataset.

    Applies conservative syntactic rules:
    - Rejects direct case-insensitive matches for 'test.jsonl'.
    - Strips Windows Alternate Data Stream (ADS) specifiers (e.g. 'filename::$DATA').
    - Normalizes Win32 trailing dots and spaces (e.g. 'test.jsonl.').
    - Conservatively rejects standard Windows 8.3 short filename patterns
      (e.g. 'TEST~1.JSO').

    Note: Syntactic pattern matching for 8.3 short names is a conservative heuristic
    and cannot establish filesystem-level identity, which depends on volume-level
    8.3 generation settings.
    """
    if not comp or comp in (".", ".."):
        return False

    # Skip Windows drive letters like 'C:'
    if len(comp) == 2 and comp[1] == ":":
        return False

    # Extract base name before any Windows alternate data stream
    # specifier (e.g. filename::$DATA)
    if ":" in comp:
        comp = comp.split(":")[0]

    # Normalize Win32 trailing dots and spaces
    clean = comp.rstrip(". ")
    if clean.lower() == PROTECTED_DATASET_NAME:
        return True

    # Conservative Windows 8.3 short filename heuristic (e.g. TEST~1.JSO)
    upper = clean.upper()
    if upper.startswith("TEST~") and upper.endswith(".JSO"):
        middle = upper[5:-4]
        if middle.isdigit():
            return True

    return False


class RunConfig(StrictModel):
    """Typed configuration for an evaluation run.

    Enforces development-only evaluation, single-attempt initial execution,
    configurable retry budget for eligible model failures, and deterministic
    JSON serialization/reconstruction.
    """

    dataset_path: Path = DEFAULT_DATASET_PATH
    split: DatasetSplit = DatasetSplit.DEV
    extractor: Literal["extract_json_object"] = DEFAULT_EXTRACTOR
    attempts_per_ticket: Literal[1] = DEFAULT_ATTEMPTS_PER_TICKET
    retry_budget: int = Field(
        default=DEFAULT_RETRY_BUDGET,
        ge=0,
        description=(
            "Configurable retry budget for eligible failures (O1-O3). "
            "Under the agreed project interpretation, retry_budget=2 specifies "
            "that up to 2 retries are allowed following the initial attempt "
            "(up to 3 total attempts per ticket)."
        ),
    )
    failure_log_path: Path | None = None

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: DatasetSplit) -> DatasetSplit:
        if value != DatasetSplit.DEV:
            raise ValueError(
                f"Evaluation is restricted to the development split "
                f"({DatasetSplit.DEV.value!r}); got {value.value!r}. "
                "Test-split evaluation is protected."
            )
        return value

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: Path | str) -> Path:
        raw = str(value).strip().replace("\\", "/")
        if not raw or raw == ".":
            raise ValueError("dataset_path must specify a valid file path.")

        pure = PurePosixPath(raw)
        # 1. Syntactic guard against path strings or aliases referencing test.jsonl.
        # Normalizes trailing dots/spaces, Alternate Data Streams, and 8.3 aliases.
        if any(_is_protected_component(part) for part in pure.parts):
            raise ValueError(
                f"dataset_path cannot target the protected test dataset: "
                f"{raw!r}."
            )

        candidate = Path(raw)

        # 2. Filesystem resolution guard (resolves symlinks and canonical paths).
        # Uses strict=False so legitimate non-existent development dataset paths remain
        # accepted without error.
        # Error policy: If an unexpected error prevents verifying the path (e.g. symlink
        # loop or PermissionError on an existing path), fail closed.
        try:
            resolved = candidate.resolve(strict=False)
            if any(_is_protected_component(part) for part in resolved.parts):
                raise ValueError(
                    f"dataset_path cannot target the protected test dataset: "
                    f"{raw!r}."
                )
        except (RuntimeError, PermissionError) as exc:
            raise ValueError(
                f"Cannot verify path safety for dataset_path {raw!r}: {exc}."
            ) from exc
        except OSError as exc:
            if not isinstance(exc, FileNotFoundError):
                raise ValueError(
                    f"Cannot verify path safety for dataset_path {raw!r}: {exc}."
                ) from exc
            resolved = None

        # 3. File identity check for co-located hardlink aliases.
        # Security boundary: For existing files with multiple hard links (st_nlink > 1),
        # verify that the candidate does not share file identity with a co-located
        # 'test.jsonl' in its directory.
        # Use candidate.stat() rather than candidate.is_file() because on Python 3.14+
        # Path.is_file() swallows OSError internally and returns False, which would
        # silently bypass permission errors and unexpected filesystem failures.
        # Error policy: If the file exists but an unexpected OS error prevents
        # inspecting its metadata (e.g. PermissionError on candidate), fail closed.
        # Residual limitation: Cross-directory hardlinks placed in distant directories
        # without referencing 'test.jsonl' cannot be discovered without exhaustive
        # volume-wide traversal. Such risks are mitigated at the workflow boundary
        # (split enforcement, fingerprint verification, and evaluation freeze).
        try:
            cand_stat = candidate.stat()
        except FileNotFoundError:
            cand_stat = None
        except PermissionError as exc:
            raise ValueError(
                f"Cannot verify path safety for dataset_path {raw!r}: {exc}."
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Cannot verify path safety for dataset_path {raw!r}: {exc}."
            ) from exc

        if cand_stat is not None and stat.S_ISREG(cand_stat.st_mode):
            if cand_stat.st_nlink > 1:
                candidate_dirs: set[Path] = {candidate.parent}
                if resolved is not None:
                    candidate_dirs.add(resolved.parent)

                for parent_dir in candidate_dirs:
                    sibling_target = parent_dir / PROTECTED_DATASET_NAME
                    try:
                        sib_stat = sibling_target.stat()
                    except FileNotFoundError:
                        sib_stat = None
                    except PermissionError as exc:
                        raise ValueError(
                            f"Cannot verify path safety for dataset_path "
                            f"{raw!r}: {exc}."
                        ) from exc
                    except OSError as exc:
                        raise ValueError(
                            f"Cannot verify path safety for dataset_path "
                            f"{raw!r}: {exc}."
                        ) from exc

                    if sib_stat is not None and stat.S_ISREG(sib_stat.st_mode):
                        try:
                            if os.path.samefile(candidate, sibling_target):
                                raise ValueError(
                                    f"dataset_path cannot target the protected "
                                    f"test dataset: {raw!r}."
                                )
                        except FileNotFoundError:
                            pass
                        except PermissionError as exc:
                            raise ValueError(
                                f"Cannot verify path safety for dataset_path "
                                f"{raw!r}: {exc}."
                            ) from exc
                        except OSError as exc:
                            raise ValueError(
                                f"Cannot verify path safety for dataset_path "
                                f"{raw!r}: {exc}."
                            ) from exc

        return Path(raw)

    @field_validator("failure_log_path")
    @classmethod
    def validate_failure_log_path(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        raw = str(value).strip().replace("\\", "/")
        if not raw:
            return None
        pure = PurePosixPath(raw)
        if any(_is_protected_component(part) for part in pure.parts):
            raise ValueError(
                f"failure_log_path cannot target the protected test dataset: {raw!r}."
            )
        return Path(raw)

    @field_serializer("dataset_path", when_used="json")
    def serialize_dataset_path(self, value: Path) -> str:
        return value.as_posix()

    @field_serializer("failure_log_path", when_used="json")
    def serialize_failure_log_path(self, value: Path | None) -> str | None:
        return value.as_posix() if value is not None else None

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize configuration to a deterministic JSON string."""
        data = self.model_dump(mode="json")
        return json.dumps(data, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_data: str | bytes) -> "RunConfig":
        """Reconstruct and validate configuration from a JSON string or bytes."""
        return cls.model_validate_json(json_data)


def _verify_ancestor_safety(
    target: Path,
    raw_display: str,
    label: str,
    *,
    require_direct_parent: bool = False,
) -> os.stat_result:
    """Verify that target's parent or nearest existing ancestor is safe.

    Returns the stat_result of the nearest existing ancestor directory.
    Fails closed on permission errors, unexpected OS errors, or if an
    existing ancestor in the chain is not a directory.
    """
    curr = target.parent
    direct_parent = True
    while True:
        try:
            st = curr.stat()
            break
        except FileNotFoundError:
            if require_direct_parent and direct_parent:
                raise ValueError(
                    f"Parent directory of {label} does not exist: "
                    f"{raw_display!r}."
                )
            parent_curr = curr.parent
            if parent_curr == curr:
                raise ValueError(
                    f"Cannot verify path safety for {label} "
                    f"{raw_display!r}: no existing directory ancestor found."
                )
            curr = parent_curr
            direct_parent = False
        except PermissionError as exc:
            raise ValueError(
                f"Cannot verify path safety for {label} {raw_display!r}: "
                f"{exc}."
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Cannot verify path safety for {label} {raw_display!r}: "
                f"{exc}."
            ) from exc

    if not stat.S_ISDIR(st.st_mode):
        if direct_parent:
            raise ValueError(
                f"Parent path of {label} is not a directory: {raw_display!r}."
            )
        raise ValueError(
            f"Ancestor path '{curr}' of {label} is not a directory: "
            f"{raw_display!r}."
        )

    return st


def validate_output_path(
    output_path: Path | str,
    dataset_path: Path | str,
    *,
    require_parent_exists: bool = False,
) -> Path:
    """Validate report output destination path safety before evaluation.

    Guarantees:
    1. Rejects output paths targeting the protected test dataset ('test.jsonl')
       syntactically or via resolved paths (including case variations, Win32
       trailing dots/spaces, Alternate Data Streams, and 8.3 aliases).
    2. Rejects output paths that match or resolve to the input dataset path,
       preventing source dataset destruction.
    3. Detects if an existing output file shares filesystem identity with the
       input dataset (via os.path.samefile).
    4. Validates parent directory and ancestor safety:
       - Nonexistent files in existing directories are verified to have valid,
         accessible parent directories.
       - Nonexistent parent directories are verified via their nearest existing
         ancestor directory, or rejected if require_parent_exists is True.
       - Parent or ancestor paths that are existing files (not directories) or
         that raise PermissionError / unexpected OSError fail closed.
    5. Fails closed on permission errors, resolution failures, or directory targets.

    Residual limitation:
    Does not probe 'evals/data/test.jsonl' or its parent for hard links, as any
    inspection of the protected test dataset is strictly prohibited. Cross-directory
    hard links to the protected dataset that do not reference 'test.jsonl' cannot
    be detected without probing the protected target.
    """
    raw = str(output_path).strip().replace("\\", "/")
    if not raw or raw == ".":
        raise ValueError("output_path must specify a valid file path.")

    pure = PurePosixPath(raw)

    # 1. Syntactic guard against path strings or aliases referencing test.jsonl
    if any(_is_protected_component(part) for part in pure.parts):
        raise ValueError(
            f"output_path cannot target the protected test dataset: {raw!r}."
        )

    candidate = Path(raw)

    # 2. Reject existing directory destinations and verify destination safety.
    # Use candidate.stat() rather than candidate.is_dir() because on Python 3.14+
    # Path.is_dir() swallows OSError internally and returns False, which would
    # silently bypass permission errors and unexpected filesystem failures.
    try:
        cand_stat = candidate.stat()
        if stat.S_ISDIR(cand_stat.st_mode):
            raise ValueError(
                f"output_path cannot be an existing directory: {raw!r}."
            )
    except FileNotFoundError:
        cand_stat = None
        # Nonexistent output file: verify parent directory or ancestor safety.
        _verify_ancestor_safety(
            candidate,
            raw,
            "output_path",
            require_direct_parent=require_parent_exists,
        )
    except PermissionError as exc:
        raise ValueError(
            f"Cannot verify path safety for output_path {raw!r}: {exc}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Cannot verify path safety for output_path {raw!r}: {exc}."
        ) from exc

    # 3. Filesystem resolution guard against protected test dataset
    try:
        resolved = candidate.resolve(strict=False)
        if any(_is_protected_component(part) for part in resolved.parts):
            raise ValueError(
                f"output_path cannot target the protected test dataset: {raw!r}."
            )
    except (RuntimeError, PermissionError) as exc:
        raise ValueError(
            f"Cannot verify path safety for output_path {raw!r}: {exc}."
        ) from exc
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError):
            raise ValueError(
                f"Cannot verify path safety for output_path {raw!r}: {exc}."
            ) from exc
        resolved = None

    # 4. Guard against overwriting the input dataset
    ds_raw = str(dataset_path).strip().replace("\\", "/")
    ds_pure = PurePosixPath(ds_raw)

    # Syntactic check against input dataset path
    if pure == ds_pure:
        raise ValueError(
            f"output_path cannot target the input dataset: {raw!r}."
        )

    # Syntactic guard preventing dataset_path from referencing protected dataset
    if any(_is_protected_component(part) for part in ds_pure.parts):
        raise ValueError(
            f"dataset_path cannot target the protected test dataset: {ds_raw!r}."
        )

    ds_candidate = Path(ds_raw)
    try:
        ds_resolved = ds_candidate.resolve(strict=False)
        if resolved is not None and resolved == ds_resolved:
            raise ValueError(
                f"output_path cannot target the input dataset: {raw!r}."
            )
    except (RuntimeError, PermissionError) as exc:
        raise ValueError(
            "Cannot verify path safety between output_path and dataset_path: "
            f"{exc}."
        ) from exc
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError):
            raise ValueError(
                "Cannot verify path safety between output_path and dataset_path: "
                f"{exc}."
            ) from exc

    # Identity check if both files exist on disk.
    # Stat ds_candidate directly; avoid Path.is_file() because it swallows
    # OSError on Python 3.14+.
    try:
        ds_stat = ds_candidate.stat()
    except FileNotFoundError:
        ds_stat = None
        # Nonexistent dataset path: verify parent/ancestor directory safety
        # without breaking legitimate nonexistent development datasets.
        _verify_ancestor_safety(
            ds_candidate,
            ds_raw,
            "dataset_path",
            require_direct_parent=False,
        )
    except PermissionError as exc:
        raise ValueError(
            f"Cannot verify path safety for dataset_path {ds_raw!r}: {exc}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Cannot verify path safety for dataset_path {ds_raw!r}: {exc}."
        ) from exc

    try:
        if (
            cand_stat is not None
            and ds_stat is not None
            and os.path.samefile(candidate, ds_candidate)
        ):
            raise ValueError(
                f"output_path cannot target the input dataset: {raw!r}."
            )
    except PermissionError as exc:
        raise ValueError(
            f"Cannot verify path safety for output_path {raw!r}: {exc}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Cannot verify path safety for output_path {raw!r}: {exc}."
        ) from exc

    return candidate

