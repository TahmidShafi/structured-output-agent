"""Safe checkpoint persistence, loading, and prompt fingerprinting.

Provides:
- compute_prompt_fingerprint: deterministic SHA-256 fingerprint for evaluation prompts.
- validate_checkpoint_path: path safety validation guarding datasets,
  reports, and protected files.
- save_checkpoint: atomic checkpoint persistence with rollback cleanup
  and corruption protection.
- load_checkpoint: strictly read-only checkpoint loader enforcing schema
  and path validity.
"""

import hashlib
import os
import stat
import uuid
from pathlib import Path, PurePosixPath

from .checkpoint import EvaluationCheckpoint
from .run_config import (
    _is_protected_component,
    _verify_ancestor_safety,
    validate_output_path,
)
from .runner import (
    DEFAULT_PROMPT_TEMPLATE,
    PromptBuilder,
    default_prompt_builder,
)


def compute_prompt_fingerprint(
    prompt: PromptBuilder | str | object | None = None,
) -> str:
    """Compute a deterministic 64-character lowercase SHA-256 hex digest for a prompt.

    Supported representations:
    1. None or default_prompt_builder: produces the canonical fingerprint of
       DEFAULT_PROMPT_TEMPLATE.
    2. str: hashes the explicit template/prompt configuration string directly.
    3. Callable or object explicitly declaring a non-empty string 'prompt_template'
       or 'template' attribute: hashes that declared template string.

    Rejection:
    - Rejects empty or whitespace-only template strings.
    - Rejects arbitrary callables that do not declare a 'prompt_template' or 'template'
      attribute with a clear ValueError.
    - Rejects unsupported argument types with TypeError.

    Guarantees:
    - Stable across runs, processes, and Python versions.
    - Does not rely on memory addresses, process IDs, or volatile runtime state.
    - Yields a 64-character lowercase hex digest matching EvaluationCheckpoint format.
    """
    decl_template: str | None = None
    if prompt is None or prompt is default_prompt_builder:
        template_str = DEFAULT_PROMPT_TEMPLATE
    elif isinstance(prompt, str):
        template_str = prompt.strip()
        if not template_str:
            raise ValueError("Prompt template string must not be empty or whitespace.")
    else:
        # Check for static prompt template declaration on callable or
        # non-callable object
        if hasattr(prompt, "prompt_template") and isinstance(
            prompt.prompt_template, str
        ):
            decl_template = prompt.prompt_template
        elif hasattr(prompt, "template") and isinstance(prompt.template, str):
            decl_template = prompt.template

        if decl_template is not None:
            template_str = decl_template.strip()
            if not template_str:
                raise ValueError(
                    "Declared prompt template attribute must not be empty or "
                    "whitespace."
                )
        elif callable(prompt):
            raise ValueError(
                f"Callable prompt builder {prompt!r} does not declare a static "
                f"'prompt_template' or 'template' attribute. A stable prompt "
                "template declaration is required for deterministic fingerprinting."
            )
        else:
            raise TypeError(
                f"Expected prompt template string, callable PromptBuilder, or None; "
                f"got {type(prompt).__name__}."
            )

    hasher = hashlib.sha256()
    hasher.update(b"PROMPT_CONFIG_V1:\n")
    hasher.update(template_str.encode("utf-8"))
    return hasher.hexdigest().lower()


def _validate_report_path_collision(
    candidate: Path,
    report_path: Path | str,
) -> None:
    """Validate that candidate path does not collide with the final report path."""
    rep_raw = str(report_path).strip().replace("\\", "/")
    if not rep_raw or rep_raw == ".":
        return

    pure = PurePosixPath(str(candidate).strip().replace("\\", "/"))
    rep_pure = PurePosixPath(rep_raw)
    if pure == rep_pure:
        raise ValueError(
            f"checkpoint_path cannot target the final report path: {rep_raw!r}."
        )

    # Syntactic guard preventing report_path from referencing protected dataset
    if any(_is_protected_component(part) for part in rep_pure.parts):
        raise ValueError(
            f"report_path cannot target the protected test dataset: {rep_raw!r}."
        )

    rep_candidate = Path(rep_raw)
    try:
        cand_resolved = candidate.resolve(strict=False)
        rep_resolved = rep_candidate.resolve(strict=False)
        if cand_resolved == rep_resolved:
            raise ValueError(
                f"checkpoint_path cannot target the final report path: {rep_raw!r}."
            )
    except (RuntimeError, PermissionError) as exc:
        raise ValueError(
            f"Cannot verify path safety between checkpoint_path and report_path: {exc}."
        ) from exc
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError):
            raise ValueError(
                "Cannot verify path safety between checkpoint_path and "
                f"report_path: {exc}."
            ) from exc

    # Filesystem identity check if both files exist on disk
    try:
        if (
            candidate.exists()
            and rep_candidate.exists()
            and os.path.samefile(candidate, rep_candidate)
        ):
            raise ValueError(
                f"checkpoint_path cannot target the final report path: {rep_raw!r}."
            )
    except PermissionError as exc:
        raise ValueError(
            f"Cannot verify path safety between checkpoint_path and report_path: {exc}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Cannot verify path safety between checkpoint_path and report_path: {exc}."
        ) from exc


def validate_checkpoint_path(
    checkpoint_path: Path | str,
    dataset_path: Path | str | None = None,
    *,
    report_path: Path | str | None = None,
    require_parent_exists: bool = False,
) -> Path:
    """Validate checkpoint destination or source path safety.

    Guarantees:
    1. Rejects paths targeting the protected test dataset ('test.jsonl') syntactically
       or via filesystem resolution (including ADS, Win32 dots/spaces, 8.3 aliases).
    2. Rejects checkpoint paths targeting existing directories.
    3. Fails closed on file ancestors, inaccessible directories, or permission errors.
    4. Rejects checkpoint paths matching or resolving to the input dataset path
       (if provided).
    5. If report_path is provided, rejects checkpoint paths matching or resolving
       to the final report path, preventing report destruction, even when
       dataset_path is None.
    """
    raw = str(checkpoint_path).strip().replace("\\", "/")
    if not raw or raw == ".":
        raise ValueError("checkpoint_path must specify a valid file path.")

    pure = PurePosixPath(raw)
    if any(_is_protected_component(part) for part in pure.parts):
        raise ValueError(
            f"checkpoint_path cannot target the protected test dataset: {raw!r}."
        )

    candidate = Path(raw)
    try:
        resolved = candidate.resolve(strict=False)
        if any(_is_protected_component(part) for part in resolved.parts):
            raise ValueError(
                f"checkpoint_path cannot target the protected test dataset: {raw!r}."
            )
    except (RuntimeError, PermissionError) as exc:
        raise ValueError(
            f"Cannot verify path safety for checkpoint_path {raw!r}: {exc}."
        ) from exc
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError):
            raise ValueError(
                f"Cannot verify path safety for checkpoint_path {raw!r}: {exc}."
            ) from exc

    # Base validation reusing validate_output_path against protected & input datasets
    if dataset_path is not None:
        candidate = validate_output_path(
            candidate,
            dataset_path,
            require_parent_exists=require_parent_exists,
        )
    else:
        # Verify candidate is not an existing directory and check ancestor safety
        try:
            cand_stat = candidate.stat()
            if stat.S_ISDIR(cand_stat.st_mode):
                raise ValueError(f"checkpoint_path cannot be a directory: {raw!r}.")
        except FileNotFoundError:
            _verify_ancestor_safety(
                candidate,
                raw,
                "checkpoint_path",
                require_direct_parent=require_parent_exists,
            )
        except PermissionError as exc:
            raise ValueError(
                f"Cannot verify path safety for checkpoint_path {raw!r}: {exc}."
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Cannot verify path safety for checkpoint_path {raw!r}: {exc}."
            ) from exc

    # Guard against colliding with final report path (checked independently of
    # dataset_path)
    if report_path is not None:
        _validate_report_path_collision(candidate, report_path)

    return candidate


def save_checkpoint(
    checkpoint: EvaluationCheckpoint,
    checkpoint_path: Path | str,
    *,
    report_path: Path | str | None = None,
) -> Path:
    """Atomically save an EvaluationCheckpoint to disk.

    Guarantees:
    1. Validates that the input is an EvaluationCheckpoint instance.
    2. Validates checkpoint path safety before directory creation or file writes.
    3. Never silently overwrites or resets an invalid or corrupted existing checkpoint.
    4. Writes deterministic UTF-8 JSON to a uniquely named temporary file in
       target directory.
    5. Atomically replaces the target destination file.
    6. Cleans up temporary files if an error occurs during write or replacement.
    7. Preserves any pre-existing valid checkpoint if write/replacement fails.
    """
    if not isinstance(checkpoint, EvaluationCheckpoint):
        raise TypeError(
            f"checkpoint must be an EvaluationCheckpoint instance; "
            f"got {type(checkpoint).__name__}."
        )

    target_path = validate_checkpoint_path(
        checkpoint_path,
        checkpoint.run_config.dataset_path,
        report_path=report_path,
        require_parent_exists=False,
    )

    # Pre-write integrity guard: if target_path already exists, verify that it is
    # NOT invalid.
    # Refuse to silently overwrite or destroy a corrupted checkpoint.
    try:
        cand_stat = target_path.stat()
    except FileNotFoundError:
        cand_stat = None
    except OSError as exc:
        raise ValueError(
            f"Cannot inspect existing checkpoint path {target_path}: {exc}."
        ) from exc

    if cand_stat is not None:
        if stat.S_ISDIR(cand_stat.st_mode):
            raise ValueError(f"checkpoint_path cannot be a directory: {target_path}.")
        try:
            existing_content = target_path.read_text(encoding="utf-8")
            EvaluationCheckpoint.from_json(existing_content)
        except Exception as exc:
            raise ValueError(
                f"Existing file at {target_path} is an invalid or corrupted checkpoint "
                f"({exc}). Refusing to overwrite corrupted checkpoint."
            ) from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = checkpoint.to_json(indent=2) + "\n"

    temp_path = target_path.parent / f".{target_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return target_path


def load_checkpoint(
    checkpoint_path: Path | str,
    dataset_path: Path | str | None = None,
    *,
    report_path: Path | str | None = None,
) -> EvaluationCheckpoint:
    """Safely load and validate an EvaluationCheckpoint from disk.

    Guarantees:
    1. Strictly read-only: does not modify or alter the checkpoint file.
    2. Validates path safety, rejecting paths targeting the protected test dataset.
    3. Rejects collisions with input dataset or report path (if provided), even if
       dataset_path is None.
    4. Rejects missing, unreadable, non-UTF-8, or directory paths with clear errors.
    5. Enforces full schema and invariant validation via
       EvaluationCheckpoint.from_json().
    6. Never silently resets or repairs an invalid or corrupted checkpoint.
    """
    candidate = validate_checkpoint_path(
        checkpoint_path,
        dataset_path,
        report_path=report_path,
        require_parent_exists=False,
    )

    try:
        cand_stat = candidate.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {str(checkpoint_path)!r}.")
    except PermissionError as exc:
        raise ValueError(
            f"Permission denied accessing checkpoint path: {str(checkpoint_path)!r}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Cannot access checkpoint path: {str(checkpoint_path)!r}: {exc}."
        ) from exc

    if stat.S_ISDIR(cand_stat.st_mode):
        raise ValueError(
            f"checkpoint_path cannot be a directory: {str(checkpoint_path)!r}."
        )

    try:
        content = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Checkpoint file is not valid UTF-8: {str(checkpoint_path)!r}: {exc}."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Failed to read checkpoint file: {str(checkpoint_path)!r}: {exc}."
        ) from exc

    try:
        return EvaluationCheckpoint.from_json(content)
    except Exception as exc:
        raise ValueError(
            f"Checkpoint file at {str(checkpoint_path)!r} is invalid or "
            f"malformed: {exc}."
        ) from exc
