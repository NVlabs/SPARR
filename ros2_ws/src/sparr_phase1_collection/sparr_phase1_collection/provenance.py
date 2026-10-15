"""Reproducible artifact and repository provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class ArtifactProvenance:
    """Identity of one input artifact."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RepositoryProvenance:
    """Best-effort identity of the source checkout."""

    root: str
    commit: Optional[str]
    branch: Optional[str]
    dirty: Optional[bool]


@dataclass(frozen=True)
class SourceTreeProvenance:
    """Deterministic identity for a bind-mounted runtime source tree."""

    root: str
    sha256: str
    file_count: int


_SOURCE_TREE_EXCLUDED_DIRECTORIES = frozenset(
    {".git", ".pytest_cache", "__pycache__", "build", "install", "log"}
)
_SOURCE_TREE_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".partial.h5", ".h5")


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_provenance(path: str | Path) -> ArtifactProvenance:
    """Collect stable provenance for a required file."""

    resolved = Path(path).expanduser().resolve(strict=True)
    return ArtifactProvenance(
        path=str(resolved),
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def source_tree_provenance(root: str | Path) -> SourceTreeProvenance:
    """Hash stable relative paths and bytes for the live development package."""

    resolved = Path(root).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Runtime source root is not a directory: {resolved}")
    candidates = []
    for path in resolved.rglob("*"):
        relative = path.relative_to(resolved)
        if any(
            part in _SOURCE_TREE_EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_dir() or path.name.endswith(_SOURCE_TREE_EXCLUDED_SUFFIXES):
            continue
        if path.is_file() or path.is_symlink():
            candidates.append((relative.as_posix(), path))

    digest = hashlib.sha256()
    for relative, path in sorted(candidates):
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8")
            digest.update(b"L")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            continue
        digest.update(b"F")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return SourceTreeProvenance(
        root=str(resolved),
        sha256=digest.hexdigest(),
        file_count=len(candidates),
    )


def find_repository_root(config_path: str | Path, explicit: str = "") -> Path:
    """The repository root that relative recording paths are resolved against.

    ``explicit`` wins (session.sh passes it); otherwise the nearest directory
    holding ``.git``, searching the CWD and then the config's ancestors.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    for candidate in (Path.cwd(), config.parent, *config.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def resolve_output_directory(
    output_directory: str | Path, repository_root: str | Path
) -> Path:
    """``recording.output_directory`` as the writer will use it."""
    output = Path(output_directory).expanduser()
    if not output.is_absolute():
        output = Path(repository_root) / output
    return output.resolve()


def resolve_artifact_path(
    value: str | Path,
    *,
    config_path: str | Path | None = None,
    additional_roots: Iterable[str | Path] = (),
) -> Path:
    """Resolve a configured artifact without assuming the launch CWD.

    Relative paths are checked against the current directory, the configuration
    directory and each of its ancestors, followed by any explicit roots. The
    first existing file wins; otherwise the original CWD-relative path is
    returned so callers can produce a useful error.
    """

    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates: list[Path] = [Path.cwd() / raw]
    if config_path is not None:
        parent = Path(config_path).expanduser().resolve().parent
        candidates.extend(ancestor / raw for ancestor in (parent, *parent.parents))
    candidates.extend(Path(root).expanduser().resolve() / raw for root in additional_roots)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def repository_provenance(root: str | Path) -> RepositoryProvenance:
    """Return best-effort git metadata without changing the checkout."""

    resolved = Path(root).expanduser().resolve()
    commit = _git(resolved, "rev-parse", "HEAD")
    branch = _git(resolved, "branch", "--show-current")
    # Untracked source can change the running code, so it counts as dirty.
    status = _git(resolved, "status", "--porcelain", "--untracked-files=normal")
    return RepositoryProvenance(
        root=str(resolved),
        commit=commit,
        branch=branch or None,
        dirty=None if status is None else bool(status),
    )


def container_image_provenance(
    manifest_path: str | Path | None = None,
) -> Optional[dict[str, Any]]:
    """Load the immutable image manifest when running in a packaged image.

    Returns ``None`` outside the image. If an explicit path (or
    ``SPARR_IMAGE_MANIFEST``) is given, missing or malformed content is fatal.
    """

    configured = manifest_path or os.environ.get("SPARR_IMAGE_MANIFEST")
    if not configured:
        return None
    resolved = Path(configured).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Container image manifest must be a mapping: {resolved}")
    result = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "manifest": dict(manifest),
    }
    runtime_image_id = os.environ.get("SPARR_RUNTIME_IMAGE_ID", "").strip()
    if runtime_image_id:
        result["runtime_image_id"] = runtime_image_id
    return result


def build_session_provenance(
    *,
    repository_root: str | Path,
    checkpoint: str | Path,
    agent_yaml: str | Path,
    env_yaml: str | Path,
) -> dict:
    """Build the serialization-safe provenance block stored in HDF5."""

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": asdict(repository_provenance(repository_root)),
        "checkpoint": asdict(artifact_provenance(checkpoint)),
        "agent_yaml": asdict(artifact_provenance(agent_yaml)),
        "env_yaml": asdict(artifact_provenance(env_yaml)),
    }
    image = container_image_provenance()
    if image is not None:
        provenance["container_image"] = image
    runtime_source_root = os.environ.get("SPARR_RUNTIME_SOURCE_ROOT", "").strip()
    if runtime_source_root:
        provenance["runtime_source"] = asdict(
            source_tree_provenance(runtime_source_root)
        )
    return provenance
