# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pathlib import Path
import json
import shutil
import subprocess

import pytest

from sparr_phase1_collection.provenance import (
    build_session_provenance,
    container_image_provenance,
    repository_provenance,
    sha256_file,
    source_tree_provenance,
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_untracked_files_make_repository_provenance_dirty(tmp_path: Path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    assert repository_provenance(tmp_path).dirty is False

    (tmp_path / "new_runtime_source.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert repository_provenance(tmp_path).dirty is True


def test_container_image_provenance_is_optional(monkeypatch):
    monkeypatch.delenv("SPARR_IMAGE_MANIFEST", raising=False)

    assert container_image_provenance() is None


def test_container_image_manifest_is_loaded_and_hashed(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "build-manifest.json"
    manifest = {
        "base_image_digest": "sha256:base",
        "sparr_source_sha256": "source",
        "flexiv_ros2_commit": "flexiv",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("SPARR_IMAGE_MANIFEST", str(manifest_path))
    monkeypatch.delenv("SPARR_RUNTIME_IMAGE_ID", raising=False)

    actual = container_image_provenance()

    assert actual == {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "manifest": manifest,
    }


def test_container_image_provenance_includes_launcher_image_id(
    tmp_path: Path, monkeypatch
):
    manifest_path = tmp_path / "build-manifest.json"
    manifest_path.write_text('{"source": "locked"}', encoding="utf-8")
    monkeypatch.setenv("SPARR_IMAGE_MANIFEST", str(manifest_path))
    monkeypatch.setenv("SPARR_RUNTIME_IMAGE_ID", "sha256:runtime-image")

    actual = container_image_provenance()

    assert actual is not None
    assert actual["runtime_image_id"] == "sha256:runtime-image"


def test_runtime_source_tree_hash_tracks_code_but_ignores_generated_files(
    tmp_path: Path,
):
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_provenance(tmp_path)

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"generated")
    egg_info = tmp_path / "package.egg-info"
    egg_info.mkdir()
    (egg_info / "SOURCES.txt").write_text("generated\n", encoding="utf-8")
    assert source_tree_provenance(tmp_path).sha256 == first.sha256

    (tmp_path / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = source_tree_provenance(tmp_path)
    assert changed.sha256 != first.sha256
    assert changed.file_count == 1


def test_session_provenance_embeds_container_manifest(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"approved")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"image": "sha256:accepted"}', encoding="utf-8")
    monkeypatch.setenv("SPARR_IMAGE_MANIFEST", str(manifest_path))

    provenance = build_session_provenance(
        repository_root=tmp_path,
        checkpoint=artifact,
        agent_yaml=artifact,
        env_yaml=artifact,
    )

    assert provenance["container_image"]["manifest"]["image"] == "sha256:accepted"


def test_session_provenance_embeds_live_runtime_source_hash(
    tmp_path: Path, monkeypatch
):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"approved")
    runtime_source = tmp_path / "runtime_source"
    runtime_source.mkdir()
    (runtime_source / "node.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.delenv("SPARR_IMAGE_MANIFEST", raising=False)
    monkeypatch.setenv("SPARR_RUNTIME_SOURCE_ROOT", str(runtime_source))

    provenance = build_session_provenance(
        repository_root=tmp_path,
        checkpoint=artifact,
        agent_yaml=artifact,
        env_yaml=artifact,
    )

    assert provenance["runtime_source"] == {
        "root": str(runtime_source.resolve()),
        "sha256": source_tree_provenance(runtime_source).sha256,
        "file_count": 1,
    }


def test_configured_invalid_container_manifest_is_fatal(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("SPARR_IMAGE_MANIFEST", str(manifest_path))

    with pytest.raises(ValueError, match="must be a mapping"):
        container_image_provenance()


def test_relative_output_directories_resolve_against_the_repository(tmp_path):
    from sparr_phase1_collection.provenance import (
        find_repository_root,
        resolve_output_directory,
    )

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    config = repo / "phase1_live.yaml"
    config.write_text("{}\n")

    root = find_repository_root(config)
    assert root == repo.resolve()
    # Relative paths follow the repository, not wherever the process started.
    assert resolve_output_directory("data/phase1", root) == repo.resolve() / "data" / "phase1"
    assert resolve_output_directory(tmp_path / "abs", root) == (tmp_path / "abs").resolve()
    assert find_repository_root(config, explicit=str(tmp_path)) == tmp_path.resolve()
