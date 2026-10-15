from pathlib import Path

import pytest

from sparr_phase1_collection.hdf5_schema import (
    SchemaContractError,
    load_declarative_schema,
    validate_declarative_schema,
)


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "sparr_phase1_v1.yaml"


def test_checked_in_declarative_schema_has_exact_runtime_parity():
    validate_declarative_schema(SCHEMA_PATH)


def test_declarative_schema_loader_rejects_duplicate_keys(tmp_path: Path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema:\n  name: one\n  name: two\n", encoding="utf-8")

    with pytest.raises(SchemaContractError, match="duplicate"):
        load_declarative_schema(duplicate)


def test_declarative_schema_identity_drift_is_rejected(tmp_path: Path):
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(
        SCHEMA_PATH.read_text(encoding="utf-8").replace(
            "version: 1.0.0", "version: 9.9.9", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(SchemaContractError, match="schema version"):
        validate_declarative_schema(drifted)
