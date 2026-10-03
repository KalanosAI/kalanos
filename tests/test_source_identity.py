"""Verifies the opt-in source byte hash, and what it refuses to vouch for."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import pytest

# Internal
from kalanos.api import compare, grade

# Local
from helpers import LEROBOT_FIXTURE


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_opt_in_byte_hash_makes_real_reports_comparable():
    old = grade(LEROBOT_FIXTURE, hash_source=True)
    new = grade(LEROBOT_FIXTURE, hash_source=True)
    assert old.run is not None and old.run.source is not None
    assert old.run.source.complete and old.run.source.covered_inputs > 0
    assert compare(old, new).comparable


def test_source_hash_refuses_symlinks(tmp_path):
    from kalanos.analysis.models.errors import SourceUnavailable
    from kalanos.analysis.source_identity import hash_local_source

    p = tmp_path / "link"
    p.symlink_to(str(LEROBOT_FIXTURE), target_is_directory=True)
    with pytest.raises(SourceUnavailable, match="symlink"):
        hash_local_source(p)


def test_change_during_analysis_withholds_report(monkeypatch):
    from kalanos import api
    from kalanos.analysis.models.errors import SourceUnavailable
    from kalanos.analysis.source_identity import hash_local_source

    original = hash_local_source(LEROBOT_FIXTURE)
    values = iter([original, original.model_copy(update={"digest": "b" * 64})])
    monkeypatch.setattr(api, "hash_local_source", lambda root: next(values))
    with pytest.raises(SourceUnavailable, match="changed during"):
        grade(LEROBOT_FIXTURE, hash_source=True)
