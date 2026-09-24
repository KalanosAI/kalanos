"""Pins how a typed path becomes a root to grade, and the guard in front of it.

A Hugging Face reference must resolve to one commit,
and an oversized remote root must be refused before any of its files is read.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import sys
from types import SimpleNamespace

# External
import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

# Internal
from kalanos.analysis.discovery.source import enforce_limits, resolve_source
from kalanos.analysis.models.discovery import SourceInfo, SourceLimits
from kalanos.analysis.models.errors import SourceTooLarge, SourceUnavailable


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


SHA = "f" * 40
SIBLINGS = [
    SimpleNamespace(rfilename="meta/info.json", size=10),
    SimpleNamespace(rfilename="data/chunk-000/file-000.parquet", size=90),
]


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture
def fake_hub(monkeypatch):
    """Answer `HfApi.dataset_info` without the network, and record what was asked."""

    calls = []

    class FakeApi:
        def dataset_info(self, repo_id, revision=None, files_metadata=False):
            calls.append((repo_id, revision))
            return SimpleNamespace(sha=SHA, siblings=SIBLINGS)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    return calls


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


@pytest.mark.parametrize(
    ("typed", "repo", "revision", "size", "count"),
    [
        (
            "https://huggingface.co/datasets/lerobot/pusht",
            "lerobot/pusht",
            None,
            100,
            2,
        ),
        (
            "https://huggingface.co/datasets/lerobot/pusht/tree/v3.0",
            "lerobot/pusht",
            "v3.0",
            100,
            2,
        ),
        ("hf://datasets/lerobot/pusht", "lerobot/pusht", None, 100, 2),
        ("hf://datasets/lerobot/pusht@abc123/meta", "lerobot/pusht", "abc123", 10, 1),
    ],
)
def test_a_hugging_face_reference_resolves_to_one_pinned_commit(
    fake_hub, typed, repo, revision, size, count
):
    """One run reads one snapshot, even if the repo is pushed while it runs."""

    root, source = resolve_source(typed)

    assert fake_hub == [(repo, revision)]
    assert str(root).startswith(f"hf://datasets/{repo}@{SHA}")
    assert (source.protocol, source.repo_id, source.revision) == ("hf", repo, SHA)
    assert (source.size_bytes, source.file_count) == (size, count)


def test_a_remote_root_over_the_byte_limit_is_refused():
    """Verify a remote root listing more bytes than the limit is refused, naming it."""

    source = SourceInfo(
        uri="hf://datasets/o/n@x", protocol="hf", size_bytes=2_000_000_000, file_count=1
    )

    with pytest.raises(SourceTooLarge, match="over the 1 GB limit"):
        enforce_limits(source, SourceLimits(max_bytes=1_000_000_000))


def test_a_remote_root_over_the_file_limit_is_refused():
    """Verify a remote root listing more files than the limit is refused, naming it."""

    source = SourceInfo(
        uri="hf://datasets/o/n@x", protocol="hf", size_bytes=1, file_count=11
    )

    with pytest.raises(SourceTooLarge, match="over the 10 file limit"):
        enforce_limits(source, SourceLimits(max_files=10))


def test_a_local_root_is_never_limited():
    """Verify the limits never refuse a local root, whose bytes are already on disk."""

    source = SourceInfo(
        uri="/data", protocol="file", size_bytes=10**15, file_count=10**7
    )

    enforce_limits(source, SourceLimits(max_bytes=1, max_files=1))


def test_a_local_root_is_described_by_its_files(tmp_path):
    """Verify a local folder is totalled over every file, nested ones included."""

    (tmp_path / "a.csv").write_text("12345")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.csv").write_text("123")

    _, source = resolve_source(tmp_path)

    assert (source.protocol, source.size_bytes, source.file_count) == ("file", 8, 2)


def test_a_missing_local_path_is_unavailable(tmp_path):
    """Verify a missing local path raises SourceUnavailable."""

    with pytest.raises(SourceUnavailable, match="does not exist"):
        resolve_source(tmp_path / "missing")


def test_a_gated_dataset_names_hf_token(monkeypatch):
    """Verify a gated dataset tells the user which variable unlocks it."""

    class GatedApi:
        def dataset_info(self, repo_id, revision=None, files_metadata=False):
            request = httpx.Request("GET", "https://huggingface.co/api/datasets/o/n")
            raise GatedRepoError("gated", response=httpx.Response(401, request=request))

    monkeypatch.setattr("huggingface_hub.HfApi", GatedApi)

    with pytest.raises(SourceUnavailable, match="HF_TOKEN"):
        resolve_source("hf://datasets/o/n")


def test_a_missing_hf_extra_names_the_extra(monkeypatch):
    """Verify grading a Hugging Face dataset without the extra names the extra."""

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(SourceUnavailable, match=r"kalanos\[hf\]"):
        resolve_source("hf://datasets/o/n")
