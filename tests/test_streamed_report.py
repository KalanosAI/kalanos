"""Verifies the streaming report writer writes what a whole-report dump would."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
from collections.abc import Iterator

# External
import pytest
from pydantic_core import to_json
from upath import UPath

# Internal
from kalanos.analysis.models.compact_keys import encode_keys
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.omit import omit_empty_defaults
from kalanos.analysis.reporting.render import render_compact_json, render_json
from kalanos.analysis.reporting.write import write_report
from kalanos.api import grade
from kalanos.core.settings import get_settings

from helpers import (
    CSV_FIXTURE,
    FIXTURES_DIR,
    HDF5_FIXTURE,
    LEROBOT_FIXTURE,
    LEROBOT_V2_0_FIXTURE,
    LEROBOT_V2_1_FIXTURE,
    MCAP_FIXTURE,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

SOURCES = [
    FIXTURES_DIR,
    CSV_FIXTURE,
    HDF5_FIXTURE,
    MCAP_FIXTURE,
    LEROBOT_FIXTURE,
    LEROBOT_V2_0_FIXTURE,
    LEROBOT_V2_1_FIXTURE,
]


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(scope="module", autouse=True)
def _clean_kalanos_env_for_module() -> Iterator[None]:
    """Strip KALANOS_ env vars around the module-scoped reports below."""

    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith("KALANOS_"):
                mp.delenv(key, raising=False)
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(scope="module", params=SOURCES, ids=lambda path: path.name)
def report(request: pytest.FixtureRequest) -> Report:
    """Grade one fixture of the corpus, or the corpus as a folder.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Carries the source to grade.

    Returns
    -------
    Report
        The graded report.
    """

    source: UPath = request.param
    return grade(source)


# ░█░█░█▀▀░█░░░█▀█░█▀▀░█▀▄░█▀▀
# ░█▀█░█▀▀░█░░░█▀▀░█▀▀░█▀▄░▀▀█
# ░▀░▀░▀▀▀░▀▀▀░▀░░░▀▀▀░▀░▀░▀▀▀


def _whole_dump(report: Report, *, compact: bool) -> bytes:
    """Dump a report as one whole-report dict.

    Parameters
    ----------
    report : Report
        The report to dump.
    compact : bool
        Replace every key with its alias.

    Returns
    -------
    bytes
        The report's JSON.
    """

    data = omit_empty_defaults(report, report.model_dump(mode="json"))
    return to_json(encode_keys(data) if compact else data, inf_nan_mode="null")


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_render_json_writes_the_whole_dump(report):
    assert render_json(report).encode() == _whole_dump(report, compact=False)


def test_render_compact_json_writes_the_whole_compact_dump(report):
    assert render_compact_json(report).encode() == _whole_dump(report, compact=True)


@pytest.mark.parametrize("compact", [False, True])
def test_write_report_writes_the_whole_dump(report, tmp_path, compact):
    path = write_report(report, tmp_path / "report.json", compact=compact)

    assert path.read_bytes() == _whole_dump(report, compact=compact)


def test_a_report_without_episodes_leaves_them_out(report):
    empty = report.model_copy(update={"episodes": []})

    assert render_json(empty).encode() == _whole_dump(empty, compact=False)
    assert render_compact_json(empty).encode() == _whole_dump(empty, compact=True)
