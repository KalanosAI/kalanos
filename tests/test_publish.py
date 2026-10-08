"""Verifies `kalanos publish` and `kalanos grade --publish`.

A local HTTP server stands in for the hub.

The key must never reach output,
so every CLI result passes through `_run`, which checks that.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# External
import pytest
from typer.testing import CliRunner

# Internal
from kalanos import api, publish
from kalanos.cli import app
from kalanos.publish import PublishError, check_hub_url

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

KEY = "klns_live_" + "ab" * 16

HF_REPORT = {
    "schema_version": "7.0.0",
    "source": {"protocol": "hf", "repo_id": "lerobot/pusht"},
    "episodes": [],
}
LOCAL_REPORT = {
    "schema_version": "7.0.0",
    "source": {"protocol": "file", "repo_id": None},
    "episodes": [],
}

ACCEPTED = (
    202,
    {},
    {
        "report_id": "rep_123",
        "status": "queued",
        "page": None,
        "dashboard": "https://hub.example/dashboard",
    },
)

runner = CliRunner()


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class Hub:
    """A scripted hub: the answers it will give, and the requests it got."""

    url: str
    answers: list[tuple[int, dict[str, str], dict | str]] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture
def hub(monkeypatch):
    """Serve scripted answers on 127.0.0.1 and point `KALANOS_HUB_URL` at them."""

    state = Hub(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            state.requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            status, headers, body = state.answers.pop(0)
            payload = body if isinstance(body, str) else json.dumps(body)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("KALANOS_HUB_URL", state.url)
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture
def waits(monkeypatch):
    """Record the retry waits instead of sleeping through them."""

    recorded: list[float] = []
    monkeypatch.setattr(publish.time, "sleep", recorded.append)
    return recorded


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _run(*args: str, env: dict[str, str] | None = None):
    """Invoke the CLI and assert the key appears in neither stream."""

    result = runner.invoke(app, list(args), env=env)
    assert KEY not in result.stdout
    assert KEY not in result.stderr
    return result


def _write(tmp_path, content) -> str:
    path = tmp_path / "report.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return str(path)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_accepted_report_prints_id_and_dashboard(hub, tmp_path):
    hub.answers.append(ACCEPTED)

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 0, result.stderr
    assert "rep_123" in result.stdout
    assert "https://hub.example/dashboard" in result.stdout
    [request] = hub.requests
    assert request["path"] == "/api/reports"
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert request["body"] == {"report": HF_REPORT}


def test_local_report_needs_a_name(hub, tmp_path):
    result = _run("publish", _write(tmp_path, LOCAL_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert "--name" in result.stderr
    assert hub.requests == []


def test_local_report_sends_its_name(hub, tmp_path):
    hub.answers.append(ACCEPTED)

    result = _run(
        "publish",
        _write(tmp_path, LOCAL_REPORT),
        "--api-key",
        KEY,
        "--name",
        "acme/set",
    )

    assert result.exit_code == 0, result.stderr
    assert hub.requests[0]["body"]["name"] == "acme/set"


def test_malformed_name_is_refused(hub, tmp_path):
    result = _run(
        "publish", _write(tmp_path, LOCAL_REPORT), "--api-key", KEY, "--name", "a b/c"
    )

    assert result.exit_code == 2
    assert hub.requests == []


def test_missing_key_is_refused(hub, tmp_path):
    result = _run("publish", _write(tmp_path, HF_REPORT))

    assert result.exit_code == 2
    assert "KALANOS_API_KEY" in result.stderr
    assert hub.requests == []


@pytest.mark.parametrize("bad", ["klns_live_not-a-real-key-at-all", KEY + "\n"])
def test_malformed_key_is_refused_without_echoing_it(hub, tmp_path, bad):

    result = _run("publish", _write(tmp_path, HF_REPORT), env={"KALANOS_API_KEY": bad})

    assert result.exit_code == 2
    assert bad not in result.stderr
    assert hub.requests == []


@pytest.mark.parametrize("content", [None, "[1]", "{"])
def test_unusable_report_file_is_refused(hub, tmp_path, content):
    path = (
        str(tmp_path / "report.json") if content is None else _write(tmp_path, content)
    )

    result = _run("publish", path, "--api-key", KEY)

    assert result.exit_code == 2
    assert hub.requests == []


def test_unauthorized_names_the_key_by_prefix(hub, tmp_path):
    hub.answers.append((401, {}, {"error": "unauthorized", "detail": "revoked"}))

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert "klns_live_abab" in result.stderr


def test_quota_refusal_states_the_plan_limit(hub, tmp_path):
    hub.answers.append(
        (
            402,
            {},
            {
                "error": "quota_exceeded",
                "detail": "limit",
                "tier": "free",
                "tier_label": "Free",
                "limit": 50,
                "used": 50,
            },
        )
    )

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert "Free plan allows 50 datasets (50 used)" in result.stderr


def test_unprocessable_report_gives_the_hub_reason(hub, tmp_path):
    hub.answers.append(
        (
            422,
            {},
            {"status": "KALANOS_UNSUPPORTED", "error": "kalanos read no episodes"},
        )
    )

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert "KALANOS_UNSUPPORTED" in result.stderr
    assert "kalanos read no episodes" in result.stderr


def test_server_error_is_retried_once(hub, tmp_path, waits):
    hub.answers.extend([(503, {}, "busy"), ACCEPTED])

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 0, result.stderr
    assert len(hub.requests) == 2
    assert waits == [2.0]


def test_server_error_twice_gives_up(hub, tmp_path, waits):
    hub.answers.extend([(503, {}, "busy"), (503, {}, "busy")])

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert len(hub.requests) == 2


def test_unreachable_hub_is_retried_once(tmp_path, monkeypatch, waits):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("KALANOS_HUB_URL", f"http://127.0.0.1:{port}")

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert "could not reach" in result.stderr
    assert waits == [2.0]


def test_unexpected_success_body_is_refused(hub, tmp_path):
    hub.answers.append((202, {}, "<html>proxy</html>"))

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert "not understood" in result.stderr


def test_redirect_is_not_followed(hub, tmp_path):
    hub.answers.append((302, {"Location": "https://elsewhere.example/"}, ""))

    result = _run("publish", _write(tmp_path, HF_REPORT), "--api-key", KEY)

    assert result.exit_code == 2
    assert len(hub.requests) == 1
    assert "https://elsewhere.example/" in result.stderr


def test_help_does_not_show_the_key():
    result = _run("publish", "--help", env={"KALANOS_API_KEY": KEY})

    assert result.exit_code == 0


def test_grade_publish_sends_the_graded_report(hub):
    hub.answers.append(ACCEPTED)

    result = _run(
        "grade",
        str(CSV_FIXTURE),
        "--publish",
        "--name",
        "acme/set",
        env={"KALANOS_API_KEY": KEY},
    )

    assert result.exit_code == 0, result.stderr
    [request] = hub.requests
    assert request["body"]["name"] == "acme/set"
    assert "Submitted rep_123" in result.stderr
    assert "Submitted" not in result.stdout


def test_grade_publish_json_keeps_stdout_one_document(hub):
    hub.answers.append(ACCEPTED)

    result = _run(
        "grade",
        str(CSV_FIXTURE),
        "--json",
        "--publish",
        "--api-key",
        KEY,
        "--name",
        "acme/set",
    )

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == hub.requests[0]["body"]["report"]
    assert "Submitted rep_123" in result.stderr


def test_grade_publish_keeps_the_gate_exit_code(hub):
    hub.answers.append(ACCEPTED)

    result = _run(
        "grade", str(FIXTURES_DIR), "--publish", "--api-key", KEY, "--name", "acme/set"
    )

    assert result.exit_code == 1, result.stderr
    assert len(hub.requests) == 1


def test_grade_publish_with_operational_errors_still_sends_then_exits_two(
    hub, monkeypatch
):
    hub.answers.append(ACCEPTED)
    graded = api.grade(str(CSV_FIXTURE))
    broken = graded.model_copy(
        update={"operational_errors": [{"stage": "test", "error": "failed"}]}
    )
    monkeypatch.setattr(api, "grade", lambda *args, **kwargs: broken)

    result = _run(
        "grade", str(CSV_FIXTURE), "--publish", "--api-key", KEY, "--name", "acme/set"
    )

    assert result.exit_code == 2
    assert len(hub.requests) == 1


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (["--name", "acme/set"], "KALANOS_API_KEY"),
        (["--api-key", KEY, "--name", "a b/c"], "is not ORG/NAME"),
    ],
    ids=["no-key", "bad-name"],
)
def test_grade_publish_refuses_before_grading(hub, monkeypatch, flags, reason):
    def fail(*args, **kwargs):
        pytest.fail("grade ran")

    monkeypatch.setattr(api, "grade", fail)

    result = _run("grade", str(CSV_FIXTURE), "--publish", *flags)

    assert result.exit_code == 2
    assert reason in result.stderr
    assert hub.requests == []


def test_grade_publish_refusal_exits_two_over_a_tripped_gate(hub):
    hub.answers.append((422, {}, {"status": "KALANOS_UNSUPPORTED", "error": "no"}))

    result = _run(
        "grade", str(FIXTURES_DIR), "--publish", "--api-key", KEY, "--name", "acme/set"
    )

    assert result.exit_code == 2
    assert "KALANOS_UNSUPPORTED" in result.stderr


def test_grade_publish_of_a_local_dataset_needs_a_name(hub, tmp_path):
    report = tmp_path / "report.json"

    result = _run(
        "grade",
        str(CSV_FIXTURE),
        "--report",
        str(report),
        "--publish",
        "--api-key",
        KEY,
    )

    assert result.exit_code == 2
    assert f"kalanos publish {report} --name" in result.stderr
    assert hub.requests == []


def test_grade_refuses_a_name_without_publish(hub):
    result = _run("grade", str(CSV_FIXTURE), "--name", "acme/set")

    assert result.exit_code == 2
    assert "--publish" in result.stderr
    assert hub.requests == []


@pytest.mark.parametrize(
    ("flags", "env"),
    [(["--api-key", KEY], None), ([], {"KALANOS_API_KEY": KEY})],
    ids=["flag", "env"],
)
def test_grade_without_publish_ignores_the_key(hub, flags, env):
    result = _run("grade", str(CSV_FIXTURE), *flags, env=env)

    assert result.exit_code == 0, result.stderr
    assert "Assessment" in result.stdout
    assert hub.requests == []


@pytest.mark.parametrize(
    "url",
    ["https://hub.kalanos.ai/", "http://localhost:8787", "http://127.0.0.1:8787"],
)
def test_hub_url_accepts_https_and_local_http(url):
    assert not check_hub_url(url).endswith("/")


@pytest.mark.parametrize("url", ["http://hub.kalanos.ai", "ftp://x"])
def test_hub_url_refuses_anything_else(url):
    with pytest.raises(PublishError):
        check_hub_url(url)


def test_tls_context_trusts_certifi_without_system_cas(tmp_path, monkeypatch):
    # Stands in for a Python whose default CA paths are missing.
    empty = tmp_path / "empty.pem"
    empty.touch()
    monkeypatch.setenv("SSL_CERT_FILE", str(empty))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))

    assert publish.tls_context().cert_store_stats()["x509_ca"] > 0
