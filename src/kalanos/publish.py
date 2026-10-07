"""Send a saved report to a Kalanos hub account.

The report goes as `kalanos grade` wrote it.
The hub decides whether it can be published, and says why when it can't.
Nothing here prints or exits:
every failure is a `PublishError` whose message is ready for the user.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.metadata
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

KEY_PATTERN = re.compile(r"^klns_(live|test)_[0-9a-f]{32}$")

# The pattern the hub enforces on a dataset name.
NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

_TIMEOUT_S = 60
_RETRY_WAIT_S = 2.0


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class PublishError(Exception):
    """A refusal or failure, worded for the user, that never contains the key."""


@dataclass(frozen=True)
class Published:
    """What the hub answered for an accepted report.

    Attributes
    ----------
    report_id : str
        The hub's id for this submission.
    page : str or None
        The report's page, when the hub gives one.
    dashboard : str
        Where the submission's status shows.
    """

    report_id: str
    page: str | None
    dashboard: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface a 3xx as an `HTTPError`.

    `urllib` copies the `Authorization` header to a redirect target,
    so following one could hand the key to another host.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def key_prefix(key: str) -> str:
    """Return the part of a key the dashboard shows, e.g. `klns_live_a1c8f4`."""

    return key[:16]


def check_key(key: str | None, hub_url: str) -> str:
    """Return `key` when it has the shape of a Kalanos API key.

    Raises
    ------
    PublishError
        When `key` is missing or malformed. The message echoes nothing of it.
    """

    if not key:
        raise PublishError(
            f"set KALANOS_API_KEY or pass --api-key; create a key at {hub_url}/settings"
        )
    if not KEY_PATTERN.fullmatch(key):
        raise PublishError(
            "KALANOS_API_KEY is not a Kalanos API key; keys start with klns_live_"
        )
    return key


def check_hub_url(url: str) -> str:
    """Return `url` without a trailing slash.

    Raises
    ------
    PublishError
        When `url` is not https, unless it is http to this machine for a local hub.
    """

    parts = urlsplit(url)
    local = parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS
    if not (parts.scheme == "https" and parts.hostname) and not local:
        raise PublishError(
            f"KALANOS_HUB_URL must be an https URL, or http to localhost; got {url!r}"
        )
    return url.rstrip("/")


def load_report(path: Path) -> dict[str, Any]:
    """Read a report JSON as the object it holds.

    Raises
    ------
    PublishError
        When `path` can't be read, isn't JSON, or holds something other than an object.
    """

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise PublishError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PublishError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise PublishError(f"{path} does not hold a report")
    return report


def check_name(report: dict[str, Any], name: str | None) -> str | None:
    """Return the dataset name to send with `report`, if any.

    A hosted dataset publishes under its own name.
    A local one has none, so it needs `name`.

    Raises
    ------
    PublishError
        When `name` is not `ORG/NAME`, or is missing for a local dataset.
    """

    if name is not None and not NAME_PATTERN.fullmatch(name):
        raise PublishError(
            f"--name {name!r} is not ORG/NAME: letters, digits, '.', '_' or '-', "
            "up to 64 each"
        )
    source = report.get("source")
    repo_id = source.get("repo_id") if isinstance(source, dict) else None
    if name is None and not repo_id:
        raise PublishError(
            "this report is of a local dataset; pass --name ORG/NAME to publish it"
        )
    return name


def publish_report(
    report: dict[str, Any],
    *,
    api_key: str,
    name: str | None,
    hub_url: str,
) -> Published:
    """Send `report` to the hub and return what it accepted.

    A network error or a 5xx is retried once after a short wait.
    The retry is safe because the hub replaces a dataset's report
    rather than adding a second one.

    Parameters
    ----------
    report : dict
        The report, sent unchanged.
    api_key : str
        A key `check_key` accepted.
    name : str or None
        `ORG/NAME` for a local dataset, sent only when given.
    hub_url : str
        A URL `check_hub_url` returned.

    Raises
    ------
    PublishError
        When the hub refused the report or could not be reached.
    """

    body: dict[str, Any] = {"report": report}
    if name is not None:
        body["name"] = name
    request = urllib.request.Request(
        f"{hub_url}/api/reports",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"kalanos/{importlib.metadata.version('kalanos')}",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)

    for attempt in (1, 2):
        try:
            with opener.open(request, timeout=_TIMEOUT_S) as response:
                answer = json.loads(response.read())
            return Published(
                report_id=answer["report_id"],
                page=answer.get("page"),
                dashboard=answer["dashboard"],
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PublishError(
                f"{hub_url} accepted the request but its answer was not understood"
            ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise PublishError(_refusal(exc, api_key, hub_url)) from exc
            if attempt == 2:
                raise PublishError(
                    f"{hub_url} failed with HTTP {exc.code} twice; try again later"
                ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 2:
                reason = getattr(exc, "reason", exc)
                raise PublishError(f"could not reach {hub_url}: {reason}") from exc
        time.sleep(_RETRY_WAIT_S)
    raise AssertionError("unreachable")


def _refusal(error: urllib.error.HTTPError, api_key: str, hub_url: str) -> str:
    """Word a 3xx or 4xx answer for the user."""

    raw = error.read().decode("utf-8", errors="replace")
    try:
        detail = json.loads(raw)
    except json.JSONDecodeError:
        detail = None
    if not isinstance(detail, dict):
        detail = {}

    match error.code:
        case 401:
            return (
                f"the hub refused key {key_prefix(api_key)} (invalid or revoked); "
                f"create a new one at {hub_url}/settings"
            )
        case 402:
            tier = detail.get("tier_label") or detail.get("tier")
            return (
                f"Dataset limit reached: {tier} plan allows "
                f"{detail.get('limit')} datasets ({detail.get('used')} used). "
                "Upgrade or delete a dataset in the dashboard."
            )
        case 422:
            return (
                f"the hub could not publish this report: "
                f"{detail.get('error')} ({detail.get('status')})"
            )
        case code if 300 <= code < 400:
            return (
                f"the hub redirected to {error.headers.get('Location')}; "
                "set KALANOS_HUB_URL to the current address"
            )
        case code:
            reason = detail.get("error") or detail.get("detail") or raw[:200]
            return f"the hub answered HTTP {code}: {reason}"
