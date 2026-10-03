"""Conservative implementation identities; shared helper changes invalidate evidence."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import inspect
import platform
from collections.abc import Iterable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Internal
from kalanos.analysis.models.provenance import content_digest


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@lru_cache(maxsize=1)
def analysis_digest() -> str:
    root = Path(__file__).parent.parent
    files = sorted(
        [
            *root.joinpath("analysis").rglob("*.py"),
            *root.joinpath("assets").glob("*.py"),
        ]
    )
    identities = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    identities["python"] = platform.python_version()
    for name in (
        "polars",
        "numpy",
        "pydantic",
        "h5py",
        "mcap",
        "mcap-ros2-support",
        "av",
    ):
        try:
            identities["dependency:" + name] = version(name)
        except PackageNotFoundError:
            identities["dependency:" + name] = "not-installed"
    return content_digest(identities)


def detector_versions() -> dict[str, str]:
    from kalanos.analysis.metrics.registry import registered_metrics

    versions = {}
    for entry in registered_metrics():
        metric_id = f"{entry.family.value}.{entry.name}"
        versions[metric_id] = "unknown"
        if not entry.module.startswith("kalanos.analysis.metrics."):
            continue  # Third-party transitive implementation identity is unknown.
        try:
            module = inspect.getmodule(entry.func)
            if module is None:
                continue  # Missing implementation identity cannot authorize a block.
            body = inspect.getsource(module)
        except (OSError, TypeError):
            continue  # Missing implementation identity cannot authorize a block.
        versions[metric_id] = content_digest(
            {
                "analysis": analysis_digest(),
                "module": body,
                "function": entry.func.__qualname__,
            }
        )
    return versions


def adapter_versions(names: Iterable[str]) -> dict[str, str]:
    builtins = {
        "csv",
        "json",
        "jsonl",
        "delimited",
        "hdf5",
        "mcap",
        "lerobot_v2",
        "lerobot_v3",
    }
    return {
        name: analysis_digest() if name in builtins else "unknown" for name in names
    }
