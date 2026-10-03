"""Opt-in complete local byte identity; never follow hidden symlink targets."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import os
from pathlib import Path
from typing import Any

# Internal
from kalanos.analysis.models.errors import SourceUnavailable
from kalanos.analysis.models.provenance import HashScope, SourceEvidence, content_digest


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

CANONICALIZATION = "sorted-relative-path-content-sha256-v1"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def hash_local_source(root: Any) -> SourceEvidence:
    if getattr(root, "protocol", "") not in ("", "file", "local"):
        raise SourceUnavailable("complete hashing requires a materialized local source")
    path = Path(root)

    def fail(error: OSError) -> None:
        raise error

    try:
        if path.is_symlink():
            raise OSError("source hashing refuses symlinks")
        files = []
        if path.is_file():
            files = [path]
        else:
            for directory, dirs, names in os.walk(
                path, onerror=fail, followlinks=False
            ):
                for name in dirs + names:
                    if (Path(directory) / name).is_symlink():
                        raise OSError("source hashing refuses symlinks")
                files.extend(Path(directory) / name for name in names)
        entries = []
        for file in sorted(files):
            if not file.is_file():
                raise OSError(f"not a regular file: {file}")
            before = file.stat()
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = file.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise OSError("source changed while hashing")
            entries.append(
                (
                    str(file.relative_to(path)) if path.is_dir() else file.name,
                    digest.hexdigest(),
                )
            )
        if not files:
            raise OSError("source contains no regular files to hash")
        return SourceEvidence(
            scope=HashScope.COMPLETE,
            canonicalization=CANONICALIZATION,
            digest=content_digest(entries),
            covered_inputs=len(entries),
            complete=True,
        )
    except OSError as exc:
        raise SourceUnavailable(f"cannot establish source identity: {exc}") from exc
