"""Load one report in this process and print the time and peak memory it took.

Run as `python -m scripts.report_size.probe PATH`. Each load gets a fresh process,
so its peak resident memory is its own. `tracemalloc` would miss pydantic-core's Rust
allocations.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import resource
import sys
import time

# Internal
from kalanos.analysis.models.legacy import load_any


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main(path: str) -> None:
    """Load the report at `path` and print the seconds and peak bytes as JSON.

    Parameters
    ----------
    path : str
        A report JSON, full or compact.
    """

    start = time.perf_counter()
    load_any(path)
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KiB on Linux and in bytes on macOS.
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    print(json.dumps({"seconds": seconds, "peak_rss_bytes": peak_bytes}))


if __name__ == "__main__":
    main(sys.argv[1])
