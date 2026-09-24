# Adapters

The `Adapter` protocol, plugin discovery, selection by confidence, the contract suite, the `kalanos new adapter` scaffold, and eight built-in adapters all exist: `csv`, `delimited`, `json` and `jsonl` (text tables, all via `TabularAdapter`); `lerobot_v2` and `lerobot_v3` (LeRobot dataset directories); and `hdf5` and `mcap` (each behind its own extra). A third-party format installs as a package through the same entry points, with no privileged path. What is *not* yet wired is loading a single-file adapter from `--plugin` or the plugin directory — that remains the target shape, noted where it appears below.

An adapter teaches Kalanos to read a format. It is the main extension point, and writing one requires no change to Kalanos itself.

## What an adapter does

**An adapter reads files and produces episodes.** It does not apply thresholds, decide whether the data is good, or drop a field it fails to recognise. A column it cannot resolve becomes a stream typed `unmapped.<name>`, which reaches the report and tells the user a sensor was there. Drop it silently and the user loses a sensor without noticing.

## The protocol

```python
from upath import UPath


class Adapter(Protocol):
    name: str

    def detect(self, path: UPath) -> float:
        """How confident this adapter is that it can read the path, from 0 to 1."""

    def describe(self, path: UPath) -> DatasetInfo:
        """Facts available before reading: episode count, declared rate, robot type."""

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield episodes, lazily, stopping early when sample is given."""
```

`describe` exists so `kalanos inspect` can say what is in a dataset without paying to read it, and so facts the format declares reach the policy without being inferred. A declared frame rate is a better target for a rate metric than one derived from the timestamps being checked.

`episodes` yields rather than returns, because a dataset can be larger than memory and `--sample` has to be able to stop early.

## Confidence

Every adapter bids on a path and the highest bid wins. A tie is an error naming the tied adapters and the path, because reading a dataset with the wrong reader produces a report that is wrong in a way nothing downstream can detect.

What the bids mean:

| Situation | Bid |
|---|---|
| The format announces itself: a magic number, a required metadata file | 0.9 and up |
| The extension matches and the header parses as expected | 0.6 to 0.8 |
| It is readable as a table and nothing contradicts that | around 0.3 |
| Anything else | 0.0 |

A generic adapter that bids high will win paths that belong to a specific reader, and the dataset then gets graded on the wrong fields. If your adapter reads a broad container that other formats also use, bid low.

## Writing one

Start with the scaffold:

```bash
kalanos new adapter myformat
```

It writes a publishable package: the source layout, the entry-point declaration, an adapter, and a test that calls the contract suite. That test is green before you edit anything — the generated `parse` reads the generated CSV fixture, so the package installs and passes on the first run. Replacing those two, `parse` and the fixture, is the work.

For a table-shaped format, subclass `TabularAdapter` and write only the parse step. Everything after parsing (time axis, subject splitting, sampling regularity, and resolving field names to taxonomy types) is shared:

```python
class MyFormatAdapter(TabularAdapter):
    name = "myformat"

    def detect(self, path):
        return 0.9 if path.suffix == ".myf" else 0.0

    def parse(self, path):
        return ParsedTable(frame=pl.read_something(path))
```

`array_stems` defaults to empty; set it only when `parse` expanded a positional array into indexed columns, so `TabularAdapter` knows those indices — and only those — are eligible to be promoted to axis letters.

For a format that declares its own schema, implement `episodes` directly. Set the timestamps, the instance and the payload from what the format tells you, and still call `inference.roles()` for the field names: knowing a field is called `observation.state` is not the same as knowing it means joint position, and only `dictionary.yaml` knows that.

## Contract tests

```python
from kalanos.testing import check_adapter


def test_contract():
    check_adapter(
        MyFormatAdapter(),
        "fixtures/sample.myf",
        "fixtures/decoy.csv",
    )
```

Both sides are required: a fixture the adapter reads, and a fixture it must decline. That asserts the properties every adapter has to have: confidence stays in range and is positive on a path you claim; `describe` names this adapter and this path, and a declared `episode_count` agrees with a full read; `episodes` returns an iterator rather than a materialised list, and iterating twice yields the same episodes in the same order; `sample=n` yields at most n; every stream has monotonic timestamps, float and non-null, and a payload whose length matches them; every stream has a taxonomy type or an explicit `unmapped.*`; and a path you do not support bids zero and either yields nothing or raises `AdapterRefusal` when read anyway.

When the contract changes, every plugin finds out at once by failing this test.

## Registration

Declare an entry point:

```toml
[project.entry-points."kalanos.adapters"]
myformat = "kalanos_myformat:MyFormatAdapter"
```

Installing the package is enough. Kalanos finds it, and `kalanos adapters` lists it with the distribution it came from and whether it loaded:

```
$ kalanos adapters
NAME       FROM                      STATUS
csv        kalanos 0.1.0             ok
delimited  kalanos 0.1.0             ok
json       kalanos 0.1.0             ok
jsonl      kalanos 0.1.0             ok
lerobot_v2 kalanos 0.1.0             ok
lerobot_v3 kalanos 0.1.0             ok
hdf5       kalanos 0.1.0             FAILED: not installed — add kalanos[hdf5]
mcap       kalanos 0.1.0             FAILED: not installed — add kalanos[mcap]
myformat   kalanos-myformat 0.2.1    ok
zed        kalanos-zed 0.1.0         FAILED: ModuleNotFoundError: No module named 'pyzed'
```

`kalanos plugins` gives the same picture across adapters, metrics and reporters at once.

### Optional dependencies

`kalanos-zed` above is the ordinary case: the package declares `pyzed` as a dependency, so a missing one means a broken install, and the import error is what leads the reader to the fix.

An optional dependency is a different state: absent by design, one install away. The row can say that instead. Guard the import and name what provides it:

```python
from kalanos.analysis.entry_points import MissingDependency

try:
    import pyzed
except ImportError as exc:
    raise MissingDependency("kalanos-zed", "zed") from exc
```

```
zed        kalanos-zed 0.1.0         FAILED: not installed — add kalanos-zed[zed]
```

Usually what to install is an extra of your own distribution, as above. When you gate on a separate package instead, one you deliberately do not require, name that package alone:

```python
raise MissingDependency("rosbags") from exc
```

```
bag        kalanos-ros 0.3.0         FAILED: not installed — add rosbags
```

It is named at the import it guards because nowhere else can know it: resolving a module name back to the distribution that provides it needs that distribution's metadata, which an uninstalled one does not have. Kalanos' own optional adapters raise `MissingDependency("kalanos", …)` through the same path.

During development, before there is a package, a single file will load — this is the target shape; `--plugin` and plugin-directory loading are not yet wired into a run:

```bash
kalanos grade data/ --plugin ./my_adapter.py
```

or drop it in the user plugin directory: `platformdirs`' user config directory for `kalanos`, plus a `plugins` subdirectory. On Linux that's `$XDG_CONFIG_HOME/kalanos/plugins`, falling back to `~/.config/kalanos/plugins` when `XDG_CONFIG_HOME` is unset; other platforms use their own convention. Overridable with `KALANOS_PLUGIN_DIR`.

Loading a plugin file executes Python from a path you named, as the user running `kalanos`. A plugin file is trusted code, like anything you'd `pip install`: do not point `--plugin` or the plugin directory at a file you would not install.

A plugin that fails to import is reported and skipped. One broken third-party package does not take down a run that did not need it.