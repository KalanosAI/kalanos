# Contributing

## Getting an environment

Install [uv](https://docs.astral.sh/uv/), then `uv sync` creates `.venv` and installs every dependency, locked by `uv.lock`. Prefix commands with `uv run` to use that environment — `uv run kalanos --help`, `uv run pytest`.

## Before you send a patch

Run the same gate CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

If you touched packaging, entry points, or dependencies, also build the wheel and exercise the installed CLI:

```bash
uv run pytest -m integration
```

## Where the design lives

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first: the domain model, the pipeline, and the decisions behind both. [docs/ADAPTERS.md](docs/ADAPTERS.md) is where to go to add a format. [docs/METRICS.md](docs/METRICS.md) is where to go to add a metric.

## Conventions a patch is judged against

- Every function and method gets a [numpydoc-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstring.
- **polars**, not pandas, for dataframes.
- **Pydantic** models at every cross-stage boundary and every YAML config.
- **uv** for dependencies: `uv add`, never a hand-edited lockfile or requirements file.
- Line width belongs to `ruff format`. Don't hand-wrap code, comments, or docstrings at a column.
- Commit messages open with a capitalized imperative sentence, no trailing period; a leading gitmoji is welcome, not required.

**Adding a metric is one decorated function, one policy entry, one contract test.** If a change needs more than that, say so in the pull request rather than working around it.

The values `docs/METRICS.md` marks *to define* stay absent rather than guessed at. Settling one takes real recordings to measure against, not a plausible-looking number.

## Licensing

Contributions come in under the Apache-2.0 license this project is released under.
