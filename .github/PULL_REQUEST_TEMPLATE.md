## Before opening this PR

- [ ] `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest` all pass
- [ ] `uv run pytest -m integration` passes, if this touches packaging, entry points, or dependencies
- [ ] `docs/` is updated, or this change needs none
- [ ] If this adds a metric: a policy entry and a contract test are included

## What does this PR do?

Write a brief description of the changes introduced by this PR. Include any relevant context or background information.

- Bug: Describe the bug being fixed and how the changes address it.
- Feature: Describe the new feature being added and its benefits.
- Refactor: Describe the code changes and the reasons for the refactor.

For more complex PRs, consider breaking down the description into smaller, digestible sections.

All sections but this one are optional, if you don't have anything to add for a section, feel free to omit it.

## Changes Outline

- `example.py` - line X-Y
- `another_file.py` - line A-B

Don't simply list the files changed; provide a brief summary of the changes made in each file.

If the PR includes multiple changes across different files:

- Consider grouping related changes together for better readability.
- List the files in a logical order (e.g., by feature or by the flow of the code).

## How do I feel about this change?

- How risky is this PR? Downstream impact, time to deploy, dependencies, etc.
- What should the reviewer pay closest attention to?

Examples:

- Low: Minimal changes; net new isolated features with no downstream dependencies.
- Medium: Changes to existing code with some downstream dependencies; requires testing.
- High: Significant changes to existing code with many downstream dependencies; requires thorough testing and review.

## What should the reviewer test?

- Explain how the reviewer can verify that the changes work as intended.
- What is the easiest way to verify changes?
- Any edge cases to consider?

## Screenshots

Attach any relevant screenshots here — a rendered HTML report, or the visual behaviour of a bug being fixed. Include a label for each screenshot to explain what it shows.
