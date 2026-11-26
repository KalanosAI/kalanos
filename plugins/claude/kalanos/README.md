# Kalanos

Kalanos helps robotics teams inspect datasets before training. Look up a public
dataset, see which checks ran, and follow findings to the episodes they affect.
The bundled connector reads published assessments; it does not start new
grading jobs, publish reports, or upload recordings. No Kalanos account or API
key is required to try the public examples.

## Install

In Claude Code: `/plugin marketplace add KalanosAI/kalanos`, then
`/plugin install kalanos@kalanos-ai`.

## Try it

- What does the published audit for `lerobot/pusht` say, and what wasn't checked?
- Find public SO-100 datasets with a published assessment. Show episode counts
  where available and link the reports.
- For `triton7777/cube_to_bowl_so100`, which episodes should I inspect first,
  and what evidence supports that choice?

Local audits require Claude Code, a local workspace, and an existing Kalanos
CLI installation. Use the [Grader quickstart](https://kalanos.ai/grader) to
set up the CLI separately. Local audit inputs and outputs stay in the local
workflow; the plugin does not send them to Kalanos MCP. The hosted connector
logs public search terms, dataset identifiers, and request metadata as described
in the [Kalanos privacy notice](https://kalanos.ai/privacy).

If a lookup returns no report, check the exact public dataset identifier. If
the connector is unavailable, see [the MCP setup guide](https://kalanos.ai/mcp)
or [open a support issue](https://github.com/KalanosAI/kalanos/issues). Do not
include private data or credentials in a public issue.
