# Kalanos

Look up published audits for public robotics datasets. Find which checks ran,
which episodes need review, and the measurements behind each finding.

## Try it

- Is there a published audit for `lerobot/pusht`? What was and wasn't checked?
- Find public SO-100 datasets with published assessments and link the reports.
- Which episodes in `triton7777/cube_to_bowl_so100` should I inspect first, and
  what evidence supports that?

## What this plugin accesses

The Directory plugin uses the read-only Kalanos MCP server. You do not need
a Kalanos account or API key for public report lookups. It cannot start new
audits or publish reports, and it does not run commands, access local files,
or upload recordings.

The hosted service may log public search terms, dataset IDs, IP addresses,
and other request metadata as described in the [Kalanos Privacy Policy](https://kalanos.ai/privacy).

## Grade your own recordings

Follow the [Kalanos Grader quickstart](https://kalanos.ai/grader) and run the
local CLI yourself in a terminal. This workflow is separate from the Directory
plugin. For help with public report lookups, see the [MCP guide](https://kalanos.ai/mcp)
or [open a GitHub issue](https://github.com/KalanosAI/kalanos/issues).
