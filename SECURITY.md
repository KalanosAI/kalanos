# Security Policy

## Supported versions

Kalanos is pre-1.0 and moves quickly. Security fixes land on `main` and in the
next release; only the latest release is supported.

| Version | Supported |
|---|---|
| latest release / `main` | :white_check_mark: |
| anything older | :x: |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting: on the **Security** tab of this
repository, choose **Report a vulnerability** (Security → Advisories → Report a
vulnerability). That opens a channel visible only to the maintainers.

Where you can, include:

- what the vulnerability is and the impact you expect,
- the version or commit you observed it on,
- steps or a proof of concept to reproduce it.

## What to expect

- We aim to acknowledge a report within a few business days.
- We will work with you on a fix and coordinate a disclosure timeline; please
  allow reasonable time to release a fix before disclosing publicly.
- Responsible disclosure is welcome, and we are glad to credit reporters who
  want it.

## Scope

Kalanos reads untrusted data files, and it can load third-party plugins named on
the command line (`--plugin`) or dropped in the plugin directory. A plugin is
trusted code, like anything you `pip install` — loading one executes its Python.

- **In scope:** parsing untrusted *data* — a crafted recording that causes a
  crash, resource exhaustion, or code execution during grading.
- **Out of scope:** a plugin you chose to install behaving maliciously; that is
  the same trust boundary as installing any package.
