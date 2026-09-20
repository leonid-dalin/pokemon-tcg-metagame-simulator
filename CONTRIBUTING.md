# Contributing

## Before you start

Read the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md). Use the issue templates for bug reports, feature requests, and data corrections. Do not disclose vulnerabilities in public issues.

## Development setup

The supported local verification path uses Python 3.11 and a clean environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
```

Docker Compose is the supported way to run the complete API, worker, Redis, and UI stack:

```bash
docker compose up -d --build
```

## Changes

Keep changes focused and preserve existing data contracts, formulas, source evidence, and deck-page behaviour. Add or update tests for behavioural changes. Use the repository's existing formatting and naming conventions.

For data or scraping changes, include the source URL or fixture that supports the change and explain how the generated data was verified.

## Pull requests

Open a pull request against `main` with:

- a short summary of the change
- the tests and other checks you ran
- relevant issue links
- any known limitations or follow-up work

Keep commits small enough to review. Do not include secrets, credentials, generated local environments, or unrelated formatting changes.

## Review

Changes require review from the repository owners listed in `CODEOWNERS`. Address review findings with a follow-up commit and keep the pull request description focused on public project information
