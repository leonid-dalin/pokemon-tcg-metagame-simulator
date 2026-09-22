# Contributing

## Before you start

Read the [Code of Conduct](CODE_OF_CONDUCT.md), [Security Policy](SECURITY.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Use the issue templates for bug reports, feature requests, and data corrections. Do not disclose vulnerabilities in public issues

## Development setup

The CI workflow uses Python 3.12 and builds the Rust extension before running tests. Reproduce that environment locally:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m maturin build --release --manifest-path src/tournament/tcg_engine/Cargo.toml
python -m pip install src/tournament/tcg_engine/target/wheels/*.whl
```

Run the complete test suite:

```bash
python -m pytest -q
```

Docker Compose runs the complete API, worker, Redis, Jaeger, and UI stack:

```bash
docker compose up -d --build
docker compose down
```

## Changes

Keep changes focused and preserve existing data contracts, formulas, source evidence, and deck-page behaviour. Add or update tests for behavioural changes. Use the repository's existing formatting and naming conventions

For data or scraping changes, include the source URL or fixture that supports the change and explain how the generated data was verified. Do not commit API keys, tokens, local databases, virtual environments, caches, or generated output

## Pull requests

Open a pull request against `main` with:

- a short summary of the change
- the tests and other checks you ran
- relevant issue links
- known limitations or follow-up work

Keep commits small enough to review. Separate behaviour changes from documentation-only changes when both are part of a larger task

## Review

Changes require review from the repository owners listed in `CODEOWNERS`. Address review findings with a follow-up commit and keep the pull request description focused on public project information

## Licence

Contributions to the project's own code are accepted under the [GNU AGPL-3.0-or-later](LICENSE). Third-party libraries, data, and other externally sourced material keep their separate terms. Check [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before adding a dependency or data source
