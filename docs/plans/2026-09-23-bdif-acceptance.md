# BDIF acceptance record

## Scope

This record covers the documentation and acceptance work for the merged BDIF CLI, posterior analytics, card model, and Tasks 11-15 UI changes

## Live Limitless acceptance

**Status: NOT RUN**

The live acceptance run requires written approval from the repository owner, Leonid Dalin, in the PR thread and access to the owner's credentials. No written approval or credentials were provided for this task, so the live run was not started

The following commands were not run:

```bash
python -m src.bdif ingest --limit 20 --db output/acceptance/limitless.db --json > output/acceptance/ingest.json
python -m src.bdif status --db output/acceptance/limitless.db --json > output/acceptance/status.json
python -m src.bdif fit --db output/acceptance/limitless.db --json > output/acceptance/fit.json
python -m src.bdif report --db output/acceptance/limitless.db --card-model --precision-tier 3 --output output/acceptance --json > output/acceptance/report.json
```

The planned bootstrap gate was not run. No Limitless ingestion, third-party network call, Docker command, or command that could expose a credential was run

## Required record when approved

After written approval, record the approver and date before starting. Use a scratch database, never `data/limitless.db`. Stop at the first HTTP 4xx other than 429 and do not retry. Stop any command that exceeds 15 minutes

Record the following verbatim from the approved run:

- Event and failure counts
- `model_status`
- Selected-card count
- Observation count
- The report `provenance` block
- `posterior.interval_status`
- The top three `field_posterior` rows
- The reason if `fit` exits 3
- The bootstrap-gate output, including any ratio outside `[0.5, 2.0]`

## Local verification

The local CLI and static verification results for this documentation task are recorded below

| Check | Result |
| --- | --- |
| Branch guard | Passed: `feat/bdif-ui` |
| Parent-change guard | Passed: Tasks 11-15 source and test changes were present before editing |
| `python -m src.bdif --help` | Passed; command and subcommands are present |
| `python -m src.bdif status` | Not run; live acceptance and database access were intentionally avoided |
| Config constant probe | Passed; values were `200 100 50 2000 50 0.05 0.95 5.0` |
| Forbidden config grep | Passed; no documentation instructs enabling ingestion through a source-code assignment |
| Documentation scope | Passed; only the five named existing pages and this acceptance record were changed |

No source files, tests, or commits were changed by this task