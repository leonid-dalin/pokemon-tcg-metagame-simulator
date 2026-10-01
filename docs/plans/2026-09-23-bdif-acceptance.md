# BDIF acceptance record

## Scope

This record covers local documentation and offline acceptance of the BDIF CLI, posterior analytics, card model, and runtime contracts

## Live Limitless acceptance

**Status: NOT RUN**

Live acceptance requires written approval from the repository owner, Leonid Dalin, in the PR thread. It also requires access to the operator-managed Limitless credential. No live request is authorised by this record

The following exact invocations were not run because their default paths require written live-run approval. The offline probe ran the same commands against scratch paths

Assumed: these listed invocations describe the approved live procedure; no live database, credential, or Limitless response was checked

```bash
python -m src.bdif ingest --limit 20
python -m src.bdif status
python -m src.bdif refit
python -m src.bdif report --output output/acceptance
```

No Limitless ingestion or third-party request was made. Do not use `data/limitless.db` for an acceptance run

## Procedure when approved

Record the approver and date before the first request. Use the project interpreter and scratch paths from the repository root

Assumed: the live command sequence was not run. The offline probe used a temporary database and artefact directory

```bash
mkdir -p output/acceptance
export BDIF_DB_PATH=output/acceptance/limitless.db
export BDIF_ARTIFACT_DIR=output/acceptance
export LIMITLESS_INGESTION_ENABLED=true
export BDIF_USE_CARD_MODEL=true
python -m src.bdif ingest --limit 20 > output/acceptance/ingest.json
python -m src.bdif status > output/acceptance/status.json
python -m src.bdif refit > output/acceptance/refit.json
python -m src.bdif report --output output/acceptance > output/acceptance/report-stdout.json
```

Stop at the first HTTP 4xx other than 429 and do not retry it. Stop any command that runs longer than 15 minutes. Never point `BDIF_DB_PATH` at `data/limitless.db`

## Record

Copy these values from the files produced by the approved run:

- `ingest.json`: `status`, `events`, `skipped_events`, failed-event count, `unmapped_deck_ids`, and `model_status`
- `status.json`: `schema`, `decks`, `pairings`, and `decklists`
- `refit.json`: `status` and the exit code when it is 3
- `bdif_report.json`: `mc_results.provenance`, `mc_results.posterior.interval_status`, the three highest `mc_results.field_posterior.expected_win_rate` rows, and every `_mc_share` above 0.8 in `mc_results.metrics`
- Exit code for each command

## Local verification

The documentation command probe ran with a temporary database and artefact directory. It made no network requests

| Check | Result |
| --- | --- |
| `python -m src.bdif status` without a database | Exit 0; status `missing`; database not created |
| `python -m src.bdif refit` without a database | Exit 3; database not created |
| `python -m src.bdif ingest` with ingestion disabled | Exit 3; status `disabled` |
| `ingest --limit 0` and `ingest --limit 1001` | Both exit 2 |
| `report --output` | Exit 0; writes `bdif_report.json` |
| `report --panel` | Exit 0; both requested rows returned; unmatched list empty |
| `python -m src.ui.cli --bdif-status` | Exit 2 |
| Status help, report help, and UI CLI help | Exit 0 |