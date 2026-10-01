# BDIF acceptance record

## Scope

This record covers local documentation and offline acceptance of the BDIF CLI, posterior analytics, card model, and runtime contracts

## Live Limitless acceptance

**Status: RUN on 2026-10-01**

Before an approval is recorded below, no live request is authorised by this record

The following commands were not run before the approval recorded below. The earlier offline probe used a temporary database and artefact directory

Assumed: these invocations described the approved live procedure; no live database, credential, or Limitless response was checked before the approval

```bash
python -m src.bdif ingest --limit 20
python -m src.bdif status
python -m src.bdif refit
python -m src.bdif report --output output/acceptance
```

Before the approved run, no Limitless ingestion or third-party request was made. Do not use `data/limitless.db` for an acceptance run

## Procedure when approved

Approval: Leonid Dalin, direct Telegram approval on 2026-10-01. The owner explicitly approved the live acceptance run and overrode the request to post approval in the merged PR #35 thread

Use the project interpreter and scratch paths from the repository root

The approved live command sequence ran once. The offline probe used a temporary database and artefact directory

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

The live command outputs remain outside the repository under `C:\\Users\\DALIN\\AppData\\Local\\Temp\\BDIF-0930\\t21-acceptance`

- `ingest.json`: `status=complete`; `events=20`; `skipped_events=0`; failed-event count `0`; `unmapped_deck_ids=["other"]`; `model_status=not identifiable`; exit code `0`
- `status.json`: `schema=current`; `decks=81`; `pairings=2230`; `decklists=970`; `db_path=output/acceptance/limitless.db`; exit code `0`
- `refit.json`: `status=not identifiable`; exit code `3`
- `bdif_report.json`: `mc_results.provenance.input_path=data/input/ea_input.json`; `input_sha256=2e6fcb89fc16beed547aca8d357f77292f638a56a212acb9ae7636174a3c7376`; `posterior.interval_status=ok`; highest `expected_win_rate` rows are `Ogerpon Meganium Arboliva=0.5785260556997609`, `Dragapult Dudunsparce=0.5714665866669941`, and `Sinistcha Ogerpon=0.5668147594928132`; the report contains 300 nested `*_mc_share` values, 110 above `0.8`, and a maximum of `1.0`; report exit code `0`
- `report-stdout.json` matched the report file apart from the final newline

The scratch directory contains the database, input artefact, report artefact and command outputs. The credential scan found no key value in any output. `data/limitless.db` was absent and no tracked or untracked file under `data/` changed

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

## Wailord matchup spike

Approval: Leonid Dalin, 2026-10-01T10:06:08Z, [PR #35 comment](https://github.com/leonid-dalin/pokemon-tcg-metagame-simulator/pull/35#issuecomment-5929201677). The comment states: "I approve T20"

Scope: fetch the Wailord matchup page and the pages for Dragapult Blaziken and Dhelmise, one request at a time, with at least two seconds between requests. The request budget is 20 pages. Save response bodies outside the repository under `%TEMP%\\wailord-spike\\`. Stop at the first HTTP 4xx other than 429 and do not retry it

Result: outcome (c), the current source data disagrees with the shipped matchup counts. The local artefact records Wailord versus Dragapult Blaziken as 25 matches and versus Dhelmise as 15, while the reverse rows contain 11 and 4 matches. The approved spike made four requests: one eligible-deck index request and three page requests. All three pages returned HTTP 200 and were saved as `wailord.html`, `dragapult_blaziken.html`, and `dhelmise.html` under `C:\\Users\\DALIN\\AppData\\Local\\Temp\\wailord-spike\\`

The Wailord URL was `https://play.limitlesstcg.com/decks/wailord-ex-pbl/matchups?format=standard&rotation=2026&set=30C`. Its table still has class `striped`, but it contains no parsed matchup records and no rows for Dragapult Blaziken or Dhelmise. The opponent pages also have `striped` tables. Dragapult Blaziken contains one Wailord row with `data-name="Wailord"`, `data-matches="1"`, and fourth-cell record `1 - 0 - 0`; Dhelmise contains no Wailord row. The parser returned the same result from the saved pages, so the evidence does not show a scraper parsing defect

Evidence: `C:\\Users\\DALIN\\AppData\\Local\\Temp\\wailord-spike\\comparison.json` contains the URLs, response sizes, parsed records and raw row checks. No database, artefact or repository data file was changed
