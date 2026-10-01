# Enable Limitless and BDIF analytics

BDIF card-model reporting is opt-in. By default, ordinary matchup simulation remains active, Limitless ingestion is disabled, and the card model is not fitted

## Configure the environment

Set boolean controls to `1`, `true`, `yes`, `on`, `0`, `false`, `no`, or `off`, case-insensitively

Assumed: these example settings were not applied to a live ingestion environment

```bash
export LIMITLESS_INGESTION_ENABLED=true
export BDIF_USE_CARD_MODEL=true
export BDIF_DB_PATH=output/bdif/limitless.db
export BDIF_ARTIFACT_DIR=output/bdif
export LIMITLESS_API_KEY=replace-with-a-secret-from-your-secret-store
```

`BDIF_DB_PATH` defaults to `data/limitless.db`. `BDIF_ARTIFACT_DIR` defaults to `data/input`. Use scratch paths in an isolated run so trial ingestion cannot replace the artefacts used by a live dashboard. Compose passes both paths from the host to `api`, `worker`, and `ui`. The default generated artefacts are ignored by Git at `data/input`; check `.gitignore` before writing them elsewhere. The client sends `LIMITLESS_API_KEY` in the `X-Access-Key` header. Keep the key out of URLs, query parameters, snapshots, logs, and committed files

The dashboard, API worker, and CLI select the same simulation input: the card-model artefact when `BDIF_USE_CARD_MODEL` is enabled and that artefact exists, otherwise `data/input/ea_input.json`. Compose passes `BDIF_USE_CARD_MODEL`, `LIMITLESS_INGESTION_ENABLED`, `BDIF_DB_PATH`, and `BDIF_ARTIFACT_DIR` from the host to `api`, `worker`, and `ui`. If you set these values per service, keep them equal or the worker rejects a request whose matrix differs from its simulation input

## Ingest data

Run bounded ingestion after setting the credential and scratch paths

Assumed: with ingestion enabled, the command requests Limitless event data; this example was not run against a live service

```bash
python -m src.bdif ingest --limit 20
```

`--limit` accepts 1 to 1000 tournaments. Without it, the command uses `LIMITLESS_BACKFILL_TOURNAMENTS`, which defaults to 200. The command writes `limitless_input.json` to `BDIF_ARTIFACT_DIR` and writes `limitless_model_input.json` there when the observations identify the card model. The result status is `complete`, `partial` when some events fail, or `failed` when every event fails

## Check local status

The status command opens the database read-only and does not migrate it

Verified: the offline probe ran this command with a temporary database path and confirmed status `missing` without creating the database

```bash
python -m src.bdif status
```

The result reports the database path, both feature switches, and whether the model artefact exists. For an existing database, `schema` is `current` with deck, pairing, and decklist counts, `legacy` when the `standings.deck_name` column is absent, or `incomplete` when the `standings` or `pairings` table is missing. A later `ingest`, `refit`, or card-model `report` prepares a legacy database

## Refit the card model

Refit using observations already stored in the configured database


```bash
python -m src.bdif refit
```

When the observations are sufficient, the command writes `limitless_model_input.json` to `BDIF_ARTIFACT_DIR` and lists `card_packages`, the cards fitted together, and `not_identified`, the cards that got no coefficient. A missing database returns status `missing` and is not created

## Generate a report

Verified: the offline probe ran this command with scratch output and confirmed the report file was written

```bash
python -m src.bdif report --output output/bdif
```

The report writes one JSON result to stdout and to `bdif_report.json` in the output directory. `--panel` accepts up to 10 unique deck names from the selected input matrix

Verified: the offline probe ran this command with scratch output and returned both panel rows with no unmatched decks

```bash
python -m src.bdif report --panel "Crustle,N's Zoroark" --output output/bdif
```

The report includes posterior matchup results, field-posterior metrics, card recommendations, H1 output, and provenance when the corresponding evidence is available

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | A result was produced |
| `1` | The command failed and logged the error to stderr |
| `2` | The arguments were invalid |
| `3` | No evidence was produced: ingestion was disabled or every event failed, refit found no database or identifiable evidence, or the report ranked no deck |

## Verify a run

Check local status and list the keys in the ingestion artefact without contacting Limitless. The second command needs a completed ingestion

Verified: the offline probe ran this command before ingestion and observed exit 1 because `limitless_input.json` was absent

```bash
python -c "import json, os; print(sorted(json.load(open(os.path.join(os.environ.get('BDIF_ARTIFACT_DIR', 'data/input'), 'limitless_input.json'), encoding='utf-8'))))"
```

Keep the source snapshot and its date with each report. Card inclusion and fitted coefficients are observational associations, not causal estimates