# Enable Limitless and BDIF analytics

BDIF card-model reporting is opt-in. The default configuration keeps the ordinary matchup workflow active, leaves Limitless ingestion disabled, and does not fit the card model

## Configure the environment

Set boolean controls with one of `1`, `true`, `yes`, `on`, `0`, `false`, `no`, or `off`, case-insensitively

```bash
export LIMITLESS_INGESTION_ENABLED=true
export BDIF_USE_CARD_MODEL=true
export BDIF_DB_PATH=output/bdif/limitless.db
export LIMITLESS_API_KEY=replace-with-a-secret-from-your-secret-store
```

`BDIF_DB_PATH` defaults to `data/limitless.db`. Use a scratch path for controlled runs. `LIMITLESS_API_KEY` is sent in the `X-Access-Key` header. Do not put the key in a URL, query parameter, snapshot, log, or committed file

## Ingest data

Run the bounded BDIF ingestion command after the environment has access to the credential and the selected database path

```bash
python -m src.bdif ingest
```

The command reads up to `LIMITLESS_BACKFILL_TOURNAMENTS` tournaments, which defaults to 200. It writes the ingestion artefact to `data/input/limitless_input.json` and writes a fitted card-model artefact to `data/input/limitless_model_input.json` when the observations identify the model

## Check local status

Use the read-only status command before fitting or reporting

```bash
python -m src.bdif status
```

The result reports whether the configured database exists, its path, the number of stored decks, and the number of player observations

## Refit the card model

Refit from the observations already stored in the configured database

```bash
python -m src.bdif refit
```

The command writes `data/input/limitless_model_input.json` when the observations are sufficient and identifiable

## Generate a report

Generate the BDIF report from the configured input data

```bash
python -m src.bdif report --output output/bdif
```

Select a panel explicitly with a comma-separated list of deck names. The request accepts at most 10 unique names that exist in the input matrix

```bash
python -m src.bdif report --panel "Crustle,N's Zoroark" --output output/bdif
```

The report includes posterior matchup results, field-posterior metrics, card recommendations, H1 output, and provenance when the corresponding evidence is available. A report that lacks required evidence exits with status 3

## Verify a run

Check local inputs and the CLI status without contacting Limitless

```bash
python -m src.bdif status
python -c "import json; print(json.load(open('data/input/limitless_input.json', encoding='utf-8')).keys())"
```

Keep the source snapshot and its date with any report. Treat card inclusion and fitted coefficients as observational associations, not causal estimates