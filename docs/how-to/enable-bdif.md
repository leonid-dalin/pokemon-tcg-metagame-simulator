# Enable Limitless and BDIF analytics

BDIF card-model reporting is opt-in. The default configuration keeps the ordinary matchup workflow active and does not fetch or fit the Limitless card model

## Enable ingestion

Set `LIMITLESS_INGESTION_ENABLED = True` in `src/core/config.py` for a controlled ingestion run. Provide `LIMITLESS_API_KEY` through the process environment. The client sends it as the `X-Access-Key` header

Do not put the key in a URL, query parameter, snapshot, log, or committed file

The ingestion task reads up to `LIMITLESS_BACKFILL_TOURNAMENTS` tournaments, which defaults to 200. It writes:

- `data/limitless.db`
- `data/input/limitless_input.json`
- `data/input/limitless_model_input.json` when the matchup observations meet the model requirements

Run the task through the worker entry point after the worker environment has access to the same data volume and credentials

## Enable the card model

Set `BDIF_USE_CARD_MODEL = True` after the model artefact exists. The worker then:

1. Reads the model input instead of the baseline input when the artefact exists
2. Selects panel decks using a 3% empirical share threshold, capped at 10 decks
3. Builds Best-60 recommendations from observed legal skeletons
4. Adds the H1 Mist Energy report when Alakazam observations are available

The report is observational. Card inclusion is not a causal estimate of card value

## Data requirements

The store resolves canonical deck names before readers query standings, matchup rows, deck weights, or decklists. Legacy stores are prepared on first read

Decklists stored as the JSON string `"null"` are treated as empty decklists. New writes use SQL `NULL` for missing decklists

The model requires usable matchup observations. When evidence is too sparse, the worker returns an insufficient-data result for the affected recommendation rather than inventing a list

## Verify a run

Check that these files exist and inspect their parsed contents before reporting a result:

```bash
python -c "import json; print(json.load(open('data/input/limitless_model_input.json', encoding='utf-8')).keys())"
```

Keep the source snapshot and its date with any report. Separate sourced 60-card decklists from model inference
