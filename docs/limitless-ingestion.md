# Limitless ingestion design

## Scope and status

The importer targets the Pokémon TCG (`PTCG`) Standard format, whose decklists contain 60 cards. It does not ingest Pokémon TCG Pocket data. Ingestion and statistical Best-60 selection are opt-in; both `LIMITLESS_INGESTION_ENABLED` and `BDIF_USE_CARD_MODEL` default to `False`.

Event ingestion and reporting remain blocked when supported event IDs cannot be resolved. Tournament IDs are required to fetch details, standings, and pairings; do not treat unresolved IDs as a usable event source. No live API call is required to inspect or report on already stored local data.

## Limitless API

The client uses `https://play.limitlesstcg.com/api`, sends `Accept: application/json`, and, when `LIMITLESS_API_KEY` is set, sends the key in the `X-Access-Key` header. It does not put credentials in query parameters.

| Request | Shape and use |
| --- | --- |
| `GET /tournaments` | Query parameters are passed through. The importer filters with `game=PTCG` and `format=STANDARD`; pagination uses `limit=50&page=1`, incrementing `page` until the requested limit or a short page. The configured backfill limit defaults to 200. |
| `GET /tournaments/{event_id}/details` | One event details object. |
| `GET /tournaments/{event_id}/standings` | Standings list; fetched only when `details.decklists` is truthy or absent. |
| `GET /tournaments/{event_id}/pairings` | Pairings list. |
| `GET /games/PTCG/decks` | Game deck catalogue, used to resolve deck identifiers to display names. |

List endpoints must return JSON lists and details must return a JSON object. The client retries one HTTP 429 response, honours `Retry-After` up to 60 seconds, applies a one-second minimum request delay, and rejects responses over 10 MB.

## Storage and artefacts

`BDIF_DB_PATH` selects the SQLite database and defaults to `data/limitless.db`. The store creates parent directories and maintains three tables:

| Table | Stored fields |
| --- | --- |
| `tournaments` | Event `id` primary key; game, format, name, date, player count, and raw details JSON. |
| `standings` | Event and player composite primary key; placing, record, deck ID and canonical name, decklist JSON, and drop round. |
| `pairings` | Event, round, phase, and player IDs composite primary key; winner ID. |

The store can add a missing `standings.deck_name` column to an older database when a writing command runs; `status` reads the database without changing it. It retains event responses and decklists as JSON text; aggregate calculations derive matchup and card observations from those records.

The ingestion aggregate is written atomically to `limitless_input.json` in `BDIF_ARTIFACT_DIR`, which defaults to `data/input`. A fitted model artefact, when identifiable, is written to `limitless_model_input.json` in the same directory. The normal simulation input remains `data/input/ea_input.json`. The BDIF settings select the baseline input unless statistical Best-60 selection is enabled and the model artefact exists. If the flag is enabled but the artefact is missing, the service logs `card_model_artifact_missing` with the expected path and uses the baseline input. It never substitutes an empty or partially fitted model.

## Archetype mapping

`data/input/limitless_archetype_map.json` explicitly maps Limitless deck IDs to the simulator's canonical archetype names. For example, `dragapult-ex` maps to `Dragapult`, `n-zoroark` to `N's Zoroark`, and `alakazam-dudunsparce` to `Alakazam Dudunsparce`. A JSON `null` value deliberately leaves an ID unmapped; `other` also resolves to no archetype.

For a known ID, the explicit mapping wins over any API display name. An unknown ID can use its non-empty display name, but an empty or absent display name remains unmapped. Unmapped rows have no canonical deck name and therefore do not enter matchup or card-model observations. The ingestion result reports and logs unresolved deck IDs; resolve them in the mapping file rather than silently treating them as a canonical archetype.

## Card model and evidence

The model uses one binary observation per completed pairing with two resolved archetypes and two usable decklists. Its logistic-regression design is the difference between the two players' archetype indicator vectors, followed by card-presence differences (`card in deck 1` minus `card in deck 2`). It has no fitted intercept. Card coefficients estimate log-odds associations conditional on the included archetype and card covariates, not causal effects.

Card selection requires at least 50 player appearances for a card in an archetype and a within-archetype inclusion rate from 0.05 through 0.95, inclusive. Fitting needs at least two archetypes, both match outcomes, and at least one selected card. A rank-deficient design, non-convergent fit, or absolute fitted coefficient above 5 makes the model unidentifiable. The ingestion service also requires at least four complete observations with both win and loss outcomes before fitting. Standard errors are model-based unless player IDs exist for every observation; then the model uses player-clustered errors, bounded below by model-based errors. Coefficient intervals are 95% intervals.

For Best-60, each card's contribution is its coefficient multiplied by the archetype's inclusion rate minus the meta-weighted field inclusion rate. The interval uses the same delta and coefficient interval. A card is a signal only when its interval excludes zero and its Benjamini-Hochberg adjusted q-value is at most 0.05. Other cards are reported as no signal. No signal means there is insufficient evidence for a directional recommendation, not evidence of no effect.

## Best-60 constraints

Best-60 is an observational recommendation built from observed lists. It is not a legality database, and it does not establish that every resulting list is tournament-ready. Its starting skeleton is the modal Pokémon and Energy core, without ACE SPEC cards, shared by at least three quarters of the archetype's legal 60-card lists. When no core reaches that share, the result has status `missing observed skeleton`. Skeleton cards are capped before selection, and banned cards are removed. Candidate additions must be both observed for that archetype and marked playable. The selector adds supported signal cards, then other playable observed cards ordered by archetype inclusion. It can fill remaining slots with basic Energy, which has no per-card copy cap.

Legality checks require exactly 60 cards, at most four copies of a regular card, unlimited basic Energy, and at most one ACE SPEC card. The ACE SPEC names are enumerated in `src/ingestion/model.py`; supply current card rules and banned-card data when calling the recommendation API, since the model does not fetch a current legality list. Banned cards are excluded during selection and rejected by validation if present. If no observed skeleton exists, the result has status `missing observed skeleton`; if legal playable observed cards cannot fill 60, the result has status `insufficient legal observed cards to complete 60`. Neither status should be presented as a complete deck.

## Operator setup

See [Enable Limitless and BDIF analytics](how-to/enable-bdif.md) for environment variables and CLI commands. This page records the data and model contract; it does not repeat enablement steps.
