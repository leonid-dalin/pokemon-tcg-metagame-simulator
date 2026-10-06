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

The default database is tracked with Git LFS. Install Git LFS before cloning, then run `git lfs pull` from the repository root. Without LFS, `data/limitless.db` is a pointer file and SQLite cannot open it.

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

## Best-60

Best-60 builds one list per archetype from that archetype's stored legal 60-card lists. It does not use the card model above, so it still reports when that model cannot be identified.

1. The consensus list is the 60 most-played card-count slots. A slot is "at least k copies of a card", and its support is the number of lists that reach it. Slots are taken in order of support, ties broken by card name, at most one ACE SPEC. Every archetype with a legal list gets a consensus 60.
2. Each list's outcome is its match record at its event, so every round counts against the field that event really fielded. Player strength is the log-odds of the player's record at their other events, shrunk by ten wins and ten losses, and is controlled in the model because strong players adopt cards first.
3. Every slot played by at least 30 lists, and missed by at least 30, gets a coefficient with a normal prior. The prior width is chosen from 0.02, 0.05, 0.1, 0.2 and 0.5 by five-fold cross-validation over whole events. When player strength alone predicts held-out events better than every width, the consensus is kept with status `consensus only: card counts did not predict held-out results`. Archetypes with fewer than 100 legal lists keep the consensus with status `consensus only: too few lists to score cards`.
4. Every card stays at counts the archetype really plays: a count is allowed when at least 5% of its lists, and at least 30, play exactly that many (`BDIF_BEST60_LEVEL_SHARE`), or when it is the consensus count. Basic Energy may take any count. A card played at 0 or 4 and rarely in between, such as Transformation Tome in N's Zoroark, therefore moves from 4 to 0 in one move, and the copies it frees are filled by the best single-copy additions; the whole move is priced together, with the covariance of every slot it touches. A move never takes back an earlier one. Otherwise a swap removes the top copy of one card and adds the next copy of another. The builder applies the largest-gain swap whose chance of helping is at least 70%, and repeats, up to 12 swaps. It never removes the last copy of a Pokémon or of a card another card in the list needs, and never adds a card whose companions (cards in at least 95% of the lists that play it, such as its Basic) are missing. Swaps between 50% and 70% are listed as leaning swaps and not applied.
5. The swaps are applied only when the procedure holds up on events it has not seen. Player strength for each fold comes from the training events only. Swaps chosen on four folds are priced by slot coefficients fitted on the fifth, and the average gain over the five folds must exceed one standard error of that average (`BDIF_BEST60_HELD_OUT_SE`; 0 accepts any positive gain). Archetypes seen at fewer than five events keep the consensus. Otherwise the consensus is kept with status `consensus kept: swaps did not hold up on held-out events`, and the swaps are reported as proposed swaps.

`BDIF_BEST60_LIST_MODE` chooses how far the list may go from what people play. `novel` (the default) may recommend a list nobody has played. `observed` applies a move only while at least 30 played lists (`BDIF_BEST60_SUPPORT_LISTS`) sit within 4 card changes (`BDIF_BEST60_SUPPORT_CHANGES`) of the result. Either way, the report counts the played lists within 2, 4 and 6 changes of the consensus and of the recommended list, and gives the chance that the whole recommended list beats the consensus.

When moves are applied, Best-60 reruns them on 30 resamples of whole events (`BDIF_BEST60_STABILITY_DRAWS`), with the same prior width, count levels and list mode. For each card it changed, the report gives the share of resamples that recommend exactly the same count and the share that move it the same way. A change that appears in fewer than half the resamples rests on a few events.

The report gives the archetype's average match win rate, the rate with the swaps in sample, and the rate with the held-out gain. Quote the held-out rate; the in-sample rate is optimistic because the swaps were chosen on the same data.

Each card also gets its play rate, average copies, and top 50%, top 25% and winner rates with and without it, counted only at events with at least 8, 12 and 2 players. Trends compare play rates in the first and last 21 days of the snapshot. A card is rising or falling when its play rate moves by 15 points or more, and a breakthrough when it is rising, its top 25% rate beats the archetype's, and the slot model does not rate its first copy negative.

All of this is observational. Player strength is measured with noise, so part of a strong player's card choice can still read as card strength.

Legality checks require exactly 60 cards, at most four copies of a regular card, unlimited basic Energy, and at most one ACE SPEC card. The ACE SPEC names are enumerated in `src/ingestion/model.py`. Lists that fail the checks are left out before anything is counted.

## Operator setup

See [Enable Limitless and BDIF analytics](how-to/enable-bdif.md) for environment variables and CLI commands. This page records the data and model contract; it does not repeat enablement steps.
