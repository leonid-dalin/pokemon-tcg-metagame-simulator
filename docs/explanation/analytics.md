# Analytics and data contracts

The simulator consumes matchup evidence and returns modelled tournament outcomes. The two should not be read as the same measurement

## Matchup matrix

Each archetype has a row and column in the input matrix. The API and scraper require:

- Unique archetype names
- A square matrix
- Win rates between `0.0` and `1.0`
- Mirror matches equal to exactly `0.5`
- Opposing cells that sum to `1.0` within `1e-9`

The loader filters archetypes below `MIN_GAMES`, which is 100 by default. It keeps matchup details when available so posterior analytics can use observed match counts

## Replicator dynamics

Replicator mode applies matchup payoffs to a population distribution. Selection pressure controls how strongly the population moves towards higher-payoff decks. Mutation keeps a small share available for reintroduction, and noise adds Gaussian variation to generation payoffs

A run reports convergence metrics, active and inactive decks, tier assignments, matchup cycles, and deck similarity. These outputs describe the supplied matrix and assumptions, not player skill distributions or individual card choices

## Tournament simulation

Tournament mode uses the matchup matrix as the result probability for each pairing. Pairings are stochastic, but the engine does not simulate hands, draws, or player skill. Championship Series mode uses the player-count structure selected by `get_variant_5_structure`

The static predictor first resolves the requested field constraints, then produces expected win rate, confidence, sample support, meta share, power score, frequency score, and base meta score for each recommendation

## BDIF card model

The BDIF path is observational and opt-in. It aggregates Limitless decklists and matchup rows, then fits a per-player card model. Archetypes use reference coding, the model has no intercept, and the coefficients describe within-archetype card associations after accounting for the observed player-level outcomes

The model selects cards with at least `BDIF_CARD_MIN_PLAYERS` appearances, which is 50 by default, and with within-archetype presence between `BDIF_CARD_MIN_WITHIN_RATE` and `BDIF_CARD_MAX_WITHIN_RATE`, which are 0.05 and 0.95. Rank guards prevent under-supported inputs from being treated as identified. The separation guard rejects a fit when any absolute logit coefficient exceeds `BDIF_CARD_MAX_ABS_LOGIT`, which is 5.0

The panel selects decks with at least `BDIF_PANEL_SHARE_THRESHOLD` empirical share, which is 0.03, and caps the result at `BDIF_PANEL_MAX_DECKS`, which is 10. Sparse evidence returns an insufficient-data result rather than a fabricated recommendation

The card fit uses an exact zero-sum outcome construction. The A1 report is an archetype-mean approximation: it estimates the expected outcome for the archetype's observed card mix rather than a causal effect for changing one card in isolation. Card inclusion, coefficients, and H1 results remain observational associations

## Posterior intervals

Posterior matchup draws use a maximum budget of `BDIF_POSTERIOR_DRAWS`, 200 matrices, and at least `BDIF_MIN_ITERATIONS_PER_DRAW`, 100 tournament iterations per draw. Intervals are reported only when at least `BDIF_MIN_INTERVAL_DRAWS`, 50 draws, are available. Otherwise the report records `interval_status` as `too few posterior draws`; the UI can show the Monte Carlo standard error as an approximation for the player view. Monte Carlo standard error is not a posterior interval

## Field posterior

The field posterior samples matchup probabilities and estimates each deck's expected win rate against the predicted field. It also reports `best_pick_probability`, the proportion of posterior draws in which that deck has the highest expected field win rate. Field-posterior sampling uses `BDIF_FIELD_POSTERIOR_DRAWS`, 2,000 draws by default

## Legacy Limitless stores

`LimitlessStore` construction is read-only. Reader methods call a guarded preparation step that adds the `deck_name` column when an older database lacks it and backfills canonical names before querying

H1 observation parsing treats a legacy JSON `"null"` decklist as an empty dictionary. New missing decklists are stored as SQL `NULL`

Card-model output is observational. A card inclusion rate or fitted coefficient does not establish that adding a card causes a win-rate change