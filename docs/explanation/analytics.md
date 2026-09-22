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

The BDIF path is observational and opt-in. It aggregates Limitless decklists and matchup rows, fits card covariates, and builds legality-aware Best-60 recommendations from observed skeletons

The model applies three evidence checks:

- `BDIF_MIN_MATCHES` requires 1,000 total matchup matches for a deck
- `BDIF_COVERAGE_RATIO` requires 60% of evidence to meet pair coverage
- `BDIF_PAIR_MIN_GAMES` marks a pair reliable at 250 matches

The panel selects decks with at least 3% empirical share and caps the result at 10 decks. Sparse evidence returns an insufficient-data result rather than a fabricated recommendation

## Legacy Limitless stores

`LimitlessStore` construction is read-only. Reader methods call a guarded preparation step that adds the `deck_name` column when an older database lacks it and backfills canonical names before querying

H1 observation parsing treats a legacy JSON `"null"` decklist as an empty dictionary. New missing decklists are stored as SQL `NULL`

Card-model output is observational. A card inclusion rate or fitted coefficient does not establish that adding a card causes a win-rate change
