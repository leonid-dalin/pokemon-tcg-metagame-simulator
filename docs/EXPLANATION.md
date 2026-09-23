# System design

The simulator has one data contract and two main simulation models. The contract keeps matchup values mathematically consistent. The models answer different questions

## Assumptions

The engines treat the supplied matchup probability as the result distribution for a pairing. They do not model player skill, opening hands, individual draws, or mid-event deck changes

Pairings and tournament outcomes remain stochastic. A close matchup can use tie convergence, and the tournament engine can apply the configured drop feature. These are model settings, not claims about why a player drew or conceded a match

## Why two engines exist

Replicator dynamics answers a long-run population question: how does a field change when decks with higher matchup payoff attract more players

Monte Carlo brackets answer a short-run event question: how often does each deck reach the configured cut when a field and tournament structure are fixed

The static solver sits before the Monte Carlo engine. It resolves exact, minimum, and maximum field constraints, normalises the remaining distribution, and returns recommendations with sample support

## Data integrity

The API rejects duplicate deck names, non-square matrices, invalid win rates, non-zero mirror values, and asymmetric matchup pairs. The scraper applies the same mirror and zero-sum rules before writing `ea_input.json`

`MIN_GAMES` filters low-volume archetypes from the ordinary loader. BDIF uses additional evidence thresholds because card-level reports need observed match counts and decklists

## Deployment boundaries

The UI does not run the heavy simulation. It sends a request to FastAPI. Huey stores the job in Redis, the worker runs the solver and Rust-backed tournament loops, and the API returns status or SSE events

Jaeger receives traces from the services through OTLP. Structured logs are emitted by the API, worker, and CLI paths

## Historical notes

Older documents described a local SQLite broker and a different set of CLI switches. The current deployment uses Redis for Huey and progress state. Historical changes remain in `docs/CHANGELOGS.md`; this page describes the current architecture
