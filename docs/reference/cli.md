# CLI reference

The entry point is `python -m src.ui.cli`. Argument parsing lives in `src/ui/cli_args.py`

## Input and output

| Flag | Default | Meaning |
| --- | --- | --- |
| `-i`, `--input` | `data/input/ea_input.json` | JSON matchup matrix |
| `-o`, `--output` | `output/` | Result directory root |
| `-m`, `--min-games` | `100` | Minimum match count for an archetype in replicator, tournament, and batch modes |

The input path must exist. The output root is created when needed

## Simulation mode

| Flag | Default | Values |
| --- | --- | --- |
| `-M`, `--mode` | `replicator` | `replicator`, `tournament` |
| `-g`, `--gens` | `100000` | Maximum generations or epochs |
| `-s`, `--seed` | `1312` | Random seed |
| `--tournament-style` | `pure_swiss` | `pure_swiss`, `championship_series` |

## Replicator options

| Flag | Default | Meaning |
| --- | --- | --- |
| `-e`, `--extinction-threshold` | `1e-6` | Frequency below which a deck is inactive |
| `-N`, `--noise` | `0.0` | Gaussian payoff noise scale |
| `-S`, `--stability-threshold` | `5e-5` | Stability delta threshold |
| `-W`, `--convergence-window` | `50` | Consecutive stable generations |
| `-X`, `--max-inactive-gens` | `1000` | Inactive generations before culling |
| `--mutation-rate` | `1e-4` | Reintroduction frequency floor |
| `--selection-pressure` | `1` | Migration pressure towards winning decks |

## Tournament options

| Flag | Default | Meaning |
| --- | --- | --- |
| `-T`, `--tournament-size` | `32` | Pilots per tournament iteration |
| `--tournaments-per-gen` | `16` | Tournaments used for a generation payoff |
| `-r`, `--rounds` | `5` | Swiss rounds outside Championship Series mode |
| `-B`, `--use-bayesian` | `True` | Sample matchup rates from Beta distributions |
| `--no-multiproc` | enabled | Force single-process tournament execution |

## Prediction and batch options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--predict` | `False` | Run the static tournament solver |
| `-P`, `--players` | `32` | Expected event field size |
| `--meta` | empty | Comma-separated `deck:share` constraints |
| `-b`, `--batch` | `False` | Run several experiments |
| `-c`, `--batch-config` | empty | Batch JSON path |
| `--no-plot` | `False` | Skip interactive HTML plots |
| `-C`, `--cluster` | `False` | Run matchup-profile clustering |
| `-l`, `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

The parser requires `--batch-config` when `--batch` is present. It rejects extinction thresholds outside `0.0` to `1.0` and negative noise

The prediction path always loads with the module-level `MIN_GAMES` value in `src/ui/cli.py`, so `-m` and `--min-games` have no effect when `--predict` is present

## Output files

A single run writes a timestamped directory containing `simulation_trace.jsonl`, `metagame_history_full.csv`, `ess_equilibrium.csv`, `final_tiers.json`, and `deck_similarity.json`. Unless `--no-plot` is present, it also writes `metagame_evolution.html`, `matchup_heatmap.html`, and `matchup_network.html`
