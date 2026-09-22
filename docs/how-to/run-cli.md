# Run the CLI

Run CLI commands from the repository root. The input path must exist before argument parsing completes

## Replicator mode

```bash
python -m src.ui.cli \
  --input data/input/ea_input.json \
  --mode replicator \
  --gens 10000 \
  --seed 1312 \
  --output output
```

The command writes a timestamped result directory below `output/`. Add `--cluster` to run matchup-profile clustering, or `--no-plot` to skip interactive HTML files

## Tournament mode

```bash
python -m src.ui.cli \
  --input data/input/ea_input.json \
  --mode tournament \
  --tournament-style pure_swiss \
  --tournament-size 32 \
  --rounds 5
```

Use `--tournament-style championship_series` to use the Variant 5 structure selected from the player count

## Static prediction mode

```bash
python -m src.ui.cli \
  --input data/input/ea_input.json \
  --predict \
  --players 512 \
  --meta "Crustle:0.10,Joltik Box:0.15"
```

Prediction mode bypasses replicator evolution and uses the static tournament solver. The `--meta` value is a comma-separated list of `deck:share` pairs

## Batch mode

Create a JSON file with an `experiments` list. Each item may override keys accepted by `Args` and may include an `experiment_id`:

```json
{
  "experiments": [
    {"experiment_id": "baseline", "gens": 1000, "seed": 1312},
    {"experiment_id": "noisy", "gens": 1000, "noise": 0.02}
  ]
}
```

Run it with:

```bash
python -m src.ui.cli \
  --input data/input/ea_input.json \
  --batch \
  --batch-config examples/batch.json
```

The batch runner writes `batch_summary.json` below the output directory

## Common validation failures

- `Input file not found`: fix `--input` or run the command from the repository root
- `--batch mode requires --batch-config`: provide a JSON batch file
- `--extinction-threshold must be between 0.0 and 1.0`: use a value in that range
- `--noise scale cannot be negative`: use zero or a positive value
