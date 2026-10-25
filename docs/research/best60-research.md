# Offline Best-60 research

The `src.ingestion.best60_research` package is an offline sidecar for evaluating card interactions, constrained list selection and model-conditional uncertainty against an additive reference

Production Best-60 behaviour and defaults remain unchanged

## Protocols

- `best60-protocol.json` preserves the full proposed search
- `best60-protocol-execution.json` specifies a smaller CPU feasibility search frozen before locked test access
- Candidate timeouts and non-converged models remain explicit failures rather than successful fits
- The execution profile does not exhaust the full estimator search

## Run

Use the project Python environment with its normal requirements installed

```powershell
$database = 'C:/path/to/limitless.db'
$output = 'output/card-value-research'
$config = 'docs/research/best60-protocol-execution.json'
python -m src.ingestion.best60_research --database $database --output $output --config $config --stage audit --archetype-id kangaskhan-bouffalant --max-seconds 600
```

Save the returned `run_id`, then run `validate`, `lock`, `evaluate`, `simulate` and `report` in that order, adding `--resume RUN_ID` to each command

Use `--max-seconds 1800` for fitting stages; an incomplete stage can be resumed with the same command

Optional `--as-of` freezes a source-time cutoff and `--field-file` supplies a JSON map of opponent archetype IDs to non-negative weights

Source data, code revision, configuration, split identities, validation selection and optional overrides are fingerprinted; changing them requires a fresh audit

The database is opened read-only; run output must be external to the repository or Git-ignored

## Interpretation

- H1 compares held-out log loss and observed-list ranking; fewer than 20 independent UTC date blocks leaves the acceptance decision inconclusive
- H2 distinguishes training-design information and local minimum detectable effects from empirical simulation power; a simulation not run is never reported as measured power
- H3 requires defensible policy identification and support; absent exact-list logging propensities, the report says `not_identified`
- H4 fixes the recommended counts during date-block refits; its interval is model-conditional, with predictive and latent calibration explicitly unmeasured until separately established
- Simulation preflight can stop structurally inconclusive designs before fitting, with requested and completed replicate counts recorded
- No new games, in-game policy model or opponent card-list features are introduced

Exit codes are 0 for an accounted-for scientific result, 2 for arguments or protocol mismatch, 3 for source or data failures, and 4 for budget-limited or incomplete execution

## Verification

The test suite includes legacy parity, poisoned-history leakage checks, mirrored-pair weights, sparse FM derivatives, known null and planted interaction cases, fixed-list uncertainty, interrupted/resumed checkpoints, protocol tamper detection and a synthetic full-stage CLI run

Run `python -m pytest -q` and the existing scripts under `tools/qa-harness` to repeat the checks
