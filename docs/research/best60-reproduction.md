# Reproduce the Best-60 research run

## Recorded computation

The completed run is `run-20261025T223336Z-a15a1e44`, with scientific status `inconclusive`

- Frozen implementation: `9bbc9c120a37e679fd7da698cb814cf89198e8a4`
- Branch: `feat/best60-research`
- Protocol: `docs/research/best60-protocol-execution.json`
- Protocol lock: `9c31452bd5a3d299ecbce0d619d048b634afb3339be4734d237b30669ecdba59`
- SQLite SHA-256: `170afff7528e6a38c77100c0d20212b601e0f845d57c9236abc50edde7e5cd59`
- Python 3.12.14; NumPy 1.26.4; SciPy 1.16.3; pytest 9.1.1; tcg-engine 0.1.0
- Windows 11, 32 logical CPUs, `OPENBLAS_NUM_THREADS=1`

The recorded output directory is `output/card-value-research/run-20261025T223336Z-a15a1e44` within this checkout

`report.json`, `state.json`, `protocol-lock.json`, `validation.json`, `evaluation.json`, `simulation.json`, the per-pair prediction file, fitted-model caches and bootstrap checkpoint retain the detailed evidence locally. The committed run summary contains aggregate results; player-level source data and fitted caches stay outside Git

The documentation commit follows the frozen implementation. The runner includes HEAD and source-file bytes in its code fingerprint: a completed historical run must be read under its recorded code state, or inspected directly as JSON. A fresh audit on the documentation-only HEAD creates a new run with the same package implementation

## Start a fresh run on this machine

The project environment below is already installed and was used for the recorded computation. A new environment needs the project's requirements; the full legacy test suite also uses its Rust extension

```powershell
$project = 'C:/path/to/pokemon-tcg-metagame-simulator'
$P = Join-Path $project '.venv/Scripts/python.exe'
$database = 'C:/path/to/limitless.db'
$config = Join-Path $project 'docs/research/best60-protocol-execution.json'
$output = Join-Path $project 'output/card-value-research'
$logs = Join-Path $output 'replay-logs'
Set-Location -LiteralPath $project
$env:PYTHONPATH = $project
$env:OPENBLAS_NUM_THREADS = '1'
$env:OTEL_SDK_DISABLED = 'true'
if ((Get-FileHash -LiteralPath $database -Algorithm SHA256).Hash.ToLowerInvariant() -ne '170afff7528e6a38c77100c0d20212b601e0f845d57c9236abc50edde7e5cd59') {
    throw 'The database differs from the recorded source'
}
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$common = @('--database', $database, '--output', $output, '--config', $config, '--archetype-id', 'kangaskhan-bouffalant')
$audit = & $P -m src.ingestion.best60_research @common --stage audit --max-seconds 600 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Audit failed' }
$runId = $audit.run_id
foreach ($stage in @('validate', 'lock', 'evaluate', 'simulate', 'report')) {
    $budget = if ($stage -in @('validate', 'evaluate')) { 1800 } else { 600 }
    & $P -m src.ingestion.best60_research @common --stage $stage --resume $runId --max-seconds $budget *> (Join-Path $logs "$stage.log")
    if ($LASTEXITCODE -ne 0) { throw "Stage $stage needs attention; read its log" }
}
$run = Join-Path $output $runId
Get-Content -LiteralPath (Join-Path $run 'state.json')
```

The invocation uses the training-derived field, with no field-file override or explicit as-of override. It preserves the production arguments exercised by the research runner

Each audit creates a unique run ID. Source and WAL bytes, configuration, split dates, selection-window metadata and code fingerprints must match when resuming. Runtime-limited candidate availability can differ across hardware or machine load; preserve the recorded checkpoints when examining this particular run

Exit 0 includes accounted-for scientific `inconclusive` and `not_identified` results. Exit 2 indicates an argument or protocol mismatch; 3 indicates source/data failure; 4 indicates incomplete or budget-limited work. A budget-limited stage can be resumed with the same run/config and a larger stage budget. A cached 90-second candidate timeout remains an excluded candidate in this frozen profile

The full proposed rank/seed/penalty grid remains in `best60-protocol.json`. It is a different study configuration and requires a new audit and lock

## Repeat the training-only diagnostics

Run these after the fresh audit, using its `$run` path. They verify the audited source snapshot and package-file hashes, and exclude all final-test events from extraction

```powershell
& $P docs/research/reproduce_lumiose_information.py --database $database --run $run
& $P docs/research/reproduce_embargo_retention.py --database $database --run $run
```

The Lumiose script measures local weighted residual information at probability 0.5 after nuisance, event and supported main-effect adjustment. It counts no prior precision and does not estimate dependence inflation or empirical power. The retention script compares seven- and fourteen-day training populations; it performs no predictive refits

The primary run's nominal splits contain 633 development, 208 validation and 208 test events. Selection uses 127 validation events dated strictly before 25 September 2026 UTC, excluding 81 events in the pre-test embargo. Final testing covers 2 to 19 October, or 18 UTC date blocks

## Cache provenance

Eighty-four validation fits were imported from the pre-test run `run-20261025T221627Z-6fecbb9f` after an evaluation-only correction matched interaction main-effect priors to the additive refit at the final training size. The import compared source/snapshot, config, split and field identities, all other package files, and the module AST outside `_evaluate_benchmark`. `validation-cache-provenance.json` records the checkpoint hash and checks. The fresh validation stage computed the remaining fits and rebuilt its own selection/lock bindings

A fresh replay can omit that cache import and compute all validation fits. All 98 candidate attempts were accounted for in the recorded study: 96 converged and the pooled pairs/FM candidates reached their 90-second budgets

## Verification commands

```powershell
& $P -m pytest -q
& $P tools/qa-harness/test_inventory.py --check
& $P tools/qa-harness/mutation_inventory.py --check
& $P tools/qa-harness/test_quality_audit.py --tests tests --package src
& $P tools/qa-harness/dead_symbol_scan.py --src src --tests tests
& $P tools/qa-harness/interval_sanity.py --spec tools/qa-harness/interval_spec_card_model.py --draws 300
& $P tools/qa-harness/interval_sanity.py --spec tools/qa-harness/interval_spec_h1.py --draws 400
& $P tools/qa-harness/cluster_se_check.py --replicates 200
& $P tools/qa-harness/flag_matrix.py --spec tools/qa-harness/flag_spec_bdif.py
& $P tools/qa-harness/contract_probe.py --spec tools/qa-harness/contract_spec_bdif.py
foreach ($matrix in @('card-model-statistics', 'best60-selection', 'minmatches', 'report-contracts')) {
    & $P tools/qa-harness/mutation_sweep.py --repo . --mutations "tools/qa-harness/mutations.$matrix.json" --pythonpath . --workers 2
    if ($LASTEXITCODE -ne 0) { throw "Mutation matrix $matrix failed" }
}
```

Mutation sweeps require a clean committed checkout and create temporary worktrees. This Windows execution used a writable TEMP directory and subprocess-capable terminal permissions for the full suite and mutation harness. The recorded verification summary reports 679 tests passed, 103 mutants caught, zero bad anchors, no interval-check disagreements and no blocking test-quality findings
